import csv

import torch
import torchaudio
import os
import torch.nn.functional as F

class VoiceRecognizer:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def enroll_user(self, username: str, audio_path: str, model_path: str):
        try:
            model = self._load_model(model_path)
        except Exception:
            return f"用户模型识别失败，请重新选择模型文件"
        try:
            embedding = self._extract_embedding(audio_path, model)
            embedding_np = embedding.squeeze(0).squeeze(0).numpy()
            embedding_str = ','.join(map(str, embedding_np.tolist()))
            file_exists = os.path.isfile('voice_users.csv')
            with open('voice_users.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['username', 'embedding'])
                writer.writerow([username, embedding_str])
            return f"用户 {username} 已成功录制声纹"
        except Exception:
            return f"用户模型识别失败，请重新选择模型文件"

    def verify_user(self, username: str, audio_path: str, model_path: str):
        try:
            model = self._load_model(model_path)
            embedding = self._extract_embedding(audio_path, model).squeeze(0).squeeze(0)
        except Exception:
            return None, "用户模型识别失败，请重新选择模型文件"

        if not os.path.isfile('voice_users.csv'):
            return None, "用户模型识别失败，请重新选择模型文件"

        with open('voice_users.csv', mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == username:
                    stored_embedding = torch.tensor([float(x) for x in row['embedding'].split(',')])
                    similarity = F.cosine_similarity(embedding, stored_embedding, dim=0).item()
                    if similarity > 0.90:
                        return similarity, True
                    else:
                        return similarity, False

        raise FileNotFoundError(f"用户 {username} 未找到")

    def _load_config(self, config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = {
            "sample_rate": config_module.sample_rate,
            "embedding_dim": config_module.embedding_dim,
            "channels": config_module.channels
        }
        return config

    def _load_model(self, model_path):
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        model = ECAPA_TDNN(
            input_size=80,
            lin_neurons=self.config["embedding_dim"],
            channels=self.config["channels"],
            kernel_sizes=[5, 3, 3, 1],
            dilations=[1, 2, 3, 1],
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _extract_embedding(self, audio_path, model):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.config["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sr, self.config["sample_rate"])
        # If stereo, convert to mono by averaging channels
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        max_len = self.config["sample_rate"] * 10
        if waveform.size(0) > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.size(0)))

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config["sample_rate"],
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )(waveform).transpose(0, 1)

        mel = mel.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = model(mel)
        return embedding.cpu()

if __name__ == '__main__':
    config_path = "config_ecapa_cnceleb.py"
    v = VoiceRecognizer(config_path)
    v.enroll_user("3号玩家", "audio/b_1.wav", "results/ecapa_cnceleb/final_model.ckpt")
    # v.verify_user("2号玩家", "audio/b_2.wav", "results/ecapa_cnceleb/final_model.ckpt")
