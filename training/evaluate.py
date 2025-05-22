import torch
from torch.utils.data import DataLoader, TensorDataset
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.nn.functional import cosine_similarity


class ModelEvaluator:
    def __init__(self, model_path):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_path = model_path
        self.batch_size = 16
        self.model = None
        self.val_loader = None

        self.valid_csv = "data/cnceleb/valid.csv"
        self.wav_dir = "data/cnceleb/wav"

    def load_model(self):
        self.model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024],
            kernel_sizes=[5, 3, 3, 1],
            dilations=[1, 2, 3, 1],
            attention_channels=128,
            lin_neurons=192,
            se_channels=128,
            res2net_scale=8
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def load_data(self):
        df = pd.read_csv(self.valid_csv)
        feats = []
        labels = []

        max_audio_len = 16000 * 7
        for _, row in df.head(300).iterrows():
            audio_path = row['wav'] if row['wav'].startswith(self.wav_dir) else f"{self.wav_dir}/{row['wav']}"
            signal, sr = torchaudio.load(audio_path)
            # Truncate audio to max 7 seconds (112000 frames at 16kHz)
            if signal.shape[1] > max_audio_len:
                signal = signal[:, :max_audio_len]
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=80
            )(signal).squeeze(0).transpose(0, 1)
            feats.append(mel_spec)
            labels.append(row['spk_id'])

        max_len = max([f.shape[0] for f in feats])
        feats_padded = torch.zeros(len(feats), max_len, 80)
        for i, f in enumerate(feats):
            feats_padded[i, :f.shape[0], :] = f
        unique_labels = list(sorted(set(labels)))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        labels = torch.tensor([label_to_index[label] for label in labels])
        dataset = TensorDataset(feats_padded, labels)
        self.val_loader = DataLoader(dataset, batch_size=self.batch_size)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        classifier = nn.Linear(192, len(set(label.item() for _, label in self.val_loader.dataset)))
        classifier.to(self.device)
        classifier.eval()

        with torch.no_grad():
            for feats, labels in self.val_loader:
                feats = feats.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(feats).squeeze(1)
                logits = classifier(embeddings)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total > 0 else 0
        print(f"✅ 模型评估完成，分类准确率: {acc:.4f}")

if __name__ == '__main__':
    evaluator = ModelEvaluator(model_path="results/ecapa_cnceleb/final_model.ckpt")
    evaluator.load_model()
    evaluator.load_data()
    evaluator.evaluate()