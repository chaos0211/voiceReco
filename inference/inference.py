'''实现功能：
	•	给定音频路径，提取 mel 特征，生成嵌入。
	•	比较两个嵌入的余弦相似度，判断是否为同一人。
	•	或将嵌入输入训练好的分类器输出预测类别
'''

import torch
import torchaudio
import torch.nn.functional as F
import os

def load_model(model_path, config, device):
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    model = ECAPA_TDNN(
        input_size=80,
        lin_neurons=config["embedding_dim"],
        channels=config["channels"],
        kernel_sizes=[5, 3, 3, 1],
        dilations=[1, 2, 3, 1],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def extract_embedding(audio_path, model, config, device):
    waveform, sr = torchaudio.load(audio_path)
    if sr != config["sample_rate"]:
        waveform = torchaudio.functional.resample(waveform, sr, config["sample_rate"])
    # If stereo, convert to mono by averaging channels
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    max_len = config["sample_rate"] * 10
    if waveform.size(0) > max_len:
        waveform = waveform[:max_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.size(0)))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80
    )(waveform).transpose(0, 1)

    mel = mel.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(mel)
    return embedding.cpu()

if __name__ == "__main__":
    import json

    # 加载配置
    with open("config_ecapa_cnceleb.py") as f:
        config = eval(f.read())

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = "results/ecapa_cnceleb/final_model.ckpt"
    audio1 = "audio/a_1.wav"
    audio2 = "audio/a_2.wav"

    # 加载模型
    model = load_model(model_path, config, device)

    # 提取两个音频的嵌入
    emb1 = extract_embedding(audio1, model, config, device).squeeze(1)
    emb2 = extract_embedding(audio2, model, config, device).squeeze(1)

    # 计算余弦相似度
    similarity = F.cosine_similarity(emb1, emb2).item()
    print(f"Cosine Similarity between {audio1} and {audio2}: {similarity:.4f}")

    # 简单分类预测（如果模型是分类器）
    with torch.no_grad():
        logits = model(emb1.unsqueeze(0).to(device))
        predicted_class = torch.argmax(logits, dim=-1).item()
        print(f"Predicted class for {audio1}: {predicted_class}")