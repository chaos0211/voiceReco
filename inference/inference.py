'''实现功能：
	•	给定音频路径，提取 mel 特征，生成嵌入。
	•	比较两个嵌入的余弦相似度，判断是否为同一人。
	•	或将嵌入输入训练好的分类器输出预测类别
'''

import torch
import torchaudio

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
        embedding = model(mel).squeeze(1)
    return embedding.cpu()