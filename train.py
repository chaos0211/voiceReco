# Ensure torchaudio uses the Python-level soundfile backend for cross-platform compatibility
import torchaudio
torchaudio.set_audio_backend("soundfile")
# config_ecapa_cnceleb.py
# Configuration for training ECAPA-TDNN model on CN-Celeb using SpeechBrain

from speechbrain.utils.parameter_transfer import Pretrainer

import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Augmentation (optional, CN-Celeb is diverse enough)
use_augmentation = False

# Save & checkpoint
output_folder = "results/ecapa_cnceleb"
checkpoints_dir = f"{output_folder}/checkpoints"
save_model_path = f"{output_folder}/final_model.ckpt"

# Loss and backend classifier
loss_type = "AAMSoftmax"  # or "softmax"
aam_margin = 0.2
aam_scale = 30

# Pretrainer (optional)
pretrainer = Pretrainer(
    collect_in="pretrained_models/ecapa_voxceleb",
    loadables={},
    paths={},
)

# Evaluation parameters
cosine_threshold = 0.65

# Logging
use_wandb = False  # set to True if you want to use Weights & Biases


# === Train function for external import ===
def train_model(config):
    import os
    import pandas as pd
    import torch
    import torchaudio
    from torch.utils.data import DataLoader, Dataset
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.dataio.encoder import CategoricalEncoder

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80
    )

    class CNCELEBDataset(Dataset):
        def __init__(self, csv_file, label_encoder):
            self.data = pd.read_csv(csv_file)
            self.label_encoder = label_encoder

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            print(">>>注意这里 Using CNCELEBDataset for waveform loading")
            path = self.data.iloc[idx]['wav']
            label = self.data.iloc[idx]['spk_id']
            waveform, sr = torchaudio.load(path)
            if sr != config["sample_rate"]:
                waveform = torchaudio.functional.resample(waveform, sr, config["sample_rate"])
            waveform = waveform.squeeze(0)
            waveform = mel_transform(waveform)  # [F, T]
            print("这里的Mel shape:", waveform.shape)
            waveform = waveform.transpose(0, 1)  # [T, F]
            label_idx = self.label_encoder.encode_label(label)
            return waveform, label_idx

    # 读取 CSV 和准备标签编码器
    df = pd.read_csv(config["train_annotation"])
    label_encoder = CategoricalEncoder()
    label_encoder.update_from_iterable(df["spk_id"])

    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        waveforms = pad_sequence(waveforms, batch_first=True)
        labels = torch.tensor(labels).long()
        return waveforms, labels

    train_dataset = CNCELEBDataset(config["train_annotation"], label_encoder)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    model = ECAPA_TDNN(
        input_size=80,
        lin_neurons=config["embedding_dim"],
        channels=config["channels"],
        kernel_sizes=[5, 3, 3, 1],
        dilations=[1, 2, 3, 1],
    ).to(config["device"])

    num_classes = len(label_encoder)
    classifier = torch.nn.Linear(config["embedding_dim"], num_classes).to(config["device"])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    model.train()
    classifier.train()
    for epoch in range(config["epochs"]):
        total_loss = 0
        for batch in train_loader:
            feats, labels = batch
            feats, labels = feats.to(config["device"]), labels.to(config["device"]).long()

            # print(f"feats shape: {feats.shape}")  # 应该是 [B, T, 80]
            # embeddings = model(feats)
            # print(f"embeddings shape: {embeddings.shape}")
            # logits = classifier(embeddings)
            # print(f"logits shape: {logits.shape}")

            feats = feats.transpose(1, 2)  # [B, T, 80] → [B, 80, T]
            embeddings = model(feats)  # [B, 192]
            logits = classifier(embeddings)
            loss = criterion(input=logits, target=labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), config["max_grad_norm"])
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs(os.path.dirname(config["save_model_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["save_model_path"])
    print(f"✅ 模型已保存至: {config['save_model_path']}")