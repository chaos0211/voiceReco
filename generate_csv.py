import os
import csv
import random
'''生成清单文件脚本'''

data_root = "data/cnceleb/wav"
output_dir = "data/cnceleb"
train_csv = os.path.join(output_dir, "train.csv")
valid_csv = os.path.join(output_dir, "valid.csv")

entries = []

for spk_id in os.listdir(data_root):
    spk_path = os.path.join(data_root, spk_id)
    if not os.path.isdir(spk_path):
        continue
    for fname in os.listdir(spk_path):
        if fname.endswith(".flac") or fname.endswith(".wav"):
            wav_path = os.path.join(spk_path, fname)
            entries.append({
                "wav": wav_path,
                "spk_id": spk_id
            })

# 随机打乱并划分验证集（10%）
random.shuffle(entries)
split = int(0.9 * len(entries))
train_entries = entries[:split]
valid_entries = entries[split:]

# 保存为 CSV
os.makedirs(output_dir, exist_ok=True)

def save_csv(filepath, data):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav", "spk_id"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

save_csv(train_csv, train_entries)
save_csv(valid_csv, valid_entries)

print(f"✅ 生成完成: {train_csv} ({len(train_entries)}条), {valid_csv} ({len(valid_entries)}条)")