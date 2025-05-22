from speechbrain.utils.parameter_transfer import Pretrainer

# -----------------------
# 数据相关配置
# -----------------------
data_folder = "data/cnceleb/wav"  # 原始数据路径
train_annotation = "data/cnceleb/train.csv"  # 训练集列表
valid_annotation = "data/cnceleb/valid.csv"  # 验证集列表
sample_rate = 16000  # 采样率

# -----------------------
# 模型结构参数
# -----------------------
embedding_dim = 192
channels = [1024, 1024, 1024, 1024]  # 每层卷积通道数

# -----------------------
# 训练参数
# -----------------------
epochs = 10
batch_size = 16
lr = 0.001
momentum = 0.9
weight_decay = 1e-5
lr_annealing = True
max_grad_norm = 5.0
max_steps_per_epoch= 500

# -----------------------
# 数据增强（可选）
# -----------------------
use_augmentation = False

# -----------------------
# 输出 & 模型保存
# -----------------------
output_folder = "results/ecapa_cnceleb"
checkpoints_dir = f"{output_folder}/checkpoints"
save_model_path = f"{output_folder}/final_model.ckpt"

# -----------------------
# 损失函数设置
# -----------------------
loss_type = "AAMSoftmax"  # 可选："softmax" 或 "AAMSoftmax"
aam_margin = 0.2
aam_scale = 30

# -----------------------
# 预训练模型加载器（可选）
# -----------------------
pretrainer = Pretrainer(
    collect_in="pretrained_models/ecapa_voxceleb",
    loadables={},
    paths={},
)

# -----------------------
# 验证设置
# -----------------------
cosine_threshold = 0.65

# -----------------------
# 日志与监控
# -----------------------
use_wandb = False  # 设置为 True 以启用 Weights & Biases 日志