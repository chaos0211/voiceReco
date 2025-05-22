from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
import sys
import torch
from config_ecapa_cnceleb import *


class VoiceprintTrainerGUI:
    def __init__(self):
        self.device = self._select_device()
        self.config = self._load_config()
        self.pretrainer = self._init_pretrainer()
        print(f"当前设备: {self.device}")
        print(f"模型输出路径: {self.config['save_model_path']}")

    def _load_config(self):
        return {
            "data_folder": data_folder,
            "train_annotation": train_annotation,
            "valid_annotation": valid_annotation,
            "sample_rate": sample_rate,
            "embedding_dim": embedding_dim,
            "channels": channels,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "lr_annealing": lr_annealing,
            "max_grad_norm": max_grad_norm,
            "use_augmentation": use_augmentation,
            "output_folder": output_folder,
            "checkpoints_dir": checkpoints_dir,
            "save_model_path": save_model_path,
            "loss_type": loss_type,
            "aam_margin": aam_margin,
            "aam_scale": aam_scale,
            "cosine_threshold": cosine_threshold,
            "use_wandb": use_wandb,
            "device": self.device,
            "max_steps_per_epoch": max_steps_per_epoch,
        }

    def _select_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _init_pretrainer(self):
        return pretrainer

    def run_training(self):
        from training.train import train_model
        print(">>> 启动训练流程")
        train_model(self.config)


# 新增 PyQt5 GUI 类
class VoiceRecoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.trainer = VoiceprintTrainerGUI()

    def initUI(self):
        self.setWindowTitle("Voice Recognition Trainer")
        self.setGeometry(100, 100, 300, 150)

        self.label = QLabel("Voice Recognition System", self)

        self.train_button = QPushButton("Start Training", self)
        self.train_button.clicked.connect(self.start_training)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.train_button)
        self.setLayout(layout)

    def start_training(self):
        self.label.setText("Training started...")
        self.trainer.run_training()
        self.label.setText("Training completed.")


if __name__ == "__main__":
    # app = VoiceprintTrainerGUI()
    # app.run_training()

    app = QApplication(sys.argv)
    mainWin = VoiceRecoApp()
    mainWin.show()
    sys.exit(app.exec_())
