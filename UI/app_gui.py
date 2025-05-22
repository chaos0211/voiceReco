

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QMessageBox, QTabWidget, QSpinBox
)
from ../inference.user_db import UserDatabase

class VoiceRecoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("声纹识别系统")
        self.resize(600, 400)

        self.db = UserDatabase()

        tabs = QTabWidget()
        tabs.addTab(self.create_record_tab(), "声音识别")
        tabs.addTab(self.create_train_tab(), "模型训练")

        self.setCentralWidget(tabs)

    def create_record_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")

        self.audio_path_label = QLabel("未选择音频文件")
        audio_btn = QPushButton("选择音频文件")
        audio_btn.clicked.connect(self.select_audio_file)

        submit_btn = QPushButton("提交音频")
        submit_btn.clicked.connect(self.submit_audio)

        layout.addWidget(QLabel("录入声音"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.audio_path_label)
        layout.addWidget(audio_btn)
        layout.addWidget(submit_btn)

        widget.setLayout(layout)
        return widget

    def create_train_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.dataset_path_label = QLabel("未选择数据集文件夹")
        dataset_btn = QPushButton("选择数据集")
        dataset_btn.clicked.connect(self.select_dataset)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(10)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 10000)
        self.steps_spin.setValue(300)

        model_btn = QPushButton("评估模型（选择 .ckpt）")
        model_btn.clicked.connect(self.select_model_for_evaluation)

        layout.addWidget(QLabel("模型训练参数"))
        layout.addWidget(self.dataset_path_label)
        layout.addWidget(dataset_btn)
        layout.addWidget(QLabel("Epochs"))
        layout.addWidget(self.epoch_spin)
        layout.addWidget(QLabel("Batch Size"))
        layout.addWidget(self.batch_size_spin)
        layout.addWidget(QLabel("Max Steps Per Epoch"))
        layout.addWidget(self.steps_spin)
        layout.addWidget(model_btn)

        widget.setLayout(layout)
        return widget

    def select_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "Audio Files (*.wav *.mp3)")
        if path:
            self.audio_path = path
            self.audio_path_label.setText(f"已选择: {path}")

    def submit_audio(self):
        username = self.username_input.text().strip()
        if not username or not hasattr(self, 'audio_path'):
            QMessageBox.warning(self, "警告", "请填写用户名并选择音频文件")
            return

        if self.db.user_exists(username):
            reply = QMessageBox.question(self, "用户已存在", "是否替换已有音频？", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.db.add_or_replace_user(username, self.audio_path)
        QMessageBox.information(self, "成功", "音频已保存")

    def select_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集文件夹")
        if path:
            self.dataset_path_label.setText(f"数据集路径: {path}")
            self.dataset_path = path

    def select_model_for_evaluation(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Checkpoint Files (*.ckpt)")
        if model_path:
            QMessageBox.information(self, "模型选择", f"已选择模型文件: {model_path}")
            # 这里你可以调用模型评估函数并传入model_path和self.dataset_path等参数

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRecoApp()
    window.show()
    sys.exit(app.exec_())