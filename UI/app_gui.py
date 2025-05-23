import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QMessageBox, QTabWidget, QSpinBox, QTextEdit
)
from inference.user_db import UserDatabase
from inference.inference import VoiceRecognizer
import os

class VoiceRecoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("声纹识别系统")
        self.resize(600, 400)

        self.db = UserDatabase()
        self.recognizer = VoiceRecognizer(config_path="config_ecapa_cnceleb.py")

        tabs = QTabWidget()
        tabs.addTab(self.create_voice_entry_tab(), "声纹录入")
        tabs.addTab(self.create_voice_verify_tab(), "声纹识别")
        tabs.addTab(self.create_train_tab(), "模型训练")
        tabs.addTab(self.create_eval_tab(), "模型评估")

        self.setCentralWidget(tabs)

    def create_voice_entry_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")

        self.audio_path_label = QLabel("未选择音频文件")
        audio_btn = QPushButton("选择音频文件")
        audio_btn.clicked.connect(self.select_audio_file)

        self.entry_model_label = QLabel("未选择模型文件")
        entry_model_btn = QPushButton("选择模型文件")
        entry_model_btn.clicked.connect(self.select_entry_model_file)

        submit_btn = QPushButton("提交音频")
        submit_btn.clicked.connect(self.submit_audio)

        layout.addWidget(QLabel("录入声音"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.audio_path_label)
        layout.addWidget(audio_btn)
        layout.addWidget(self.entry_model_label)
        layout.addWidget(entry_model_btn)
        layout.addWidget(submit_btn)

        self.entry_result_label = QLabel("")
        layout.addWidget(self.entry_result_label)

        widget.setLayout(layout)
        return widget

    def create_voice_verify_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.verify_username_input = QLineEdit()
        self.verify_username_input.setPlaceholderText("请输入用户名以验证")

        self.verify_audio_label = QLabel("未选择验证音频")
        verify_audio_btn = QPushButton("选择音频文件")
        verify_audio_btn.clicked.connect(self.select_verify_audio)

        self.verify_model_label = QLabel("未选择模型文件")
        verify_model_btn = QPushButton("选择模型文件")
        verify_model_btn.clicked.connect(self.select_verify_model_file)

        verify_btn = QPushButton("开始识别")
        verify_btn.clicked.connect(self.verify_audio_identity)

        layout.addWidget(QLabel("声音识别"))
        layout.addWidget(self.verify_username_input)
        layout.addWidget(self.verify_audio_label)
        layout.addWidget(verify_audio_btn)
        layout.addWidget(self.verify_model_label)
        layout.addWidget(verify_model_btn)
        layout.addWidget(verify_btn)

        self.verify_result_label = QLabel("")
        layout.addWidget(self.verify_result_label)

        widget.setLayout(layout)
        return widget

    def create_train_tab(self):
        widget = QWidget()
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.dataset_path_label = QLabel("未选择数据集文件夹")
        dataset_btn = QPushButton("选择数据集")
        dataset_btn.clicked.connect(self.select_dataset)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(10)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(16)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(100, 10000)
        self.max_steps_spin.setValue(300)

        layout_label = QLabel("模型训练参数")
        left_layout.addWidget(layout_label)
        left_layout.addWidget(self.dataset_path_label)
        left_layout.addWidget(dataset_btn)
        left_layout.addWidget(QLabel("Epochs"))
        left_layout.addWidget(self.epoch_spin)
        left_layout.addWidget(QLabel("Batch Size"))
        left_layout.addWidget(self.batch_size_spin)
        left_layout.addWidget(QLabel("Max Steps Per Epoch"))
        left_layout.addWidget(self.max_steps_spin)

        # 新增：开始训练按钮
        train_btn = QPushButton("开始训练")
        train_btn.clicked.connect(self.train_model_from_gui)
        left_layout.addWidget(train_btn)

        # 新增：另存为模型按钮
        self.save_model_btn = QPushButton("另存为模型")
        self.save_model_btn.clicked.connect(self.save_trained_model)
        left_layout.addWidget(self.save_model_btn)

        right_layout.addWidget(QLabel("训练日志"))
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        right_layout.addWidget(self.train_log)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        widget.setLayout(layout)
        return widget
    def train_model_from_gui(self):
        try:
            from training.train import train_model
            import config_ecapa_cnceleb as config

            config_dict = config.__dict__
            config_dict["epochs"] = self.epoch_spin.value()
            config_dict["batch_size"] = self.batch_size_spin.value()
            config_dict["max_steps_per_epoch"] = self.max_steps_spin.value()

            import torch
            if torch.backends.mps.is_available():
                config_dict["device"] = "mps"
            elif torch.cuda.is_available():
                config_dict["device"] = "cuda"
            else:
                config_dict["device"] = "cpu"

            self.train_log.clear()
            self.train_log.append(f"开始训练模型...")
            self.train_log.append(f"Epochs: {config_dict['epochs']}")
            self.train_log.append(f"Batch Size: {config_dict['batch_size']}")
            self.train_log.append(f"Max Steps Per Epoch: {config_dict['max_steps_per_epoch']}")

            import io, contextlib
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                train_model(config_dict)
            self.train_log.append(buffer.getvalue())

            QMessageBox.information(self, "训练完成", "✅ 模型训练已完成")
        except Exception as e:
            QMessageBox.critical(self, "训练错误", f"❌ 模型训练失败: {str(e)}")

    def save_trained_model(self):
        try:
            if not os.path.exists("results/ecapa_cnceleb/final_model.ckpt"):
                QMessageBox.warning(self, "警告", "找不到训练好的模型文件")
                return
            save_path, _ = QFileDialog.getSaveFileName(self, "另存为模型", filter="Checkpoint Files (*.ckpt)")
            if save_path and not save_path.endswith(".ckpt"):
                save_path += ".ckpt"
            if save_path:
                import shutil
                shutil.copy("results/ecapa_cnceleb/final_model.ckpt", save_path)
                QMessageBox.information(self, "另存成功", f"模型已保存至: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型另存失败: {str(e)}")

    def create_eval_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.eval_model_label = QLabel("未选择模型文件")
        eval_model_btn = QPushButton("选择模型（.ckpt）")
        eval_model_btn.clicked.connect(self.select_model_for_evaluation)

        layout.addWidget(QLabel("模型评估"))
        layout.addWidget(self.eval_model_label)
        layout.addWidget(eval_model_btn)

        eval_btn = QPushButton("开始模型评估")
        eval_btn.clicked.connect(self.run_model_evaluation)
        layout.addWidget(eval_btn)

        self.eval_result_label = QLabel("")
        layout.addWidget(self.eval_result_label)

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
        if not hasattr(self, 'entry_model_path'):
            QMessageBox.warning(self, "警告", "请选择用于提取特征的模型文件")
            return

        # recognizer = self.recognizer
        try:
            self.recognizer.enroll_user(username, self.audio_path, self.entry_model_path)
            self.entry_result_label.setText(f"用户 [{username}] 已成功录制声纹")
        except Exception as e:
            self.entry_result_label.setText(f"录音失败: {str(e)}")

    def select_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集文件夹")
        if path:
            self.dataset_path_label.setText(f"数据集路径: {path}")
            self.dataset_path = path
            self.train_log.append(f"已选择数据集路径: {path}")

    def select_model_for_evaluation(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Checkpoint Files (*.ckpt)")
        if model_path:
            self.eval_model_label.setText(f"模型路径: {model_path}")
            self.eval_model_path = model_path
            # 可调用评估逻辑

    def select_verify_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "Audio Files (*.wav *.mp3)")
        if path:
            self.verify_audio_path = path
            self.verify_audio_label.setText(f"已选择: {path}")

    def verify_audio_identity(self):
        username = self.verify_username_input.text().strip()
        if not username or not hasattr(self, 'verify_audio_path'):
            QMessageBox.warning(self, "警告", "请填写用户名并选择音频文件")
            return
        if not hasattr(self, 'verify_model_path'):
            QMessageBox.warning(self, "警告", "请选择用于识别的模型文件")
            return

        # recognizer = self.recognizer
        try:
            similarity, passed = self.recognizer.verify_user(username, self.verify_audio_path, self.verify_model_path)
            if passed:
                self.verify_result_label.setText(f"用户 [{username}] 模型识别计算结果为 : {similarity:.4f}，用户识别成功")
            else:
                self.verify_result_label.setText(f"用户 [{username}] 模型识别计算结果为 : {similarity:.4f}，用户识别失败")
        except FileNotFoundError:
            self.verify_result_label.setText(f"该用户 [{username}] 未录入声音")
        except Exception as e:
            self.verify_result_label.setText(f"识别失败: {str(e)}")

    def select_entry_model_file(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Checkpoint Files (*.ckpt)")
        if model_path:
            self.entry_model_path = model_path
            self.entry_model_label.setText(f"模型路径: {model_path}")

    def select_verify_model_file(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Checkpoint Files (*.ckpt)")
        if model_path:
            self.verify_model_path = model_path
            self.verify_model_label.setText(f"模型路径: {model_path}")

    def run_model_evaluation(self):
        if not hasattr(self, 'eval_model_path'):
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return
        try:
            from training.evaluate import ModelEvaluator

            evaluator = ModelEvaluator(model_path=self.eval_model_path)
            evaluator.load_model()
            evaluator.load_data()
            acc, msg = evaluator.evaluate()
            if acc is None:
                self.eval_result_label.setText("模型评估失败，未返回结果")
                return
            self.eval_result_label.setText(msg)
        except Exception as e:
            self.eval_result_label.setText(f"模型评估错误: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRecoApp()
    window.show()
    sys.exit(app.exec_())