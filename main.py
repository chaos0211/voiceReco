from PyQt5.QtWidgets import QApplication
import sys
import torch
from config_ecapa_cnceleb import *
from UI.app_gui import VoiceRecoApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = VoiceRecoApp()
    mainWin.show()
    sys.exit(app.exec_())
