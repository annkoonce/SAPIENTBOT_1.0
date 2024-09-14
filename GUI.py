# gui.py
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys
import threading
from main import run_bot  # import the function to run your bot

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Start Bot")
        button.clicked.connect(self.start_bot)

        self.setCentralWidget(button)

    def start_bot(self):
        threading.Thread(target=run_bot, daemon=True).start()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
