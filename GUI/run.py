import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFrame
from PyQt5.QtGui import QPalette, QColor

import qdarktheme

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Autoposture")
        self.setGeometry(100, 100, 500, 600)  # (x, y, width, height)

        # Create a central widget and set it
        video_display_canvas = QWidget()
        self.setCentralWidget(video_display_canvas)

        # Create a vertical layout for the central widget
        layout = QVBoxLayout()
        video_display_canvas.setLayout(layout)

        # Create a frame for video display
        video_frame = QFrame(self)
        # Set the background color to white
        video_frame.setStyleSheet("background-color: white;")
        layout.addWidget(video_frame)

        # Create a button to start
        self.start_button = QPushButton("START", self)
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_camera)

    def start_camera(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

