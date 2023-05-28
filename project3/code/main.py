import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QLabel
from PyQt5 import QtCore, QtGui

import MainWindow
if __name__ == '__main__':
    app = QApplication(sys.argv)
    work = QMainWindow()
    ui = MainWindow.Ui_MainWindow()
    ui.setupUi(work)
    work.show()
    sys.exit(app.exec_())