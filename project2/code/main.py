import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import uiresolution

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Resolution = QMainWindow()
    ui = uiresolution.Ui_Resolution()
    ui.setupUi(Resolution)
    Resolution.show()
    sys.exit(app.exec_())