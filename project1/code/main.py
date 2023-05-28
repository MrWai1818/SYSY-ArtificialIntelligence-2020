import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import toRomania

if __name__ == '__main__':
    app = QApplication(sys.argv)
    toThere = QMainWindow()
    ui = toRomania.Ui_toThere()
    ui.setupUi(toThere)
    toThere.show()
    sys.exit(app.exec_())