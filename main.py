import sys

from PyQt5.QtWidgets import *
from GUI.MainFrm import MainWindow

from E_Brain import E_Agent


app = QApplication([])

window = MainWindow()
window.setFixedSize(1300, 800)
window.show()
sys.exit(app.exec_())

#
# brain = E_Agent()
# print(brain)
