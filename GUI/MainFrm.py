from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Manager.E_Manager import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

#Brain Tes
from E_Brain import E_Agent

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("EJ DQN Project")

        #Widgets
        self.m_centralWidget = QWidget()
        self.m_vtkWidget = [0, 0]
        for i in range(2):
            self.m_vtkWidget[i] = QVTKRenderWindowInteractor()

        #Managers
        self.Mgr = E_Manager(self)

        #Initialize Toolbar
        self.InitToolbar()

        #Initialize Central Widget
        self.InitCentralWidget()

        #Initialize Manager
        self.Mgr.Initialize()


    def InitToolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        truthAction = QAction("Ground Truth", self)
        truthAction.triggered.connect(self.OnClickGroundTruth)
        toolbar.addAction(truthAction)

        trainAction = QAction("Train", self)
        trainAction.triggered.connect(self.OnClickTraining)
        toolbar.addAction(trainAction)

        SaveAction = QAction("Save Weights", self)
        SaveAction.triggered.connect(self.OnClickSave)
        toolbar.addAction(SaveAction)

        LoadAction = QAction("Load Weights", self)
        LoadAction.triggered.connect(self.OnClickRestore)
        toolbar.addAction(LoadAction)

        testAction = QAction("Test", self)
        testAction.triggered.connect(self.OnClickTest)
        toolbar.addAction(testAction)





    def InitCentralWidget(self):
        #Set Central Widget
        self.setCentralWidget(self.m_centralWidget)

        #Set Central Layout
        mainLayout = QHBoxLayout()
        self.m_centralWidget.setLayout(mainLayout)

        for i in range(2):
            mainLayout.addWidget(self.m_vtkWidget[i])


    def OnClickGroundTruth(self):
        print("Ground Truth Action")

    def OnClickTraining(self):
        print("Training Action")

    def OnClickSave(self):
        print("Save Action")

    def OnClickRestore(self):
        print("Restore Action")

    def OnClickTest(self):
        agent = E_Agent()