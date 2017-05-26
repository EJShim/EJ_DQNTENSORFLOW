import vtk
from vtk.util.numpy_support import vtk_to_numpy

class E_Manager:
    def __init__(self, window):
        self.m_mainFrm = window

        #Two Renderers
        self.m_renderer = [0, 0]




    def Initialize(self):
        for i in range(2):
            interactor = vtk.vtkInteractorStyleSwitch()

            self.m_renderer[i] = vtk.vtkRenderer()
            self.m_renderer[i].SetBackground(0.0, 0.0, 0.0)
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().AddRenderer(self.m_renderer[i])
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().Render()
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)


        self.InitDefaultObjects()

    def InitDefaultObjects(self):
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(1, 0, 0)
        planeSource.SetNormal(0, 0, 1)

        planeMapper = vtk.vtkPolyDataMapper()
        planeMapper.SetInputConnection(planeSource.GetOutputPort())

        planeActor = vtk.vtkActor()
        planeActor.SetMapper(planeMapper)

        self.m_renderer[0].AddActor(planeActor)
        self.m_renderer[0].ResetCamera()


        planeActor2 = vtk.vtkActor()
        planeActor2.SetMapper(planeMapper)

        self.m_renderer[1].AddActor(planeActor2)
        self.m_renderer[1].ResetCamera()

        self.Redraw()


    def Redraw(self):
        for i in range(2):
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().Render()
