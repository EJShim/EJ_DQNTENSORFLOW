import vtk
from vtk.util.numpy_support import vtk_to_numpy

from Manager.E_Interactor import *

class E_Manager:
    def __init__(self, window):
        self.m_mainFrm = window

        #Two Renderers
        self.m_renderer = [0, 0]

        #Camera Actor
        self.m_cameraActor = vtk.vtkActor()




    def Initialize(self):
        for i in range(2):
            self.m_renderer[i] = vtk.vtkRenderer()
            self.m_renderer[i].SetBackground(0.0, 0.0, 0.0)
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().AddRenderer(self.m_renderer[i])
            # self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().Render()

            interactor = E_InteractorStyle(self)
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)
            self.m_renderer[i].GetActiveCamera().ParallelProjectionOff()

        self.m_renderer[1].GetActiveCamera().SetPosition(3, 10, 2)

        self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor().GetInteractorStyle().Disable()



        self.InitDefaultObjects()

    def InitDefaultObjects(self):

        #Texture Image
        imgReader = vtk.vtkPNGReader()
        imgReader.SetFileName("/Users/sim-eungjun/documents/projects/EJ_DQNTENSORFLOW/image/four.png")

        texture = vtk.vtkTexture()
        texture.SetInputConnection(imgReader.GetOutputPort())

        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(1, 0, 0)
        planeSource.SetNormal(0, 0, 1)

        texturePlaneSource = vtk.vtkTextureMapToPlane()
        texturePlaneSource.SetInputConnection(planeSource.GetOutputPort())

        planeMapper = vtk.vtkPolyDataMapper()
        planeMapper.SetInputConnection(texturePlaneSource.GetOutputPort())

        planeActor = vtk.vtkActor()
        planeActor.SetMapper(planeMapper)
        planeActor.SetTexture(texture)

        self.m_renderer[0].AddActor(planeActor)
        self.m_renderer[0].ResetCamera()


        planeActor2 = vtk.vtkActor()
        planeActor2.SetMapper(planeMapper)
        planeActor2.SetTexture(texture)

        self.m_renderer[1].AddActor(planeActor2)


        #Add Camera Actor
        cameraSource = vtk.vtkConeSource()
        cameraMapper = vtk.vtkPolyDataMapper()
        cameraMapper.SetInputConnection(cameraSource.GetOutputPort())
        self.m_cameraActor.SetMapper(cameraMapper)
        camTrans = self.m_renderer[0].GetActiveCamera().GetViewTransformObject()
        self.m_cameraActor.SetUserTransform(camTrans)

        self.m_renderer[1].AddActor(self.m_cameraActor)
        self.m_renderer[1].ResetCamera()


        self.Redraw()

    def GetCamera(self, idx):
        return self.m_renderer[idx].GetActiveCamera()



    def Redraw(self):
        self.m_cameraActor.SetPosition( self.m_renderer[0].GetActiveCamera().GetPosition() )

        self.m_renderer[1].ResetCamera()

        for i in range(2):
            self.m_renderer[i].GetRenderWindow().Render()
            self.m_renderer[i].ResetCameraClippingRange ()
