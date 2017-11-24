import os
import math

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from Manager.E_Interactor import *
from Manager.E_Registration import *


from E_Brain import *
# from E_TestBrain import *


class E_Manager:
    def __init__(self, window):
        self.m_mainFrm = window

        #Two Renderers
        self.m_renderer = [0, 0]

        #Camera Actor
        self.m_cameraActor = vtk.vtkActor()

        self.m_agent = E_Agent(8, 6)


        self.m_prevErr = 0.0



    def Initialize(self):
        for i in range(2):
            self.m_renderer[i] = vtk.vtkRenderer()
            self.m_renderer[i].SetBackground(0.0, 0.0, 0.0)
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().AddRenderer(self.m_renderer[i])
            # self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().Render()

            interactor = E_InteractorStyle(self)
            self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)
            self.m_renderer[i].GetActiveCamera().ParallelProjectionOff()

        self.GetCamera(0).SetPosition(0, 1, 0)
        self.GetCamera(1).SetPosition(0, 0.2, 1)

        self.m_mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor().GetInteractorStyle().Disable()



        #Initialize Registration Manager
        #Registration Manager
        self.RegMgr = E_Registration(self)
        self.m_bInitialized = True;

        self.InitDefaultObjects()


    def InitDefaultObjects(self):

        #Texture Image
        fileDir = os.path.dirname(os.path.realpath(__file__))
        fileDir = os.path.join(fileDir, "../image/four.png")
        imgReader = vtk.vtkPNGReader()
        imgReader.SetFileName(fileDir)

        texture = vtk.vtkTexture()
        texture.SetInputConnection(imgReader.GetOutputPort())

        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(0, 0, 0)
        planeSource.SetNormal(0, 1, 0)

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
        self.m_cameraSource = vtk.vtkConeSource()
        self.m_cameraSource.SetDirection(0, 2, 0)
        self.m_cameraSource.SetHeight(1.0)
        self.m_cameraSource.SetCenter(0, -1, 0)
        self.m_cameraSource.SetResolution(4)

        cameraMapper = vtk.vtkPolyDataMapper()
        cameraMapper.SetInputConnection(self.m_cameraSource.GetOutputPort())
        self.m_cameraActor.SetMapper(cameraMapper)
        self.m_cameraActor.GetProperty().SetColor(0.0, 1.0, 0.0)
        self.m_cameraActor.GetProperty().SetRepresentationToWireframe();

        self.m_renderer[1].AddActor(self.m_cameraActor)
        self.m_renderer[1].ResetCamera()

        #Add 2D actors?
        plot = vtk.vtkXYPlotActor()
        plot.ExchangeAxesOff();
        plot.SetLabelFormat( "%g" );
        plot.SetXTitle( "Level" );
        plot.SetYTitle( "Frequency" );
        plot.SetXValuesToValue();

        # self.m_renderer[0].AddActor(plot)
        self.m_log = vtk.vtkTextActor()
        self.m_log.SetInput("log")
        self.m_log.SetPosition(10, 40)
        self.m_log.GetTextProperty().SetFontSize(15)
        self.m_log.GetTextProperty().SetColor(0.0, 1.0, 0.0)
        self.m_renderer[0].AddActor2D(self.m_log)


        self.Redraw()
        self.RegMgr.Add3DPoints()


    def GetCamera(self, idx):
        return self.m_renderer[idx].GetActiveCamera()



    def Redraw(self):

        position = self.GetCamera(0).GetPosition()
        self.m_cameraActor.SetPosition( position )
        self.RegMgr.Update()
        #Initialize TF Trainable Variable



        for i in range(2):
            self.m_renderer[i].GetRenderWindow().Render()
            self.m_renderer[i].ResetCameraClippingRange ()

        far =   self.GetCamera(0).GetClippingRange()[1]
        self.m_cameraSource.SetCenter(0.0, -far/2.0, 0.0)
        # self.m_cameraSource.SetHeight(far)
        self.m_renderer[1].ResetCamera()


    def SetGroundTruth(self):

        self.GetCamera(0).DeepCopy(self.RegMgr.m_gCamera)

        self.Redraw()

    def RunTraining(self):

        max_episodes = 1000
        max_steps = 1000

        i = 0
        while 1:
            self.GoToRandom()            
            done = False

            rewards = []
            transitions = []
            rAll = 0

            state =  self.RegMgr.GetState()


            while 1:

                perr = self.RegMgr.GetError()
                action = self.m_agent.Forward( state )

                #Forward Action
                self.ForwardActions(action)
                self.Redraw()

                #Calculate Error
                state1 = self.RegMgr.GetState()
                err = self.RegMgr.GetError()


                reward = 0.0
                if err > perr:
                    reward = -1.0
                elif err < perr:
                    reward = 1.0

                done =  self.IsDone(err)
                
                if done:
                    reward *= 100
                    print("done", reward)
                    break

                done = True
                

                self.m_agent.Backward(state, action, reward, state1, done)                
                state = state1




    def ForwardActions(self, idx):
        interactorStyle = self.m_renderer[0].GetRenderWindow().GetInteractor().GetInteractorStyle()

        if idx == 0:
            interactorStyle.CamTop()
        elif idx == 1:
            interactorStyle.CamBottom()
        elif idx == 2:
            interactorStyle.CamRight()
        elif idx == 3:
            interactorStyle.CamLeft()
        elif idx == 4:
            interactorStyle.CamForward()
        elif idx == 5:
            interactorStyle.CamBackward()

    def IsOutOfRange(self):
        position = np.array(self.GetCamera(0).GetPosition())
        hold = position[1] * math.tan(math.radians(15)) - 0.5

        if abs(position[0]) > hold or abs(position[2]) > hold:
            return True
        else:
            return False

    def IsDone(self, err):
        position = np.array(self.GetCamera(0).GetPosition())
        hold = position[1] * math.tan(math.radians(15)) - 0.5

        if abs(position[0]) > hold or abs(position[2]) > hold or err < 1.0:
            return True
        else:
            return False

    def GoToRandom(self):
        y = random.randrange(50, 100)

        hold = math.floor(y * math.tan(math.radians(15)) -0.6)
        x = random.randrange(-hold, hold)
        z = random.randrange(-hold, hold)

        self.GetCamera(0).SetPosition(x/10, y/10, z/10)
        self.GetCamera(0).SetFocalPoint(x/10, 0, z/10)

    def SaveBrain(self):
        self.m_agent.SaveWeights()
