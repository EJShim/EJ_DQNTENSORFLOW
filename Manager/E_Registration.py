import vtk
import numpy as np
from Manager.Data.E_Point import *

class E_Registration:
    def __init__(self, Mgr):
        self.Mgr = Mgr
        self.m_gCamera = vtk.vtkCamera()

        self.m_gCamera.DeepCopy(Mgr.GetCamera(0) )


        factor = 0.27
        self.m_3DPositions = [[factor, 0.0, factor],[-factor, 0.0, factor],[factor, 0.0, -factor],[-factor, 0.0, -factor]]
        self.m_bInitialized = False


    def Add3DPoints(self):
        #Screen Coordinate Points
        self.m_2DPositions = [[149.74413489247786, 542.7558651075221, 0.4711253694667498], [483.2558651075222, 542.7558651075221, 0.4711253694667498], [149.74413489247786, 209.24413489247783, 0.4711253694667498], [483.2558651075222, 209.24413489247783, 0.4711253694667498]]
        self.m_projectedPositions = self.m_2DPositions



        dPoints = self.GetPointWorldPosition()

        self.m_gtPoint = [0, 0, 0, 0]
        self.m_textActor = [0, 0, 0, 0]

        for i in range(len(self.m_3DPositions)):
            point = E_Point(self.m_3DPositions[i])
            self.Mgr.m_renderer[0].AddActor(point)

            self.m_gtPoint[i] = E_Point(dPoints[i])
            self.m_gtPoint[i].SetColor(0.0, 0.0, 1.0)
            self.Mgr.m_renderer[0].AddActor(self.m_gtPoint[i])

            #Add text actor
            vPosition = self.m_2DPositions[i]
            self.m_textActor[i] = vtk.vtkTextActor()
            self.m_textActor[i].SetInput("3D Position")
            self.m_textActor[i].SetPosition(vPosition[0], vPosition[1])
            self.m_textActor[i].GetTextProperty().SetFontSize(10)
            self.m_textActor[i].GetTextProperty().SetColor(0.0, 1.0, 0.0)
            self.Mgr.m_renderer[0].AddActor2D(self.m_textActor[i])



        self.m_bInitialized = True


    def GetProjectedPosition(self):
        results = []
        renderer = self.Mgr.m_renderer[0]

        for i in range(len(self.m_3DPositions)):
            position = [0, 0, 0]
            wPosition = self.m_3DPositions[i]
            vtk.vtkInteractorObserver.ComputeWorldToDisplay(renderer, wPosition[0], wPosition[1], wPosition[2], position)
            results.append(position)

        return results

    def GetPointWorldPosition(self):
        results = []
        renderer = self.Mgr.m_renderer[0]
        for i in range(4):
            position = [0, 0, 0, 0]
            sPosition = self.m_2DPositions[i]
            vtk.vtkInteractorObserver.ComputeDisplayToWorld(renderer, sPosition[0], sPosition[1], sPosition[2], position)
            results.append(position[0:3])

        return results

    def Update(self):
        if not self.m_bInitialized: return

        vPosition = self.GetPointWorldPosition()
        self.m_projectedPositions = self.GetProjectedPosition()

        for i in range(4):
            self.m_gtPoint[i].SetPosition(vPosition[i])

            #Update Screen Messages
            screen = self.m_projectedPositions[i]
            log = str(round(screen[0], 2)) + "," + str(round(screen[1], 2))
            self.m_textActor[i].SetInput(log)
            self.m_textActor[i].SetPosition(screen[0],screen[1])




    def GetError(self):
        if not self.m_bInitialized: return

        sub = np.array(self.m_projectedPositions) - np.array(self.m_2DPositions)
        return np.linalg.norm(sub)


    def GetState(self):
        wPos = np.array(self.m_projectedPositions)[0:4,0:2]
        vPos = np.array(self.m_3DPositions)[0:4,0:2]

        state = np.subtract(wPos, vPos)
        state = np.reshape(state, [8, ])

        return state
