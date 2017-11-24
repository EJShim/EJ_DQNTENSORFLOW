import vtk
import numpy as np

class E_InteractorStyle(vtk.vtkInteractorStyle):
    def __init__(self, Manager):
        self.Mgr = Manager;
        self.m_bProopagateEvent = True

        self.m_bKeyDown = False

        self.m_TransFactor = 0.01

        self.AddObserver("MouseMoveEvent", self.MouseMoveEvent)
        self.AddObserver("RightButtonPressEvent", self.RightButtonPressEvent)
        self.AddObserver("RightButtonReleaseEvent", self.RightButtonReleaseEvent)
        self.AddObserver("LeftButtonPressEvent", self.LeftButtonPressEvent)
        self.AddObserver("LeftButtonReleaseEvent", self.LeftButtonReleaseEvent)
        self.AddObserver("MiddleButtonPressEvent", self.MiddleButtonPressEvent)
        self.AddObserver("MiddleButtonReleaseEvent", self.MiddleButtonReleaseEvent)
        self.AddObserver("MouseWheelForwardEvent", self.MouseWheelForwardEvent)
        self.AddObserver("MouseWheelBackwardEvent", self.MouseWheelBackwardEvent)
        self.AddObserver("KeyPressEvent", self.KeyPressEvent)
        self.AddObserver("CharEvent", self.DummyFunc)

    def DummyFunc(self, obj, event):
        return


    def KeyPressEvent(self, obj, event):
        keycode = self.GetInteractor().GetKeySym()


        if keycode == 'w':
            self.CamTop()

        elif keycode == 'a':
            self.CamLeft()

        elif keycode == 's':
            self.CamBottom()

        elif keycode == 'd':
            self.CamRight()


        elif keycode == 'space':
            self.CamForward()

        elif keycode == 'c':
            self.CamBackward()




    def MouseMoveEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnMouseMove()

    def RightButtonPressEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnRightButtonDown()

    def RightButtonReleaseEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnRightButtonUp()

    def LeftButtonPressEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnLeftButtonDown()

    def LeftButtonReleaseEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnLeftButtonUp()

    def MiddleButtonPressEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnMiddleButtonDown()

    def MiddleButtonReleaseEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnMiddleButtonUp()

    def MouseWheelForwardEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnMouseWheelForward()

    def MouseWheelBackwardEvent(self, obj, event):
        if self.m_bProopagateEvent :
            self.OnMouseWheelBackward()


    def Disable(self):
        self.m_bProopagateEvent = False

    def Enable(self):
        self.m_bProopagateEvent = True


    def CamTop(self):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()

        #View Up
        camera.OrthogonalizeViewUp()
        up = np.array(camera.GetViewUp())

        self.MoveCamera(up*self.m_TransFactor)

    def CamBottom(self):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()

        #View Up
        camera.OrthogonalizeViewUp()
        up = np.array(camera.GetViewUp())

        self.MoveCamera(up*(-self.m_TransFactor))

    def CamLeft(self):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()

        #View Up
        camera.OrthogonalizeViewUp()
        up = np.array(camera.GetViewUp())
        viewDir = np.array(camera.GetDirectionOfProjection())

        left = np.cross(up, viewDir)
        self.MoveCamera(left*self.m_TransFactor)

    def CamRight(self):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()

        #View Up
        camera.OrthogonalizeViewUp()
        up = np.array(camera.GetViewUp())
        viewDir = np.array(camera.GetDirectionOfProjection())

        left = np.cross(up, viewDir)
        self.MoveCamera(left*(-self.m_TransFactor))

    def CamForward(self):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
        viewDir = np.array(camera.GetDirectionOfProjection())
        self.MoveCamera(viewDir * self.m_TransFactor)


    def CamBackward(self):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
        viewDir = np.array(camera.GetDirectionOfProjection())
        self.MoveCamera(viewDir * (-self.m_TransFactor))

    def MoveCamera(self, vector):
        camera = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()

        #Get Camera Position
        position = np.array(camera.GetPosition())
        focal = np.array(camera.GetFocalPoint())

        updatePosition = position + vector
        updateFocal = focal + vector

        camera.SetPosition(updatePosition)
        camera.SetFocalPoint(updateFocal)

        self.Mgr.Redraw()
