import vtk

class E_Registration:
    def __init__(self, Mgr):
        self.Mgr = Mgr
        self.m_gCamera = vtk.vtkCamera()

        self.m_gCamera.DeepCopy(Mgr.GetCamera(0) )
