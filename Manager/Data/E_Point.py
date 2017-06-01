import vtk

class E_Point(vtk.vtkActor):
    def __init__(self, position=[0.0, 0.0, 0.0]):
        super(E_Point, self).__init__()
        self.m_position = position

        # Create the geometry of a point (the coordinate)
        self.points = vtk.vtkPoints()
        # Create the topology of the point (a vertex)
        vertices = vtk.vtkCellArray()

        id = self.points.InsertNextPoint(self.m_position)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()

        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(self.points)
        point.SetVerts(vertices)

        # Visualize
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(point)

        self.SetMapper(mapper)
        self.GetProperty().SetPointSize(10)
        self.GetProperty().SetColor(0.0, 1.0, 0.0)

    def SetColor(self, x, y, z):
        self.GetProperty().SetColor(x, y, z)

    def SetPosition(self, x, y, z):
        self.m_position = [x, y, z]

        self.points.Initialize()
        self.points.InsertNextPoint(self.m_position)

    def SetPosition(self, position):
        self.m_position = position

        self.points.Initialize()
        self.points.InsertNextPoint(self.m_position)
