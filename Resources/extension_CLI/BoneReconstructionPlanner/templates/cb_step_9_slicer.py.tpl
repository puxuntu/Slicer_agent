# Turn on slice intersection visibility, interaction mode, Translate, and Rotate

appLogic = slicer.app.applicationLogic()

# Enable slice intersection visibility
appLogic.SetIntersectingSlicesEnabled(
    slicer.vtkMRMLApplicationLogic.IntersectingSlicesVisibility, True)

# Enable slice intersection interaction mode
appLogic.SetIntersectingSlicesEnabled(
    slicer.vtkMRMLApplicationLogic.IntersectingSlicesInteractive, True)

# Enable Translate handles
appLogic.SetIntersectingSlicesEnabled(
    slicer.vtkMRMLApplicationLogic.IntersectingSlicesTranslation, True)

# Enable Rotate handles
appLogic.SetIntersectingSlicesEnabled(
    slicer.vtkMRMLApplicationLogic.IntersectingSlicesRotation, True)