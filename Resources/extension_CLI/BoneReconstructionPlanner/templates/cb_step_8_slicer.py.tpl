# Toggle FOV spacing match 2D for Red slice view

# Get the Red slice node
sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")

# Determine current mode
currentMode = sliceNode.GetSliceResolutionMode()
match2DMode = slicer.vtkMRMLSliceNode.SliceResolutionMatch2DView
matchVolumesMode = slicer.vtkMRMLSliceNode.SliceResolutionMatchVolumes

# Toggle: if currently match 2D, revert to match volumes; otherwise set to match 2D
if currentMode == match2DMode:
    sliceNode.SetSliceResolutionMode(matchVolumesMode)
else:
    sliceNode.SetSliceResolutionMode(match2DMode)