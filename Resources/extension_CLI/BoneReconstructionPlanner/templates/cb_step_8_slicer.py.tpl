# Toggle on FOV/Spacing match 2D for the Red view
redSliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
if redSliceNode:
    redSliceNode.SetSliceResolutionMode(slicer.vtkMRMLSliceNode.SliceResolutionMatch2DView)