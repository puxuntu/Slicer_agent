# Toggle slice visibility in 3D view for the Red slice view
sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed")
if sliceNode:
    sliceNode.SetSliceVisible(not sliceNode.GetSliceVisible())