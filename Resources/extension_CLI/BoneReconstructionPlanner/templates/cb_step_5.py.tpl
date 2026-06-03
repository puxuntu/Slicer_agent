try:
    _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

parameterNode = _bonereconstructionplanner_logic.getParameterNode()

# Resolve fibula segmentation node reference
fibulaSegmentation = parameterNode.GetNodeReference("fibulaSegmentation")
if fibulaSegmentation is None:
    nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for i in range(nodes.GetNumberOfItems()):
        n = nodes.GetItemAsObject(i)
        if "fibula" in n.GetName().lower():
            fibulaSegmentation = n
            break
    if fibulaSegmentation is not None:
        parameterNode.SetNodeReferenceID("fibulaSegmentation", fibulaSegmentation.GetID())

# Resolve mandibular segmentation node reference
mandibularSegmentation = parameterNode.GetNodeReference("mandibularSegmentation")
if mandibularSegmentation is None:
    nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for i in range(nodes.GetNumberOfItems()):
        n = nodes.GetItemAsObject(i)
        if "mandib" in n.GetName().lower():
            mandibularSegmentation = n
            break
    if mandibularSegmentation is not None:
        parameterNode.SetNodeReferenceID("mandibularSegmentation", mandibularSegmentation.GetID())

# Ensure useNonDecimatedBoneModelsForPreview is set (default 'True')
useNonDecimated = parameterNode.GetParameter("useNonDecimatedBoneModelsForPreview")
if useNonDecimated == "":
    parameterNode.SetParameter("useNonDecimatedBoneModelsForPreview", "True")

# Call the method
_bonereconstructionplanner_logic.makeModels()

print("BoneReconstructionPlanner step 5 (makeModels) completed.")