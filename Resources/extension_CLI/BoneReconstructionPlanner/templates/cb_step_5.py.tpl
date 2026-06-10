from BoneReconstructionPlanner import BoneReconstructionPlannerLogic

# precondition:begin
try:
    slicer.util.selectModule("BoneReconstructionPlanner")
except Exception as _module_enter_error:
    print(f"Warning: could not activate module 'BoneReconstructionPlanner': {_module_enter_error}")
# precondition:end

try:
    _bonereconstructionplanner_logic
except NameError:
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

logic = _bonereconstructionplanner_logic
parameterNode = logic.getParameterNode()

# Ensure parameterNode is valid
if parameterNode is None:
    print("Error: Could not get parameter node for BoneReconstructionPlanner")
    raise RuntimeError("Parameter node not available")

# --- Set up required segmentation nodes ---
# fibulaSegmentation
fibulaSeg = parameterNode.GetNodeReference("fibulaSegmentation")
if fibulaSeg is None:
    segNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for i in range(segNodes.GetNumberOfItems()):
        n = segNodes.GetItemAsObject(i)
        if "fibula" in n.GetName().lower():
            fibulaSeg = n
            break
if fibulaSeg is not None:
    parameterNode.SetNodeReferenceID("fibulaSegmentation", fibulaSeg.GetID())
else:
    print("Warning: Could not find fibula segmentation node. Attempting step anyway.")

# mandibularSegmentation
mandibularSeg = parameterNode.GetNodeReference("mandibularSegmentation")
if mandibularSeg is None:
    segNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for i in range(segNodes.GetNumberOfItems()):
        n = segNodes.GetItemAsObject(i)
        if "mandible" in n.GetName().lower():
            mandibularSeg = n
            break
if mandibularSeg is not None:
    parameterNode.SetNodeReferenceID("mandibularSegmentation", mandibularSeg.GetID())
else:
    print("Warning: Could not find mandibular segmentation node. Attempting step anyway.")

# --- Set scalar parameters if not already set ---
# useNonDecimatedBoneModelsForPreview defaults to "True"
currentVal = parameterNode.GetParameter("useNonDecimatedBoneModelsForPreview")
if currentVal == "":
    parameterNode.SetParameter("useNonDecimatedBoneModelsForPreview", "True")

# Call the method
print("[BoneReconstructionPlanner] Executing makeModels()...")
logic.makeModels()
print("[BoneReconstructionPlanner] makeModels() completed.")