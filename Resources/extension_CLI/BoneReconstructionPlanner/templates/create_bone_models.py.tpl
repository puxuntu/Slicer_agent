import slicer

# Try to reuse existing logic instance
try:
    logic = _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    logic = BoneReconstructionPlannerLogic()
    _bonereconstructionplanner_logic = logic

# Get parameter node
parameterNode = logic.getParameterNode()

# Helper: Find segmentation node by name substring
def findSegmentationNode(substring):
    nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for i in range(nodes.GetNumberOfItems()):
        n = nodes.GetItemAsObject(i)
        if substring.lower() in n.GetName().lower():
            return n
    return None

# Find fibula segmentation
fibulaSeg = parameterNode.GetNodeReference("fibulaSegmentation")
if fibulaSeg is None:
    fibulaSeg = findSegmentationNode("fibula")
    if fibulaSeg is not None:
        parameterNode.SetNodeReferenceID("fibulaSegmentation", fibulaSeg.GetID())

# Find mandibular segmentation
mandibularSeg = parameterNode.GetNodeReference("mandibularSegmentation")
if mandibularSeg is None:
    mandibularSeg = findSegmentationNode("mandib")
    if mandibularSeg is not None:
        parameterNode.SetNodeReferenceID("mandibularSegmentation", mandibularSeg.GetID())

# Optionally set useNonDecimatedBoneModelsForPreview if not already set (default False)
if not parameterNode.GetParameter("useNonDecimatedBoneModelsForPreview"):
    parameterNode.SetParameter("useNonDecimatedBoneModelsForPreview", "False")

# Call the method
logic.makeModels()

print("Bone models created from segmentations.")