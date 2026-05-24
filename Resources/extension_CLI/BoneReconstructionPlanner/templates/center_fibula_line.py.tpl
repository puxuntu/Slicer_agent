import slicer
from BoneReconstructionPlanner import BoneReconstructionPlannerLogic

# Reuse or create logic instance
try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()
    _bonereconstructionplanner_logic = logic

# Get parameter node (creates if not exists)
parameterNode = logic.getParameterNode()

# Find fibula model node
fibulaModelNode = parameterNode.GetNodeReference("fibulaModelNode")
if fibulaModelNode is None:
    # Search scene for a model node with "fibula" in name
    modelNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
    for i in range(modelNodes.GetNumberOfItems()):
        node = modelNodes.GetItemAsObject(i)
        if "fibula" in node.GetName().lower():
            fibulaModelNode = node
            break
    if fibulaModelNode is None:
        raise ValueError("Fibula model node not found in the scene.")
    parameterNode.SetNodeReferenceID("fibulaModelNode", fibulaModelNode.GetID())

# Find fibula line (should be a markups line node)
fibulaLine = parameterNode.GetNodeReference("fibulaLine")
if fibulaLine is None:
    # Search for a markups line node with "fibula" in name
    markupsNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode")
    for i in range(markupsNodes.GetNumberOfItems()):
        node = markupsNodes.GetItemAsObject(i)
        if "fibula" in node.GetName().lower():
            fibulaLine = node
            break
    if fibulaLine is None:
        raise ValueError("Fibula line node not found in the scene.")
    parameterNode.SetNodeReferenceID("fibulaLine", fibulaLine.GetID())

# Call the method
logic.centerFibulaLine()

print("[BoneReconstructionPlanner] Fibula line centered successfully.")