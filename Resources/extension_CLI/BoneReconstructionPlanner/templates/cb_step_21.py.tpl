import slicer
import numpy as np
import vtk

# Reuse or create logic instance
try:
    _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

logic = _bonereconstructionplanner_logic

# Ensure parameter node exists
parameterNode = logic.getParameterNode()

# Helper: search for a node by class and name substring
def _findNodeByClassAndName(nodeClass, nameSubstring):
    nodes = slicer.mrmlScene.GetNodesByClass(nodeClass)
    for i in range(nodes.GetNumberOfItems()):
        n = nodes.GetItemAsObject(i)
        if nameSubstring.lower() in n.GetName().lower():
            return n
    return None

# Set up fibulaLine reference
fibulaLine = parameterNode.GetNodeReference("fibulaLine")
if fibulaLine is None:
    fibulaLine = _findNodeByClassAndName("vtkMRMLMarkupsLineNode", "fibulaLine")
    if fibulaLine is None:
        # Create a temporary line if none exists (fallback)
        fibulaLine = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "FibulaLine")
    parameterNode.SetNodeReferenceID("fibulaLine", fibulaLine.GetID())

# Set up fibulaModelNode reference
fibulaModelNode = parameterNode.GetNodeReference("fibulaModelNode")
if fibulaModelNode is None:
    fibulaModelNode = _findNodeByClassAndName("vtkMRMLModelNode", "fibulaModel")
    if fibulaModelNode is None:
        fibulaModelNode = _findNodeByClassAndName("vtkMRMLModelNode", "fibula")
    if fibulaModelNode is None:
        raise ValueError("No fibula model node found in the scene.")
    parameterNode.SetNodeReferenceID("fibulaModelNode", fibulaModelNode.GetID())

# Ensure mandible reconstruction folder exists (needed by getMandibleReconstructionFolderItemID)
shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
sceneItemID = shNode.GetSceneItemID()
# Create the folder directly; the original recursive search is replaced because
# GetItemChild and HasChildItems are not available on this Slicer build.
reconstructionFolderID = shNode.CreateFolderItem(sceneItemID, "Reconstruction")

# Now call the method
logic.centerFibulaLine()

print("[BoneReconstructionPlanner] Center fibula line step completed.")