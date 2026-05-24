# --- BoneReconstructionPlanner: Draw a markup line along the fibula diaphysis (distal to proximal) to define the fibula axis. (Setup) ---
import slicer

# Create the markup node for user interaction
node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Add Fibula Line")
displayNode = node.GetDisplayNode()
if displayNode is None:
    displayNode = node.CreateDefaultDisplayNode()
displayNode.SetVisibility(True)

print("[BoneReconstructionPlanner] Please Switch to line placement mode. Click two points along the fibula segmentation to define the fibula axis.")
print("When finished, press the 'Done' button in the workflow panel.")

# Enter placement mode
slicer.modules.markups.logic().SetActiveListID(node)
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToPersistentPlaceMode()

_bonereconstructionplanner_add_fibula_line_id = node.GetID()
