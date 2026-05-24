# --- BoneReconstructionPlanner: Place a markup plane on the mandible model; adjacent planes define bone pieces for reconstruction. (Setup) ---
import slicer

# Create the markup node for user interaction
node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "Add Cut Plane")
displayNode = node.GetDisplayNode()
if displayNode is None:
    displayNode = node.CreateDefaultDisplayNode()
displayNode.SetVisibility(True)

print("[BoneReconstructionPlanner] Please Switch to persistent plane placement mode. Click three points on the mandibular model to define each cut plane. Continue placing planes until all resections are defined.")
print("When finished, press the 'Done' button in the workflow panel.")

# Enter placement mode
slicer.modules.markups.logic().SetActiveListID(node)
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToPersistentPlaceMode()

_bonereconstructionplanner_add_cut_plane_id = node.GetID()
