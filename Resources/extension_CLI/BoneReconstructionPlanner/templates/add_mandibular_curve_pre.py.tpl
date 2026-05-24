# --- BoneReconstructionPlanner: Place a markup curve along the ideal mandibular arch to guide initial cut plane positioning. (Setup) ---
import slicer

# Create the markup node for user interaction
node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Add Mandibular Curve")
displayNode = node.GetDisplayNode()
if displayNode is None:
    displayNode = node.CreateDefaultDisplayNode()
displayNode.SetVisibility(True)

print("[BoneReconstructionPlanner] Please Switch to curve placement mode. Click on the mandibular segmentation to define a curve. Use at least three points for an accurate curve representation.")
print("When finished, press the 'Done' button in the workflow panel.")

# Enter placement mode
slicer.modules.markups.logic().SetActiveListID(node)
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToPersistentPlaceMode()

_bonereconstructionplanner_add_mandibular_curve_id = node.GetID()
