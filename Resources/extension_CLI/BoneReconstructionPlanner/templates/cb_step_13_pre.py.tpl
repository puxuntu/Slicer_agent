# --- BoneReconstructionPlanner: 13. Manually click and draw on the "Red" view to create a curve along the mandible. (Setup) ---
import slicer

# Reuse the markup node created by addMandibularCurve() in the previous step.
nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsCurveNode")
node = None
for i in range(nodes.GetNumberOfItems() - 1, -1, -1):
    candidate = nodes.GetItemAsObject(i)
    if candidate is not None:
        node = candidate
        break
if node is None:
    raise RuntimeError("No vtkMRMLMarkupsCurveNode found from previous placement step.")

displayNode = node.GetDisplayNode()
if displayNode is not None:
    displayNode.SetVisibility(True)
slicer.modules.markups.logic().SetActiveListID(node)
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
if interactionNode is not None:
    interactionNode.SwitchToPersistentPlaceMode()
_bonereconstructionplanner_cb_step_13_id = node.GetID()

print("[BoneReconstructionPlanner] Please Click and draw on the Red view to create a curve along the mandible")
print("When finished, press the 'Done' button in the workflow panel.")
