# --- BoneReconstructionPlanner: Manually click and draw on the "Red" view to create a curve along the mandible. (Setup) ---
import slicer
from SlicerAIAgentLib.workflow_state import remember_interaction_node

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
_bonereconstructionplanner_cb_step_13_id = node.GetID()
remember_interaction_node(_workflow_runtime_extension, _workflow_runtime_id, "cb_step_13", _bonereconstructionplanner_cb_step_13_id, _workflow_runtime_repeat_index)

print("[BoneReconstructionPlanner] Please Click along the mandible in the Red view to place curve control points")
print("When finished, press the 'Done' button in the workflow panel.")
