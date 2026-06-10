# --- BoneReconstructionPlanner: Draw a line over the fibula in "3D View 2", starting with the first point distally and the last point proximally. (Setup) ---
import slicer
from SlicerAIAgentLib.workflow_state import remember_interaction_node

# Reuse the markup node created by addFibulaLine() in the previous step.
nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode")
node = None
for i in range(nodes.GetNumberOfItems() - 1, -1, -1):
    candidate = nodes.GetItemAsObject(i)
    if candidate is not None:
        node = candidate
        break
if node is None:
    raise RuntimeError("No vtkMRMLMarkupsLineNode found from previous placement step.")

displayNode = node.GetDisplayNode()
if displayNode is not None:
    displayNode.SetVisibility(True)
_bonereconstructionplanner_cb_step_20_id = node.GetID()
remember_interaction_node(_workflow_runtime_extension, _workflow_runtime_id, "cb_step_20", _bonereconstructionplanner_cb_step_20_id, _workflow_runtime_repeat_index)

print("[BoneReconstructionPlanner] Please Place first point distally, last point proximally on the fibula in 3D View 2")
print("When finished, press the 'Done' button in the workflow panel.")
