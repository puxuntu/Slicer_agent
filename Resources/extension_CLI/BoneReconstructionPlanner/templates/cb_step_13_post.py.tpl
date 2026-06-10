# --- BoneReconstructionPlanner: Manually click and draw on the "Red" view to create a curve along the mandible. (Process) ---
import slicer
from SlicerAIAgentLib.workflow_state import resolve_interaction_node

node = resolve_interaction_node(_workflow_runtime_extension, _workflow_runtime_id, "cb_step_13", "vtkMRMLMarkupsCurveNode", _workflow_runtime_repeat_index)
if node is None:
    node = slicer.mrmlScene.GetNodeByID(_bonereconstructionplanner_cb_step_13_id)
if node is None:
    raise RuntimeError("Node not found for step 'cb_step_13'")

# Exit placement mode
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToViewTransformMode()

print("[BoneReconstructionPlanner] Step 'cb_step_13' processed with %d control points." % node.GetNumberOfControlPoints())
