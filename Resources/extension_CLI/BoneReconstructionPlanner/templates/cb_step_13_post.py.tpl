# --- BoneReconstructionPlanner: 13. Manually click and draw on the "Red" view to create a curve along the mandible. (Process) ---
import slicer

node = slicer.mrmlScene.GetNodeByID(_bonereconstructionplanner_cb_step_13_id)
if node is None:
    raise RuntimeError("Node not found for step 'cb_step_13'")

# Validate user input
numPoints = node.GetNumberOfControlPoints()
if numPoints < 2:
    raise RuntimeError("Need at least 2 control points, got %d. Please add more." % numPoints)

# Store the placed node on the extension parameter node for later steps
from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()
parameterNode = logic.getParameterNode()
parameterNode.SetNodeReferenceID("mandibleCurve", node.GetID())
_bonereconstructionplanner_logic = logic

# Exit placement mode
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToViewTransformMode()

print("[BoneReconstructionPlanner] Step 'cb_step_13' processed with %d control points." % node.GetNumberOfControlPoints())
