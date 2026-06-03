# --- BoneReconstructionPlanner: 18. Place one mandibular cut plane using the extension's Add cut plane workflow. If the user requested N cut planes, repeat the Add cut plane + place plane interaction N times. Do not store these planes as a rotation plane; they are mandibular cut planes managed by the extension. (Process) ---
import slicer

node = slicer.mrmlScene.GetNodeByID(_bonereconstructionplanner_cb_step_18_id)
if node is None:
    raise RuntimeError("Node not found for step 'cb_step_18'")

# Validate user input
numPoints = node.GetNumberOfControlPoints()
if numPoints < 1:
    raise RuntimeError("Need at least 1 control points, got %d. Please add more." % numPoints)

# Exit placement mode
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToViewTransformMode()

print("[BoneReconstructionPlanner] Step 'cb_step_18' processed with %d control points." % node.GetNumberOfControlPoints())
