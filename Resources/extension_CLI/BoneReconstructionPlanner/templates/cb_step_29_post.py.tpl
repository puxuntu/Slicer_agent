# --- BoneReconstructionPlanner: Manually adjust the mandibular cut planes in the mandible 3D view by dragging the visible plane interaction handles until the positions and rotations look correct. (Done) ---
import slicer

interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
if interactionNode is not None:
    interactionNode.SwitchToViewTransformMode()

print("[BoneReconstructionPlanner] Step 'cb_step_29' view adjustment completed.")
