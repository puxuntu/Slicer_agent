# --- BoneReconstructionPlanner: Manually adjust the slice intersection position by translate and rotate of the cross lines in each view. (Done) ---
import slicer

interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
if interactionNode is not None:
    interactionNode.SwitchToViewTransformMode()

print("[BoneReconstructionPlanner] Step 'cb_step_10' view adjustment completed.")
