# --- PelvicFracturePlanning: Manually adjust the position and rotation of the screw trajectories. (Done) ---
import slicer

interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
if interactionNode is not None:
    interactionNode.SwitchToViewTransformMode()

print("[PelvicFracturePlanning] Step 'cb_step_14' view adjustment completed.")
