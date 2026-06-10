# --- BoneReconstructionPlanner: Manually adjust the mandibular cut planes in the mandible 3D view by dragging the visible plane interaction handles until the positions and rotations look correct. (Setup) ---
import slicer

# Ensure interaction mode is view transform so user can drag handles
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
if interactionNode is not None:
    interactionNode.SwitchToViewTransformMode()

print("[BoneReconstructionPlanner] Please drag the handles in the 3D view to adjust plane positions and rotations")
print("When finished, press the 'Done' button in the workflow panel.")