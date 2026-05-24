try:
    _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

# Retrieve the markup plane node the user just placed points on
node = slicer.mrmlScene.GetNodeByID(_bonereconstructionplanner_add_cut_plane_id)

# Validate that the user placed at least 3 control points
if node.GetNumberOfControlPoints() < 3:
    raise RuntimeError("Cut plane requires at least 3 control points (placed {}).".format(node.GetNumberOfControlPoints()))

# Set up required state on the logic instance (the method may read e.g. self.currentPlaneNode)
# According to the extension's source, addCutPlane() creates a new plane, but we do this anyway
# in case the method expects the node to be stored.
_bonereconstructionplanner_logic.currentPlaneNode = node

# Call the method (no arguments expected)
_bonereconstructionplanner_logic.addCutPlane()

# Exit placement mode (if the method entered it)
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToViewTransformMode()

# Print completion message
numPoints = node.GetNumberOfControlPoints()
print(f"[BoneReconstructionPlanner] Cut plane step completed. User placed {numPoints} control points.")