try:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
except ImportError:
    raise ImportError("BoneReconstructionPlanner extension is not installed.")

# Retrieve the markups line node that the user placed control points on
node = slicer.mrmlScene.GetNodeByID(_bonereconstructionplanner_add_fibula_line_id)
if node is None:
    raise ValueError("Could not find the fibula line node.")

numControlPoints = node.GetNumberOfControlPoints()
if numControlPoints < 2:
    raise ValueError(f"Fibula line must have at least 2 control points, but found {numControlPoints}.")

# Reuse or create the logic instance
try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()

# Call the method (it creates a new line node and enters placement mode)
logic.addFibulaLine()

# Exit placement mode
interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToViewTransformMode()

# Store the logic instance for subsequent steps
_bonereconstructionplanner_logic = logic

print(f"[BoneReconstructionPlanner] Fibula line finalized with {numControlPoints} control points.")