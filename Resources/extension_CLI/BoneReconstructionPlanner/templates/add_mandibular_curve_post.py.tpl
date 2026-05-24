import BoneReconstructionPlanner

try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlanner.BoneReconstructionPlannerLogic()

node = slicer.mrmlScene.GetNodeByID(_bonereconstructionplanner_add_mandibular_curve_id)
if node.GetNumberOfControlPoints() < 3:
    raise ValueError("At least 3 control points required for mandibular curve.")

logic.inputMarkupNode = node

logic.addMandibularCurve()

interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
interactionNode.SwitchToViewTransformMode()

_bonereconstructionplanner_logic = logic

print(f"Mandibular curve placed with {node.GetNumberOfControlPoints()} control points.")