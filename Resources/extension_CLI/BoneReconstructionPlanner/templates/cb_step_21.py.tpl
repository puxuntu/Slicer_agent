# step: cb_step_21
# precondition:begin
try:
    slicer.util.selectModule("BoneReconstructionPlanner")
except Exception as _module_enter_error:
    print(f"Warning: could not activate module 'BoneReconstructionPlanner': {_module_enter_error}")
# precondition:end

try:
    _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()
_logic = _bonereconstructionplanner_logic

paramNode = _logic.getParameterNode()
if paramNode is None:
    paramNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode")
    _logic.setParameterNode(paramNode)

# Ensure fibulaLine reference exists in parameter node
fibulaLine = paramNode.GetNodeReference("fibulaLine")
if fibulaLine is None:
    # search for a fibula line markups line node (not curve)
    lines = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode")
    for i in range(lines.GetNumberOfItems()):
        c = lines.GetItemAsObject(i)
        if "fibula" in c.GetName().lower():
            fibulaLine = c
            break
    if fibulaLine is None:
        raise RuntimeError("Fibula line markups line node not found in scene. Run earlier steps first.")
    paramNode.SetNodeReferenceID("fibulaLine", fibulaLine.GetID())

# Ensure fibulaModelNode reference exists
fibulaModelNode = paramNode.GetNodeReference("fibulaModelNode")
if fibulaModelNode is None:
    models = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
    for i in range(models.GetNumberOfItems()):
        m = models.GetItemAsObject(i)
        if "fibula" in m.GetName().lower():
            fibulaModelNode = m
            break
    if fibulaModelNode is None:
        raise RuntimeError("Fibula model node not found in scene. Run earlier steps first.")
    paramNode.SetNodeReferenceID("fibulaModelNode", fibulaModelNode.GetID())

# Call the method
_logic.centerFibulaLine()

print("[BoneReconstructionPlanner] cb_step_21: fibula line centered.")