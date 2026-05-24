try:
    _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

logic = _bonereconstructionplanner_logic

# Ensure parameterNode is set
if not hasattr(logic, 'parameterNode') or logic.parameterNode is None:
    # Find the parameter node in the scene
    paramNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScriptedModuleNode")
    for i in range(paramNodes.GetNumberOfItems()):
        node = paramNodes.GetItemAsObject(i)
        if "BoneReconstructionPlanner" in node.GetName():
            logic.parameterNode = node
            break

# If still not found, create a new parameter node (as fallback)
if logic.parameterNode is None:
    logic.parameterNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "BoneReconstructionPlannerParameters")

# Force hard VSP update
logic.hardVSPUpdate()

print("[BoneReconstructionPlanner] Hard VSP update completed.")