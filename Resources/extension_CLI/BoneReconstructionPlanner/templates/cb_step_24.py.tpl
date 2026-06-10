try:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
except ImportError:
    slicer.util.pip_install("BoneReconstructionPlanner")
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
# precondition:begin
try:
    slicer.util.selectModule("BoneReconstructionPlanner")
except Exception as _module_enter_error:
    print(f"Warning: could not activate module 'BoneReconstructionPlanner': {_module_enter_error}")
# precondition:end

try:
    _bonereconstructionplanner_logic
except NameError:
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

# Get or create the parameter node
_parameterNode = None
try:
    _parameterNode = _bonereconstructionplanner_logic.parameterNode
except AttributeError:
    pass
if _parameterNode is None:
    _nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScriptedModuleNode")
    for i in range(_nodes.GetNumberOfItems()):
        n = _nodes.GetItemAsObject(i)
        if "BoneReconstructionPlanner" in n.GetName():
            _parameterNode = n
            break
if _parameterNode is None:
    _parameterNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "BoneReconstructionPlannerParameters")
    _bonereconstructionplanner_logic.parameterNode = _parameterNode

# Set default parameters if not already present
_defaults = {
    "additionalBetweenSpaceOfFibulaPlanes": "1.5",
    "fibulaCentroidX": "0.0",
    "fibulaCentroidY": "0.0",
    "fibulaCentroidZ": "0.0",
    "fibulaSegmentsMeasurementMode": "center2center",
    "fixCutGoesThroughTheMandibleTwice": "False",
    "fixCutGoesThroughTheMandibleTwiceCheckBoxChanged": "False",
    "initialSpace": "0.0",
    "kindOfMandibleResection": "Segmental Mandibulectomy",
    "lockVSP": "False",
    "makeAllMandiblePlanesRotateTogether": "True",
    "mandibleCentroidX": "0.0",
    "mandibleCentroidY": "0.0",
    "mandibleCentroidZ": "0.0",
    "mandiblePlanesPositioningForMaximumBoneContact": "True",
    "mandibleSideToRemove": "Removing right side",
    "rightSideLegFibula": "False",
    "useMoreExactVersionOfPositioningAlgorithm": "False",
    "useNonDecimatedBoneModelsForPreview": "True",
}
for param, value in _defaults.items():
    existing = _parameterNode.GetParameter(param)
    if existing == "" or existing is None:
        _parameterNode.SetParameter(param, value)

# Set up node references (fuzzy matching)
# Mandible segmentation node
_mandibleSegmentationNode = _parameterNode.GetNodeReference("mandibleSegmentationNode")
if _mandibleSegmentationNode is None:
    _nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for i in range(_nodes.GetNumberOfItems()):
        n = _nodes.GetItemAsObject(i)
        if "mandible" in n.GetName().lower():
            _parameterNode.SetNodeReferenceID("mandibleSegmentationNode", n.GetID())
            break

# Fibula segmentation nodes (may be multiple)
_fibulaSegmentationNodes = []
_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
for i in range(_nodes.GetNumberOfItems()):
    n = _nodes.GetItemAsObject(i)
    if "fibula" in n.GetName().lower():
        _fibulaSegmentationNodes.append(n)
if _fibulaSegmentationNodes:
    _parameterNode.SetNodeReferenceID("fibulaSegmentationNode", _fibulaSegmentationNodes[0].GetID())

# Attach parameter node to logic if not already
if not hasattr(_bonereconstructionplanner_logic, "parameterNode") or _bonereconstructionplanner_logic.parameterNode != _parameterNode:
    _bonereconstructionplanner_logic.parameterNode = _parameterNode

# Call the method
_bonereconstructionplanner_logic.hardVSPUpdate()
print("[BoneReconstructionPlanner] hardVSPUpdate completed successfully.")