# --- BoneReconstructionPlanner: Click "Update fibula planes over fibula line; update fibula bone pieces and transform them to mandible" to regenerate the reconstruction. ---
from BoneReconstructionPlanner import BoneReconstructionPlannerLogic

# precondition:begin
try:
    slicer.util.selectModule("BoneReconstructionPlanner")
except Exception as _module_enter_error:
    print(f"Warning: could not activate module 'BoneReconstructionPlanner': {_module_enter_error}")
# precondition:end

try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()

# Attempt to find the parameter node
parameterNode = None
nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScriptedModuleNode")
for i in range(nodes.GetNumberOfItems()):
    n = nodes.GetItemAsObject(i)
    if "BoneReconstructionPlanner" in n.GetName():
        parameterNode = n
        break

if parameterNode is None:
    print("Warning: Could not find BoneReconstructionPlanner parameter node.")
else:
    # Default scalar parameters to set if not already set
    scalar_defaults = {
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
        "useNonDecimatedBoneModelsForPreview": "True"
    }
    for param, default_value in scalar_defaults.items():
        if not parameterNode.GetParameter(param):
            parameterNode.SetParameter(param, default_value)

    # Set node references if missing (using fuzzy name matching)
    node_refs = {
        "mandibleSegmentationNode": "vtkMRMLSegmentationNode",
        "fibulaSegmentationNode": "vtkMRMLSegmentationNode",
        "mandibleModelNode": "vtkMRMLModelNode",
        "fibulaModelNode": "vtkMRMLModelNode",
    }
    for ref_name, class_name in node_refs.items():
        existing_id = parameterNode.GetNodeReferenceID(ref_name)
        if existing_id is None:
            # Search scene for a node with matching name fragment
            nodes = slicer.mrmlScene.GetNodesByClass(class_name)
            for i in range(nodes.GetNumberOfItems()):
                n = nodes.GetItemAsObject(i)
                name_lower = n.GetName().lower()
                if ref_name.lower().replace("segmentation","").replace("model","").replace("node","").strip() in name_lower:
                    parameterNode.SetNodeReferenceID(ref_name, n.GetID())
                    break
            else:
                print(f"Warning: Could not find node reference for '{ref_name}'.")

# Also ensure logic attributes are set from parameter node (if logic expects them)
if hasattr(logic, "mandibleSegmentationNode") and logic.mandibleSegmentationNode is None:
    node_id = parameterNode.GetNodeReferenceID("mandibleSegmentationNode") if parameterNode else None
    if node_id:
        logic.mandibleSegmentationNode = slicer.mrmlScene.GetNodeByID(node_id)
if hasattr(logic, "fibulaSegmentationNode") and logic.fibulaSegmentationNode is None:
    node_id = parameterNode.GetNodeReferenceID("fibulaSegmentationNode") if parameterNode else None
    if node_id:
        logic.fibulaSegmentationNode = slicer.mrmlScene.GetNodeByID(node_id)

# Call the method
logic.hardVSPUpdate()

# Store logic for subsequent steps
_bonereconstructionplanner_logic = logic

print("[BoneReconstructionPlanner] hardVSPUpdate() completed.")