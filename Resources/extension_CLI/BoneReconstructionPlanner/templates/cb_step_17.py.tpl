# --- BoneReconstructionPlanner: Click "Add cut plane" button. (Setup) ---
import slicer
from SlicerAIAgentLib.workflow_state import remember_interaction_node
from BoneReconstructionPlanner import BoneReconstructionPlannerLogic

# precondition:begin
# Ensure the extension module is active so module.enter() has run.
try:
    slicer.util.selectModule('BoneReconstructionPlanner')
except Exception as _module_enter_error:
    print(f"Warning: could not activate module 'BoneReconstructionPlanner': {_module_enter_error}")
# precondition:end

try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()

_workflow_before_ids = set()
_workflow_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsPlaneNode")
_workflow_before_count = _workflow_nodes.GetNumberOfItems()
for _workflow_i in range(_workflow_before_count):
    _workflow_n = _workflow_nodes.GetItemAsObject(_workflow_i)
    if _workflow_n is not None:
        _workflow_before_ids.add(_workflow_n.GetID())

logic.addCutPlane()

_workflow_created_node = None
_workflow_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsPlaneNode")
for _workflow_i in range(_workflow_nodes.GetNumberOfItems() - 1, -1, -1):
    _workflow_n = _workflow_nodes.GetItemAsObject(_workflow_i)
    if _workflow_n is not None and _workflow_n.GetID() not in _workflow_before_ids:
        _workflow_created_node = _workflow_n
        break
if _workflow_created_node is not None:
    remember_interaction_node(_workflow_runtime_extension, _workflow_runtime_id, "cb_step_18", _workflow_created_node.GetID(), _workflow_runtime_repeat_index)
_bonereconstructionplanner_logic = logic

print("[BoneReconstructionPlanner] Placement started for step 'cb_step_17'.")