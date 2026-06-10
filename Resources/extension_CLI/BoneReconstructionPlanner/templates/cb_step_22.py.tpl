# --- BoneReconstructionPlanner: Tick the "Automatic mandibular planes positioning for maximum bones contact area" checkbox. ---
import slicer
from BoneReconstructionPlanner import BoneReconstructionPlannerLogic

# precondition:begin
# Ensure the extension module is loaded so module.enter() has run.
_active_module = slicer.app.moduleManager().module('BoneReconstructionPlanner')
if _active_module is None:
    try:
        slicer.util.selectModule('BoneReconstructionPlanner')
    except Exception as _module_enter_error:
        print(f"Warning: could not activate module 'BoneReconstructionPlanner': {_module_enter_error}")
# precondition:end

try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()
    _bonereconstructionplanner_logic = logic

parameterNode = logic.getParameterNode()
parameterNode.SetParameter('mandiblePlanesPositioningForMaximumBoneContact', 'True')
try:
    parameterNode.Modified()
except Exception:
    pass
_bonereconstructionplanner_logic = logic
print("[BoneReconstructionPlanner] Step 'cb_step_22' completed.")