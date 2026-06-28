# --- PelvicFracturePlanning: Choose which fragment needs adjustment in the "Fragment" selection box. ---
import slicer
from PelvicFracturePlanning import cal_BBox

# precondition:begin
# Ensure the extension module is active so module.enter() has run.
_active_module_name = slicer.util.selectedModule()
if _active_module_name != 'PelvicFracturePlanning':
    try:
        slicer.util.selectModule('PelvicFracturePlanning')
    except Exception as _module_enter_error:
        print(f"Warning: could not activate module 'PelvicFracturePlanning': {_module_enter_error}")
# precondition:end

try:
    logic = _pelvicfractureplanning_logic
except NameError:
    from PelvicFracturePlanning import PelvicFracturePlanningLogic
    logic = PelvicFracturePlanningLogic()
    _pelvicfractureplanning_logic = logic

# Use the reduction segmentation node produced by previous reduction step
_fragmentSegNode = logic._reductionNode if hasattr(logic, '_reductionNode') else None
if _fragmentSegNode is None:
    raise RuntimeError("No reduction segmentation node found. Ensure reduction step (cb_step_8) has been completed.")

cal_BBox(_fragmentSegNode)

print("[PelvicFracturePlanning] Step 'cb_step_9' completed.")
