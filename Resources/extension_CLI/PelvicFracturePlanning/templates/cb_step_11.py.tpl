# --- PelvicFracturePlanning: Click the "Apply adjustments" button. ---
import slicer
from PelvicFracturePlanning import Apply_transform_to_polydata

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

# Retrieve required nodes from the scene by known names
_fragmentModelNode = slicer.util.getFirstNodeByName('FragmentModel')
_adjustTransformNode = slicer.util.getFirstNodeByName('AdjustTransform')
_adjustedModelNode = slicer.util.getFirstNodeByName('AdjustedModel')
if _fragmentModelNode is None or _adjustTransformNode is None or _adjustedModelNode is None:
    raise RuntimeError("Could not find required nodes (FragmentModel, AdjustTransform, AdjustedModel) in the scene. Ensure previous steps are completed.")

Apply_transform_to_polydata(_fragmentModelNode, _adjustTransformNode, _adjustedModelNode)

print("[PelvicFracturePlanning] Step 'cb_step_11' completed.")
