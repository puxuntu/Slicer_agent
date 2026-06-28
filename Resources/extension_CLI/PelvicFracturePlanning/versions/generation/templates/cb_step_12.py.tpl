# --- PelvicFracturePlanning: Click the "Plan Screws" button. ---
import slicer

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

# Create a new model node for the screw plan output (parent folder for individual screws)
_outputScrew = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'ScrewPlan')
_screw_id = _outputScrew.GetID()

logic.plan_screws(_outputScrew, None)

print("[PelvicFracturePlanning] Step 'cb_step_12' completed.")
