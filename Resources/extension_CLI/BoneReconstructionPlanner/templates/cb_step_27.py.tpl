# --- BoneReconstructionPlanner: Enter the desired value in "Initial space (mm)". ---
import slicer
from BoneReconstructionPlanner import BoneReconstructionPlannerLogic

# precondition:begin
# Ensure the extension module is active so module.enter() has run.
try:
    slicer.util.selectModule('BoneReconstructionPlanner')
except Exception:
    pass
# precondition:end

try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()
    _bonereconstructionplanner_logic = logic

parameterNode = logic.getParameterNode()
initial_space = {initial_space: 0.0}
parameterNode.SetParameter('initialSpace', str(initial_space))
try:
    parameterNode.Modified()
except Exception:
    pass
_bonereconstructionplanner_logic = logic
print("[BoneReconstructionPlanner] Step 'cb_step_27' completed.")