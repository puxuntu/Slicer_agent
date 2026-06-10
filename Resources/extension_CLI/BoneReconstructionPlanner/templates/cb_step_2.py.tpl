# --- BoneReconstructionPlanner: In the "Select mandibular segmentation" section, choose the mandibular segmentation. ---
import slicer
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
    _bonereconstructionplanner_logic = logic

parameterNode = logic.getParameterNode()
mandibular_segmentation_node_name = {mandibular_segmentation_node_name: ''}
mandibular_segmentation_node = None
if mandibular_segmentation_node_name:
    try:
        mandibular_segmentation_node = slicer.util.getNode(mandibular_segmentation_node_name)
    except Exception:
        pass
if mandibular_segmentation_node is None:
    _nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLSegmentationNode')
    _keywords = ['mandibularsegmentation', 'mandibular']
    for _i in range(_nodes.GetNumberOfItems()):
        _candidate = _nodes.GetItemAsObject(_i)
        _name = (_candidate.GetName() or '').lower()
        if not _keywords or any(_kw in _name for _kw in _keywords):
            mandibular_segmentation_node = _candidate
            break
if mandibular_segmentation_node is None:
    raise RuntimeError('Could not find node for parameter reference mandibularSegmentation')
parameterNode.SetNodeReferenceID('mandibularSegmentation', mandibular_segmentation_node.GetID())
try:
    parameterNode.Modified()
except Exception:
    pass
_bonereconstructionplanner_logic = logic
print("[BoneReconstructionPlanner] Step 'cb_step_2' completed.")