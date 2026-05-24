import logging
try:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
except ImportError:
    raise ImportError("BoneReconstructionPlanner module not found. Please install the extension.")

try:
    logic = _bonereconstructionplanner_logic
except NameError:
    logic = BoneReconstructionPlannerLogic()
    _bonereconstructionplanner_logic = logic

logging.info("Calling onGenerateFibulaPlanesTimerTimeout to compute fibula planes, bone pieces, and transforms.")
logic.onGenerateFibulaPlanesTimerTimeout()
print("[BoneReconstructionPlanner] Fibula planes and bone pieces generated and transformed to mandible.")