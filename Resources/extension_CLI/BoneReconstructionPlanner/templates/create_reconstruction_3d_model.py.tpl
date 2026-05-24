import logging

# Reuse existing logic instance if available
try:
    logic = _bonereconstructionplanner_logic
    logging.info("Reusing existing BoneReconstructionPlannerLogic instance.")
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    logic = BoneReconstructionPlannerLogic()
    _bonereconstructionplanner_logic = logic
    logging.info("Created new BoneReconstructionPlannerLogic instance.")

# The method create3DModelOfTheReconstruction() uses the parameter node and
# expects the scene to already contain necessary input data (transformed fibula pieces, cut bones, etc.).
# No additional state setup is required on the logic object itself; it will retrieve
# the parameter node internally.
logic.create3DModelOfTheReconstruction()

print("[BoneReconstructionPlanner] Step 'Create 3D model of the reconstruction for 3D printing' completed.")