# --- BoneReconstructionPlanner: Manually adjust the slice intersection position by translate and rotate of the cross lines in each view. (Setup) ---
import slicer

# Ensure slice intersection visibility and interaction are enabled (re-apply if previously set)
appLogic = slicer.app.applicationLogic()
if appLogic:
    appLogic.SetIntersectingSlicesEnabled(appLogic.IntersectingSlicesVisibility, True)
    appLogic.SetIntersectingSlicesEnabled(appLogic.IntersectingSlicesInteractive, True)
    appLogic.SetIntersectingSlicesEnabled(appLogic.IntersectingSlicesTranslation, True)
    appLogic.SetIntersectingSlicesEnabled(appLogic.IntersectingSlicesRotation, True)

print("[BoneReconstructionPlanner] Please adjust slice intersection cross lines in each view.")
print("When finished, press the 'Done' button in the workflow panel.")