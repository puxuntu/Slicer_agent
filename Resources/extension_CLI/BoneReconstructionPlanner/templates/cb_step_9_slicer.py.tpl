# Enable slice intersection visibility and interaction modes (Translate, Rotate)

appLogic = slicer.app.applicationLogic()
if appLogic:
    # Show the slice intersection lines
    appLogic.SetIntersectingSlicesEnabled(
        appLogic.IntersectingSlicesVisibility, True)
    # Enable interactive handle mode (prerequisite for translate/rotate)
    appLogic.SetIntersectingSlicesEnabled(
        appLogic.IntersectingSlicesInteractive, True)
    # Enable the translate interaction handle
    appLogic.SetIntersectingSlicesEnabled(
        appLogic.IntersectingSlicesTranslation, True)
    # Enable the rotate interaction handle
    appLogic.SetIntersectingSlicesEnabled(
        appLogic.IntersectingSlicesRotation, True)