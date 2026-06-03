try:
    _bonereconstructionplanner_logic
except NameError:
    from BoneReconstructionPlanner import BoneReconstructionPlannerLogic
    _bonereconstructionplanner_logic = BoneReconstructionPlannerLogic()

logic = _bonereconstructionplanner_logic
parameterNode = logic.getParameterNode()

# Initialize parameter defaults if not already set
if not parameterNode.GetParameter("additionalBetweenSpaceOfFibulaPlanes"):
    parameterNode.SetParameter("additionalBetweenSpaceOfFibulaPlanes", "1.5")
if not parameterNode.GetParameter("fibulaCentroidX"):
    parameterNode.SetParameter("fibulaCentroidX", "0.0")
if not parameterNode.GetParameter("fibulaCentroidY"):
    parameterNode.SetParameter("fibulaCentroidY", "0.0")
if not parameterNode.GetParameter("fibulaCentroidZ"):
    parameterNode.SetParameter("fibulaCentroidZ", "0.0")
if not parameterNode.GetParameter("fibulaSegmentsMeasurementMode"):
    parameterNode.SetParameter("fibulaSegmentsMeasurementMode", "center2center")
if not parameterNode.GetParameter("fixCutGoesThroughTheMandibleTwice"):
    parameterNode.SetParameter("fixCutGoesThroughTheMandibleTwice", "False")
if not parameterNode.GetParameter("fixCutGoesThroughTheMandibleTwiceCheckBoxChanged"):
    parameterNode.SetParameter("fixCutGoesThroughTheMandibleTwiceCheckBoxChanged", "False")
if not parameterNode.GetParameter("initialSpace"):
    parameterNode.SetParameter("initialSpace", "0.0")
if not parameterNode.GetParameter("kindOfMandibleResection"):
    parameterNode.SetParameter("kindOfMandibleResection", "Segmental Mandibulectomy")
if not parameterNode.GetParameter("mandibleCentroidX"):
    parameterNode.SetParameter("mandibleCentroidX", "0.0")
if not parameterNode.GetParameter("mandibleCentroidY"):
    parameterNode.SetParameter("mandibleCentroidY", "0.0")
if not parameterNode.GetParameter("mandibleCentroidZ"):
    parameterNode.SetParameter("mandibleCentroidZ", "0.0")
if not parameterNode.GetParameter("mandibleSideToRemove"):
    parameterNode.SetParameter("mandibleSideToRemove", "Removing right side")
if not parameterNode.GetParameter("rightSideLegFibula"):
    parameterNode.SetParameter("rightSideLegFibula", "False")
if not parameterNode.GetParameter("useMoreExactVersionOfPositioningAlgorithm"):
    parameterNode.SetParameter("useMoreExactVersionOfPositioningAlgorithm", "False")
if not parameterNode.GetParameter("useNonDecimatedBoneModelsForPreview"):
    parameterNode.SetParameter("useNonDecimatedBoneModelsForPreview", "True")

# Ensure node references exist for mandible model nodes
for ref_name in ["mandibleModelNode", "decimatedMandibleModelNode"]:
    if parameterNode.GetNodeReference(ref_name) is None:
        nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
        for i in range(nodes.GetNumberOfItems()):
            n = nodes.GetItemAsObject(i)
            if "mandible" in n.GetName().lower():
                parameterNode.SetNodeReferenceID(ref_name, n.GetID())
                break

# Ensure subject hierarchy folder IDs are set
shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
parentFolderID = shNode.GetSceneItemID()
if not parameterNode.GetParameter("parentFolderItemID"):
    parameterNode.SetParameter("parentFolderItemID", str(parentFolderID))

mandiblePlanesFolderID = shNode.GetItemByName("Mandible planes")
if mandiblePlanesFolderID == 0:
    mandiblePlanesFolderID = shNode.CreateFolderItem(parentFolderID, "Mandible planes")
if not parameterNode.GetParameter("mandiblePlanesFolderItemID"):
    parameterNode.SetParameter("mandiblePlanesFolderItemID", str(mandiblePlanesFolderID))

# Execute the step method
logic.generateFibulaPlanesFibulaBonePiecesAndTransformThemToMandible()

print("[Step 22] Fibula planes, bone pieces, and transformation to mandible generated.")
