# Configure display of mandibular curve for View 1 and Red view
# Find the curve node by fuzzy name matching
curveNode = None
for i in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsCurveNode")):
    node = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLMarkupsCurveNode")
    if "mandibular" in node.GetName().lower():
        curveNode = node
        break

if curveNode is None:
    raise ValueError("Mandibular curve node not found. "
                     "Expected a vtkMRMLMarkupsCurveNode with 'mandibular' in its name.")

# Ensure a display node exists
displayNode = curveNode.GetMarkupsDisplayNode()
if displayNode is None:
    curveNode.CreateDefaultDisplayNodes()
    displayNode = curveNode.GetMarkupsDisplayNode()

# Resolve the Red slice view node by layout name
redSliceNode = None
sliceNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSliceNode")
for i in range(sliceNodes.GetNumberOfItems()):
    sn = sliceNodes.GetItemAsObject(i)
    if sn.GetLayoutName() == "Red":
        redSliceNode = sn
        break

# Resolve View 1 (the first 3D view node)
viewNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLViewNode")

if redSliceNode is None or viewNode is None:
    raise ValueError("Could not find Red slice view or 3D view node.")

# Enable overall visibility and 2D (slice) visibility for markups
displayNode.SetVisibility(True)
displayNode.SetVisibility2D(True)

# Restrict display to only the specified views
displayNode.RemoveAllViewNodeIDs()
displayNode.AddViewNodeID(redSliceNode.GetID())
displayNode.AddViewNodeID(viewNode.GetID())