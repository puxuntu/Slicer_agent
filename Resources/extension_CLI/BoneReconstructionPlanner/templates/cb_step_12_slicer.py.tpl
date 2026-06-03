"""
Configure display settings of a markups curve named "mandibular curve"
to show only in "View 1" (3D view) and "Red" (Red slice view).
"""

# Find the mandibular curve markups node by fuzzy name matching
curveNode = None
for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsCurveNode"):
    if "mandibular" in node.GetName().lower() and "curve" in node.GetName().lower():
        curveNode = node
        break

if curveNode is None:
    # Try a broader search if no curve-specific node found
    for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode"):
        if "mandibular" in node.GetName().lower() and "curve" in node.GetName().lower():
            curveNode = node
            break

if curveNode is None:
    raise RuntimeError("Markups curve node containing 'mandibular' and 'curve' not found.")

# Get the display node
displayNode = curveNode.GetDisplayNode()
if displayNode is None:
    raise RuntimeError("Display node not found for the curve node.")

# Collect view node IDs for "View 1" (3D view) and "Red" (Red slice view)
layoutManager = slicer.app.layoutManager()
viewNodeIDs = []

# Add the 3D view "View 1" (first 3D view widget)
threeDViewNode = layoutManager.threeDWidget(0).mrmlViewNode()
viewNodeIDs.append(threeDViewNode.GetID())

# Add the Red slice view
redSliceNode = layoutManager.sliceWidget("Red").mrmlSliceNode()
viewNodeIDs.append(redSliceNode.GetID())

# Restrict the display node to show only in the specified views
displayNode.SetViewNodeIDs(viewNodeIDs)