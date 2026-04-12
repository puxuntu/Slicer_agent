"""
SlicerCodeTemplates - Common code patterns and templates for Slicer operations.

Provides reusable code snippets that the AI can reference or use directly.
"""

from typing import Dict, List


class SlicerCodeTemplates:
    """
    Collection of common Slicer Python code patterns.
    
    These templates serve as:
    - Examples for the AI to follow
    - Quick insertion for common operations
    - Best practice references
    """
    
    TEMPLATES = {
        "load_volume": {
            "description": "Load a volume from file",
            "code": '''# Load a scalar volume
volumeNode = slicer.util.loadVolume("/path/to/volume.nrrd")

# Check if loaded successfully
if volumeNode:
    print(f"Loaded volume: {volumeNode.GetName()}")
else:
    print("Failed to load volume")''',
            "tags": ["volume", "load", "file"],
        },
        
        "volume_to_numpy": {
            "description": "Convert volume to numpy array and modify",
            "code": '''# Get volume as numpy array
import numpy as np

# Get the volume node (replace with your node's name)
volumeNode = slicer.util.getNode('MyVolume')

# Get array (returns a view, not a copy)
volumeArray = slicer.util.arrayFromVolume(volumeNode)

# Modify the array (example: invert intensity)
volumeArray[:] = volumeArray.max() - volumeArray

# IMPORTANT: Notify Slicer that the array was modified
slicer.util.arrayFromVolumeModified(volumeNode)

# Update display
volumeNode.GetDisplayNode().Modified()''',
            "tags": ["volume", "numpy", "array", "modify"],
        },
        
        "create_segmentation": {
            "description": "Create a new segmentation and add data",
            "code": '''# Create a new segmentation node
segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'MySegmentation')

# Create a segment
segmentation = segmentationNode.GetSegmentation()
segment = vtk.vtkSegment()
segment.SetName('Segment_1')

# Add the segment
segmentation.AddSegment(segment)

# Import labelmap to segment (example)
# labelmapVolumeNode = slicer.util.getNode('MyLabelmap')
# slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
#     labelmapVolumeNode, segmentationNode)

# Update display
segmentationNode.CreateDefaultDisplayNodes()''',
            "tags": ["segmentation", "segment", "create"],
        },
        
        "volume_rendering": {
            "description": "Setup volume rendering for a volume",
            "code": '''# Get the volume node
volumeNode = slicer.util.getNode('MyVolume')

# Create volume rendering property node
vrDisplayNode = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLVolumeRenderingDisplayNode')

# Set volume node
vrDisplayNode.SetAndObserveVolumeNodeID(volumeNode.GetID())

# Set preset (optional)
# presetNode = slicer.util.getFirstNodeByClassByName(
#     'vtkMRMLVolumePropertyNode', 'CT-AAA')
# vrDisplayNode.SetAndObserveVolumePropertyNodeID(presetNode.GetID())

# Create ROI for cropping (optional)
roiNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode')
vrDisplayNode.SetAndObserveROINodeID(roiNode.GetID())

# Show volume rendering
volumeNode.AddAndObserveDisplayNodeID(vrDisplayNode.GetID())
vrDisplayNode.SetVisibility(True)''',
            "tags": ["volume rendering", "vr", "display", "3d"],
        },
        
        "load_transform": {
            "description": "Load and apply a transform",
            "code": '''# Load a transform from file
transformNode = slicer.util.loadTransform("/path/to/transform.h5")

# Get the node to transform
volumeNode = slicer.util.getNode('MyVolume')

# Apply transform
volumeNode.SetAndObserveTransformNodeID(transformNode.GetID())

# Harden the transform (apply permanently)
slicer.modules.transforms.logic().hardenTransform(volumeNode)''',
            "tags": ["transform", "registration", "apply"],
        },
        
        "create_fiducials": {
            "description": "Create fiducial markups",
            "code": '''# Create a fiducial markup node
markupsNode = slicer.mrmlScene.AddNewNodeByClass(
    'vtkMRMLMarkupsFiducialNode', 'MyFiducials')

# Add fiducial points
markupsNode.AddControlPoint([10.5, 20.3, 15.0], 'Point1')
markupsNode.AddControlPoint([25.0, 30.5, 12.5], 'Point2')
markupsNode.AddControlPoint([15.0, 25.0, 20.0], 'Point3')

# Configure display
markupsNode.GetDisplayNode().SetColor(1, 0, 0)  # Red
markupsNode.GetDisplayNode().SetSelectedColor(0, 1, 0)  # Green when selected
markupsNode.GetDisplayNode().SetGlyphScale(3)

# Show in 3D view
markupsNode.GetDisplayNode().SetVisibility(True)''',
            "tags": ["markup", "fiducial", "point", "landmark"],
        },
        
        "export_to_stl": {
            "description": "Export model to STL file",
            "code": '''# Get the model node
modelNode = slicer.util.getNode('MyModel')

# Export to STL
success = slicer.util.saveNode(modelNode, '/path/to/output.stl')

if success:
    print("Export successful")
else:
    print("Export failed")''',
            "tags": ["model", "export", "stl", "file"],
        },
        
        "segment_editor_threshold": {
            "description": "Use Segment Editor to threshold a volume",
            "code": '''# Get volume and segmentation nodes
volumeNode = slicer.util.getNode('MyVolume')
segmentationNode = slicer.util.getNode('MySegmentation')

# Get Segment Editor widget
segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor

# Set volume and segmentation
segmentEditorWidget.setSourceVolumeNode(volumeNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)

# Create segment if needed
segmentEditorWidget.setCurrentSegmentID('Segment_1')

# Set active effect
segmentEditorWidget.setActiveEffectByName('Threshold')
effect = segmentEditorWidget.activeEffect()

# Configure threshold
effect.setParameter("MinimumThreshold", "100")
effect.setParameter("MaximumThreshold", "500")

# Apply
effect.self().onApply()''',
            "tags": ["segment editor", "segmentation", "threshold", "effect"],
        },
        
        "batch_process": {
            "description": "Batch process multiple files",
            "code": '''import os
import glob

# Get list of files
input_dir = "/path/to/input"
output_dir = "/path/to/output"
file_pattern = os.path.join(input_dir, "*.nrrd")
files = glob.glob(file_pattern)

# Process each file
for i, filepath in enumerate(files):
    print(f"Processing {i+1}/{len(files)}: {filepath}")
    
    # Load
    volumeNode = slicer.util.loadVolume(filepath)
    if not volumeNode:
        print(f"  Failed to load")
        continue
    
    # Process (example: apply filter)
    # ... your processing code here ...
    
    # Save result
    basename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, basename)
    slicer.util.saveNode(volumeNode, output_path)
    
    # Clean up
    slicer.mrmlScene.RemoveNode(volumeNode)
    
    # Process events to keep UI responsive
    slicer.app.processEvents()

print("Batch processing complete")''',
            "tags": ["batch", "loop", "process", "multiple"],
        },
        
        "screenshot": {
            "description": "Capture screenshot of current view",
            "code": '''# Capture screenshot of the 3D view
view = slicer.app.layoutManager().threeDWidget(0).threeDView()

# Create screenshot
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(view.renderWindow())
capture.Update()

# Save to file
writer = vtk.vtkPNGWriter()
writer.SetFileName('/path/to/screenshot.png')
writer.SetInputData(capture.GetOutput())
writer.Write()

print("Screenshot saved")''',
            "tags": ["screenshot", "capture", "image", "export"],
        },
    }
    
    @classmethod
    def getTemplate(cls, name: str) -> Dict:
        """Get a specific template by name."""
        return cls.TEMPLATES.get(name, {})
        
    @classmethod
    def getAllTemplates(cls) -> Dict:
        """Get all templates."""
        return cls.TEMPLATES.copy()
        
    @classmethod
    def findByTag(cls, tag: str) -> List[Dict]:
        """Find templates by tag."""
        results = []
        for name, template in cls.TEMPLATES.items():
            if tag.lower() in [t.lower() for t in template.get("tags", [])]:
                results.append({"name": name, **template})
        return results
        
    @classmethod
    def findByKeyword(cls, keyword: str) -> List[Dict]:
        """Find templates matching a keyword."""
        results = []
        keyword_lower = keyword.lower()
        
        for name, template in cls.TEMPLATES.items():
            # Check name
            if keyword_lower in name.lower():
                results.append({"name": name, **template})
                continue
                
            # Check description
            if keyword_lower in template.get("description", "").lower():
                results.append({"name": name, **template})
                continue
                
            # Check tags
            if any(keyword_lower in tag.lower() for tag in template.get("tags", [])):
                results.append({"name": name, **template})
                
        return results
        
    @classmethod
    def formatForPrompt(cls, templates: List[Dict]) -> str:
        """Format templates for inclusion in a prompt."""
        if not templates:
            return ""
            
        lines = ["## RELEVANT CODE EXAMPLES:\n"]
        
        for t in templates[:3]:  # Limit to 3 examples
            lines.append(f"### {t.get('description', t.get('name', 'Example'))}")
            lines.append("```python")
            lines.append(t.get('code', ''))
            lines.append("```\n")
            
        return "\n".join(lines)
