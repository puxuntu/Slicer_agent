"""
SkillContextManager - Manages skill-based context for LLM prompts.

Integrates with the slicer-skill-full knowledge base located alongside the
extension at: Resources/Skills/slicer-skill-full (relative to the extension root).

The AI agent uses tools (Grep, Glob, ReadFile) to search the skill directly.
This manager provides:
1. Skill mode detection (full/lightweight/web)
2. API guidance hints based on prompt keywords
3. Current Slicer MRML scene context
"""

import json
import logging
import os
from typing import Dict, List, Optional

import slicer

logger = logging.getLogger(__name__)


class SkillContextManager:
    """
    Manages context for the Slicer skill-based knowledge system.
    
    Features:
    - Detects skill mode from .setup-stamp.json
    - Provides API guidance hints for the AI
    - Provides current MRML scene context
    """
    
    # Path to the skill directory (resolved relative to this file's location)
    SKILL_PATH = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Resources', 'Skills', 'slicer-skill-full'
    ))
    
    # API guidance hints based on keywords
    # These tell the AI which Slicer APIs to use - NOT search commands
    TOPIC_SEARCH_HINTS = {
        # Volumes
        "volume": [
            "Use SampleData.SampleDataLogic().downloadMRHead() for example volumes",
            "Or use slicer.util.loadVolume(filepath) for loading from file"
        ],
        "volumes": [
            "Use slicer.util.loadVolume() to load volumes",
            "Use slicer.util.arrayFromVolume() for numpy access - call arrayFromVolumeModified() after changes"
        ],
        "image": [
            "Use slicer.util.loadVolume() for image volumes"
        ],
        "nrrd": [
            "Use slicer.util.loadVolume('path.nrrd') for NRRD files"
        ],
        "nifti": [
            "Use slicer.util.loadVolume('path.nii.gz') for NIfTI files"
        ],
        "load": [
            "Use slicer.util.loadVolume(), loadModel(), loadSegmentation(), loadTransform()",
            "Or use SampleData.SampleDataLogic().downloadMRHead() for examples"
        ],
        "save": [
            "Use slicer.util.saveNode(node) or slicer.util.exportNode(node, filepath)"
        ],
        "export": [
            "Use slicer.util.exportNode(node, filepath) to export to new file"
        ],
        "rendering": [
            "Use slicer.modules.volumerendering.logic() for volume rendering"
        ],
        "volume rendering": [
            "Create vtkMRMLVolumeRenderingDisplayNode, set volume and property"
        ],
        "numpy": [
            "Use slicer.util.arrayFromVolume() to get numpy array view (KJI order)",
            "IMPORTANT: Call slicer.util.arrayFromVolumeModified(volumeNode) after modifying"
        ],
        "array": [
            "Volume arrays are in KJI order (slice, row, column)",
            "Always call slicer.util.arrayFromVolumeModified() after modifying arrays"
        ],
        
        # Segmentations
        "segmentation": [
            "Create vtkMRMLSegmentationNode, use segmentations.logic() for import/export",
            "Use slicer.modules.segmenteditor.widgetRepresentation() for editor effects"
        ],
        "segmentations": [
            "Use slicer.modules.segmentations.logic() for segmentation operations",
            "Create vtkMRMLSegmentationNode for new segmentations"
        ],
        "segment": [
            "Use segmentation.AddSegmentFromClosedSurfaceRepresentation() or similar"
        ],
        "labelmap": [
            "Use slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode()"
        ],
        "mask": [
            "Use segmentation operations or numpy array masking on volume arrays"
        ],
        "editor": [
            "Use slicer.modules.segmenteditor.widgetRepresentation().self().editor"
        ],
        "segment editor": [
            "Access editor via slicer.modules.segmenteditor.widgetRepresentation()",
            "Use editor.setActiveEffectByName() to select effects"
        ],
        "effect": [
            "Use segment editor effects via setActiveEffectByName()",
            "Effects: Threshold, Paint, Draw, LevelTracing, GrowFromSeeds, etc."
        ],
        "threshold": [
            "Use Threshold effect in Segment Editor: setActiveEffectByName('Threshold')"
        ],
        
        # Markups
        "markups": [
            "Use slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'Name')",
            "Or use slicer.util.loadMarkups() for loading from file"
        ],
        "fiducial": [
            "Create vtkMRMLMarkupsFiducialNode, use AddControlPoint([x,y,z], 'name')"
        ],
        "point list": [
            "Use vtkMRMLMarkupsFiducialNode for point lists",
            "Use GetNumberOfControlPoints() and GetNthControlPointPosition()"
        ],
        "control point": [
            "Use AddControlPoint(), SetNthControlPointPosition(), RemoveNthControlPoint()"
        ],
        "curve": [
            "Create vtkMRMLMarkupsCurveNode for curves"
        ],
        "plane": [
            "Create vtkMRMLMarkupsPlaneNode for planes"
        ],
        "roi": [
            "Create vtkMRMLMarkupsROINode for ROI"
        ],
        
        # Models
        "model": [
            "Use slicer.util.loadModel(filepath) for loading models",
            "Or use slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')"
        ],
        "models": [
            "Use slicer.util.loadModel() for STL, OBJ, PLY files"
        ],
        "mesh": [
            "Models are vtkPolyData - access via modelNode.GetMesh()",
            "Use slicer.util.arrayFromModelPoints() for point arrays"
        ],
        "stl": [
            "Use slicer.util.loadModel('path.stl') for STL files"
        ],
        "polydata": [
            "Use modelNode.GetMesh() to get vtkPolyData",
            "Use slicer.util.arrayFromModelPoints() for points array"
        ],
        
        # Transforms
        "transform": [
            "Create vtkMRMLTransformNode, use SetMatrixTransformToParent()",
            "Use slicer.modules.transforms.logic().hardenTransform(node) to apply"
        ],
        "transforms": [
            "Use slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')",
            "Apply to nodes via node.SetAndObserveTransformNodeID(transform.GetID())"
        ],
        "registration": [
            "Use BRAINSFit, Elastix, or ANTs modules for registration",
            "Check Modules/CLI/ for registration CLI modules"
        ],
        "linear": [
            "Use vtkMatrix4x4 for linear transforms, set via transformNode.SetMatrixTransformToParent()"
        ],
        "harden": [
            "Use slicer.modules.transforms.logic().hardenTransform(node) to bake transform"
        ],
        
        # DICOM
        "dicom": [
            "Use DICOM module: slicer.modules.DICOM.widgetRepresentation()",
            "Or use slicer.util.loadVolume() if already imported to database"
        ],
        "pacs": [
            "Use DICOMweb or DIMSE for PACS - check DICOM module"
        ],
        "import": [
            "Use slicer.util.loadVolume() for files, DICOM module for DICOM"
        ],
        
        # GUI
        "gui": [
            "Use slicer.util.selectModule('ModuleName') to switch modules",
            "Use slicer.app.layoutManager() for layout operations"
        ],
        "layout": [
            "Use slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayout...)"
        ],
        "view": [
            "Use slicer.app.layoutManager().sliceWidget('Red') for slice views",
            "Use threeDWidget for 3D views"
        ],
        "module": [
            "Use slicer.util.selectModule('ModuleName') to switch modules"
        ],
        "extension": [
            "Use Extension Manager or slicer.app.applicationLogic().GetExtensionManager()"
        ],
        
        # Plots
        "plot": [
            "Create vtkMRMLPlotSeriesNode and vtkMRMLPlotChartNode",
            "Add to scene and set in layout"
        ],
        "chart": [
            "Use vtkMRMLPlotChartNode and vtkMRMLPlotSeriesNode for plotting"
        ],
        
        # Other topics
        "screenshot": [
            "Use slicer.util.captureScreenshot() or slicer.app.layoutManager().capture()"
        ],
        "sequence": [
            "Use vtkMRMLSequenceNode for 4D data",
            "Use SequenceBrowser module for playback"
        ],
        "subject hierarchy": [
            "Use slicer.mrmlScene.GetSubjectHierarchyNode() for hierarchy operations"
        ],
        "tractography": [
            "Use vtkMRMLFiberBundleNode for DTI tractography",
            "Load via slicer.util.loadFiberBundle()"
        ],
        "batch": [
            "Use Python loops with slicer.util.loadVolume() etc. for batch processing",
            "Call slicer.app.processEvents() periodically to keep UI responsive"
        ],
        "web": [
            "Use slicer.modules.webserver for web server functionality"
        ],
        
        # MRML
        "mrml": [
            "Use slicer.mrmlScene for scene operations",
            "Node types: vtkMRMLScalarVolumeNode, vtkMRMLModelNode, etc."
        ],
        "scene": [
            "Use slicer.mrmlScene for scene-wide operations",
            "Use GetNodesByClass() to find nodes"
        ],
        "node": [
            "Use node.GetID() for unique identification (names are not unique!)",
            "Use slicer.mrmlScene.GetNodeByID() to get by ID"
        ],
        
        # VTK/ITK
        "vtk": [
            "Use vtkSmartPointer for proper reference counting",
            "Common classes: vtkMatrix4x4, vtkTransform, vtkPolyData"
        ],
        "itk": [
            "Use via slicer.modules.SimpleITK or vtkITK bridge"
        ],
    }
    
    def __init__(self):
        """
        Initialize the skill context manager.
        """
        self.skill_path = self.SKILL_PATH
        self._skill_mode = self._detect_skill_mode()
        
        if not os.path.exists(self.skill_path):
            logger.warning(f"Skill not found at {self.skill_path}")
        else:
            logger.info(f"SkillContextManager initialized with skill at: {self.skill_path}")
            logger.info(f"Skill mode: {self._skill_mode}")
    
    def _detect_skill_mode(self) -> str:
        """
        Detect the skill mode by reading .setup-stamp.json.
        
        Returns:
            "full", "lightweight", "web", or "unknown"
        """
        stamp_path = os.path.join(self.skill_path, ".setup-stamp.json")
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, 'r', encoding='utf-8') as f:
                    stamp = json.load(f)
                return stamp.get("mode", "unknown")
            except Exception as e:
                logger.warning(f"Failed to read setup stamp: {e}")
        return "unknown"
    
    def get_skill_mode(self) -> str:
        """Get the current skill mode."""
        return self._skill_mode
    
    def _generate_search_hints(self, prompt: str) -> List[str]:
        """
        Generate API guidance hints based on prompt keywords.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            List of API guidance hints for the AI
        """
        prompt_lower = prompt.lower()
        hints = []
        matched_topics = set()
        
        # Match keywords to hints
        for keyword, hint_list in sorted(self.TOPIC_SEARCH_HINTS.items(), key=lambda x: -len(x[0])):
            if keyword in prompt_lower and keyword not in matched_topics:
                hints.extend(hint_list)
                matched_topics.add(keyword)
        
        # Add default hints if no specific matches
        if not hints:
            hints = [
                "Use slicer.util functions for common operations",
                "Check if SampleData module has example data for your task"
            ]
        
        return hints[:10]  # Limit to top 10 hints
    
    def _get_scene_context(self) -> Optional[Dict]:
        """
        Get context about the current Slicer scene.
        
        Returns:
            Dictionary with node counts and sample node names
        """
        try:
            scene = slicer.mrmlScene
            
            context = {
                "node_counts": {},
                "sample_node_names": [],
            }
            
            node_classes = [
                ("vtkMRMLScalarVolumeNode", "Volume"),
                ("vtkMRMLLabelMapVolumeNode", "LabelMap"),
                ("vtkMRMLModelNode", "Model"),
                ("vtkMRMLSegmentationNode", "Segmentation"),
                ("vtkMRMLTransformNode", "Transform"),
                ("vtkMRMLMarkupsFiducialNode", "Fiducial"),
                ("vtkMRMLMarkupsCurveNode", "Curve"),
                ("vtkMRMLMarkupsPlaneNode", "Plane"),
                ("vtkMRMLMarkupsROINode", "ROI"),
            ]
            
            for class_name, display_name in node_classes:
                nodes = scene.GetNodesByClass(class_name)
                count = nodes.GetNumberOfItems()
                if count > 0:
                    context["node_counts"][display_name] = count
            
            all_nodes = scene.GetNodes()
            for i in range(min(5, all_nodes.GetNumberOfItems())):
                node = all_nodes.GetItemAsObject(i)
                if node and node.GetName():
                    context["sample_node_names"].append(node.GetName())
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to get scene context: {e}")
            return None
    
    def buildContext(self, prompt: str) -> Dict:
        """
        Build context for a given prompt.
        
        The context includes:
        - Skill path and mode
        - API guidance hints
        - Current MRML scene state
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Dictionary with skill info, hints, and scene context
        """
        context = {
            "skill_path": self.skill_path,
            "skill_mode": self._skill_mode,
            "api_hints": self._generate_search_hints(prompt),
            "scene": None,
        }
        
        # Add scene context
        scene_context = self._get_scene_context()
        if scene_context:
            context["scene"] = scene_context
        
        return context
    
    def formatContextForPrompt(self, context: Dict) -> str:
        """
        Format the context dictionary into a string suitable for LLM prompt.
        
        Args:
            context: Context dictionary from buildContext()
            
        Returns:
            Formatted context string
        """
        lines = []
        
        # Add API guidance
        hints = context.get("api_hints", [])
        if hints:
            lines.append("## API GUIDANCE")
            for hint in hints:
                lines.append(f"- {hint}")
            lines.append("")
        
        # Add scene context
        if context.get("scene"):
            scene = context["scene"]
            lines.append("## CURRENT SLICER SCENE")
            if scene.get("node_counts"):
                lines.append("Nodes in scene:")
                for node_type, count in scene["node_counts"].items():
                    lines.append(f"  - {node_type}: {count}")
            if scene.get("sample_node_names"):
                lines.append(f"Sample node names: {', '.join(scene['sample_node_names'][:3])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_status(self) -> Dict:
        """
        Get current status of the skill context manager.
        
        Returns:
            Dictionary with skill status information
        """
        return {
            "skill_path": self.skill_path,
            "skill_exists": os.path.exists(self.skill_path),
            "skill_mode": self._skill_mode,
        }
