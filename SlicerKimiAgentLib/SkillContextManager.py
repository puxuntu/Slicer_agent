"""
SkillContextManager - Manages skill-based context for LLM prompts.

Reads code examples from the bundled Slicer script repository located at:
SlicerKimiAgent/Resources/Skills/script_repository/

This includes 15 markdown files covering all major Slicer operations:
volumes, segmentations, markups, models, transforms, dicom, gui, etc.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import slicer

logger = logging.getLogger(__name__)


class SkillContextManager:
    """
    Manages context from the bundled Slicer script repository.
    
    Features:
    - Reads from bundled script repository markdown files (Resources/Skills/script_repository/)
    - Extracts code examples from ```python blocks
    - Keyword-based topic matching covering 15 major Slicer topics
    - Scene-aware context enrichment
    
    The script repository is bundled with the extension at:
    SlicerKimiAgent/Resources/Skills/script_repository/
    """
    
    # Path to bundled script repository (relative to this file)
    # Located at: SlicerKimiAgent/Resources/Skills/script_repository/
    SCRIPT_REPO_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Resources', 'Skills', 'script_repository'
    )
    
    # Topic to file mapping - COMPLETE mapping for all 15 script repository files
    TOPIC_FILE_MAPPING = {
        # volumes.md
        "volume": "volumes.md",
        "volumes": "volumes.md",
        "image": "volumes.md",
        "nrrd": "volumes.md",
        "nifti": "volumes.md",
        "dicom": "volumes.md",
        "load": "volumes.md",
        "save": "volumes.md",
        "export": "volumes.md",
        "voxel": "volumes.md",
        "slice": "volumes.md",
        "rendering": "volumes.md",
        "volume rendering": "volumes.md",
        "mip": "volumes.md",
        "window": "volumes.md",
        "level": "volumes.md",
        "colormap": "volumes.md",
        "dti": "volumes.md",
        "tensor": "volumes.md",
        "simpleitk": "volumes.md",
        "sitk": "volumes.md",
        "slab": "volumes.md",
        "clone": "volumes.md",
        "numpy": "volumes.md",
        "array": "volumes.md",
        "modify": "volumes.md",
        
        # segmentations.md
        "segmentation": "segmentations.md",
        "segmentations": "segmentations.md",
        "segment": "segmentations.md",
        "segments": "segmentations.md",
        "labelmap": "segmentations.md",
        "label": "segmentations.md",
        "mask": "segmentations.md",
        "contour": "segmentations.md",
        "3d": "segmentations.md",
        "surface": "segmentations.md",
        "editor": "segmentations.md",
        "segment editor": "segmentations.md",
        "effect": "segmentations.md",
        "threshold": "segmentations.md",
        "paint": "segmentations.md",
        "draw": "segmentations.md",
        "scissors": "segmentations.md",
        "level tracing": "segmentations.md",
        "grow from seeds": "segmentations.md",
        "statistics": "segmentations.md",
        "centroid": "segmentations.md",
        "bounding box": "segmentations.md",
        "obb": "segmentations.md",
        "hollow": "segmentations.md",
        "slicerio": "segmentations.md",
        
        # markups.md
        "markups": "markups.md",
        "markup": "markups.md",
        "fiducial": "markups.md",
        "fiducials": "markups.md",
        "point list": "markups.md",
        "control point": "markups.md",
        "control points": "markups.md",
        "curve": "markups.md",
        "curves": "markups.md",
        "line": "markups.md",
        "lines": "markups.md",
        "plane": "markups.md",
        "planes": "markups.md",
        "roi": "markups.md",
        "region of interest": "markups.md",
        "placement": "markups.md",
        "angle": "markups.md",
        "distance": "markups.md",
        "measurement": "markups.md",
        "measure": "markups.md",
        "json": "markups.md",
        "fcsv": "markups.md",
        "csv": "markups.md",
        
        # models.md
        "model": "models.md",
        "models": "models.md",
        "mesh": "models.md",
        "surface": "models.md",
        "polydata": "models.md",
        "stl": "models.md",
        "obj": "models.md",
        "gltf": "models.md",
        "vrml": "models.md",
        "ply": "models.md",
        "vertex": "models.md",
        "vertices": "models.md",
        "triangle": "models.md",
        "cell": "models.md",
        "cells": "models.md",
        "texture": "models.md",
        "curvature": "models.md",
        "filter": "models.md",
        "decimate": "models.md",
        "smooth": "models.md",
        "distance": "models.md",
        "comparison": "models.md",
        "rasterize": "models.md",
        
        # transforms.md
        "transform": "transforms.md",
        "transforms": "transforms.md",
        "linear": "transforms.md",
        "grid": "transforms.md",
        "displacement": "transforms.md",
        "rotation": "transforms.md",
        "translate": "transforms.md",
        "translation": "transforms.md",
        "rigid": "transforms.md",
        "affine": "transforms.md",
        "bspline": "transforms.md",
        "matrix": "transforms.md",
        "harden": "transforms.md",
        "itk": "transforms.md",
        "lps": "transforms.md",
        "ras": "transforms.md",
        "trajectory": "transforms.md",
        
        # registration.md
        "registration": "registration.md",
        "register": "registration.md",
        "align": "registration.md",
        "alignment": "registration.md",
        "brainsfit": "registration.md",
        "elastix": "registration.md",
        "ants": "registration.md",
        "resample": "registration.md",
        
        # dicom.md
        "pacs": "dicom.md",
        "dicomweb": "dicom.md",
        "import": "dicom.md",
        "tags": "dicom.md",
        "query": "dicom.md",
        "retrieve": "dicom.md",
        "dimse": "dicom.md",
        "study": "dicom.md",
        "studies": "dicom.md",
        "series": "dicom.md",
        "patient": "dicom.md",
        "patients": "dicom.md",
        "orthanc": "dicom.md",
        "kheops": "dicom.md",
        "database": "dicom.md",
        "browser": "dicom.md",
        "export dicom": "dicom.md",
        "dcmtk": "dicom.md",
        
        # gui.md
        "gui": "gui.md",
        "ui": "gui.md",
        "layout": "gui.md",
        "view": "gui.md",
        "views": "gui.md",
        "3d view": "gui.md",
        "slice view": "gui.md",
        "crosshair": "gui.md",
        "mouse": "gui.md",
        "keyboard": "gui.md",
        "shortcut": "gui.md",
        "color": "gui.md",
        "legend": "gui.md",
        "hanging protocol": "gui.md",
        "module": "gui.md",
        "select module": "gui.md",
        "install": "gui.md",
        "extension": "gui.md",
        "pip": "gui.md",
        "scene": "gui.md",
        "mrml": "gui.md",
        
        # plots.md
        "plot": "plots.md",
        "plots": "plots.md",
        "chart": "plots.md",
        "charts": "plots.md",
        "graph": "plots.md",
        "histogram": "plots.md",
        "matplotlib": "plots.md",
        "svg": "plots.md",
        "series": "plots.md",
        "table": "plots.md",
        
        # screencapture.md
        "screenshot": "screencapture.md",
        "capture": "screencapture.md",
        "image": "screencapture.md",
        "png": "screencapture.md",
        "video": "screencapture.md",
        "animation": "screencapture.md",
        "sweep": "screencapture.md",
        "transparent": "screencapture.md",
        "background": "screencapture.md",
        
        # sequences.md
        "sequence": "sequences.md",
        "sequences": "sequences.md",
        "4d": "sequences.md",
        "time series": "sequences.md",
        "cine": "sequences.md",
        "volume sequence": "sequences.md",
        "browser": "sequences.md",
        "temporal": "sequences.md",
        "concatenate": "sequences.md",
        "intensity plot": "sequences.md",
        
        # subjecthierarchy.md
        "subject hierarchy": "subjecthierarchy.md",
        "hierarchy": "subjecthierarchy.md",
        "folder": "subjecthierarchy.md",
        "folders": "subjecthierarchy.md",
        "tree": "subjecthierarchy.md",
        "organization": "subjecthierarchy.md",
        "organize": "subjecthierarchy.md",
        "reparent": "subjecthierarchy.md",
        "item": "subjecthierarchy.md",
        "items": "subjecthierarchy.md",
        "children": "subjecthierarchy.md",
        "parent": "subjecthierarchy.md",
        "visibility": "subjecthierarchy.md",
        
        # tractography.md
        "tractography": "tractography.md",
        "fiber": "tractography.md",
        "fibers": "tractography.md",
        "streamline": "tractography.md",
        "streamlines": "tractography.md",
        "dti": "tractography.md",
        "dmri": "tractography.md",
        "diffusion": "tractography.md",
        "blender": "tractography.md",
        
        # batch.md
        "batch": "batch.md",
        "batch processing": "batch.md",
        "iterate": "batch.md",
        "loop": "batch.md",
        "machine learning": "batch.md",
        "ml": "batch.md",
        "deep learning": "batch.md",
        "patch": "batch.md",
        "patches": "batch.md",
        "monai": "batch.md",
        "crop": "batch.md",
        "bounding box": "batch.md",
        
        # webserver.md
        "web": "webserver.md",
        "web server": "webserver.md",
        "http": "webserver.md",
        "rest": "webserver.md",
        "api": "webserver.md",
        "endpoint": "webserver.md",
        "static": "webserver.md",
        "serve": "webserver.md",
    }
    
    # File descriptions for better context
    FILE_DESCRIPTIONS = {
        "volumes.md": "Working with image volumes (load, save, render, numpy arrays)",
        "segmentations.md": "Working with segmentations and Segment Editor",
        "markups.md": "Working with fiducials, curves, lines, planes, ROIs",
        "models.md": "Working with surface models and meshes",
        "transforms.md": "Working with transforms and coordinate systems",
        "registration.md": "Image registration and alignment",
        "dicom.md": "DICOM import, export, and PACS integration",
        "gui.md": "User interface, layouts, views, and interaction",
        "plots.md": "Creating charts and plots",
        "screencapture.md": "Screenshots, videos, and image export",
        "sequences.md": "4D volume sequences and time series",
        "subjecthierarchy.md": "Data organization and subject hierarchy",
        "tractography.md": "Diffusion MRI and fiber tracking",
        "batch.md": "Batch processing and machine learning",
        "webserver.md": "Web server and REST API",
    }
    
    def __init__(self):
        """
        Initialize the skill context manager.
        Reads from the bundled script repository at Resources/Skills/script_repository/
        """
        self.script_repo_path = self.SCRIPT_REPO_PATH
        self._file_cache: Dict[str, str] = {}  # Cache for file contents
        self._code_cache: Dict[str, List[str]] = {}  # Cache for extracted code
        
        # Verify path exists
        if not os.path.exists(self.script_repo_path):
            logger.warning(f"Script repository not found at {self.script_repo_path}")
        else:
            logger.info(f"SkillContextManager initialized with path: {self.script_repo_path}")
            self._preloadCache()
    
    def _preloadCache(self):
        """Preload all markdown files into cache for faster access."""
        try:
            for filename in os.listdir(self.script_repo_path):
                if filename.endswith('.md'):
                    filepath = os.path.join(self.script_repo_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            self._file_cache[filename] = f.read()
                        # Pre-extract code blocks
                        self._code_cache[filename] = self._extractCodeBlocks(self._file_cache[filename])
                        logger.info(f"Cached {filename}: {len(self._code_cache[filename])} code examples")
                    except Exception as e:
                        logger.warning(f"Failed to cache {filename}: {e}")
        except Exception as e:
            logger.warning(f"Failed to preload cache: {e}")
    
    def _extractCodeBlocks(self, content: str) -> List[str]:
        """
        Extract python code blocks from markdown content.
        
        Args:
            content: Markdown file content
            
        Returns:
            List of code strings
        """
        code_blocks = []
        
        # Match ```python ... ``` blocks
        python_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_pattern, content, re.DOTALL)
        code_blocks.extend(matches)
        
        # Also match ``` ... ``` blocks (without language specifier)
        generic_pattern = r'```\s*\n(.*?)\n```'
        generic_matches = re.findall(generic_pattern, content, re.DOTALL)
        # Filter out non-Python blocks (heuristic: check for common Python patterns)
        for match in generic_matches:
            if any(keyword in match for keyword in ['slicer.', 'vtk.', 'import ', 'def ', 'class ']):
                code_blocks.append(match)
        
        return code_blocks
    
    def _identifyTopics(self, prompt: str) -> List[str]:
        """
        Identify relevant topics from the prompt.
        
        Args:
            prompt: User's input prompt (lowercase)
            
        Returns:
            List of matching filenames
        """
        prompt_lower = prompt.lower()
        matched_files = set()
        
        # Check for multi-word matches first (to prioritize "segment editor" over "editor")
        for keyword, filename in sorted(self.TOPIC_FILE_MAPPING.items(), key=lambda x: -len(x[0])):
            if keyword in prompt_lower:
                matched_files.add(filename)
        
        return list(matched_files)
    
    def _getCodeExamples(self, filename: str, max_examples: int = 5) -> List[str]:
        """
        Get code examples from a file.
        
        Args:
            filename: Markdown filename
            max_examples: Maximum number of examples to return
            
        Returns:
            List of code example strings
        """
        if filename in self._code_cache:
            return self._code_cache[filename][:max_examples]
        
        # Fallback: read from disk
        filepath = os.path.join(self.script_repo_path, filename)
        if not os.path.exists(filepath):
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._extractCodeBlocks(content)[:max_examples]
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return []
    
    def _getFileSummary(self, filename: str) -> str:
        """Get a brief summary of what the file covers."""
        return self.FILE_DESCRIPTIONS.get(filename, "Slicer operations")
    
    def _getSceneContext(self) -> Optional[Dict]:
        """Get context about the current Slicer scene."""
        try:
            scene = slicer.mrmlScene
            
            context = {
                "node_counts": {},
                "sample_node_names": [],
            }
            
            # Count nodes by class
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
            
            # Get a few sample node names
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
        Build context for a given prompt by reading relevant script repository files.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Dictionary with relevant examples, APIs, and scene context
        """
        # Identify relevant topics/files
        relevant_files = self._identifyTopics(prompt)
        
        # Build context
        context = {
            "topics": [],
            "examples": [],
            "file_descriptions": {},
            "scene": None,
        }
        
        # Collect examples from relevant files
        for filename in relevant_files[:3]:  # Limit to top 3 files
            context["topics"].append(filename.replace('.md', ''))
            context["file_descriptions"][filename] = self._getFileSummary(filename)
            
            examples = self._getCodeExamples(filename, max_examples=3)
            for example in examples:
                context["examples"].append({
                    "source": filename,
                    "code": example.strip()
                })
        
        # Deduplicate examples (by full code content, normalized)
        seen_codes = set()
        unique_examples = []
        for ex in context["examples"]:
            # Normalize code: strip whitespace and use full content for hashing
            normalized = ' '.join(ex["code"].split())
            code_hash = hash(normalized)
            if code_hash not in seen_codes:
                seen_codes.add(code_hash)
                unique_examples.append(ex)
        context["examples"] = unique_examples[:5]  # Limit total examples
        
        # Add scene context if available
        scene_context = self._getSceneContext()
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
        
        # Add file descriptions
        if context.get("file_descriptions"):
            lines.append("## RELEVANT TOPICS:")
            for filename, description in context["file_descriptions"].items():
                lines.append(f"- {filename.replace('.md', '')}: {description}")
            lines.append("")
        
        # Add code examples
        if context.get("examples"):
            lines.append("## RELEVANT CODE EXAMPLES:")
            for i, example in enumerate(context["examples"][:3], 1):
                lines.append(f"\n### Example {i} (from {example['source']}):")
                lines.append("```python")
                lines.append(example["code"])
                lines.append("```")
            lines.append("")
        
        # Add scene context
        if context.get("scene"):
            scene = context["scene"]
            lines.append("## CURRENT SLICER SCENE:")
            if scene.get("node_counts"):
                lines.append("Node counts:")
                for node_type, count in scene["node_counts"].items():
                    lines.append(f"  - {node_type}: {count}")
            if scene.get("sample_node_names"):
                lines.append(f"Sample nodes: {', '.join(scene['sample_node_names'][:3])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def refreshCache(self):
        """Reload all files from disk into cache."""
        self._file_cache.clear()
        self._code_cache.clear()
        self._preloadCache()
        logger.info("Skill cache refreshed")
