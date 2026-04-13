"""
SkillContextManager - Manages skill-based context for LLM prompts.

Hybrid retrieval system combining:
1. Local script repository (fast, reliable, works offline)
2. Online knowledge base (GitHub code search, up-to-date)

Reads code examples from the bundled Slicer script repository and 
supplements with online search when enabled.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import slicer

# Import online knowledge client (optional dependency)
try:
    from .OnlineKnowledgeClient import OnlineKnowledgeClient
    ONLINE_AVAILABLE = True
except ImportError:
    ONLINE_AVAILABLE = False
    OnlineKnowledgeClient = None

logger = logging.getLogger(__name__)


class SkillContextManager:
    """
    Manages context from local script repository and online sources.
    
    Features:
    - Reads from bundled script repository markdown files (local)
    - Online code search from Slicer GitHub repositories (optional)
    - Keyword-based topic matching covering 15 major Slicer topics
    - Scene-aware context enrichment
    - Smart result merging and deduplication
    """
    
    # Path to bundled script repository
    SCRIPT_REPO_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Resources', 'Skills', 'script_repository'
    )
    
    # Topic to file mapping
    TOPIC_FILE_MAPPING = {
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
        
        "model": "models.md",
        "models": "models.md",
        "mesh": "models.md",
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
        
        "registration": "registration.md",
        "register": "registration.md",
        "align": "registration.md",
        "alignment": "registration.md",
        "brainsfit": "registration.md",
        "elastix": "registration.md",
        "ants": "registration.md",
        "resample": "registration.md",
        
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
        
        "screenshot": "screencapture.md",
        "capture": "screencapture.md",
        "image": "screencapture.md",
        "png": "screencapture.md",
        "video": "screencapture.md",
        "animation": "screencapture.md",
        "sweep": "screencapture.md",
        "transparent": "screencapture.md",
        "background": "screencapture.md",
        
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
        
        "tractography": "tractography.md",
        "fiber": "tractography.md",
        "fibers": "tractography.md",
        "streamline": "tractography.md",
        "streamlines": "tractography.md",
        "dti": "tractography.md",
        "dmri": "tractography.md",
        "diffusion": "tractography.md",
        "blender": "tractography.md",
        
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
        
        "web": "webserver.md",
        "web server": "webserver.md",
        "http": "webserver.md",
        "rest": "webserver.md",
        "api": "webserver.md",
        "endpoint": "webserver.md",
        "static": "webserver.md",
        "serve": "webserver.md",
    }
    
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
    
    def __init__(self, enable_online: bool = True, github_token: Optional[str] = None):
        """
        Initialize the skill context manager.
        
        Args:
            enable_online: Whether to enable online knowledge search
            github_token: GitHub API token for higher rate limits
        """
        self.script_repo_path = self.SCRIPT_REPO_PATH
        self._file_cache: Dict[str, str] = {}
        self._code_cache: Dict[str, List[str]] = {}
        
        # Online knowledge client
        self._online_client: Optional[OnlineKnowledgeClient] = None
        self._online_enabled = enable_online and ONLINE_AVAILABLE
        
        if self._online_enabled:
            try:
                self._online_client = OnlineKnowledgeClient(
                    github_token=github_token,
                    enabled=True
                )
                logger.info("Online knowledge client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize online client: {e}")
                self._online_enabled = False
        
        # Verify local path
        if not os.path.exists(self.script_repo_path):
            logger.warning(f"Script repository not found at {self.script_repo_path}")
        else:
            logger.info(f"SkillContextManager initialized with path: {self.script_repo_path}")
            self._preload_cache()
    
    def set_online_enabled(self, enabled: bool):
        """Enable or disable online search."""
        self._online_enabled = enabled and ONLINE_AVAILABLE
        if self._online_client:
            self._online_client.set_enabled(self._online_enabled)
    
    def set_github_token(self, token: str):
        """Set GitHub API token."""
        if self._online_client:
            self._online_client.set_token(token)
    
    def _preload_cache(self):
        """Preload all markdown files into cache."""
        try:
            for filename in os.listdir(self.script_repo_path):
                if filename.endswith('.md'):
                    filepath = os.path.join(self.script_repo_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            self._file_cache[filename] = f.read()
                        self._code_cache[filename] = self._extract_code_blocks(self._file_cache[filename])
                        logger.info(f"Cached {filename}: {len(self._code_cache[filename])} code examples")
                    except Exception as e:
                        logger.warning(f"Failed to cache {filename}: {e}")
        except Exception as e:
            logger.warning(f"Failed to preload cache: {e}")
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract python code blocks from markdown content."""
        code_blocks = []
        
        python_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_pattern, content, re.DOTALL)
        code_blocks.extend(matches)
        
        generic_pattern = r'```\s*\n(.*?)\n```'
        generic_matches = re.findall(generic_pattern, content, re.DOTALL)
        for match in generic_matches:
            if any(keyword in match for keyword in ['slicer.', 'vtk.', 'import ', 'def ', 'class ']):
                code_blocks.append(match)
        
        return code_blocks
    
    def _identify_topics(self, prompt: str) -> List[str]:
        """Identify relevant topics from the prompt."""
        prompt_lower = prompt.lower()
        matched_files = set()
        
        for keyword, filename in sorted(self.TOPIC_FILE_MAPPING.items(), key=lambda x: -len(x[0])):
            if keyword in prompt_lower:
                matched_files.add(filename)
        
        return list(matched_files)
    
    def _get_code_examples(self, filename: str, max_examples: int = 5) -> List[str]:
        """Get code examples from a local file."""
        if filename in self._code_cache:
            return self._code_cache[filename][:max_examples]
        
        filepath = os.path.join(self.script_repo_path, filename)
        if not os.path.exists(filepath):
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._extract_code_blocks(content)[:max_examples]
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return []
    
    def _get_file_summary(self, filename: str) -> str:
        """Get a brief summary of what the file covers."""
        return self.FILE_DESCRIPTIONS.get(filename, "Slicer operations")
    
    def _get_scene_context(self) -> Optional[Dict]:
        """Get context about the current Slicer scene."""
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
    
    def _search_online_knowledge(
        self, 
        prompt: str, 
        local_topics: List[str],
        max_results: int = 3
    ) -> Tuple[List[Dict], List[str], Optional[str]]:
        """
        Search for additional examples from online sources.
        
        Returns:
            Tuple of (examples, sources, error)
        """
        if not self._online_enabled or not self._online_client:
            return [], [], "Online search disabled"
        
        try:
            result = self._online_client.search_code_examples(
                query=prompt,
                context={"topics": local_topics},
                max_results=max_results
            )
            
            return (
                result.get("examples", []),
                result.get("sources", []),
                result.get("error")
            )
        except Exception as e:
            logger.warning(f"Online search failed: {e}")
            return [], [], str(e)
    
    def _merge_examples(
        self, 
        local_examples: List[Dict], 
        online_examples: List[Dict]
    ) -> List[Dict]:
        """
        Merge local and online examples, removing duplicates.
        Online examples are added after local ones.
        """
        seen_codes = set()
        merged = []
        
        # Add local examples first
        for ex in local_examples:
            code = ex.get("code", "")
            normalized = ' '.join(code.split())
            code_hash = hash(normalized)
            if code_hash not in seen_codes:
                seen_codes.add(code_hash)
                merged.append(ex)
        
        # Add online examples that aren't duplicates
        for ex in online_examples:
            code = ex.get("code", "")
            normalized = ' '.join(code.split())
            code_hash = hash(normalized)
            if code_hash not in seen_codes:
                seen_codes.add(code_hash)
                merged.append(ex)
        
        return merged
    
    def buildContext(self, prompt: str, use_online: bool = True) -> Dict:
        """
        Build context for a given prompt using local and optional online sources.
        
        Args:
            prompt: User's input prompt
            use_online: Whether to supplement with online search
            
        Returns:
            Dictionary with relevant examples, APIs, and scene context
        """
        # Identify relevant topics/files
        relevant_files = self._identify_topics(prompt)
        
        # Build base context
        context = {
            "topics": [],
            "examples": [],
            "file_descriptions": {},
            "scene": None,
            "online_examples": [],
            "online_sources": [],
            "online_error": None,
        }
        
        # Collect local examples
        local_examples = []
        for filename in relevant_files[:3]:
            context["topics"].append(filename.replace('.md', ''))
            context["file_descriptions"][filename] = self._get_file_summary(filename)
            
            examples = self._get_code_examples(filename, max_examples=3)
            for example in examples:
                local_examples.append({
                    "source": filename,
                    "code": example.strip(),
                    "type": "local"
                })
        
        # Online search if enabled and needed
        if use_online and self._online_enabled:
            # Only search online if we have few local examples or query seems complex
            should_search_online = (
                len(local_examples) < 3 or
                any(term in prompt.lower() for term in [
                    'how to', 'example', 'sample', 'advanced', 'complex'
                ])
            )
            
            if should_search_online:
                online_examples, sources, error = self._search_online_knowledge(
                    prompt, relevant_files, max_results=3
                )
                context["online_examples"] = online_examples
                context["online_sources"] = sources
                context["online_error"] = error
        
        # Merge examples
        all_examples = self._merge_examples(local_examples, context["online_examples"])
        context["examples"] = all_examples[:5]  # Limit total examples
        
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
        
        # Add file descriptions
        if context.get("file_descriptions"):
            lines.append("## RELEVANT TOPICS:")
            for filename, description in context["file_descriptions"].items():
                lines.append(f"- {filename.replace('.md', '')}: {description}")
            lines.append("")
        
        # Add local code examples
        local_examples = [ex for ex in context.get("examples", []) if ex.get("type") == "local"]
        if local_examples:
            lines.append("## CODE EXAMPLES FROM SLICER SCRIPT REPOSITORY:")
            for i, example in enumerate(local_examples[:3], 1):
                lines.append(f"\n### Example {i} (from {example['source']}):")
                lines.append("```python")
                lines.append(example['code'])
                lines.append("```")
            lines.append("")
        
        # Add online code examples
        online_examples = [ex for ex in context.get("examples", []) if ex.get("type") == "github"]
        if online_examples:
            lines.append("## ADDITIONAL CODE EXAMPLES FROM SLICER SOURCE:")
            for i, example in enumerate(online_examples[:2], 1):
                lines.append(f"\n### Example {i} (from {example.get('file', 'Slicer source')}):")
                lines.append("```python")
                lines.append(example['code'])
                lines.append("```")
                if example.get('url'):
                    lines.append(f"[Source: {example['url']}]")
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
        
        # Add online search status if there was an error
        if context.get("online_error") and self._online_enabled:
            lines.append(f"<!-- Note: Online knowledge search unavailable: {context['online_error']} -->")
        
        return "\n".join(lines)
    
    def refreshCache(self):
        """Reload all files from disk into cache."""
        self._file_cache.clear()
        self._code_cache.clear()
        self._preload_cache()
        if self._online_client:
            self._online_client.clear_cache()
        logger.info("Skill cache refreshed")
    
    def get_status(self) -> Dict:
        """Get current status including online availability."""
        status = {
            "local_available": os.path.exists(self.script_repo_path),
            "online_enabled": self._online_enabled,
        }
        
        if self._online_client:
            status["online_status"] = self._online_client.get_status()
        
        return status
