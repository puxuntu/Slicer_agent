"""
ExtensionCLIAnalyzer - 8-stage pipeline for analyzing Slicer extensions
and generating operation CLIs (tool schemas + code templates).

Uses the same LLM provider as the main agent to analyze extension source code,
identify operations, and generate validated code templates that integrate with
the SlicerAIAgent tool system.
"""

import ast
import json
import logging
import os
import textwrap
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Path to the analyzer system prompt
_ANALYZER_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Resources", "Prompts", "extension_cli_analyzer_prompt.md",
)

# Maximum source file size to send to LLM (chars)
_MAX_SOURCE_FOR_LLM = 30_000

# Maximum revision attempts for failed validation
_MAX_REVISION_ATTEMPTS = 3


def _tokenize_name(name: str) -> set:
    """Split a CamelCase/underscore name into lowercase tokens."""
    import re
    parts = re.split(r'(?<=[a-z])(?=[A-Z])|_|(?<=[A-Z])(?=[A-Z][a-z])', name)
    return {p.lower() for p in parts if p}


def _name_similarity(name_a: str, name_b: str) -> float:
    """Jaccard similarity between tokens of two names."""
    ta = _tokenize_name(name_a)
    tb = _tokenize_name(name_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_default_value(default_str: str):
    """Parse a default value string from AST into a Python value."""
    if default_str == "True":
        return True
    if default_str == "False":
        return False
    if default_str == "None":
        return None
    try:
        return int(default_str)
    except (ValueError, TypeError):
        pass
    try:
        return float(default_str)
    except (ValueError, TypeError):
        pass
    return default_str


class _AddNodeVisitor(ast.NodeVisitor):
    """AST visitor that detects AddNode/AddNewNodeByClass calls on method parameters."""

    def __init__(self, param_names: set):
        self.param_names = param_names
        self.params_added_to_scene = set()   # params passed to AddNode()
        self.has_addnewnodebyclass = False    # method calls AddNewNodeByClass anywhere
        self._added_node_args = []            # raw args to AddNode calls

    def visit_Call(self, node):
        func_name = self._get_qualified_name(node.func)
        # slicer.mrmlScene.AddNode(param)
        if func_name and func_name.endswith("AddNode"):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self.param_names:
                    self.params_added_to_scene.add(arg.id)
                elif isinstance(arg, ast.Name):
                    self._added_node_args.append(arg.id)
            # Don't recurse into the call's args
            return
        # slicer.mrmlScene.AddNewNodeByClass(...)
        if func_name and func_name.endswith("AddNewNodeByClass"):
            self.has_addnewnodebyclass = True
            return
        self.generic_visit(node)

    @staticmethod
    def _get_qualified_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts)) if parts else None


class ExtensionCLIAnalyzer:
    """
    Analyzes a Slicer extension's source code and generates operation CLIs.

    8-stage pipeline:
    1. Extension Scanning (AST, no LLM)
    2. Logic Class Analysis (LLM-assisted)
    3. State Dependency Analysis (programmatic + optional LLM)
    4. Node Lifecycle Analysis (LLM-assisted)
    5. Tool Schema Generation (LLM-assisted)
    6. Code Template Generation (LLM-assisted)
    7. Prompt Fragment Generation (LLM-assisted)
    8. Validation (CodeValidator, no LLM)
    """

    def __init__(
        self,
        llm_client,
        output_base_dir: Optional[str] = None,
        code_validator=None,
        on_progress: Optional[Callable[[int, str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            llm_client: LLMClient instance for making LLM calls.
            output_base_dir: Base directory for saving CLI packages.
                             Defaults to Resources/extension_CLI/.
            code_validator: CodeValidator instance. Created if not provided.
            on_progress: Callback(stage_num, stage_name, detail) for progress updates.
            on_error: Callback(error_message) for error reporting.
        """
        self.llm_client = llm_client
        self.output_base_dir = output_base_dir or self._default_base_dir()
        self.code_validator = code_validator
        self.on_progress = on_progress or (lambda n, s, d: None)
        self.on_error = on_error or (lambda e: None)
        self._analyzer_prompt = self._load_analyzer_prompt()
        self._cancelled = False
        # Pipeline-scoped state (reset in analyze_and_generate)
        self._readme_content: Optional[str] = None
        self._ui_workflow: Optional[Dict] = None
        self._debug_dir: Optional[str] = None
        self._llm_call_counter: int = 0
        self._current_stage_label: str = ""

    @staticmethod
    def _default_base_dir() -> str:
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(module_dir, "Resources", "extension_CLI")

    def _load_analyzer_prompt(self) -> str:
        try:
            with open(_ANALYZER_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            logger.warning("Could not load analyzer prompt, using minimal fallback")
            return "You are a code analysis assistant for Slicer extensions."

    def cancel(self):
        """Cancel the analysis pipeline."""
        self._cancelled = True

    # ================================================================
    # Main Entry Point
    # ================================================================

    def analyze_and_generate(
        self,
        extension_name: str,
        source_path: str,
        source_type: str = "",
        force_overwrite: bool = False,
    ) -> Dict:
        """
        Run the full 8-stage analysis pipeline.

        Args:
            extension_name: Name for the generated CLI directory.
            source_path: Path to the extension's source code root.
            source_type: How the extension was discovered
                         ("extension_manager", "additional_paths", "loaded_modules").
            force_overwrite: If True, overwrite existing CLI.

        Returns:
            Dict with 'success', 'cli_dir', 'manifest', 'stages_completed',
            'validation_result', 'error' keys.
        """
        self._cancelled = False
        self._readme_content = None
        self._ui_workflow = None
        self._debug_dir = None
        self._llm_call_counter = 0
        self._current_stage_label = ""
        result = {
            "success": False,
            "cli_dir": None,
            "manifest": None,
            "stages_completed": [],
            "validation_result": None,
            "error": None,
        }

        ext_dir = os.path.join(self.output_base_dir, extension_name)
        if os.path.isdir(ext_dir) and not force_overwrite:
            result["error"] = f"CLI for '{extension_name}' already exists. Use force_overwrite=True."
            return result

        try:
            # Set up debug directory (lazily created on first LLM call)
            self._debug_dir = os.path.join(ext_dir, "debug")

            # Stage 1: Scanning
            self._current_stage_label = "1"
            scan_result = self._stage1_scan(source_path)
            result["stages_completed"].append(1)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 1"
                return result

            if not scan_result.get("logic_class"):
                result["error"] = (
                    f"No ScriptedLoadableModuleLogic subclass found in {source_path}. "
                    "The extension may be C++-only or have no Python logic class."
                )
                return result

            # README discovery (supplementary context for LLM stages)
            self._readme_content = self._find_readme(
                extension_name, source_path, source_type
            )
            if self._readme_content:
                self.on_progress(
                    1, "Scanning extension files",
                    f"README found ({len(self._readme_content)} chars)"
                )
            else:
                self.on_progress(
                    1, "Scanning extension files",
                    f"No README found for '{extension_name}'"
                )

            # Stage 1.5: UI Workflow Extraction
            self._current_stage_label = "1.5"
            self._ui_workflow = self._stage1_5_extract_workflow(scan_result)
            if self._ui_workflow:
                step_count = sum(
                    len(s.get("steps", []))
                    for s in self._ui_workflow.get("ui_sections", [])
                )
                self.on_progress(
                    1.5, "UI workflow extraction",
                    f"Extracted {step_count} steps from UI analysis"
                )
            result["stages_completed"].append("1.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 1.5"
                return result

            # Stage 2: Logic Class Analysis (LLM)
            self._current_stage_label = "2"
            logic_analysis = self._stage2_analyze_logic(scan_result)
            result["stages_completed"].append(2)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 2"
                return result

            # Stage 2.5: AST Signature Verification
            self._current_stage_label = "2.5"
            self._verify_signatures_ast(logic_analysis, scan_result)
            result["stages_completed"].append("2.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 2.5"
                return result

            # Stage 3: State Dependency Analysis
            self._current_stage_label = "3"
            stage_map = self._stage3_state_dependencies(logic_analysis)
            result["stages_completed"].append(3)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 3"
                return result

            # Stage 3.5: Cross-Stage Parameter Mapping
            self._current_stage_label = "3.5"
            cross_stage_map = self._map_cross_stage_params(stage_map, extension_name)
            result["stages_completed"].append("3.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 3.5"
                return result

            # Stage 4: Node Lifecycle Analysis (AST-based)
            self._current_stage_label = "4"
            node_lifecycle = self._stage4_node_lifecycle(scan_result, logic_analysis)
            result["stages_completed"].append(4)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 4"
                return result

            # Stage 4.5: Interactive Pattern Detection
            self._current_stage_label = "4.5"
            interactive_patterns = self._stage4b_detect_interactive_patterns(
                scan_result, logic_analysis
            )
            result["stages_completed"].append("4.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 4.5"
                return result

            # Stage 4.7: Workflow Graph Construction (if interactive patterns found)
            self._current_stage_label = "4.7"
            workflow_graph = None
            if interactive_patterns.get("has_interactive"):
                workflow_graph = self._stage4c_build_workflow_graph(
                    interactive_patterns, logic_analysis, stage_map, extension_name
                )
            result["stages_completed"].append("4.7")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 4.7"
                return result

            # Stage 4.9: Workflow Validation (if workflow graph was built)
            self._current_stage_label = "4.9"
            if workflow_graph:
                workflow_graph = self._stage4d_validate_workflow(
                    workflow_graph, logic_analysis
                )
            result["stages_completed"].append("4.9")

            # Stage 5: Tool Schema Generation (user-facing params only)
            self._current_stage_label = "5"
            tool_schemas = self._stage5_generate_schemas(
                extension_name, stage_map, logic_analysis,
                node_lifecycle=node_lifecycle,
                cross_stage_map=cross_stage_map,
                workflow_graph=workflow_graph,
            )
            result["stages_completed"].append(5)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 5"
                return result

            # Stage 6: Code Template Generation (with cross-stage wiring)
            self._current_stage_label = "6"
            templates = self._stage6_generate_templates(
                extension_name, stage_map, node_lifecycle, scan_result, logic_analysis,
                cross_stage_map=cross_stage_map,
                workflow_graph=workflow_graph,
            )
            result["stages_completed"].append(6)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 6"
                return result

            # Stage 6.5: LLM Review of Templates
            self._current_stage_label = "6.5"
            templates = self._stage6b_review_templates(
                templates, logic_analysis, node_lifecycle,
            )
            result["stages_completed"].append("6.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 6.5"
                return result

            # Stage 7: Prompt Fragment Generation
            self._current_stage_label = "7"
            prompt_fragment = self._stage7_generate_prompt(
                extension_name, tool_schemas, stage_map, logic_analysis,
                workflow_graph=workflow_graph,
            )
            result["stages_completed"].append(7)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 7"
                return result

            # Stage 8: Validation (CodeValidator + semantic) + Save
            self._current_stage_label = "8"
            manifest, generators = self._build_manifest_and_generators(
                extension_name, scan_result, stage_map,
                workflow_graph=workflow_graph,
            )
            validation_result = self._stage8_validate(
                templates, generators, logic_analysis=logic_analysis,
            )
            result["stages_completed"].append(8)
            result["validation_result"] = validation_result

            # Save CLI package
            from .ExtensionCLILoader import save_cli_package
            cli_dir = save_cli_package(
                extension_name=extension_name,
                manifest=manifest,
                tool_schemas=tool_schemas,
                code_generators=generators,
                templates=templates,
                prompt_fragment=prompt_fragment,
                generation_log_entry={
                    "attempt": 1,
                    "timestamp": datetime.now().isoformat(),
                    "stage": "initial_generation",
                    "trigger": "user_request",
                    "analysis_stages_completed": result["stages_completed"],
                    "validation_result": validation_result,
                },
            )
            result["cli_dir"] = cli_dir
            result["manifest"] = manifest

            if validation_result.get("valid"):
                result["success"] = True
            else:
                result["error"] = (
                    f"Validation failed: {validation_result.get('errors', [])}. "
                    "Use revise to fix."
                )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("ExtensionCLIAnalyzer failed: %s\n%s", e, tb)
            result["error"] = str(e)
            self.on_error(str(e))

        finally:
            self._debug_dir = None

        return result

    # ================================================================
    # Stage 1: Extension Scanning (AST, no LLM)
    # ================================================================

    def _stage1_scan(self, source_path: str) -> Dict:
        """Scan extension source tree, parse AST, find Logic class."""
        self.on_progress(1, "Scanning extension files", "Walking directory tree...")

        if not os.path.isdir(source_path):
            raise ValueError(f"Source path does not exist: {source_path}")

        # Walk and collect Python and .ui files
        py_files = []
        ui_files = []
        for root, dirs, files in os.walk(source_path):
            # Skip hidden dirs, __pycache__, build dirs
            dirs[:] = [d for d in dirs if not d.startswith((".", "__")) and d != "build"]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
                elif f.endswith(".ui"):
                    ui_files.append(os.path.join(root, f))

        self.on_progress(
            1, "Scanning extension files",
            f"Found {len(py_files)} Python files"
        )

        # Parse each file's AST
        file_inventory = {}
        logic_candidates = []
        widget_candidates = []

        for fpath in py_files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                tree = ast.parse(source)
            except Exception:
                continue

            classes = []
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [self._ast_name(b) for b in node.bases]
                    methods = [
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    classes.append({
                        "name": node.name,
                        "bases": bases,
                        "methods": methods,
                        "line": node.lineno,
                    })
                    # Detect Logic class
                    is_logic = (
                        "ScriptedLoadableModuleLogic" in bases
                        or node.name.endswith("Logic")
                    )
                    if is_logic:
                        logic_candidates.append({
                            "file": fpath,
                            "class_name": node.name,
                            "methods": methods,
                            "bases": bases,
                            "line": node.lineno,
                        })
                    # Detect Widget class
                    is_widget = (
                        "ScriptedLoadableModuleWidget" in bases
                        or node.name.endswith("Widget")
                    )
                    if is_widget:
                        widget_candidates.append({
                            "file": fpath,
                            "class_name": node.name,
                            "methods": methods,
                            "bases": bases,
                            "line": node.lineno,
                        })
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(node.name)

            file_inventory[fpath] = {
                "classes": classes,
                "functions": functions,
            }

        # Pick best Logic candidate (prefer ScriptedLoadableModuleLogic subclass,
        # then prefer one with "process" or "run" methods)
        logic_class = None
        if logic_candidates:
            scored = []
            for cand in logic_candidates:
                score = 0
                # Strong preference for the Slicer-standard base class
                if "ScriptedLoadableModuleLogic" in cand["bases"]:
                    score += 100
                # Preference for classes with more methods (likely the main logic, not a helper)
                score += min(len(cand["methods"]), 20)
                for m in cand["methods"]:
                    if m.startswith(("process", "run", "compute", "execute")):
                        score += 10
                    if m.startswith("__init__"):
                        score += 1
                scored.append((score, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            logic_class = scored[0][1]

        # Pick best Widget candidate (prefer ScriptedLoadableModuleWidget subclass)
        widget_class = None
        if widget_candidates:
            scored_w = []
            for cand in widget_candidates:
                score = 0
                if "ScriptedLoadableModuleWidget" in cand["bases"]:
                    score += 100
                score += min(len(cand["methods"]), 20)
                if "setup" in cand["methods"]:
                    score += 50
                scored_w.append((score, cand))
            scored_w.sort(key=lambda x: x[0], reverse=True)
            widget_class = scored_w[0][1]

        # Find the entry point module (the main module file)
        entry_module = None
        if logic_class:
            entry_module = logic_class["file"]

        self.on_progress(
            1, "Scanning extension files",
            f"Logic class: {logic_class['class_name'] if logic_class else 'None'} "
            f"in {os.path.basename(entry_module) if entry_module else 'N/A'}"
            f", Widget class: {widget_class['class_name'] if widget_class else 'None'}"
            f", UI files: {len(ui_files)}"
        )

        return {
            "source_path": source_path,
            "py_files": py_files,
            "ui_files": ui_files,
            "file_inventory": file_inventory,
            "logic_class": logic_class,
            "widget_class": widget_class,
            "entry_module": entry_module,
        }

    @staticmethod
    def _ast_name(node) -> str:
        """Extract a readable name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{ExtensionCLIAnalyzer._ast_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    # ================================================================
    # .ui File Parsing (Qt Designer XML)
    # ================================================================

    def _parse_ui_file(self, ui_path: str) -> Optional[Dict]:
        """Parse a Qt .ui file and extract sections and buttons."""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(ui_path)
        except ET.ParseError:
            logger.debug("Failed to parse .ui file: %s", ui_path)
            return None

        root = tree.getroot()
        sections = []
        # Find all collapsible sections and their button children
        for widget in root.iter("widget"):
            widget_class = widget.get("class", "")
            if widget_class in ("ctkCollapsibleButton", "ctkCollapsibleGroupBox"):
                section_name = ""
                # Find the text property
                for prop in widget.findall("property"):
                    if prop.get("name") == "text":
                        string_el = prop.find("string")
                        if string_el is not None and string_el.text:
                            section_name = string_el.text.strip()
                if not section_name:
                    section_name = widget_class

                buttons = self._extract_buttons_from_widget(widget)
                if buttons:
                    sections.append({
                        "name": section_name,
                        "buttons": buttons,
                    })

        # If no collapsible sections found, look for top-level buttons
        if not sections:
            all_buttons = []
            for widget in root.iter("widget"):
                wc = widget.get("class", "")
                if wc in ("QPushButton", "ctkCheckablePushButton"):
                    btn = self._parse_button_widget(widget)
                    if btn:
                        all_buttons.append(btn)
            if all_buttons:
                sections.append({"name": "Buttons", "buttons": all_buttons})

        return {"sections": sections} if sections else None

    def _extract_buttons_from_widget(self, parent_widget) -> List[Dict]:
        """Extract buttons from a UI widget element, recursing into all nested layouts/frames."""
        buttons = []
        for child in parent_widget:
            if child.tag == "widget":
                wc = child.get("class", "")
                if wc in ("QPushButton", "ctkCheckablePushButton"):
                    btn = self._parse_button_widget(child)
                    if btn:
                        buttons.append(btn)
                else:
                    # Recurse into any non-button widget (QFrame, QGroupBox, etc.)
                    buttons.extend(self._extract_buttons_from_widget(child))
            elif child.tag == "layout":
                buttons.extend(self._extract_buttons_from_widget(child))
            elif child.tag == "item":
                buttons.extend(self._extract_buttons_from_widget(child))
        return buttons

    @staticmethod
    def _parse_button_widget(widget_el) -> Optional[Dict]:
        """Parse a single button widget element."""
        name = widget_el.get("name", "")
        label = ""
        for prop in widget_el.findall("property"):
            if prop.get("name") == "text":
                string_el = prop.find("string")
                if string_el is not None and string_el.text:
                    label = string_el.text.strip()
        if name:
            return {"widget_name": name, "label": label}
        return None

    # ================================================================
    # Widget Signal Connection Extraction (AST)
    # ================================================================

    def _extract_widget_connections(self, widget_source: str) -> List[Dict]:
        """Extract button→handler→logic_method mappings from Widget class source."""
        try:
            tree = ast.parse(widget_source)
        except SyntaxError:
            return []

        connections = []
        # Find the setup() method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "setup":
                        connections = self._find_clicked_connections(item, node)

        return connections

    def _find_clicked_connections(self, setup_node, class_node) -> List[Dict]:
        """Find .clicked.connect(self.XXX) patterns in setup() method."""
        connections = []
        # Build handler→logic_method map from all methods in the class
        handler_logic_map = {}
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                logic_calls = self._find_logic_calls_in_method(item)
                if logic_calls:
                    handler_logic_map[item.name] = logic_calls

        for stmt in ast.walk(setup_node):
            if not isinstance(stmt, ast.Call):
                continue
            func = stmt.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "connect":
                continue

            button_name = ""
            handler_name = ""

            # Pattern 1: something.clicked.connect(self.handlerMethod)
            receiver = func.value
            if isinstance(receiver, ast.Attribute) and receiver.attr == "clicked":
                button_name = self._get_attribute_chain(receiver.value)
                if stmt.args:
                    arg = stmt.args[0]
                    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                        if arg.value.id == "self":
                            handler_name = arg.attr

            # Pattern 2: something.connect('clicked(bool)', self.handlerMethod)
            #            something.connect("clicked(bool)", self.handlerMethod)
            if not button_name and stmt.args:
                first_arg = stmt.args[0]
                if (isinstance(first_arg, ast.Constant)
                        and isinstance(first_arg.value, str)
                        and "clicked" in first_arg.value):
                    button_name = self._get_attribute_chain(func.value)
                    if len(stmt.args) > 1:
                        second_arg = stmt.args[1]
                        if isinstance(second_arg, ast.Attribute) and isinstance(second_arg.value, ast.Name):
                            if second_arg.value.id == "self":
                                handler_name = second_arg.attr

            if button_name and handler_name:
                logic_methods = handler_logic_map.get(handler_name, [])
                connections.append({
                    "button_widget_name": button_name,
                    "handler_method": handler_name,
                    "logic_methods": logic_methods,
                })

        return connections

    @staticmethod
    def _get_attribute_chain(node) -> str:
        """Build dotted name from chained attribute access (e.g. self.buttonName)."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = ExtensionCLIAnalyzer._get_attribute_chain(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _find_logic_calls_in_method(self, method_node) -> List[str]:
        """Find self.logic.XXX() calls in a method body."""
        calls = []
        for node in ast.walk(method_node):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Attribute)):
                    if (func.value.attr == "logic"
                            and isinstance(func.value.value, ast.Name)
                            and func.value.value.id == "self"):
                        calls.append(func.attr)
        return calls

    # ================================================================
    # Stage 2: Logic Class Analysis (LLM-assisted)
    # ================================================================

    def _stage2_analyze_logic(self, scan_result: Dict) -> Dict:
        """Use LLM to analyze the Logic class methods in detail."""
        logic_info = scan_result["logic_class"]
        logic_file = logic_info["file"]
        class_name = logic_info["class_name"]

        self.on_progress(
            2, "Analyzing logic class",
            f"Reading {class_name} from {os.path.basename(logic_file)}..."
        )

        # Extract Logic class source
        logic_source = self._extract_class_source(logic_file, class_name)
        if not logic_source:
            raise RuntimeError(f"Could not extract source for {class_name} from {logic_file}")

        # Truncate if too large
        if len(logic_source) > _MAX_SOURCE_FOR_LLM:
            logic_source = logic_source[:_MAX_SOURCE_FOR_LLM] + "\n# ... [truncated for LLM analysis] ..."

        # Build prompt
        prompt = textwrap.dedent(f"""\
Analyze the following Slicer extension Logic class and return a JSON object with this exact structure:

{{
  "class_name": "{class_name}",
  "source_file": "{os.path.basename(logic_file)}",
  "methods": [
    {{
      "name": "method_name",
      "purpose": "one-line description",
      "parameters": [
        {{"name": "param_name", "type": "vtkMRML... or str or int etc", "required": true, "description": "what it is"}}
      ],
      "return_value": "description or null",
      "state_reads": ["self.field1", "self.field2"],
      "state_writes": ["self.field3"],
      "calls_addnode": true/false,
      "adds_output_to_scene": true/false,
      "side_effects": "description"
    }}
  ],
  "public_api_methods": ["method1", "method2"],
  "internal_methods": ["_helper1"],
  "pipeline_methods": ["method1", "method2"],
  "state_fields": [
    {{"name": "self.field1", "type": "description", "set_by": "method_name", "read_by": ["other_method"]}}
  ]
}}

Focus on public methods that perform meaningful operations (process, run, compute, execute).
Skip trivial getters/setters and Qt signal handlers.
For each method, be precise about:
- Whether it calls slicer.mrmlScene.AddNode() on its output parameters
- Whether it reads state from self.* that must be set by a prior method call
- Whether it writes state to self.* that future method calls depend on""")

        # Inject README context if available
        if self._readme_content:
            prompt += textwrap.dedent(f"""\

Extension README (for understanding the implementation pipeline):
Focus on the pipeline, algorithm, and workflow descriptions below.
Ignore setup, installation, build instructions, and dependency lists.

{self._readme_content}

""")

        # Inject UI workflow context if available
        if self._ui_workflow:
            prompt += textwrap.dedent(f"""\
Extracted UI Workflow (from .ui file and Widget class analysis):
```json
{json.dumps(self._ui_workflow, indent=2)}
```
Use this workflow to understand the intended user-facing sequence of operations.
Match method descriptions to their corresponding UI workflow steps.

""")

        prompt += textwrap.dedent(f"""\
Logic class source:
```python
{logic_source}
```

Return ONLY the JSON object, no markdown fences or explanation.""")

        response = self._call_llm(prompt)
        analysis = self._parse_json_response(response)

        if not analysis or "methods" not in analysis:
            raise RuntimeError(
                f"LLM analysis returned invalid structure. Response: {response[:500]}"
            )

        self.on_progress(
            2, "Analyzing logic class",
            f"Analyzed {len(analysis.get('methods', []))} methods"
        )

        analysis["_logic_source"] = logic_source
        analysis["_logic_file"] = logic_file
        return analysis

    def _verify_signatures_ast(self, logic_analysis: Dict, scan_result: Dict) -> None:
        """Cross-check LLM-extracted method signatures against actual AST."""
        logic_file = logic_analysis.get("_logic_file", "")
        methods = logic_analysis.get("methods", [])
        corrections = 0

        for method in methods:
            mname = method.get("name", "")
            method_source = self._extract_method_source(logic_file, mname)
            if not method_source:
                continue
            method_source = textwrap.dedent(method_source)

            try:
                tree = ast.parse(method_source)
            except SyntaxError:
                continue

            # Find the FunctionDef
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == mname:
                        func_def = node
                        break
            if not func_def:
                continue

            # Extract actual params from AST
            ast_params = []
            args = func_def.args
            all_args = args.args[:]
            # Skip 'self' if present
            if all_args and getattr(all_args[0], 'arg', None) == 'self':
                all_args = all_args[1:]

            defaults = args.defaults[:]
            # Pad defaults with None for params without defaults
            padded_defaults = [None] * (len(all_args) - len(defaults)) + defaults

            for arg_obj, default_val in zip(all_args, padded_defaults):
                annotation = ""
                if arg_obj.annotation:
                    if isinstance(arg_obj.annotation, ast.Name):
                        annotation = arg_obj.annotation.id
                    elif isinstance(arg_obj.annotation, ast.Attribute):
                        annotation = self._ast_name(arg_obj.annotation) or ""
                    elif isinstance(arg_obj.annotation, ast.Subscript):
                        annotation = self._ast_name(arg_obj.annotation.value) or ""
                default_str = None
                if default_val is not None:
                    try:
                        default_str = ast.unparse(default_val)
                    except Exception:
                        default_str = "..."
                ast_params.append({
                    "name": arg_obj.arg,
                    "type": annotation,
                    "default": default_str,
                })

            # Compare with LLM params
            llm_params = method.get("parameters", [])
            llm_param_names = [p.get("name", "") for p in llm_params]
            ast_param_names = [p["name"] for p in ast_params]

            if llm_param_names != ast_param_names:
                logger.info(
                    "Signature mismatch for %s: LLM=%s AST=%s — correcting",
                    mname, llm_param_names, ast_param_names,
                )
                # Rebuild parameters from AST, preserving LLM descriptions where names match
                llm_desc_map = {p.get("name", ""): p for p in llm_params}
                new_params = []
                for ap in ast_params:
                    if ap["name"] in llm_desc_map:
                        # Keep LLM description, update name/type
                        entry = dict(llm_desc_map[ap["name"]])
                        entry["name"] = ap["name"]
                        if ap["type"]:
                            entry["type"] = ap["type"]
                        if ap["default"] is not None:
                            entry["required"] = False
                            entry["default"] = _parse_default_value(ap["default"])
                        new_params.append(entry)
                    else:
                        # New param from AST not in LLM output
                        entry = {
                            "name": ap["name"],
                            "type": ap["type"] or "Any",
                            "required": ap["default"] is None,
                            "description": "",
                        }
                        if ap["default"] is not None:
                            entry["default"] = _parse_default_value(ap["default"])
                        new_params.append(entry)

                method["parameters"] = new_params
                corrections += 1

        if corrections:
            self.on_progress(
                2, "Verifying signatures",
                f"Corrected {corrections} method signature(s) via AST"
            )

    # ================================================================
    # Stage 3: State Dependency Analysis (programmatic + optional LLM)
    # ================================================================

    @staticmethod
    def _is_auxiliary_method(method: Dict) -> bool:
        """Check if a method is auxiliary (setup/install/cache/download) rather than a primary operation."""
        name = method.get("name", "").lower()
        purpose = method.get("purpose", "").lower()
        # Auxiliary keyword patterns
        aux_patterns = [
            "install", "setup", "setuppython", "clearcache", "clear_cache",
            "download", "check", "isinstalled", "ismodel", "torchversion",
            "incompatible", "installed", "disallowed",
        ]
        for pat in aux_patterns:
            if pat in name or pat in purpose:
                return True
        # Methods with no vtkMRML parameters and no state writes are likely auxiliary
        has_vtkmrml_param = any(
            "vtkMRML" in p.get("type", "")
            for p in method.get("parameters", [])
            if p.get("name") != "self"
        )
        has_state_write = bool(method.get("state_writes", []))
        if not has_vtkmrml_param and not has_state_write:
            return True
        return False

    def _stage3_state_dependencies(self, logic_analysis: Dict) -> Dict:
        """Build stage map from state dependencies."""
        self.on_progress(3, "Analyzing state dependencies", "Building dependency graph...")

        methods = logic_analysis.get("methods", [])
        pipeline_methods = logic_analysis.get("pipeline_methods", [])
        state_fields = logic_analysis.get("state_fields", [])

        # Classify methods as primary or auxiliary
        primary_methods = []
        auxiliary_methods = []
        for m in methods:
            if self._is_auxiliary_method(m):
                auxiliary_methods.append(m["name"])
            else:
                primary_methods.append(m)

        # Only include primary methods in the dependency graph
        dep_graph = {}
        for method in primary_methods:
            name = method["name"]
            writes = set(method.get("state_writes", []))
            dep_graph[name] = {"writes": writes, "reads": set(method.get("state_reads", []))}

        # Compute dependencies among primary methods
        method_deps = {}
        for name, info in dep_graph.items():
            deps = set()
            for other_name, other_info in dep_graph.items():
                if other_name == name:
                    continue
                if info["reads"] & other_info["writes"]:
                    deps.add(other_name)
            method_deps[name] = deps

        # Identify stages (topological grouping) from primary methods only
        stages = []
        assigned = set()
        remaining = set(dep_graph.keys())

        while remaining:
            current_stage = []
            for name in sorted(remaining):
                if method_deps.get(name, set()).issubset(assigned):
                    current_stage.append(name)

            if not current_stage:
                current_stage = sorted(remaining)

            stages.append(current_stage)
            assigned.update(current_stage)
            remaining -= set(current_stage)

        # Build stage map — each stage gets its primary methods + any auxiliary methods
        # that should run as setup (folded into the first stage that has primary methods)
        stage_map = []
        for i, stage_methods in enumerate(stages):
            method_infos = []
            for mname in stage_methods:
                for m in primary_methods:
                    if m["name"] == mname:
                        method_infos.append(m)
                        break

            stage_name = self._infer_stage_name(stage_methods, i, len(stages))

            input_nodes = []
            output_nodes = []
            for mi in method_infos:
                for p in mi.get("parameters", []):
                    ptype = p.get("type", "")
                    if "vtkMRML" in ptype and p.get("required", True):
                        if "output" in p.get("name", "").lower() or "output" in p.get("description", "").lower():
                            output_nodes.append(p)
                        else:
                            input_nodes.append(p)

            stage_map.append({
                "stage_index": i,
                "stage_name": stage_name,
                "methods": stage_methods,
                "method_details": method_infos,
                "depends_on": [stages[j] for j in range(i) if j < len(stages)],
                "input_nodes": input_nodes,
                "output_nodes": output_nodes,
            })

        self.on_progress(
            3, "Analyzing state dependencies",
            f"Found {len(stage_map)} stages: {[s['stage_name'] for s in stage_map]}"
        )

        return {"stages": stage_map, "dep_graph": dep_graph, "method_deps": method_deps}

    @staticmethod
    def _infer_stage_name(methods: List[str], index: int, total: int) -> str:
        """Infer a semantic name for a stage from its method names."""
        names_lower = " ".join(m.lower() for m in methods)
        if "seg" in names_lower:
            return "segmentation"
        if "plan" in names_lower or "reduc" in names_lower or "screw" in names_lower:
            return "planning"
        if "regist" in names_lower or "align" in names_lower:
            return "registration"
        if "detect" in names_lower or "find" in names_lower:
            return "detection"
        if "mesh" in names_lower or "model" in names_lower or "generat" in names_lower:
            return "generation"
        if "visual" in names_lower or "render" in names_lower or "display" in names_lower:
            return "visualization"
        if total == 1:
            return "full"
        return f"stage_{index + 1}"

    @staticmethod
    def _map_cross_stage_params(stage_map: Dict, extension_name: str = "") -> Dict:
        """
        Map parameters in later stages to output parameters from earlier stages.

        Returns:
            {stage_index: {param_name: {"source_stage": int, "source_param": str}},
             "_extension_name": str}
        """
        stages = stage_map.get("stages", [])
        cross_map = {}
        if extension_name:
            cross_map["_extension_name"] = extension_name

        for i, stage in enumerate(stages):
            if i == 0:
                continue

            # Collect all output params from prior stages
            prior_outputs = []
            for j in range(i):
                for mi in stages[j].get("method_details", []):
                    for p in mi.get("parameters", []):
                        pname = p.get("name", "")
                        ptype = p.get("type", "")
                        if "vtkMRML" in ptype and (
                            "output" in pname.lower()
                            or "result" in pname.lower()
                            or "out" in pname.lower()
                        ):
                            prior_outputs.append((j, pname, ptype))

            # Match this stage's vtkMRML params against prior outputs
            stage_map_entry = {}
            for mi in stage.get("method_details", []):
                for p in mi.get("parameters", []):
                    pname = p.get("name", "")
                    ptype = p.get("type", "")
                    if "vtkMRML" not in ptype:
                        continue

                    # Try to match against prior outputs by type + name similarity
                    best_match = None
                    best_score = 0.0
                    for (src_stage, src_name, src_type) in prior_outputs:
                        if src_type != ptype:
                            continue
                        score = _name_similarity(pname, src_name)
                        if score > best_score:
                            best_score = score
                            best_match = (src_stage, src_name)

                    if best_match and best_score > 0.4:
                        stage_map_entry[pname] = {
                            "source_stage": best_match[0],
                            "source_param": best_match[1],
                        }

            if stage_map_entry:
                cross_map[i] = stage_map_entry

        return cross_map

    # ================================================================
    # Stage 4: Node Lifecycle Analysis (LLM-assisted)
    # ================================================================

    def _stage4_node_lifecycle(self, scan_result: Dict, logic_analysis: Dict) -> Dict:
        """Determine node creation mode and param role for each vtkMRML parameter."""
        self.on_progress(4, "Analyzing node lifecycle", "Determining node creation patterns via AST...")

        node_lifecycle = {}
        methods = logic_analysis.get("methods", [])
        logic_file = logic_analysis.get("_logic_file", "")

        for method in methods:
            mname = method["name"]
            params = method.get("parameters", [])

            # Build set of all param names for this method
            all_param_names = {p.get("name", "") for p in params}

            # Extract and parse method source
            method_source = self._extract_method_source(logic_file, mname)
            if not method_source:
                method_source = ""
            else:
                method_source = textwrap.dedent(method_source)

            # Run AST visitor
            visitor = _AddNodeVisitor(all_param_names)
            try:
                tree = ast.parse(method_source)
                visitor.visit(tree)
            except SyntaxError:
                logger.warning("Could not parse method %s for lifecycle analysis", mname)

            for p in params:
                ptype = p.get("type", "")
                pname = p.get("name", "")
                if "vtkMRML" not in ptype:
                    continue

                is_output = (
                    "output" in pname.lower()
                    or "result" in pname.lower()
                    or "out" in pname.lower()
                    or p == params[-1]
                )
                if not is_output:
                    continue

                key = f"{mname}:{pname}"

                # Determine create_mode using AST results
                if pname in visitor.params_added_to_scene:
                    # Method calls AddNode(param) — caller should pre-create without adding
                    create_mode = "CreateNodeByClass"
                    reason = f"AST: method passes '{pname}' to AddNode() — pre-create without adding to scene"
                elif visitor.has_addnewnodebyclass:
                    # Method uses AddNewNodeByClass for other nodes, but not this param
                    # Be safe: use AddNewNodeByClass so the node is in the scene
                    create_mode = "AddNewNodeByClass"
                    reason = "AST: method uses AddNewNodeByClass for internal nodes — create and add to scene"
                else:
                    # No AddNode call targets this param — caller must create AND add
                    create_mode = "AddNewNodeByClass"
                    reason = "AST: no AddNode() call targets this param — caller creates and adds to scene"

                node_lifecycle[key] = {
                    "create_mode": create_mode,
                    "reason": reason,
                    "node_class": ptype,
                    "param_role": "output",
                }

        # If no output nodes found from parameters, ask LLM
        if not node_lifecycle:
            self.on_progress(
                4, "Analyzing node lifecycle",
                "Asking LLM about node creation patterns..."
            )
            node_lifecycle = self._llm_node_lifecycle(logic_analysis)

        self.on_progress(
            4, "Analyzing node lifecycle",
            f"Analyzed {len(node_lifecycle)} output nodes via AST"
        )

        return node_lifecycle

    def _llm_node_lifecycle(self, logic_analysis: Dict) -> Dict:
        """Use LLM to determine node lifecycle for ambiguous cases."""
        methods = logic_analysis.get("methods", [])
        method_summaries = []
        for m in methods:
            params_str = ", ".join(
                f"{p['name']}: {p['type']}"
                for p in m.get("parameters", [])
            )
            method_summaries.append(
                f"  {m['name']}({params_str})\n"
                f"    adds_output_to_scene: {m.get('adds_output_to_scene', 'unknown')}\n"
                f"    calls_addnode: {m.get('calls_addnode', 'unknown')}"
            )

        prompt = textwrap.dedent(f"""\
For each method below, determine whether its output node parameters should be created
with CreateNodeByClass (creates WITHOUT adding to scene) or AddNewNodeByClass (creates AND adds).

Rules:
- If the method calls slicer.mrmlScene.AddNode() on the parameter → use CreateNodeByClass (template should NOT add to scene)
- If the method does NOT add to scene itself → use AddNewNodeByClass (template must add)
- When unsure, default to CreateNodeByClass (safer — avoids "Node already added" errors)

Methods:
{chr(10).join(method_summaries)}

Return JSON:
{{
  "nodes": [
    {{"method": "method_name", "param": "param_name", "node_class": "vtkMRML...Node", "create_mode": "CreateNodeByClass" or "AddNewNodeByClass", "reason": "why"}}
  ]
}}""")

        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        if not parsed:
            return {}

        result = {}
        for node in parsed.get("nodes", []):
            key = f"{node['method']}:{node['param']}"
            result[key] = {
                "create_mode": node.get("create_mode", "CreateNodeByClass"),
                "reason": node.get("reason", ""),
                "node_class": node.get("node_class", ""),
            }
        return result

    # ================================================================
    # Stage 4.5: Interactive Pattern Detection
    # ================================================================

    # MRML markup node class prefixes for AST scanning
    _MARKUP_NODE_CLASSES = {
        "vtkMRMLMarkupsCurveNode": "curve",
        "vtkMRMLMarkupsPlaneNode": "plane",
        "vtkMRMLMarkupsLineNode": "line",
        "vtkMRMLMarkupsFiducialNode": "fiducial",
        "vtkMRMLMarkupsROINode": "roi",
    }

    _INTERACTION_PATTERNS = [
        "StartPlaceMode", "SetPlaceModeEnabled", "SwitchToSinglePlaceMode",
        "SwitchToPersistentPlaceMode", "PlaceModeEnabled",
        "AddObserver", "PointModifiedEvent", "PointAddedEvent",
        "PointPositionDefinedEvent", "InteractionEvent",
        "DynamicModeler", "vtkSlicerDynamicModelerModuleLogic",
        "HandlesInteractive", "RotationHandleVisibility",
        "TranslationHandleVisibility", "ScaleHandleVisibility",
        "QTimer", "singleShot",
    ]

    def _stage4b_detect_interactive_patterns(
        self, scan_result: Dict, logic_analysis: Dict
    ) -> Dict:
        """
        Detect interactive markup placement patterns in the extension source.

        Scans both the Logic class and the Widget class (if found in the same file)
        for markup node creation, placement mode entry, observer setup, and
        debounce timer patterns.

        Returns:
            Dict with:
            - has_interactive: bool
            - patterns: list of detected interactive pattern descriptors
            - widget_source: str (Widget class source, if found)
        """
        self.on_progress("4.5", "Interactive Pattern Detection", "Scanning for markup nodes...")

        result = {"has_interactive": False, "patterns": [], "widget_source": ""}

        # Read the full extension source to scan for Widget class
        entry_module = scan_result.get("entry_module")
        if not entry_module or not os.path.isfile(entry_module):
            return result

        with open(entry_module, "r", encoding="utf-8", errors="ignore") as f:
            full_source = f.read()

        try:
            full_tree = ast.parse(full_source)
        except SyntaxError:
            return result

        # --- AST-based detection ---
        detected_markup_refs = set()
        detected_interaction_calls = set()
        detected_observer_patterns = set()
        detected_timer_patterns = set()

        for node in ast.walk(full_tree):
            # Detect string literals referencing markup node classes
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                for cls_name, interaction_type in self._MARKUP_NODE_CLASSES.items():
                    if cls_name in val:
                        detected_markup_refs.add((cls_name, interaction_type))
                if "Markups" in val and "Node" in val and val not in self._MARKUP_NODE_CLASSES:
                    detected_markup_refs.add((val, "unknown"))

            # Detect AddNewNodeByClass calls with markup node types
            if isinstance(node, ast.Call):
                func_str = self._get_call_name(node)
                if func_str and "AddNewNodeByClass" in func_str:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            for cls_name, interaction_type in self._MARKUP_NODE_CLASSES.items():
                                if cls_name in arg.value:
                                    detected_markup_refs.add((cls_name, interaction_type))

                # Detect placement mode calls
                if func_str:
                    for pattern in ["StartPlaceMode", "SetPlaceModeEnabled",
                                    "SwitchToSinglePlaceMode", "SwitchToPersistentPlaceMode"]:
                        if pattern in func_str:
                            detected_interaction_calls.add(pattern)

        # String-based detection for patterns harder to catch with AST
        for pattern in self._INTERACTION_PATTERNS:
            if pattern in full_source:
                if "Observer" in pattern or "Event" in pattern:
                    detected_observer_patterns.add(pattern)
                elif "Timer" in pattern or "singleShot" in pattern:
                    detected_timer_patterns.add(pattern)

        has_interactive = bool(detected_markup_refs)
        result["has_interactive"] = has_interactive

        if not has_interactive:
            self.on_progress("4.5", "Interactive Pattern Detection", "No interactive patterns found")
            return result

        # --- LLM-assisted classification ---
        result["patterns"] = [
            {"class": cls, "interaction_type": itype}
            for cls, itype in detected_markup_refs
        ]
        result["interaction_calls"] = list(detected_interaction_calls)
        result["observer_patterns"] = list(detected_observer_patterns)
        result["timer_patterns"] = list(detected_timer_patterns)

        # Extract Widget class source for LLM context
        widget_source = self._extract_widget_source(full_tree, full_source)
        result["widget_source"] = widget_source[:_MAX_SOURCE_FOR_LLM]

        # Ask LLM to classify interactive patterns into phases
        logic_class_name = scan_result.get("logic_class", {}).get("class_name", "Logic")
        method_names = [
            m.get("name", "") for m in logic_analysis.get("methods", [])
            if not self._is_auxiliary_method(m)
        ]

        classification_prompt = textwrap.dedent(f"""\
        Analyze the following Slicer extension for interactive 3D user interaction patterns.

        Extension Logic class: {logic_class_name}
        Logic methods: {json.dumps(method_names[:30])}

        Detected markup node types: {json.dumps(list(detected_markup_refs))}
        Detected interaction calls: {json.dumps(list(detected_interaction_calls))}
        Detected observer patterns: {json.dumps(list(detected_observer_patterns))}
        Detected timer patterns: {json.dumps(list(detected_timer_patterns))}

        Widget class source (excerpt):
        ```python
        {widget_source[:20000]}
        ```

        Classify each detected interactive pattern into a structured phase.
        For each phase, determine:
        1. phase_name: a short snake_case identifier
        2. interaction_type: "curve", "plane", "line", "fiducial", or "unknown"
        3. description: what the user does in this phase
        4. node_class: the vtkMRML node class used
        5. placement_instructions: what to tell the user to do
        6. min_control_points: minimum control points needed (0 if unknown)
        7. has_reactive_chain: true if observer triggers recomputation
        8. reactive_description: what recomputation happens (if has_reactive_chain)
        9. is_optional: true if this is an optional/experimental phase
        10. depends_on: list of phase_names this phase depends on

        Respond with ONLY a JSON array of phase objects, no markdown fences.
        """)

        self.on_progress("4.5", "Interactive Pattern Detection", "LLM classifying patterns...")
        try:
            llm_response = self._call_llm(classification_prompt)
            phases = self._parse_json_response(llm_response)
            if isinstance(phases, list):
                result["phases"] = phases
                self.on_progress(
                    "4.5", "Interactive Pattern Detection",
                    f"Classified {len(phases)} interactive phases"
                )
        except Exception as e:
            logger.warning(f"Stage 4.5 LLM classification failed: {e}")
            result["phases"] = []

        return result

    def _extract_widget_source(self, tree, full_source: str) -> str:
        """Extract the Widget class source from the full module AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if "Widget" in base_name:
                        try:
                            lines = full_source.split("\n")
                            start = node.lineno - 1
                            end = node.end_lineno if hasattr(node, "end_lineno") else len(lines)
                            return "\n".join(lines[start:end])
                        except Exception:
                            return ""
        return ""

    @staticmethod
    def _get_call_name(node) -> str:
        """Get the dotted name of a Call node's function."""
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    # ================================================================
    # Stage 4.7: Workflow Graph Construction
    # ================================================================

    def _stage4c_build_workflow_graph(
        self,
        interactive_patterns: Dict,
        logic_analysis: Dict,
        stage_map: Dict,
        extension_name: str,
    ) -> Dict:
        """
        Construct a workflow graph from detected interactive patterns.

        When self._ui_workflow is available (from Stage 1.5), uses it as the
        primary step source, enriching interactive steps with detected pattern
        metadata (node_class, reactive_chains) from Stage 4.5.

        Otherwise falls back to merging stage_map + interactive patterns.
        """
        self.on_progress("4.7", "Workflow Graph Construction", "Building workflow graph...")

        phases = interactive_patterns.get("phases", [])

        # Build a lookup of interactive pattern metadata by logic method name
        pattern_by_method = {}
        for p in phases:
            # phase_name may match a logic method; also check interaction_type
            pattern_by_method[p.get("phase_name", "").lower()] = p

        # ---- UI Workflow path (preferred when available) ----
        if self._ui_workflow:
            steps = self._build_steps_from_ui_workflow(
                self._ui_workflow, pattern_by_method, phases
            )
            if steps:
                return self._finalize_workflow_graph(steps, "4.7")

        # ---- Fallback: original merge logic ----
        if not phases:
            self.on_progress("4.7", "Workflow Graph Construction", "No phases to build")
            return None

        steps = []
        automated_stages = stage_map.get("stages", [])

        # First, add any automated stages that aren't covered by interactive phases
        phase_names = [p.get("phase_name") for p in phases]

        added_automated = set()
        for stage_idx, stage_info in enumerate(automated_stages):
            stage_methods = stage_info.get("methods", [])
            # Check if this stage's purpose overlaps with any interactive phase
            stage_name = stage_info.get("semantic_name", stage_info.get("stage_name", f"stage_{stage_idx}"))
            is_covered = any(
                p.get("phase_name") == stage_name
                for p in phases
            )
            if not is_covered and stage_idx not in added_automated:
                step = {
                    "step_id": stage_name,
                    "phase": stage_name,
                    "step_type": "automated",
                    "description": f"Automated step: {stage_name}",
                    "method_name": stage_methods[0] if stage_methods else None,
                    "depends_on": [],
                    "produces_nodes": [],
                }
                # Add dependency on previous step
                if steps:
                    step["depends_on"] = [steps[-1]["step_id"]]
                steps.append(step)
                added_automated.add(stage_idx)

        # Add interactive phases
        for phase in phases:
            phase_name = phase.get("phase_name", "unknown")
            interaction_type = phase.get("interaction_type", "unknown")
            node_class_map = {
                "curve": "vtkMRMLMarkupsCurveNode",
                "plane": "vtkMRMLMarkupsPlaneNode",
                "line": "vtkMRMLMarkupsLineNode",
                "fiducial": "vtkMRMLMarkupsFiducialNode",
            }
            node_class = phase.get("node_class") or node_class_map.get(interaction_type, "")

            step = {
                "step_id": phase_name,
                "phase": phase_name,
                "step_type": "interactive",
                "description": phase.get("description", ""),
                "interaction_type": interaction_type,
                "node_class": node_class,
                "placement_instructions": phase.get("placement_instructions", ""),
                "min_control_points": phase.get("min_control_points", 0),
                "validation_rules": [f"minimum {phase.get('min_control_points', 0)} control points"],
                "is_optional": phase.get("is_optional", False),
                "produces_nodes": [],
            }

            # Reactive chains
            if phase.get("has_reactive_chain"):
                step["reactive_chains"] = [{
                    "trigger_event": "PointModifiedEvent",
                    "recompute_description": phase.get("reactive_description", ""),
                    "debounce_ms": 300,
                }]
            else:
                step["reactive_chains"] = []

            # Dependencies
            deps = phase.get("depends_on", [])
            if isinstance(deps, list):
                step["depends_on"] = deps
            elif isinstance(deps, str):
                step["depends_on"] = [deps]
            else:
                # Auto-link to previous step
                if steps:
                    step["depends_on"] = [steps[-1]["step_id"]]

            steps.append(step)

        return self._finalize_workflow_graph(steps, "4.7")

    def _build_steps_from_ui_workflow(
        self,
        ui_workflow: Dict,
        pattern_by_method: Dict,
        phases: list,
    ) -> list:
        """Build workflow steps from the Stage 1.5 UI workflow, enriched with
        detected interactive pattern metadata from Stage 4.5."""
        node_class_map = {
            "curve": "vtkMRMLMarkupsCurveNode",
            "plane": "vtkMRMLMarkupsPlaneNode",
            "line": "vtkMRMLMarkupsLineNode",
            "fiducial": "vtkMRMLMarkupsFiducialNode",
        }

        steps = []
        step_id_set = set()

        for section in ui_workflow.get("ui_sections", []):
            is_section_optional = section.get("is_optional", False)
            for ui_step in section.get("steps", []):
                step_id = ui_step.get("step_id", "")
                logic_method = ui_step.get("logic_method", "")
                step_type = ui_step.get("step_type", "automated")
                is_optional = ui_step.get("is_optional", is_section_optional)

                # Cross-reference with Stage 4.5 detected patterns
                matched_pattern = self._match_pattern(
                    logic_method, step_id, pattern_by_method, phases
                )

                if step_type == "interactive":
                    interaction_type = ui_step.get("interaction_type") or (
                        matched_pattern.get("interaction_type", "") if matched_pattern else ""
                    )
                    node_class = (
                        ui_step.get("node_class")
                        or (matched_pattern.get("node_class", "") if matched_pattern else "")
                        or node_class_map.get(interaction_type, "")
                    )
                    min_cp = (
                        ui_step.get("min_control_points")
                        or (matched_pattern.get("min_control_points", 0) if matched_pattern else 0)
                        or 0
                    )
                    step = {
                        "step_id": step_id,
                        "phase": step_id,
                        "step_type": "interactive",
                        "description": ui_step.get("description", ""),
                        "method_name": logic_method,
                        "interaction_type": interaction_type,
                        "node_class": node_class,
                        "placement_instructions": ui_step.get("placement_instructions", "")
                            or (matched_pattern.get("placement_instructions", "") if matched_pattern else ""),
                        "min_control_points": min_cp,
                        "validation_rules": [f"minimum {min_cp} control points"],
                        "is_optional": is_optional,
                        "produces_nodes": [],
                    }
                    # Reactive chains from matched pattern
                    if matched_pattern and matched_pattern.get("has_reactive_chain"):
                        step["reactive_chains"] = [{
                            "trigger_event": "PointModifiedEvent",
                            "recompute_description": matched_pattern.get("reactive_description", ""),
                            "debounce_ms": 300,
                        }]
                    else:
                        step["reactive_chains"] = []
                else:
                    # Automated step
                    step = {
                        "step_id": step_id,
                        "phase": step_id,
                        "step_type": "automated",
                        "description": ui_step.get("description", f"Automated: {step_id}"),
                        "method_name": logic_method,
                        "depends_on": [],
                        "produces_nodes": [],
                    }
                    if is_optional:
                        step["is_optional"] = True

                # Dependencies from UI workflow
                deps = ui_step.get("depends_on", [])
                if isinstance(deps, list) and deps:
                    step["depends_on"] = deps
                elif not deps and steps:
                    step["depends_on"] = [steps[-1]["step_id"]]

                step_id_set.add(step_id)
                steps.append(step)

        return steps

    @staticmethod
    def _match_pattern(
        logic_method: str, step_id: str, pattern_by_method: Dict, phases: list
    ) -> Optional[Dict]:
        """Find the best matching Stage 4.5 pattern for a UI workflow step."""
        if not logic_method and not step_id:
            return None
        # Try exact match on logic method name
        if logic_method:
            for p in phases:
                pn = p.get("phase_name", "")
                if pn.lower() == logic_method.lower():
                    return p
        # Try step_id match
        if step_id:
            for p in phases:
                pn = p.get("phase_name", "")
                if pn.lower() == step_id.lower():
                    return p
        # Try token overlap
        lm_tokens = set(logic_method.lower().split("_")) if logic_method else set()
        si_tokens = set(step_id.lower().split("_")) if step_id else set()
        search_tokens = lm_tokens | si_tokens
        if search_tokens:
            for p in phases:
                pn_tokens = set(p.get("phase_name", "").lower().split("_"))
                if search_tokens & pn_tokens:
                    return p
        return None

    def _finalize_workflow_graph(self, steps: list, stage_label: str) -> Dict:
        """Apply optional→branch transformation and build the workflow_graph dict."""
        # Mark optional phases as branch steps
        for i, step in enumerate(steps):
            if step.get("is_optional"):
                step["step_type"] = "branch"
                step["condition"] = step.get("description", "Optional step")
                next_steps = [
                    s["step_id"] for s in steps[i+1:]
                    if not s.get("is_optional")
                ]
                step["branches"] = {
                    "yes": step["step_id"],
                    "no": next_steps[0] if next_steps else "",
                }

        workflow_graph = {
            "steps": steps,
            "phases": [
                {
                    "name": s["step_id"],
                    "optional": s.get("is_optional", False),
                }
                for s in steps
            ],
        }

        self.on_progress(
            stage_label, "Workflow Graph Construction",
            f"Built graph with {len(steps)} steps"
        )
        return workflow_graph

    # ================================================================
    # Stage 4.9: Workflow Validation (LLM-assisted)
    # ================================================================

    def _stage4d_validate_workflow(
        self, workflow_graph: Dict, logic_analysis: Dict
    ) -> Dict:
        """LLM reviews the workflow graph for completeness and correctness."""
        self.on_progress("4.9", "Workflow Validation", "LLM reviewing workflow...")

        steps_summary = [
            {
                "step_id": s["step_id"],
                "step_type": s["step_type"],
                "description": s.get("description", ""),
                "interaction_type": s.get("interaction_type"),
                "depends_on": s.get("depends_on", []),
            }
            for s in workflow_graph.get("steps", [])
        ]

        method_names = [
            m.get("name", "") for m in logic_analysis.get("methods", [])
        ]

        validation_prompt = textwrap.dedent(f"""\
        Review this workflow graph for a Slicer extension CLI tool.
        Check for:
        1. Every interactive step has a clear user action described
        2. Every automated step has a callable method that exists
        3. Dependencies form a connected DAG (no orphans, no cycles)
        4. Step order is sensible (data loading before processing, etc.)

        IMPORTANT: Do NOT remove automated steps. Automated steps (like "create models",
        "center line", "update plan") are valid workflow steps that the user triggers
        by clicking a button. Only suggest removing steps that are clearly not part of
        the user workflow (e.g., internal timer callbacks, error handlers, email/feedback).

        Workflow steps: {json.dumps(steps_summary, indent=2)}
        Available logic methods: {json.dumps(method_names[:30])}

        If the graph is valid, respond with: {{"valid": true}}
        If there are issues, respond with: {{"valid": false, "fixes": [list of fixes to apply]}}
        Each fix should be: {{"step_id": "...", "field": "...", "new_value": ...}}
        To remove a step entirely, use: {{"step_id": "...", "field": "step_type", "new_value": "removed"}}
        """)

        try:
            llm_response = self._call_llm(validation_prompt)
            review = self._parse_json_response(llm_response)
            if isinstance(review, dict) and not review.get("valid", True):
                # Apply fixes
                steps_to_remove = set()
                for fix in review.get("fixes", []):
                    step_id = fix.get("step_id")
                    field = fix.get("field")
                    new_value = fix.get("new_value")
                    if field == "step_type" and new_value == "removed":
                        steps_to_remove.add(step_id)
                        continue
                    for step in workflow_graph.get("steps", []):
                        if step.get("step_id") == step_id:
                            step[field] = new_value
                # Remove marked steps
                if steps_to_remove:
                    # Build dependency map for rewiring
                    dep_map = {}
                    for s in workflow_graph.get("steps", []):
                        dep_map[s.get("step_id", "")] = s.get("depends_on", [])

                    # Rewire: if step A depends on removed step B,
                    # replace B with B's own dependencies
                    for s in workflow_graph.get("steps", []):
                        new_deps = []
                        for dep in s.get("depends_on", []):
                            if dep in steps_to_remove:
                                # Replace with the removed step's dependencies
                                replaced = dep_map.get(dep, [])
                                for r in replaced:
                                    if r not in steps_to_remove and r not in new_deps:
                                        new_deps.append(r)
                            else:
                                new_deps.append(dep)
                        s["depends_on"] = new_deps

                    workflow_graph["steps"] = [
                        s for s in workflow_graph.get("steps", [])
                        if s.get("step_id") not in steps_to_remove
                    ]
                    workflow_graph["phases"] = [
                        p for p in workflow_graph.get("phases", [])
                        if p.get("name") not in steps_to_remove
                    ]
                fix_count = len(review.get("fixes", []))
                remove_count = len(steps_to_remove)
                self.on_progress(
                    "4.9", "Workflow Validation",
                    f"Applied {fix_count - remove_count} fixes, removed {remove_count} steps"
                )
            else:
                self.on_progress("4.9", "Workflow Validation", "Workflow validated")
        except Exception as e:
            logger.warning(f"Stage 4.9 LLM validation failed: {e}")
            self.on_progress("4.9", "Workflow Validation", "Validation skipped (LLM error)")

        return workflow_graph

    # ================================================================
    # Stage 5: Tool Schema Generation (LLM-assisted)
    # ================================================================

    def _generate_workflow_schemas(
        self, extension_name: str, workflow_graph: Dict, logic_analysis: Dict,
    ) -> List[Dict]:
        """Generate tool schema for an interactive workflow extension."""
        steps = workflow_graph.get("steps", [])
        # Filter out removed/invalid steps
        valid_types = {"automated", "interactive", "branch"}
        steps = [s for s in steps if s.get("step_type") in valid_types]
        step_ids = [s["step_id"] for s in steps]
        automated_steps = [s for s in steps if s["step_type"] == "automated"]
        interactive_steps = [s for s in steps if s["step_type"] == "interactive"]
        branch_steps = [s for s in steps if s["step_type"] == "branch"]

        # Build enum of step IDs for the schema
        step_enum = step_ids

        # Build descriptions for each step
        step_descriptions = []
        for s in steps:
            desc = f"'{s['step_id']}': {s['description']}"
            if s["step_type"] == "interactive":
                desc += f" (interactive: {s.get('interaction_type', 'unknown')})"
            elif s["step_type"] == "branch":
                desc += " (optional — ask user first)"
            step_descriptions.append(desc)

        schema = {
            "type": "function",
            "function": {
                "name": extension_name,
                "description": (
                    f"Guided interactive workflow for {extension_name}. "
                    f"Execute steps in order. Interactive steps require user 3D interaction. "
                    f"Steps: {'; '.join(step_descriptions[:10])}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_step": {
                            "type": "string",
                            "enum": step_enum,
                            "description": "Which workflow step to execute",
                        },
                        "user_action": {
                            "type": "string",
                            "enum": ["start", "proceed", "skip", "cancel"],
                            "description": (
                                "Action: 'start' to begin a step, 'proceed' after user "
                                "completes interaction, 'skip' for optional steps, "
                                "'cancel' to abort workflow"
                            ),
                        },
                    },
                    "required": ["workflow_step", "user_action"],
                },
            },
        }

        self.on_progress(
            5, "Generating tool schemas",
            f"Generated interactive workflow schema with {len(steps)} steps"
        )
        return [schema]

    def _stage5_generate_schemas(
        self,
        extension_name: str,
        stage_map: Dict,
        logic_analysis: Dict,
        node_lifecycle: Optional[Dict] = None,
        cross_stage_map: Optional[Dict] = None,
        workflow_graph: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate OpenAI function-calling tool schemas."""
        self.on_progress(5, "Generating tool schemas", "Building tool definitions...")

        # Interactive workflow schema generation
        if workflow_graph:
            return self._generate_workflow_schemas(extension_name, workflow_graph, logic_analysis)

        stages = stage_map.get("stages", [])
        if not stages:
            raise RuntimeError("No stages identified — cannot generate tool schemas")

        # Determine which params are internal (output nodes, cross-stage inputs)
        internal_params = set()
        if node_lifecycle:
            for key, info in node_lifecycle.items():
                if info.get("param_role") == "output":
                    internal_params.add(key.split(":", 1)[1])
        if cross_stage_map:
            for k, stage_map_entry in cross_stage_map.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                internal_params.update(stage_map_entry.keys())

        # Build context for LLM — only include user-facing params
        stages_desc = []
        for s in stages:
            methods_desc = []
            for m in s.get("method_details", []):
                params_desc = []
                for p in m.get("parameters", []):
                    pname = p.get("name", "")
                    ptype = p.get("type", "")
                    # Skip params the template handles internally
                    if pname in internal_params:
                        params_desc.append(
                            f"    {pname}: {ptype} (INTERNAL — template creates this)"
                        )
                    elif "vtkMRML" in ptype:
                        # Input node from user (e.g., inputVolume)
                        params_desc.append(
                            f"    {pname}: {ptype} ({'required' if p.get('required') else 'optional'}) — {p.get('description', '')}"
                        )
                    elif "progress" in pname.lower() or pname == "qd":
                        params_desc.append(
                            f"    {pname}: {ptype} (INTERNAL — auto-filled by template)"
                        )
                    else:
                        params_desc.append(
                            f"    {pname}: {ptype} ({'required' if p.get('required') else 'optional'}) — {p.get('description', '')}"
                        )
                methods_desc.append(
                    f"  Method: {m['name']}\n"
                    f"  Purpose: {m.get('purpose', '')}\n"
                    f"  Parameters:\n" + "\n".join(params_desc)
                )
            stages_desc.append(
                f"Stage: {s['stage_name']}\n"
                f"Methods:\n" + "\n".join(methods_desc)
            )

        # Determine if we need a stage parameter (multiple stages) or single tool
        has_multiple_stages = len(stages) > 1

        # Reference schema format — minimal, only user-facing params
        example_schema = textwrap.dedent("""\
{
  "type": "function",
  "function": {
    "name": "PelvicFracturePlanning",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {
        "stage": {"type": "string", "enum": ["segmentation", "planning", "full"], "description": "..."},
        "volume_node_name": {"type": "string", "description": "..."}
      },
      "required": ["stage"]
    }
  }
}""")

        prompt = textwrap.dedent(f"""\
Generate an OpenAI function-calling tool schema for a Slicer extension named "{extension_name}".

Extension stages:
{chr(10).join(stages_desc)}

{'The extension has multiple stages, so include a "stage" enum parameter.' if has_multiple_stages else 'The extension has a single stage, so no "stage" parameter is needed.'}""")

        if self._readme_content:
            prompt += textwrap.dedent(f"""\

Extension README context (for designing descriptive tool parameters):
Focus on the pipeline and workflow descriptions; ignore setup and installation.

{self._readme_content}

""")

        if self._ui_workflow:
            prompt += textwrap.dedent(f"""\
Extracted UI Workflow (reflects the intended user-facing operation sequence):
```json
{json.dumps(self._ui_workflow, indent=2)}
```
Use this workflow to design tool schemas that match the actual user-facing steps.

""")

        prompt += textwrap.dedent("""\

IMPORTANT RULES:
- Do NOT include parameters marked as "INTERNAL" — the code template handles these automatically.
- Do NOT include any vtkMRML node parameters — the template creates/resolves them internally.
- Only include parameters that the LLM caller needs to provide: the "stage" enum, "volume_node_name", and any user-configurable options (thresholds, text prompts, flags, etc.).
- Boolean parameters with defaults should be optional with the default value.
- Filesystem paths (model_path, download_dir, output_path, etc.) MUST be optional — the template auto-discovers them. Do NOT mark them as required.
- The "required" list should be MINIMAL — only include what the LLM can reasonably fill from the user's request (e.g., text prompts). Never require paths, callbacks, or internal config.
- For text-prompted segmentation tools, include "text_prompts" (or similar) as required since the LLM derives it from the user's request (e.g., "segment the spine" → ["spine"]).

Also add an optional "volume_node_name" string parameter if the extension operates on a CT/volume input.

Reference schema format:
{example_schema}

Return a JSON array containing exactly one tool schema object.
The tool name should be based on the extension name (CamelCase, no spaces).
The description should explain what the tool does and when to use it.

Return ONLY the JSON array, no markdown fences.""")

        response = self._call_llm(prompt)
        schemas = self._parse_json_response(response)

        if isinstance(schemas, dict):
            schemas = [schemas]
        if not isinstance(schemas, list) or not schemas:
            raise RuntimeError(f"Invalid tool schema response: {str(response)[:300]}")

        # Validate schema structure
        for schema in schemas:
            if "function" not in schema:
                schema["function"] = schema
            if "type" not in schema:
                schema["type"] = "function"

        # Post-process: strip any remaining vtkMRML output params from schema
        # and demote filesystem path params from required
        _path_param_patterns = ("path", "dir", "directory", "folder")
        for schema in schemas:
            func = schema.get("function", {})
            params_obj = func.get("parameters", {})
            props = params_obj.get("properties", {})
            required = params_obj.get("required", [])

            to_remove = []
            for pname, pdef in props.items():
                ptype = pdef.get("type", "")
                desc = pdef.get("description", "")
                # Remove vtkMRML node params and progress/booleans the template handles
                if "vtkMRML" in str(ptype) or "vtkMRML" in desc:
                    to_remove.append(pname)
                elif pname in internal_params:
                    to_remove.append(pname)
                # Demote filesystem path params from required → optional
                elif pname.lower() != "volume_node_name":
                    plower = pname.lower()
                    if any(pat in plower for pat in _path_param_patterns):
                        if pname in required:
                            required.remove(pname)

            for pname in to_remove:
                del props[pname]
                if pname in required:
                    required.remove(pname)

            if required:
                params_obj["required"] = required
            elif "required" in params_obj and not required:
                del params_obj["required"]

        self.on_progress(
            5, "Generating tool schemas",
            f"Generated {len(schemas)} tool schema(s): "
            f"{[s.get('function', {}).get('name', '?') for s in schemas]}"
        )

        return schemas

    # ================================================================
    # Stage 6: Code Template Generation (LLM-assisted)
    # ================================================================

    def _generate_workflow_templates(
        self,
        extension_name: str,
        workflow_graph: Dict,
        scan_result: Dict,
        logic_analysis: Dict,
    ) -> Dict[str, str]:
        """
        Generate split templates for interactive workflow steps.

        For interactive steps: generates pre-interaction (node creation, placement mode)
        and post-interaction (validation, processing) templates.
        For automated steps: generates a single code template.
        Also generates the workflow.json file.
        """
        steps = workflow_graph.get("steps", [])
        templates = {}
        logic_class_name = scan_result.get("logic_class", {}).get("class_name", "")
        entry_module = scan_result.get("entry_module", "")
        module_name = os.path.splitext(os.path.basename(entry_module))[0] if entry_module else extension_name

        for step in steps:
            step_id = step["step_id"]
            step_type = step["step_type"]

            if step_type == "automated":
                # Single code template for automated steps
                tpl = self._generate_automated_workflow_template(
                    extension_name, step, logic_class_name, module_name, logic_analysis,
                )
                templates[f"templates/{step_id}.py.tpl"] = tpl
                step["code_template"] = f"templates/{step_id}.py.tpl"

            elif step_type == "interactive":
                # Pre-interaction template
                pre_tpl = self._generate_pre_interaction_template(
                    extension_name, step, logic_class_name, module_name,
                )
                templates[f"templates/{step_id}_pre.py.tpl"] = pre_tpl
                step["pre_template"] = f"templates/{step_id}_pre.py.tpl"

                # Post-interaction template
                post_tpl = self._generate_post_interaction_template(
                    extension_name, step, logic_class_name, module_name, logic_analysis,
                )
                templates[f"templates/{step_id}_post.py.tpl"] = post_tpl
                step["post_template"] = f"templates/{step_id}_post.py.tpl"

            elif step_type == "branch":
                # Branch steps don't need templates — handled by the orchestrator
                pass

        # Store workflow graph as JSON template (only valid steps)
        valid_types = {"automated", "interactive", "branch"}
        clean_graph = {k: v for k, v in workflow_graph.items() if k != "steps"}
        clean_graph["steps"] = [s for s in steps if s.get("step_type") in valid_types]
        templates["workflow.json"] = json.dumps(clean_graph, indent=2)

        self.on_progress(
            6, "Generating code templates",
            f"Generated {len(templates)} workflow templates"
        )
        return templates

    def _generate_automated_workflow_template(
        self, extension_name, step, logic_class_name, module_name, logic_analysis,
    ) -> str:
        """Generate a code template for an automated workflow step.

        Uses LLM to generate proper state setup. Falls back to a static template
        on LLM failure.
        """
        method_name = step.get("method_name", "")
        step_id = step.get("step_id", "")
        description = step.get("description", step_id)

        # Try LLM-assisted generation
        tpl = self._generate_automated_template_llm(
            extension_name, step, logic_class_name, module_name, logic_analysis,
        )
        if tpl:
            return tpl

        # Fallback: static template
        return textwrap.dedent(f"""\
            # --- {extension_name}: {description} ---
            try:
                from {module_name} import {logic_class_name}
            except ImportError:
                raise RuntimeError("{extension_name} extension is not installed.")

            logic = _{extension_name.lower()}_logic if '_{extension_name.lower()}_logic' in dir() else {logic_class_name}()

            # Execute the automated step
            result = logic.{method_name}() if '{method_name}' in dir(logic) else None

            _{extension_name.lower()}_logic = logic
            print("[{extension_name}] Step '{step_id}' completed.")
            """)

    def _generate_automated_template_llm(
        self, extension_name, step, logic_class_name, module_name, logic_analysis,
    ) -> Optional[str]:
        """Use LLM to generate an automated workflow template with proper state setup."""
        method_name = step.get("method_name", "")
        if not method_name:
            return None

        # Gather method info
        method_info = None
        for m in logic_analysis.get("methods", []):
            if m.get("name") == method_name:
                method_info = m
                break

        # Get method source
        logic_file = logic_analysis.get("_logic_file", "")
        method_source = self._extract_method_source(logic_file, method_name) or ""

        if not method_source and not method_info:
            return None

        # Truncate source if needed
        if len(method_source) > 5000:
            method_source = method_source[:5000] + "\n# ... [truncated]"

        # Build method signature info
        params_desc = ""
        if method_info:
            params = method_info.get("parameters", [])
            if params:
                params_desc = "Parameters:\n" + "\n".join(
                    f"  - {p.get('name')}: {p.get('type', '?')} ({'required' if p.get('required') else 'optional'}) — {p.get('description', '')}"
                    for p in params
                )
            state_reads = method_info.get("state_reads", [])
            state_writes = method_info.get("state_writes", [])
            if state_reads:
                params_desc += f"\nState reads: {', '.join(state_reads)}"
            if state_writes:
                params_desc += f"\nState writes: {', '.join(state_writes)}"

        # UI workflow context
        ui_context = ""
        if self._ui_workflow:
            for sec in self._ui_workflow.get("ui_sections", []):
                for s in sec.get("steps", []):
                    if s.get("step_id") == step.get("step_id") or s.get("logic_method") == method_name:
                        ui_context = f"Button label: '{s.get('button_label', '')}'\nDescription: {s.get('description', '')}"
                        break

        prompt = textwrap.dedent(f"""\
            Generate a Python code snippet for a 3D Slicer extension workflow step.

            Extension: {extension_name}
            Logic class: `{logic_class_name}` (import from `{module_name}`)
            Step: {step.get('step_id', '')}
            Method to call: `{method_name}()`
            {ui_context}

            {params_desc}

            Method source code:
            ```python
            {method_source}
            ```

            The code must:
            1. Import the logic class from `{module_name}`
            2. Reuse the existing logic instance `_{extension_name.lower()}_logic` if it exists, otherwise create a new `{logic_class_name}()`
            3. Set up any required state on the logic instance BEFORE calling the method (e.g., if the method reads `self.mandibleSegmentationNode`, find the node in the scene and assign it)
            4. To find scene nodes, use robust fuzzy matching — NEVER rely on exact node names. Use this pattern:
               ```python
               nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
               for i in range(nodes.GetNumberOfItems()):
                   n = nodes.GetItemAsObject(i)
                   if "fibula" in n.GetName().lower():
                       fibulaNode = n
                       break
               ```
               Or for parameter node references: first check `parameterNode.GetNodeReference("refName")`, and if None, search the scene by class + name substring, then set via `parameterNode.SetNodeReferenceID("refName", node.GetID())`.
            5. Call the method with correct arguments
            6. Store the logic instance as `_{extension_name.lower()}_logic` for subsequent steps
            7. Print a completion message

            IMPORTANT restrictions:
            - Do NOT use `dir()`, `eval()`, `exec()`, `globals()`, or `locals()` — these are blocked in the execution sandbox.
            - Use `try/except NameError` to check if a variable exists, NOT `if 'var' in dir()`.
            - Do NOT use curly brace template placeholders. Write actual Python values (strings, numbers, etc.). If you need a node name, hardcode a reasonable default like `slicer.util.getNode('FibulaModel')`.
            - Escape all braces in f-strings and .format() calls by doubling them: use doubled-braces for literal braces in output strings.
            - Return ONLY raw Python code. Do NOT wrap it in markdown fences (```python ... ```).""")

        try:
            response = self._call_llm(prompt)
            response = self._strip_markdown_fences(response) if response else None
            if response and "import" in response:
                return response
        except Exception:
            logger.debug("LLM automated template generation failed", exc_info=True)
        return None

    def _generate_pre_interaction_template(
        self, extension_name, step, logic_class_name, module_name,
    ) -> str:
        """Generate the pre-interaction template for an interactive step."""
        interaction_type = step.get("interaction_type", "unknown")
        node_class = step.get("node_class", "vtkMRMLMarkupsFiducialNode")
        instructions = step.get("placement_instructions", step.get("description", ""))
        node_name = step["step_id"].replace("_", " ").title()
        min_points = step.get("min_control_points", 0)

        return textwrap.dedent(f"""\
            # --- {extension_name}: {step.get('description', step['step_id'])} (Setup) ---
            import slicer

            # Create the markup node for user interaction
            node = slicer.mrmlScene.AddNewNodeByClass("{node_class}", "{node_name}")
            displayNode = node.GetDisplayNode()
            if displayNode is None:
                displayNode = node.CreateDefaultDisplayNode()
            displayNode.SetVisibility(True)

            print("[{extension_name}] Please {instructions}")
            print("When finished, press the 'Done' button in the workflow panel.")

            # Enter placement mode
            slicer.modules.markups.logic().SetActiveListID(node)
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            interactionNode.SwitchToPersistentPlaceMode()

            _{extension_name.lower()}_{step['step_id']}_id = node.GetID()
            """)

    def _generate_post_interaction_template(
        self, extension_name, step, logic_class_name, module_name, logic_analysis,
    ) -> str:
        """Generate the post-interaction template for an interactive step."""
        # Try LLM-assisted generation first
        llm_template = self._generate_post_interaction_template_llm(
            extension_name, step, logic_class_name, module_name, logic_analysis,
        )
        if llm_template:
            return llm_template

        # Fallback: static template
        min_points = step.get("min_control_points", 0)
        node_var = f"_{extension_name.lower()}_{step['step_id']}_id"

        validation_code = ""
        if min_points > 0:
            validation_code = textwrap.dedent(f"""\
                # Validate user input
                numPoints = node.GetNumberOfControlPoints()
                if numPoints < {min_points}:
                    raise RuntimeError(f"Need at least {min_points} control points, got {{numPoints}}. Please add more.")
                """)

        reactive_code = ""
        if step.get("reactive_chains"):
            for chain in step["reactive_chains"]:
                reactive_code += f"# Reactive chain: {chain.get('recompute_description', '')}\n"

        return textwrap.dedent(f"""\
            # --- {extension_name}: {step.get('description', step['step_id'])} (Process) ---
            import slicer

            node = slicer.mrmlScene.GetNodeByID({node_var})
            if node is None:
                raise RuntimeError("Node not found for step '{step['step_id']}'")

            {validation_code}
            # Exit placement mode
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            interactionNode.SwitchToViewTransformMode()

            {reactive_code}
            print(f"[{extension_name}] Step '{step['step_id']}' processed with {{node.GetNumberOfControlPoints()}} control points.")
            """)

    def _generate_post_interaction_template_llm(
        self, extension_name, step, logic_class_name, module_name, logic_analysis,
    ) -> Optional[str]:
        """Use LLM to generate a post-interaction template that calls the logic method."""
        method_name = step.get("method_name", "")
        if not method_name:
            return None

        # Gather method info
        method_info = None
        for m in logic_analysis.get("methods", []):
            if m.get("name") == method_name:
                method_info = m
                break

        # Get method source
        logic_file = logic_analysis.get("_logic_file", "")
        method_source = self._extract_method_source(logic_file, method_name) or ""

        if not method_source and not method_info:
            return None

        if len(method_source) > 5000:
            method_source = method_source[:5000] + "\n# ... [truncated]"

        # Build parameter / state info
        params_desc = ""
        if method_info:
            params = method_info.get("parameters", [])
            if params:
                params_desc = "Parameters:\n" + "\n".join(
                    f"  - {p.get('name')}: {p.get('type', '?')} ({'required' if p.get('required') else 'optional'}) — {p.get('description', '')}"
                    for p in params
                )
            state_reads = method_info.get("state_reads", [])
            state_writes = method_info.get("state_writes", [])
            if state_reads:
                params_desc += f"\nState reads: {', '.join(state_reads)}"
            if state_writes:
                params_desc += f"\nState writes: {', '.join(state_writes)}"

        node_class = step.get("node_class", "vtkMRMLMarkupsFiducialNode")
        node_var = f"_{extension_name.lower()}_{step['step_id']}_id"
        min_points = step.get("min_control_points", 0)

        # UI workflow context
        ui_context = ""
        if self._ui_workflow:
            for sec in self._ui_workflow.get("ui_sections", []):
                for s in sec.get("steps", []):
                    if s.get("step_id") == step.get("step_id") or s.get("logic_method") == method_name:
                        ui_context = f"Button label: '{s.get('button_label', '')}'\nDescription: {s.get('description', '')}"
                        break

        prompt = textwrap.dedent(f"""\
            Generate a Python code snippet for a 3D Slicer extension workflow step.
            This is the POST-INTERACTION part — the user has finished placing control points on a {node_class}.

            Extension: {extension_name}
            Logic class: `{logic_class_name}` (import from `{module_name}`)
            Step: {step.get('step_id', '')}
            Method to call: `{method_name}()`
            {ui_context}

            {params_desc}

            Method source code:
            ```python
            {method_source}
            ```

            Context: The user just placed control points on a markup node. The node ID is stored in variable `{node_var}`.

            The code must:
            1. Import the logic class from `{module_name}`
            2. Retrieve the markup node by its ID: `node = slicer.mrmlScene.GetNodeByID({node_var})`
            3. Validate the user placed enough control points ({min_points} minimum)
            4. Reuse the existing logic instance `_{extension_name.lower()}_logic` if it exists in `dir()`, otherwise create a new `{logic_class_name}()`
            5. Set up any required state on the logic instance BEFORE calling the method (e.g., if the method reads `self.inputMarkupNode`, assign the retrieved node to it)
            6. Call the method `{method_name}()` with correct arguments — pass the markup node if the method expects it
            7. Exit placement mode: `interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")` then `interactionNode.SwitchToViewTransformMode()`
            8. Store the logic instance as `_{extension_name.lower()}_logic` for subsequent steps
            9. Print a completion message with the number of control points

            IMPORTANT restrictions:
            - Do NOT use `dir()`, `eval()`, `exec()`, `globals()`, or `locals()` — these are blocked in the execution sandbox.
            - Use `try/except NameError` to check if a variable exists, NOT `if 'var' in dir()`.
            - Do NOT use curly brace template placeholders. Write actual Python values (strings, numbers, etc.). If you need a node name, use a hardcoded lookup like `slicer.util.getNode('NodeName')`.
            - Escape all braces in f-strings and .format() calls by doubling them: use doubled-braces for literal braces in output strings.
            - Return ONLY raw Python code. Do NOT wrap it in markdown fences (```python ... ```).""")

        try:
            response = self._call_llm(prompt)
            response = self._strip_markdown_fences(response) if response else None
            if response and "import" in response:
                return response
        except Exception:
            logger.debug("LLM post-interaction template generation failed", exc_info=True)
        return None

    def _stage6_generate_templates(
        self,
        extension_name: str,
        stage_map: Dict,
        node_lifecycle: Dict,
        scan_result: Dict,
        logic_analysis: Dict,
        cross_stage_map: Optional[Dict] = None,
        workflow_graph: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """Generate Python code templates for each stage."""
        # Interactive workflow template generation
        if workflow_graph:
            return self._generate_workflow_templates(
                extension_name, workflow_graph, scan_result, logic_analysis,
            )

        stages = stage_map.get("stages", [])
        templates = {}
        if cross_stage_map is None:
            cross_stage_map = {}

        for i, stage in enumerate(stages):
            stage_name = stage["stage_name"]
            self.on_progress(
                6, "Generating code templates",
                f"Generating template for stage '{stage_name}' ({i+1}/{len(stages)})..."
            )

            template = self._generate_single_template(
                extension_name, stage, node_lifecycle, scan_result, logic_analysis,
                stage_index=i, cross_stage_map=cross_stage_map,
            )
            templates[f"{stage_name}.py.tpl"] = template

        # Also generate "full" template if multiple stages
        if len(stages) > 1:
            self.on_progress(
                6, "Generating code templates",
                "Generating combined 'full' template..."
            )
            full_template = self._generate_full_template(
                extension_name, stages, node_lifecycle, scan_result, logic_analysis,
                cross_stage_map=cross_stage_map,
            )
            templates["full.py.tpl"] = full_template

        self.on_progress(
            6, "Generating code templates",
            f"Generated {len(templates)} templates: {list(templates.keys())}"
        )

        return templates

    def _stage6b_review_templates(
        self,
        templates: Dict[str, str],
        logic_analysis: Dict,
        node_lifecycle: Dict,
    ) -> Dict[str, str]:
        """LLM review of generated templates against actual method source."""
        self.on_progress(
            6, "Reviewing templates",
            "Sending templates to LLM for correctness review..."
        )

        logic_file = logic_analysis.get("_logic_file", "")
        class_name = logic_analysis.get("class_name", "")
        methods = logic_analysis.get("methods", [])
        reviewed = dict(templates)
        corrections_count = 0

        for tpl_name, tpl_code in templates.items():
            # Extract stage name from template filename
            stage_name = tpl_name.replace(".py.tpl", "")

            # Collect relevant method sources for this stage
            method_sources = []
            for m in methods:
                mname = m["name"]
                # Include method if its name appears in the template
                if f"logic.{mname}(" in tpl_code:
                    src = self._extract_method_source(logic_file, mname)
                    if src:
                        params_str = ", ".join(
                            f"{p['name']}: {p['type']}"
                            for p in m.get("parameters", [])
                        )
                        method_sources.append(
                            f"ACTUAL SIGNATURE: {mname}({params_str})\n"
                            f"Source:\n```python\n{src}\n```"
                        )

            if not method_sources:
                continue

            prompt = textwrap.dedent(f"""\
You are reviewing a generated code template for calling methods of a Slicer extension.

TEMPLATE NAME: {tpl_name}
TEMPLATE CODE:
```python
{tpl_code}
```

ACTUAL METHOD SIGNATURES AND SOURCES:
{chr(10).join(method_sources)}

NODE LIFECYCLE (how the template should create nodes):
{json.dumps(node_lifecycle, indent=2)}

CRITICAL CONSTRAINTS — do NOT violate these:
- Lines containing "slicer.mrmlScene.GetNodeByID(...)" are CROSS-STAGE wiring
  that resolves parameters from earlier pipeline stages by immutable node ID.
  Do NOT replace them with CreateNodeByClass/AddNewNodeByClass/slicer.util.getNode.
  They MUST stay as-is.
- Lines containing "_id = " followed by ".GetID()" are node ID caching for cross-stage
  lookups. Do NOT remove or modify them.
- Lines containing "# from prior stage" are variable aliasing in the full pipeline.
  Do NOT replace them.
- Lines containing "CreateNodeByClass" or "AddNewNodeByClass" are intentional node lifecycle
  choices based on AST analysis. Do NOT change the create mode (CreateNodeByClass vs AddNewNodeByClass)
  unless the method clearly does the opposite of what the lifecycle says.
- Lines containing template placeholders like {{param_name}} or {{param_name: default}}
  are dynamic fill points for the runtime template engine. Do NOT replace them with
  hardcoded values (e.g., do NOT change "textPrompts = {{text_prompts}}" to
  "textPrompts = []").
- Double-brace expressions like {{{{expr}}}} inside f-strings are intentional
  literal braces. Do NOT simplify {{{{expr}}}} to
  {{expr}} — that would create an invalid template placeholder.

Verify the template for these issues ONLY:
1. Are all logic.methodName() calls using the CORRECT parameter NAMES and COUNT?
2. Are all variables DEFINED before they are used (no NameError at runtime)?
3. Are boolean parameters set to valid literal values (True/False), not bare variable names?
4. Is the try/except for cached logic correct? (logic should be assigned in the except block, not after it)

Do NOT change: node creation mode, cross-stage wiring, or display setup code.

Return JSON:
{{
  "issues": [
    {{"line": 0, "problem": "description", "fix": "description of fix"}}
  ],
  "corrected_template": "the corrected full template string, or null if no changes needed"
}}

If the template is correct with no issues, return:
{{"issues": [], "corrected_template": null}}""")

            response = self._call_llm(prompt)
            review = self._parse_json_response(response)

            if not review:
                logger.warning("LLM review returned unparseable response for %s", tpl_name)
                continue

            issues = review.get("issues", [])
            corrected = review.get("corrected_template")

            if issues:
                issue_desc = "; ".join(
                    f"L{i.get('line', '?')}: {i.get('problem', '')}" for i in issues
                )
                logger.info("LLM review found %d issue(s) in %s: %s",
                            len(issues), tpl_name, issue_desc)

            if corrected and isinstance(corrected, str) and corrected.strip():
                # Validate the corrected template: fill placeholders then parse
                sample = corrected.replace(
                    "{vol_lookup}",
                    "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')"
                )
                sample = self._fill_remaining_placeholders(sample)
                try:
                    ast.parse(sample)
                    # Also verify the placeholder is preserved in the raw template
                    if "{vol_lookup}" not in corrected:
                        logger.warning(
                            "LLM correction for %s removed {vol_lookup} placeholder, keeping original",
                            tpl_name,
                        )
                        continue
                    reviewed[tpl_name] = corrected
                    corrections_count += 1
                    logger.info("Applied LLM correction to %s", tpl_name)
                except SyntaxError as e:
                    logger.warning(
                        "LLM correction for %s has syntax error, keeping original: %s",
                        tpl_name, e,
                    )

        if corrections_count:
            self.on_progress(
                6, "Reviewing templates",
                f"LLM corrected {corrections_count} template(s)"
            )
        else:
            self.on_progress(
                6, "Reviewing templates",
                "All templates passed LLM review"
            )

        return reviewed

    def _generate_single_template(
        self,
        extension_name: str,
        stage: Dict,
        node_lifecycle: Dict,
        scan_result: Dict,
        logic_analysis: Dict,
        stage_index: int = 0,
        cross_stage_map: Optional[Dict] = None,
    ) -> str:
        """Generate a code template for a single stage."""
        stage_name = stage["stage_name"]
        method_details = stage.get("method_details", [])
        module_name = os.path.splitext(os.path.basename(scan_result["entry_module"]))[0]
        class_name = logic_analysis["class_name"]
        if cross_stage_map is None:
            cross_stage_map = {}

        stage_cross = cross_stage_map.get(stage_index, {})
        ext_slug = cross_stage_map.get("_extension_name", extension_name).lower()

        # Build node creation / retrieval code
        node_creations = []
        output_param_names = set()
        for m in method_details:
            for p in m.get("parameters", []):
                ptype = p.get("type", "")
                pname = p.get("name", "")
                if "vtkMRML" not in ptype:
                    continue

                # Cross-stage inputs take priority — matched to prior stage outputs
                if pname in stage_cross:
                    src_param = stage_cross[pname]["source_param"]
                    # Use node ID lookup instead of name (methods may rename nodes)
                    node_creations.append(
                        f"{pname} = slicer.mrmlScene.GetNodeByID(_{ext_slug}_{src_param}_id)"
                    )
                    continue

                is_output = (
                    "output" in pname.lower()
                    or "result" in pname.lower()
                    or "out" in pname.lower()
                )
                if not is_output:
                    continue

                output_param_names.add(pname)
                key = f"{m['name']}:{pname}"
                lifecycle = node_lifecycle.get(key, {})
                mode = lifecycle.get("create_mode", "CreateNodeByClass")

                if mode == "CreateNodeByClass":
                    node_creations.append(
                        f"{pname} = slicer.mrmlScene.CreateNodeByClass(\"{ptype}\")"
                    )
                else:
                    node_creations.append(
                        f"{pname} = slicer.mrmlScene.AddNewNodeByClass(\"{ptype}\")"
                    )

        # Build param defaults for all non-vtkMRML, non-progress, non-self params
        param_defaults = []
        # Track which param names are already handled (vtkMRML nodes, inputVolume)
        handled_params = {"self"}
        for m in method_details:
            for p in m.get("parameters", []):
                pname = p.get("name", "")
                ptype = p.get("type", "")
                if "vtkMRML" in ptype:
                    handled_params.add(pname)
                if "progress" in pname.lower() or pname == "qd":
                    handled_params.add(pname)

        # Detect volume param and map to inputVolume
        volume_param_name = None
        for m in method_details:
            for p in m.get("parameters", []):
                pt = p.get("type", "")
                pn = p.get("name", "")
                if pn == "self":
                    continue
                if "vtkMRMLScalarVolumeNode" in pt or "vtkMRMLVolumeNode" in pt:
                    volume_param_name = pn
                    handled_params.add(pn)
                    if pn != "inputVolume":
                        param_defaults.append(f"{pn} = inputVolume  # alias for volume param")
                    break
            if volume_param_name:
                break

        for m in method_details:
            for p in m.get("parameters", []):
                ptype = p.get("type", "")
                pname = p.get("name", "")
                default = p.get("default")
                if pname in handled_params:
                    continue

                # Bool params — optional with detected or True default
                if ptype == "bool":
                    if default is not None:
                        param_defaults.append(f"{pname} = {{{pname}: {default}}}")
                    else:
                        param_defaults.append(f"{pname} = {{{pname}: True}}")
                # Callback / callable params — never fillable from arguments
                elif "callback" in pname.lower() or "callable" in ptype.lower():
                    param_defaults.append(f"{pname} = None")
                # Params with known defaults (from AST analysis)
                elif default is not None:
                    param_defaults.append(f"{pname} = {{{pname}: {default}}}")
                # String params without default → auto-discovery for paths, required otherwise
                elif ptype in ("str", "string"):
                    plower = pname.lower()
                    if "modelpath" in plower or "model_path" in plower:
                        # Auto-discover model path via logic
                        param_defaults.append(
                            f"{pname} = {{{pname}: logic.defaultModelPath() "
                            f"if hasattr(logic, 'defaultModelPath') else ''}}"
                        )
                    elif "path" in plower or "dir" in plower:
                        # Other path params — optional with empty default
                        param_defaults.append(f"{pname} = {{{pname}: ''}}")
                    else:
                        # General string params — required placeholder (LLM must provide)
                        param_defaults.append(f"{pname} = {{{pname}}}")
                # List/array params — required placeholder (LLM must provide the list)
                elif ptype in ("list", "array", "list[str]"):
                    param_defaults.append(f"{pname} = {{{pname}}}")
                # Numeric params — optional with 0 default
                elif ptype in ("int", "float"):
                    param_defaults.append(f"{pname} = {{{pname}: 0}}")
                # Everything else — optional with None default
                else:
                    param_defaults.append(f"{pname} = {{{pname}: None}}")

                handled_params.add(pname)

        # Build method call code
        method_calls = []
        for m in method_details:
            params = m.get("parameters", [])
            param_names = []
            for p in params:
                pn = p["name"]
                if pn == "self":
                    continue
                if "progress" in pn.lower() or pn == "qd":
                    param_names.append("_ProgressStub()")
                elif pn == volume_param_name and volume_param_name != "inputVolume":
                    param_names.append("inputVolume")
                else:
                    param_names.append(pn)
            method_calls.append(
                f"logic.{m['name']}({', '.join(param_names)})"
            )

        # Determine if this stage depends on prior state
        depends_on_prior = bool(stage.get("depends_on"))
        has_state_reads = any(
            m.get("state_reads") for m in method_details
        )

        # Build the template
        lines = [
            f"# --- {extension_name}: {stage_name.replace('_', ' ').title()} ---",
            f"# Auto-generated CLI template for {extension_name}.",
            "",
            "{vol_lookup}",
            "if inputVolume is None:",
            "    raise RuntimeError(\"No volume found in the scene. Load the required data first.\")",
            f'print(f"[{extension_name}] Using volume: {{{{inputVolume.GetName()}}}}")',
            "",
        ]

        # Import
        lines.extend([
            "try:",
            f"    from {module_name} import {class_name}",
            "except ImportError:",
            "    raise RuntimeError(",
            f"        \"{extension_name} extension is not installed. \"",
            "        \"Please install it via Slicer's Extension Manager first.\"",
            "    )",
            "",
        ])

        # State dependency check (only if there's a real prior stage)
        if stage_index > 0 and (depends_on_prior or has_state_reads):
            lines.extend([
                "# Retrieve cached state from prior stage",
                "try:",
                f"    logic = _{extension_name.lower()}Logic",
                "    print(\"Reusing cached logic instance from prior stage.\")",
                "except NameError:",
                f"    logic = {class_name}()",
                "",
            ])

        # Progress stub
        lines.extend([
            "class _ProgressStub:",
            "    def setMaximum(self, v): pass",
            "    def setValue(self, v): pass",
            "",
        ])

        # Logic instantiation (only if not already instantiated in try/except)
        if not (stage_index > 0 and (depends_on_prior or has_state_reads)):
            lines.extend([
                f"logic = {class_name}()",
                "",
            ])

        # Param defaults (boolean etc.)
        if param_defaults:
            for pd in param_defaults:
                lines.append(pd)
            lines.append("")

        # Node creation
        if node_creations:
            lines.append("# Create output nodes")
            for nc in node_creations:
                lines.append(nc)
            lines.append("")

        # Method calls
        lines.append(f"print(\"[{extension_name}] Running {stage_name}...\")")
        for mc in method_calls:
            lines.append(mc)
        lines.append("")

        # Cache state
        state_writes = []
        for m in method_details:
            state_writes.extend(m.get("state_writes", []))
        if state_writes:
            lines.extend([
                "# Cache for potential re-use",
                f"_{extension_name.lower()}Logic = logic",
                "",
            ])

        # Cache node IDs for cross-stage lookups (IDs are immutable, names are not)
        if output_param_names:
            lines.append("# Cache node IDs for subsequent stages")
            for pname in output_param_names:
                lines.append(f"_{ext_slug}_{pname}_id = {pname}.GetID()")
            lines.append("")

        # Display results for output segmentation nodes
        for pname in output_param_names:
            # Find the param type
            for m in method_details:
                for p in m.get("parameters", []):
                    if p["name"] == pname:
                        ptype = p.get("type", "")
                        if "vtkMRMLSegmentationNode" in ptype:
                            lines.extend([
                                f"{pname}.CreateClosedSurfaceRepresentation()",
                                f"_display = {pname}.GetDisplayNode()",
                                "if _display:",
                                "    _display.SetVisibility(True)",
                                "",
                            ])
                        elif "vtkMRMLModelNode" in ptype:
                            lines.extend([
                                f"_display = {pname}.GetDisplayNode()",
                                "if _display is None:",
                                f"    _display = {pname}.CreateDefaultDisplayNode()",
                                "_display.SetVisibility(True)",
                                "",
                            ])

        lines.extend([
            f"print(\"[{extension_name}] {stage_name.replace('_', ' ').title()} complete.\")",
        ])

        return "\n".join(lines)

    def _generate_full_template(
        self,
        extension_name: str,
        stages: List[Dict],
        node_lifecycle: Dict,
        scan_result: Dict,
        logic_analysis: Dict,
        cross_stage_map: Optional[Dict] = None,
    ) -> str:
        """Generate a combined template that runs all stages sequentially."""
        module_name = os.path.splitext(os.path.basename(scan_result["entry_module"]))[0]
        class_name = logic_analysis["class_name"]
        if cross_stage_map is None:
            cross_stage_map = {}

        # Track all created variables across stages for cross-stage wiring
        created_vars = {}  # var_name -> node_class

        lines = [
            f"# --- Full Pipeline: {extension_name} ---",
            f"# Auto-generated CLI template — runs all stages sequentially.",
            "",
            "{vol_lookup}",
            "if inputVolume is None:",
            "    raise RuntimeError(\"No volume found in the scene. Load the required data first.\")",
            f'print(f"[{extension_name}] Using volume: {{{{inputVolume.GetName()}}}}")',
            "",
            "try:",
            f"    from {module_name} import {class_name}",
            "except ImportError:",
            "    raise RuntimeError(",
            f"        \"{extension_name} extension is not installed. \"",
            "        \"Please install it via Slicer's Extension Manager first.\"",
            "    )",
            "",
            "class _ProgressStub:",
            "    def setMaximum(self, v): pass",
            "    def setValue(self, v): pass",
            "",
            f"logic = {class_name}()",
            "",
        ]

        # Collect param defaults across all methods (deduplicated)
        seen_default_names = set()
        all_defaults = []

        # Detect volume param name across all methods
        volume_param_name = None
        for stage in stages:
            for m in stage.get("method_details", []):
                for p in m.get("parameters", []):
                    pt = p.get("type", "")
                    pn = p.get("name", "")
                    if pn == "self":
                        continue
                    if "vtkMRMLScalarVolumeNode" in pt or "vtkMRMLVolumeNode" in pt:
                        if volume_param_name is None:
                            volume_param_name = pn
                        break

        # Track handled params
        handled_defaults = {"self", "inputvolume"}
        for stage in stages:
            for m in stage.get("method_details", []):
                for p in m.get("parameters", []):
                    pname = p.get("name", "")
                    ptype = p.get("type", "")
                    if "vtkMRML" in ptype:
                        handled_defaults.add(pname)
                    if "progress" in pname.lower() or pname == "qd":
                        handled_defaults.add(pname)

        # Volume param alias
        if volume_param_name and volume_param_name != "inputVolume":
            all_defaults.append(f"{volume_param_name} = inputVolume  # alias for volume param")
            handled_defaults.add(volume_param_name)

        for stage in stages:
            for m in stage.get("method_details", []):
                for p in m.get("parameters", []):
                    ptype = p.get("type", "")
                    pname = p.get("name", "")
                    default = p.get("default")
                    if pname in handled_defaults or pname in seen_default_names:
                        continue
                    seen_default_names.add(pname)

                    if ptype == "bool":
                        if default is not None:
                            all_defaults.append(f"{pname} = {default}")
                        else:
                            all_defaults.append(f"{pname} = True")
                    elif "callback" in pname.lower() or "callable" in ptype.lower():
                        all_defaults.append(f"{pname} = None")
                    elif default is not None:
                        all_defaults.append(f"{pname} = {default}")
                    elif ptype in ("str", "string"):
                        plower = pname.lower()
                        if "modelpath" in plower or "model_path" in plower:
                            all_defaults.append(
                                f"{pname} = logic.defaultModelPath() "
                                f'if hasattr(logic, "defaultModelPath") else ""'
                            )
                        else:
                            all_defaults.append(f'{pname} = ""')
                    elif ptype in ("list", "array", "list[str]"):
                        all_defaults.append(f"{pname} = []")
                    elif ptype in ("int", "float"):
                        all_defaults.append(f"{pname} = 0")
                    else:
                        all_defaults.append(f"{pname} = None")
        if all_defaults:
            for d in all_defaults:
                lines.append(d)
            lines.append("")

        # Generate each stage
        for i, stage in enumerate(stages):
            stage_name = stage["stage_name"]
            method_details = stage.get("method_details", [])
            stage_cross = cross_stage_map.get(i, {})

            lines.append(f"# === STAGE {i+1}: {stage_name.replace('_', ' ').title()} ===")

            # Node creations for this stage
            for m in method_details:
                for p in m.get("parameters", []):
                    ptype = p.get("type", "")
                    pname = p.get("name", "")
                    if "vtkMRML" not in ptype:
                        continue
                    # Cross-stage: check if this param matches a prior stage output
                    if pname in stage_cross:
                        src_param = stage_cross[pname]["source_param"]
                        lines.append(f"{pname} = {src_param}  # from prior stage")
                        continue
                    is_output = (
                        "output" in pname.lower()
                        or "result" in pname.lower()
                        or "out" in pname.lower()
                    )
                    if not is_output:
                        continue

                    key = f"{m['name']}:{pname}"
                    lc = node_lifecycle.get(key, {})
                    mode = lc.get("create_mode", "CreateNodeByClass")
                    if mode == "CreateNodeByClass":
                        lines.append(
                            f'{pname} = slicer.mrmlScene.CreateNodeByClass("{ptype}")'
                        )
                    else:
                        lines.append(
                            f'{pname} = slicer.mrmlScene.AddNewNodeByClass("{ptype}")'
                        )
                    created_vars[pname] = ptype

            # Method calls
            for m in method_details:
                params = m.get("parameters", [])
                param_names = []
                for p in params:
                    pn = p["name"]
                    if pn == "self":
                        continue
                    if "progress" in pn.lower() or pn == "qd":
                        param_names.append("_ProgressStub()")
                    elif pn == volume_param_name and volume_param_name != "inputVolume":
                        param_names.append("inputVolume")
                    else:
                        param_names.append(pn)
                lines.append(f"logic.{m['name']}({', '.join(param_names)})")

            lines.append(f"print(f'  Stage {i+1} complete.')")
            lines.append("")

        # Cache
        lines.extend([
            f"_{extension_name.lower()}Logic = logic",
            "",
            f"print(\"[{extension_name}] === Pipeline Complete ===\")",
        ])

        return "\n".join(lines)

    # ================================================================
    # Stage 7: Prompt Fragment Generation (LLM-assisted)
    # ================================================================

    def _generate_workflow_prompt_fragment(
        self, extension_name: str, tool_schemas: List[Dict], workflow_graph: Dict,
    ) -> str:
        """Generate prompt fragment for an interactive workflow tool."""
        steps = workflow_graph.get("steps", [])
        tool_name = tool_schemas[0]["function"]["name"] if tool_schemas else extension_name

        # Determine the actual first step ID
        first_step_id = steps[0]["step_id"] if steps else ""

        lines = [
            f"### Interactive Workflow: {extension_name}",
            "",
            f"**Tool name:** `{tool_name}`",
            f"**Type:** Guided interactive workflow",
            "",
            "This tool orchestrates a multi-step workflow where some steps require the user to",
            "perform 3D interactions (drawing curves, positioning planes, placing fiducials).",
            "Execute steps sequentially, ONE STEP PER TURN. After each interactive step, relay instructions to the user",
            "and wait for them to complete the interaction before proceeding.",
            "",
            "**Workflow Steps:**",
        ]

        for i, step in enumerate(steps):
            step_type = step["step_type"]
            desc = step.get("description", step["step_id"])
            marker = "[automated]" if step_type == "automated" else "[interactive]" if step_type == "interactive" else "[optional]"
            lines.append(f"{i+1}. `{step['step_id']}` {marker} — {desc}")
            if step_type == "interactive":
                lines.append(f"   - Interaction: {step.get('interaction_type', 'unknown')}")
                if step.get("placement_instructions"):
                    lines.append(f"   - Tell user: {step['placement_instructions']}")

        lines.extend([
            "",
            "**Protocol:**",
            f"1. Call `{tool_name}` with `workflow_step='{first_step_id}'` and `user_action='start'` to begin",
            "2. For **automated** steps: output the returned `code` verbatim in a ```python block. Then call the next step.",
            "3. For **interactive** steps: output the returned `pre_code` verbatim in a ```python block. Relay instructions to the user. Wait for them to click 'Done'.",
            "4. For **optional** steps: ask user if they want to proceed. If yes, call with `user_action='start'`. If no, call with `user_action='skip'`.",
            "5. After each step completes, call the tool with the NEXT step's `step_id` and `user_action='start'`.",
            "6. Continue until all steps are done.",
            "",
            "**CRITICAL RULES:**",
            "- Execute ONE step per turn. Do NOT call multiple steps in a single turn.",
            "- Do NOT skip automated steps. Their code MUST be output and executed.",
            "- Always start from step 1 (`" + first_step_id + "`) and proceed in order.",
        ])

        fragment = "\n".join(lines)
        self.on_progress(7, "Generating prompt fragment", "Generated workflow prompt")
        return fragment

    def _stage7_generate_prompt(
        self,
        extension_name: str,
        tool_schemas: List[Dict],
        stage_map: Dict,
        logic_analysis: Dict,
        workflow_graph: Optional[Dict] = None,
    ) -> str:
        """Generate markdown prompt fragment for system prompt injection."""
        self.on_progress(7, "Generating prompt fragment", "Building usage instructions...")

        # Interactive workflow prompt fragment
        if workflow_graph:
            return self._generate_workflow_prompt_fragment(
                extension_name, tool_schemas, workflow_graph,
            )

        stages = stage_map.get("stages", [])
        tool_name = tool_schemas[0]["function"]["name"] if tool_schemas else extension_name

        # Use LLM to generate a user-facing capability summary
        capability = self._llm_capability_summary(extension_name, logic_analysis, stages)

        # Build stage descriptions — one concise line per stage
        # Pick the most "primary" method's purpose for each stage
        stage_lines = []
        for s in stages:
            sname = s["stage_name"]
            stage_desc = self._stage_description(s)
            stage_lines.append(
                f'  - `stage="{sname}"` — {stage_desc}'
            )

        # Build full stage description
        if len(stages) > 1:
            full_line = (
                f'  - `stage="full"` — Run the complete pipeline: '
                + " + ".join(s["stage_name"] for s in stages)
            )
        else:
            full_line = ""

        # Prerequisites
        prereqs = [
            f"{extension_name} Slicer extension must be installed",
            "Required data (e.g., CT volume) must be loaded in the scene",
        ]

        # Check for GPU requirements
        logic_source = logic_analysis.get("_logic_source", "")
        if "torch" in logic_source or "cuda" in logic_source or "gpu" in logic_source.lower():
            prereqs.append("CUDA GPU recommended (CPU fallback may be very slow)")

        # Check for model files
        if "model" in logic_source.lower() and ("load" in logic_source.lower() or "path" in logic_source.lower()):
            prereqs.append("Pre-trained model files must be present in the extension's Resources directory")

        fragment = textwrap.dedent(f"""\
### {extension_name} Extension

- **{extension_name}**: If the user asks to {capability}, call `{tool_name}` with the appropriate `stage` parameter rather than writing custom code.
{chr(10).join(stage_lines)}
{full_line}
  - Prerequisites: {"; ".join(prereqs)}
  **CRITICAL**: After receiving the `{tool_name}` result, your very next response must be exactly one ```agent_plan JSON block followed by one ```python code block containing the tool's `code` string verbatim. Do NOT modify the generated code. Do NOT write analysis or planning text before the code blocks.
""")

        self.on_progress(7, "Generating prompt fragment", "Prompt fragment generated")
        return fragment.strip()

    def _llm_capability_summary(self, extension_name: str, logic_analysis: Dict, stages: List[Dict]) -> str:
        """Use LLM to generate a concise user-facing capability description."""
        methods = logic_analysis.get("methods", [])
        class_name = logic_analysis.get("class_name", "")

        # Collect method names and their docstrings/purposes
        method_info = []
        for m in methods:
            name = m.get("name", "")
            purpose = m.get("purpose", "")
            params = [p.get("name", "") for p in m.get("parameters", []) if p.get("name") != "self"]
            method_info.append(f"- {name}({', '.join(params)}): {purpose}")

        stage_info = []
        for s in stages:
            sname = s.get("stage_name", "")
            sdesc = s.get("description", "")
            method_names = [m.get("name", "") for m in s.get("method_details", [])]
            stage_info.append(f"- Stage '{sname}': {sdesc} (methods: {', '.join(method_names)})")

        prompt = textwrap.dedent(f"""\
You are writing a trigger phrase for a Slicer extension tool.

Extension name: {extension_name}
Logic class: {class_name}

Methods in {class_name}:
{chr(10).join(method_info)}

Pipeline stages:
{chr(10).join(stage_info)}

Task: Write ONE concise sentence (under 20 words) describing what this extension does FROM THE USER'S PERSPECTIVE.
Focus on the END RESULT the user wants (e.g., "segment anatomical structures from CT volumes using text prompts").
Do NOT mention internal steps like installing packages, downloading models, or caching.
Do NOT mention the extension name.

The sentence will be used in this context: "If the user asks to [YOUR SENTENCE], call the tool."

Examples of good outputs:
- "segment bones, organs, or other structures from a CT or MRI volume using text prompts"
- "segment pelvic fractures and plan surgical screw placement"
- "register two volumes using rigid or affine transformation"
- "measure distances and angles between markup points"

Return ONLY the sentence, nothing else.""")

        response = self._call_llm(prompt)
        if response:
            summary = response.strip().strip('"').strip("'")
            # Truncate if too long
            if len(summary) > 150:
                summary = summary[:147] + "..."
            return summary

        # Fallback
        return f"use {extension_name} on a loaded volume"

    @staticmethod
    def _stage_description(stage: Dict) -> str:
        """Generate a concise user-facing description for a single pipeline stage."""
        method_details = stage.get("method_details", [])
        # Find the "primary" method: the one that produces output nodes or
        # has the most relevant name (run, process, segment, etc.)
        primary = None
        for m in method_details:
            name_lower = m.get("name", "").lower()
            if any(kw in name_lower for kw in ("run", "process", "segment", "execute", "perform")):
                primary = m
                break
        if primary is None and method_details:
            primary = method_details[0]

        if primary:
            purpose = primary.get("purpose", "")
            if purpose:
                return purpose

        # Fallback to stage name
        return stage.get("stage_name", "unknown").replace("_", " ")

    # ================================================================
    # Stage 8: Validation (CodeValidator, no LLM)
    # ================================================================

    def _stage8_validate(
        self,
        templates: Dict[str, str],
        generators: List[Dict],
        logic_analysis: Optional[Dict] = None,
    ) -> Dict:
        """Validate all templates with CodeValidator + semantic checks."""
        self.on_progress(8, "Validating templates", "Running CodeValidator...")

        if not self.code_validator:
            from .CodeValidator import CodeValidator
            self.code_validator = CodeValidator()

        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "per_template": {},
        }

        for tpl_name, tpl_content in templates.items():
            # Skip non-Python files (e.g., workflow.json)
            if not tpl_name.endswith((".py.tpl", ".py")):
                continue

            # Fill with sample values for validation
            sample_code = tpl_content.replace(
                "{vol_lookup}",
                "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')"
            )
            # Fill any remaining placeholders with defaults
            sample_code = self._fill_remaining_placeholders(sample_code)

            # CodeValidator (security + syntax)
            validation = self.code_validator.validate(sample_code)

            # Semantic validation (undefined vars, arg count)
            if logic_analysis:
                semantic = self._semantic_validate(sample_code, logic_analysis)
                if semantic.get("errors"):
                    validation["valid"] = False
                    existing_reason = validation.get("reason") or ""
                    new_reasons = "; ".join(semantic["errors"])
                    validation["reason"] = (
                        f"{existing_reason}; {new_reasons}" if existing_reason
                        else new_reasons
                    )
                if semantic.get("warnings"):
                    validation.setdefault("warnings", []).extend(semantic["warnings"])

            results["per_template"][tpl_name] = validation

            if not validation.get("valid", True):
                results["valid"] = False
                results["errors"].append(
                    f"{tpl_name}: {validation.get('reason', 'unknown error')}"
                )
            if validation.get("warnings"):
                results["warnings"].extend(
                    f"{tpl_name}: {w}" for w in validation.get("warnings", [])
                )

        self.on_progress(
            8, "Validating templates",
            "PASS" if results["valid"] else f"FAIL: {results['errors']}"
        )

        return results

    def _semantic_validate(self, code: str, logic_analysis: Dict) -> Dict:
        """Check for undefined variables, wrong arg counts, invalid node types."""
        result = {"errors": [], "warnings": []}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result["errors"].append("Syntax error in generated code")
            return result

        # Collect defined names (assignments, imports, function/class defs, for-loop targets)
        defined = set()
        # Builtins and implicit Slicer names that are always available
        defined.update({
            "slicer", "qt", "vtk", "ctk", "inputVolume", "logic",
            "True", "False", "None", "print", "range", "len", "int",
            "float", "str", "bool", "list", "dict", "set", "tuple",
            "isinstance", "type", "super", "property", "staticmethod",
            "classmethod", "hasattr", "getattr", "callable", "abs",
            "min", "max", "sum", "any", "all", "sorted", "reversed",
            "enumerate", "zip", "map", "filter", "round", "hex", "oct",
            "Exception", "ValueError", "RuntimeError",
            "ImportError", "NameError", "TypeError", "KeyError",
            "AttributeError", "IndexError", "FileNotFoundError",
            "os", "json", "math", "time", "path",
            "_ProgressStub",
        })

        # Collect names from assignments and imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined.add(elt.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
                for arg in node.args.args:
                    defined.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)
            elif isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.name:
                        defined.add(handler.name)

        # Find undefined variables (names used but never defined)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined and not node.id.startswith("_"):
                    # Skip common patterns
                    if node.id in ("self", "cls"):
                        continue
                    result["errors"].append(f"Undefined variable: '{node.id}'")

        # Check method call arg counts
        method_signatures = {}
        for m in logic_analysis.get("methods", []):
            param_count = len(m.get("parameters", []))
            # Subtract 'self' if present
            params = m.get("parameters", [])
            if params and params[0].get("name") == "self":
                param_count -= 1
            method_signatures[m["name"]] = param_count

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "logic"
                        and node.func.attr in method_signatures):
                    expected = method_signatures[node.func.attr]
                    actual = len(node.args)
                    if actual != expected:
                        result["errors"].append(
                            f"logic.{node.func.attr}() called with {actual} args, "
                            f"expected {expected}"
                        )

        # Check node class strings are valid MRML types
        valid_prefixes = (
            "vtkMRMLScalar", "vtkMRMLSegmentation", "vtkMRMLModel",
            "vtkMRMLMarkup", "vtkMRMLTransform", "vtkMRMLVolume",
            "vtkMRMLLabelMap", "vtkMRMLTable", "vtkMRMLChart",
            "vtkMRMLView", "vtkMRMLLayout", "vtkMRMLCamera",
            "vtkMRMLClip", "vtkMRMLColor", "vtkMRMLDisplay",
            "vtkMRMLStorage", "vtkMRMLSubjectHierarchy",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in ("CreateNodeByClass", "AddNewNodeByClass"):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            cls = arg.value
                            if not cls.startswith(valid_prefixes):
                                result["warnings"].append(
                                    f"Unknown MRML node class: '{cls}'"
                                )

        return result

    @staticmethod
    def _fill_remaining_placeholders(code: str) -> str:
        """Fill any remaining {placeholder} or {placeholder: default} patterns with safe defaults."""
        import re
        def _replace(match):
            full = match.group(0)
            name = match.group(1)
            default = match.group(4)  # may be None (group 4 = actual default value, group 3 = ': ' separator)
            # If the placeholder has an inline default (e.g. {text_prompts: ["seg"]}), use it
            if default is not None:
                return default
            # Common placeholders
            if "name" in name.lower():
                return '"SampleNode"'
            if "radius" in name.lower() or "size" in name.lower():
                return "1.5"
            if "path" in name.lower():
                return '"/tmp/sample"'
            return '""'
        # Replace single-brace placeholders that aren't double-brace escapes
        # First, temporarily replace {{ }} with a sentinel
        sentinel = "\x00LBRACE\x00"
        code = code.replace("{{", sentinel + "{")
        code = code.replace("}}", "}" + sentinel)
        # Match {name} or {name: default_value} — the default can contain brackets, strings, etc.
        # We use a balanced-brace match for the default portion.
        code = re.sub(r'\{(\w+)((: )(.*?))?\}', _replace, code)
        # Restore literal braces
        code = code.replace(sentinel + "{", "{{")
        code = code.replace("}" + sentinel, "}}")
        return code

    # ================================================================
    # Revision System
    # ================================================================

    def revise(
        self,
        extension_name: str,
        errors: List[str],
        max_attempts: int = _MAX_REVISION_ATTEMPTS,
    ) -> Dict:
        """
        Revise failed templates using LLM feedback.

        Args:
            extension_name: Name of the CLI to revise.
            errors: List of error messages from validation or testing.

        Returns:
            Dict with 'success', 'validation_result', 'attempts' keys.
        """
        from .ExtensionCLILoader import get_cli_base_dir

        cli_dir = os.path.join(get_cli_base_dir(), extension_name)
        if not os.path.isdir(cli_dir):
            return {"success": False, "error": f"No CLI found for {extension_name}"}

        # Load existing CLI data
        manifest_path = os.path.join(cli_dir, "manifest.json")
        generators_path = os.path.join(cli_dir, "code_generators.json")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        with open(generators_path, "r") as f:
            generators = json.load(f)

        result = {
            "success": False,
            "validation_result": None,
            "attempts": 0,
            "error": None,
        }

        for attempt in range(max_attempts):
            result["attempts"] = attempt + 1
            self.on_progress(
                0, "Revising templates",
                f"Revision attempt {attempt + 1}/{max_attempts}..."
            )

            # Read all templates
            templates = {}
            for gen in generators:
                # Collect all template file references from this generator
                tpl_files = []
                if gen.get("template_file"):
                    tpl_files.append(gen["template_file"])
                if gen.get("pre_template_file"):
                    tpl_files.append(gen["pre_template_file"])
                if gen.get("post_template_file"):
                    tpl_files.append(gen["post_template_file"])
                for tpl_file in tpl_files:
                    tpl_path = os.path.join(cli_dir, tpl_file)
                    if os.path.isfile(tpl_path):
                        with open(tpl_path, "r") as f:
                            templates[tpl_file] = f.read()

            # Build revision prompt
            templates_text = "\n\n".join(
                f"--- {name} ---\n{content}"
                for name, content in templates.items()
            )

            prompt = textwrap.dedent(f"""\
The following code templates for the "{extension_name}" extension failed validation.
Please fix ALL errors while maintaining the template format (use {{placeholder}} for dynamic values, {{{{ }}}} for literal braces).

ERRORS:
{chr(10).join(f'- {e}' for e in errors)}

CONSTRAINTS (CodeValidator):
- BLOCKED: os, subprocess, sys, socket, urllib, http, pickle, ctypes, mmap
- BLOCKED: eval, exec, compile, __import__, open, file, input, getattr, setattr, delattr
- BLOCKED: globals, locals, vars, dir
- ALLOWED: slicer, vtk, qt, ctk, numpy, SimpleITK, math, json, re, copy

TEMPLATES TO FIX:
{templates_text}

Return a JSON object with this structure:
{{
  "templates": {{
    "template_file_name.py.tpl": "fixed template content",
    ...
  }},
  "fix_description": "what was changed and why"
}}

Return ONLY the JSON, no markdown fences.""")

            response = self._call_llm(prompt)
            fixed = self._parse_json_response(response)

            if not fixed or "templates" not in fixed:
                self.on_error(f"Revision attempt {attempt + 1}: LLM returned invalid response")
                continue

            # Save fixed templates — ensure .py.tpl files go into templates/ subdir
            for tpl_name, tpl_content in fixed["templates"].items():
                if tpl_name.endswith(".py.tpl") and not tpl_name.startswith("templates/"):
                    tpl_name = f"templates/{tpl_name}"
                tpl_path = os.path.join(cli_dir, tpl_name)
                os.makedirs(os.path.dirname(tpl_path), exist_ok=True)
                with open(tpl_path, "w", encoding="utf-8") as f:
                    f.write(tpl_content)

            # Re-validate
            if not self.code_validator:
                from .CodeValidator import CodeValidator
                self.code_validator = CodeValidator()

            all_valid = True
            new_errors = []
            for gen in generators:
                # Collect all template files to re-validate
                tpl_files = []
                if gen.get("template_file"):
                    tpl_files.append(gen["template_file"])
                if gen.get("pre_template_file"):
                    tpl_files.append(gen["pre_template_file"])
                if gen.get("post_template_file"):
                    tpl_files.append(gen["post_template_file"])
                for tpl_file in tpl_files:
                    if not tpl_file.endswith((".py.tpl", ".py")):
                        continue
                    tpl_path = os.path.join(cli_dir, tpl_file)
                    if not os.path.isfile(tpl_path):
                        continue
                    with open(tpl_path, "r") as f:
                        content = f.read()

                    sample = content.replace(
                        "{vol_lookup}",
                        "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')"
                    )
                    sample = self._fill_remaining_placeholders(sample)
                    validation = self.code_validator.validate(sample)

                    if not validation.get("valid", True):
                        all_valid = False
                        new_errors.append(f"{tpl_file}: {validation.get('reason', 'unknown')}")

            if all_valid:
                # Update manifest status
                manifest["status"] = "validated"
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

                # Append to generation log
                log_path = os.path.join(cli_dir, "generation_log.json")
                log_entries = []
                if os.path.isfile(log_path):
                    with open(log_path, "r") as f:
                        log_entries = json.load(f)
                log_entries.append({
                    "attempt": len(log_entries) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "stage": "revision",
                    "trigger": "validation_failure",
                    "error": "; ".join(errors),
                    "fix": fixed.get("fix_description", ""),
                    "validation_result": {"valid": True},
                })
                with open(log_path, "w") as f:
                    json.dump(log_entries, f, indent=2)

                from .ExtensionCLILoader import invalidate_cache
                invalidate_cache()

                result["success"] = True
                result["validation_result"] = {"valid": True}
                return result

            errors = new_errors

        result["error"] = f"Revision failed after {max_attempts} attempts"
        return result

    def _build_workflow_manifest_and_generators(
        self,
        extension_name: str,
        scan_result: Dict,
        workflow_graph: Dict,
    ) -> Tuple[Dict, List[Dict]]:
        """Build manifest and generators for an interactive workflow."""
        steps = workflow_graph.get("steps", [])
        stage_names = [s["step_id"] for s in steps]

        manifest = {
            "extension_name": extension_name,
            "extension_module_name": os.path.splitext(os.path.basename(scan_result.get("entry_module", "")))[0],
            "logic_class_name": scan_result.get("logic_class", {}).get("class_name", ""),
            "version": "1.0.0",
            "status": "validated",
            "workflow_type": "interactive",
            "workflow_graph_file": "workflow.json",
            "stages": stage_names,
        }

        generators = []
        for step in steps:
            step_id = step["step_id"]
            step_type = step["step_type"]

            gen = {
                "tool_name": extension_name,
                "param_signature": {"workflow_step": step_id},
                "description": step.get("description", step_id),
                "requirements": [f"{extension_name} extension must be installed"],
                "step_type": step_type,
            }

            if step_type == "automated" and step.get("code_template"):
                gen["template_file"] = step["code_template"]
            elif step_type == "interactive":
                gen["pre_template_file"] = step.get("pre_template", "")
                gen["post_template_file"] = step.get("post_template", "")
                gen["interaction_descriptor"] = {
                    "interaction_type": step.get("interaction_type"),
                    "node_class": step.get("node_class"),
                    "placement_instructions": step.get("placement_instructions", ""),
                }
            elif step_type == "branch":
                gen["condition"] = step.get("condition", "")
                gen["branches"] = step.get("branches", {})

            generators.append(gen)

        return manifest, generators

    # ================================================================
    # Helpers
    # ================================================================

    def _build_manifest_and_generators(
        self,
        extension_name: str,
        scan_result: Dict,
        stage_map: Dict,
        workflow_graph: Optional[Dict] = None,
    ) -> Tuple[Dict, List[Dict]]:
        """Build manifest.json and code_generators.json contents."""
        # Interactive workflow manifest
        if workflow_graph:
            return self._build_workflow_manifest_and_generators(
                extension_name, scan_result, workflow_graph,
            )

        stages = stage_map.get("stages", [])
        has_multiple = len(stages) > 1

        # Build stage enum values
        stage_names = [s["stage_name"] for s in stages]
        if has_multiple:
            stage_names.append("full")

        # Build generators list
        generators = []
        for s in stages:
            sname = s["stage_name"]
            methods = s.get("method_details", [])
            descriptions = [m.get("purpose", "") for m in methods]
            requirements = [
                f"{extension_name} Slicer extension must be installed",
            ]
            # Check for GPU requirement
            for m in methods:
                for p in m.get("parameters", []):
                    if "progress" in p.get("name", "").lower():
                        break

            generators.append({
                "tool_name": extension_name,
                "param_signature": {"stage": sname} if has_multiple else {},
                "template_file": f"templates/{sname}.py.tpl",
                "description": "; ".join(descriptions) if descriptions else sname,
                "requirements": requirements,
            })

        if has_multiple:
            generators.append({
                "tool_name": extension_name,
                "param_signature": {"stage": "full"},
                "template_file": "templates/full.py.tpl",
                "description": f"Complete {extension_name} pipeline: " + " + ".join(
                    s["stage_name"] for s in stages
                ),
                "requirements": [
                    f"{extension_name} Slicer extension must be installed",
                ],
            })

        manifest = {
            "extension_name": extension_name,
            "extension_module_name": os.path.splitext(
                os.path.basename(scan_result.get("entry_module", ""))
            )[0],
            "logic_class_name": scan_result.get("logic_class", {}).get("class_name", ""),
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "source_type": "analyzed_extension",
            "source_path": scan_result.get("source_path", ""),
            "status": "validated",
            "tool_count": 1,
            "stages": stage_names,
        }

        return manifest, generators

    def _call_llm(self, user_prompt: str) -> str:
        """Make an isolated LLM call and return the text response.

        If self._debug_dir is set, also saves the full input/output/thinking
        to a JSON file in the debug directory.
        """
        messages = [
            {"role": "system", "content": self._analyzer_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.chatIsolated(messages)
        message_text = response.get("message", "")

        # Debug saving
        if self._debug_dir:
            try:
                self._save_debug_call(messages, response)
            except Exception:
                logger.debug("Failed to save debug call", exc_info=True)

        return message_text

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove surrounding ```python ... ``` or ``` ... ``` fences from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1:]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()
        return text

    def _save_debug_call(self, messages: list, response: dict) -> None:
        """Save a single LLM call's debug info to a JSON file."""
        if not os.path.isdir(self._debug_dir):
            os.makedirs(self._debug_dir, exist_ok=True)

        call_index = self._llm_call_counter
        self._llm_call_counter += 1

        debug_entry = {
            "call_index": call_index,
            "timestamp": datetime.now().isoformat(),
            "stage": self._current_stage_label,
            "input": {
                "system_prompt": messages[0].get("content", "") if len(messages) > 0 else "",
                "user_prompt": messages[1].get("content", "") if len(messages) > 1 else "",
            },
            "output": {
                "message": response.get("message", ""),
                "reasoning_content": response.get("reasoning_content", ""),
            },
            "usage": response.get("usage", {}),
        }

        filename = f"stage_{self._current_stage_label}_call_{call_index:03d}.json"
        filepath = os.path.join(self._debug_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(debug_entry, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _parse_json_response(text: str) -> Any:
        """Extract and parse JSON from an LLM response."""
        if not text:
            return None

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            import re
            # Look for JSON array or object
            for pattern in [
                r'\[[\s\S]*\]',
                r'\{[\s\S]*\}',
            ]:
                match = re.search(pattern, text)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        continue
            logger.warning("Could not parse JSON from LLM response: %s", text[:300])
            return None

    def _extract_class_source(self, file_path: str, class_name: str) -> Optional[str]:
        """Extract the full source of a class from a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except Exception:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                lines = source.split("\n")
                start = node.lineno - 1
                # Find end: last line of the class body
                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
                return "\n".join(lines[start:end])

        return None

    def _extract_method_source(self, file_path: str, method_name: str) -> Optional[str]:
        """Extract the source of a specific method from the logic file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except Exception:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    lines = source.split("\n")
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 50
                    return "\n".join(lines[start:end])

        return None

    # ================================================================
    # README Discovery
    # ================================================================

    _README_NAMES = ("README.md", "README", "Readme.md", "README.MD", "README.rst")

    def _find_readme(
        self, extension_name: str, source_path: str, source_type: str
    ) -> Optional[str]:
        """Locate and read a README for the extension.

        Strategy depends on *source_type*:
        - "extension_manager" -> look up the README in the knowledge base
          (Resources/Skills/slicer-skill-full/slicer-extensions/) using the
          JSON metadata mapping.
        - "additional_paths" / "loaded_modules" -> look for a README directly
          inside *source_path* (the source folder itself may contain one).

        Returns the README text (truncated), or None if not found.
        """
        if source_type == "extension_manager":
            return self._find_readme_in_knowledge_base(extension_name)
        return self._find_readme_in_directory(source_path)

    def _find_readme_in_knowledge_base(self, extension_name: str) -> Optional[str]:
        """Find README via the slicer-extensions knowledge base JSON mapping."""
        try:
            ext_index_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Resources", "Skills", "slicer-skill-full", "slicer-extensions",
            )
            if not os.path.isdir(ext_index_dir):
                return None

            json_path = os.path.join(ext_index_dir, f"{extension_name}.json")
            if not os.path.isfile(json_path):
                return None

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            scm_url = meta.get("scm_url", "")
            if not scm_url:
                return None

            repo_name = scm_url.rstrip("/").split("/")[-1].replace(".git", "")
            repo_dir = os.path.join(ext_index_dir, repo_name)
            if not os.path.isdir(repo_dir):
                return None

            return self._read_readme_from_dir(repo_dir)
        except Exception:
            logger.debug("README lookup failed for %s", extension_name, exc_info=True)
            return None

    def _find_readme_in_directory(self, directory: str) -> Optional[str]:
        """Find README directly in the given source directory."""
        try:
            if not os.path.isdir(directory):
                return None
            return self._read_readme_from_dir(directory)
        except Exception:
            logger.debug("README lookup failed in %s", directory, exc_info=True)
            return None

    def _read_readme_from_dir(self, directory: str) -> Optional[str]:
        """Search for a README file in *directory* and return its content."""
        entries = os.listdir(directory)
        for name in self._README_NAMES:
            if name in entries:
                readme_path = os.path.join(directory, name)
                if os.path.isfile(readme_path):
                    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
        # Case-insensitive fallback (e.g. readme.md)
        for entry in entries:
            if entry.upper().startswith("README") and os.path.isfile(os.path.join(directory, entry)):
                readme_path = os.path.join(directory, entry)
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        return None

    # ================================================================
    # Stage 1.5: UI Workflow Extraction
    # ================================================================

    def _stage1_5_extract_workflow(self, scan_result: Dict) -> Optional[Dict]:
        """Extract the user-facing workflow from UI (.ui file) + Widget class + README.

        Returns a structured workflow dict, or None if insufficient UI data.
        """
        widget_class = scan_result.get("widget_class")
        ui_files = scan_result.get("ui_files", [])

        # Parse .ui file(s)
        ui_sections = None
        for ui_path in ui_files:
            parsed = self._parse_ui_file(ui_path)
            if parsed and parsed.get("sections"):
                ui_sections = parsed
                break

        # Extract Widget signal connections
        widget_connections = []
        widget_source = None
        if widget_class:
            widget_source = self._extract_class_source(
                widget_class["file"], widget_class["class_name"]
            )
            if widget_source:
                widget_connections = self._extract_widget_connections(widget_source)

        # If no UI data at all, skip this stage
        if not ui_sections and not widget_connections and not self._readme_content:
            self.on_progress(1.5, "UI workflow extraction", "No UI/Widget/README data — skipping")
            return None

        # Build the LLM prompt for workflow synthesis
        prompt_parts = [
            "## Task: Synthesize Extension Workflow from UI Analysis\n",
            "You are analyzing a 3D Slicer extension's user-facing workflow.",
            "Based on the UI layout, Widget signal connections, and README description below,",
            "produce a structured JSON workflow that reflects the actual user-facing workflow.\n",
        ]

        # UI sections and buttons
        if ui_sections:
            prompt_parts.append("### UI Layout (from .ui file)\n")
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(ui_sections, indent=2))
            prompt_parts.append("```\n")

        # Widget signal connections
        if widget_connections:
            prompt_parts.append("### Widget Signal Connections (from AST)\n")
            prompt_parts.append("Each entry maps a UI button to its handler method and the logic methods it calls.\n")
            prompt_parts.append("Some buttons may not appear in the UI Layout above (created programmatically).\n")
            prompt_parts.append("You MUST include steps for these buttons too — match them by their handler/logic method names.\n")
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(widget_connections, indent=2))
            prompt_parts.append("```\n")

        # README workflow
        if self._readme_content:
            prompt_parts.append("### README Workflow Description\n")
            prompt_parts.append(self._readme_content)
            prompt_parts.append("\n")

        # Logic class methods (for context)
        logic_class = scan_result.get("logic_class")
        if logic_class:
            prompt_parts.append("### Logic Class Methods\n")
            prompt_parts.append(f"Class: `{logic_class['class_name']}`\n")
            prompt_parts.append("Methods: " + ", ".join(f"`{m}`" for m in logic_class.get("methods", [])))
            prompt_parts.append("\n")

        # Output schema instructions
        prompt_parts.append("### Required Output\n")
        prompt_parts.append(textwrap.dedent("""\
            Return a single JSON object with this structure:
            ```json
            {
              "ui_sections": [
                {
                  "section_name": "Section Name from UI",
                  "is_optional": false,
                  "steps": [
                    {
                      "step_id": "snake_case_id",
                      "button_label": "Button text from UI",
                      "logic_method": "methodName",
                      "description": "What this step does",
                      "step_type": "automated" or "interactive",
                      "interaction_type": "fiducial|curve|line|plane|null",
                      "depends_on": ["previous_step_id"],
                      "is_optional": false
                    }
                  ]
                }
              ]
            }
            ```

            Rules:
            1. Use the UI section order as the workflow sequence.
            2. Match button widget names to logic methods using the signal connections.
            3. Include ALL buttons from the signal connections, even those not in the UI Layout — they are created programmatically.
            4. If the README describes a workflow, use it to validate and enrich descriptions.
            5. `step_type` is "interactive" if the button triggers user 3D interaction (placing markups, drawing curves), "automated" for buttons that just trigger computation.
            6. `interaction_type` should be one of: fiducial, curve, line, plane, or null for automated steps.
            7. `depends_on` should list step_ids of prerequisite steps (sequential by default).
            8. Mark optional/experimental sections with `is_optional: true`.
            9. Use descriptive snake_case step_ids that reflect the button's purpose.
            10. Return ONLY the JSON object, no other text.
        """))

        full_prompt = "\n".join(prompt_parts)
        response_text = self._call_llm(full_prompt)

        if not response_text:
            self.on_progress(1.5, "UI workflow extraction", "LLM returned empty response")
            return None

        workflow = self._parse_json_response(response_text)
        if not workflow or "ui_sections" not in workflow:
            self.on_progress(1.5, "UI workflow extraction", "Failed to parse workflow JSON from LLM")
            return None

        return workflow
