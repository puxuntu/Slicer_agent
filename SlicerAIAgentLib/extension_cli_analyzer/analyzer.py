from .common import *
from .scan import AnalyzerScanMixin
from .logic_analysis import AnalyzerLogicAnalysisMixin
from .stage4_decomposition import AnalyzerStage4DecompositionMixin
from .cross_stage import AnalyzerCrossStageMixin
from .workflow_detection import AnalyzerWorkflowDetectionMixin
from .schemas import AnalyzerSchemasMixin
from .workflow_templates import AnalyzerWorkflowTemplatesMixin
from .template_helpers import AnalyzerTemplateHelpersMixin
from .api_probe import AnalyzerApiProbeMixin
from .template_generation import AnalyzerTemplateGenerationMixin
from .prompt_validation import AnalyzerPromptValidationMixin
from .validation_contracts import AnalyzerValidationContractsMixin
from .validation_semantics import AnalyzerValidationSemanticsMixin
from .live_revision import AnalyzerLiveRevisionMixin
from .cookbook_mapping import AnalyzerCookbookMappingMixin
from .parameter_metadata import AnalyzerParameterMetadataMixin
from .workflow_contracts import AnalyzerWorkflowContractsMixin
from .slicer_op_manifest import AnalyzerSlicerOpManifestMixin


class ExtensionCLIAnalyzer(
    AnalyzerScanMixin,
    AnalyzerLogicAnalysisMixin,
    AnalyzerStage4DecompositionMixin,
    AnalyzerCrossStageMixin,
    AnalyzerWorkflowDetectionMixin,
    AnalyzerSchemasMixin,
    AnalyzerWorkflowTemplatesMixin,
    AnalyzerTemplateHelpersMixin,
    AnalyzerApiProbeMixin,
    AnalyzerTemplateGenerationMixin,
    AnalyzerPromptValidationMixin,
    AnalyzerValidationContractsMixin,
    AnalyzerValidationSemanticsMixin,
    AnalyzerLiveRevisionMixin,
    AnalyzerCookbookMappingMixin,
    AnalyzerParameterMetadataMixin,
    AnalyzerWorkflowContractsMixin,
    AnalyzerSlicerOpManifestMixin,
):
    """
    Analyzes a Slicer extension's source code and generates operation CLIs.

    9-stage cookbook-driven pipeline:
    1. AST Scanning (no LLM)
    2. Cookbook Detection & Parsing (regex-based, no LLM, REQUIRED)
    3. Logic Class Analysis (LLM)
    3.5. AST Signature Verification (no LLM)
    4. Cookbook Stage Map — LLM Decomposition into sub-operations
    4.5. Cross-Stage Parameter Mapping (LLM)
    5C. Workflow Graph Building
    5T. Slicer Op Templates (KB search + LLM)
    6. Tool Schema Generation (LLM)
    7. Code Template Generation (LLM, includes internal LLM review)
    8. Prompt Fragment Generation (LLM)
    9. Validation + Save (CodeValidator, semantic checks, optional revision)
    """

    def __init__(
        self,
        llm_client,
        output_base_dir: Optional[str] = None,
        code_validator=None,
        on_progress: Optional[Callable[[int, str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        method_keyword_map: Optional[Dict[str, str]] = None,
        live_probe_executor: Optional[Callable[[str], Any]] = None,
    ):
        """
        Args:
            llm_client: LLMClient instance for making LLM calls.
            output_base_dir: Base directory for saving CLI packages.
                             Defaults to Resources/extension_CLI/.
            code_validator: CodeValidator instance. Created if not provided.
            on_progress: Callback(stage_num, stage_name, detail) for progress updates.
            on_error: Callback(error_message) for error reporting.
            method_keyword_map: Deprecated compatibility argument. Semantic
                cookbook interpretation is performed by the Stage 4 LLM and
                validated against source-derived candidates.
            live_probe_executor: Optional callable that executes a probe snippet
                on Slicer's main thread and returns the probe result.
        """
        self.llm_client = llm_client
        self.output_base_dir = output_base_dir or self._default_base_dir()
        self.code_validator = code_validator
        self.on_progress = on_progress or (lambda n, s, d: None)
        self.on_error = on_error or (lambda e: None)
        self._method_keyword_map = {}
        self._live_probe_executor = live_probe_executor
        self._analyzer_prompt = self._load_analyzer_prompt()
        self._cancelled = False
        # Pipeline-scoped state (reset in analyze_and_generate)
        self._ui_workflow: Optional[Dict] = None  # only used by legacy fallback methods
        self._debug_dir: Optional[str] = None
        self._llm_call_counter: int = 0
        self._current_stage_label: str = ""
        self._cookbook_def = None                # Parsed CookbookDef when cookbook found
        self._widget_connections: List[Dict] = []
        self._ui_parameter_bindings: Dict[str, Dict] = {}
        self._slicer_op_templates: Dict = {}     # Pre-generated slicer_op templates
        self._slicer_op_evidence: Dict = {}      # API evidence for pre-generated slicer_op templates
        self._placement_starter_methods: Dict = {}
        self._workflow_metadata: Dict = {}
        self._last_logic_analysis: Optional[Dict] = None
        self._last_api_probe_result: Optional[Dict] = None

    @staticmethod
    def _default_base_dir() -> str:
        # __file__ may not point to the actual source in some Slicer setups
        # (e.g., when loaded from a staged build directory). Try multiple paths.
        candidates = []
        try:
            candidates.append(_PROJECT_ROOT)
        except Exception:
            pass
        # Fallback: look relative to the SlicerAIAgent module
        try:
            import SlicerAIAgent as _root_mod
            candidates.append(os.path.dirname(os.path.abspath(_root_mod.__file__)))
        except Exception:
            pass
        for module_dir in candidates:
            cli_dir = os.path.join(module_dir, "Resources", "extension_CLI")
            if os.path.isdir(cli_dir):
                return cli_dir
        # Last resort: return the first candidate anyway (will fail later
        # with a clear error if the path doesn't exist)
        if candidates:
            return os.path.join(candidates[0], "Resources", "extension_CLI")
        return ""

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
        Run the full 9-stage cookbook-driven analysis pipeline.

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
        self._debug_dir = None
        self._llm_call_counter = 0
        self._current_stage_label = ""
        self._cookbook_def = None
        self._slicer_op_templates = {}
        self._slicer_op_evidence = {}
        self._placement_starter_methods = {}
        self._widget_connections = []
        self._ui_parameter_bindings = {}
        self._workflow_metadata = {}
        self._last_logic_analysis = None
        self._last_api_probe_result = None

        # Validate extension name (prevent path traversal)
        extension_name = _validate_extension_name(extension_name)

        result = {
            "success": False,
            "cli_dir": None,
            "manifest": None,
            "stages_completed": [],
            "validation_result": None,
            "error": None,
        }

        if not self.output_base_dir:
            self.output_base_dir = self._default_base_dir()
        if not self.output_base_dir:
            result["error"] = (
                "output_base_dir is not set. "
                "Ensure the analyzer is constructed with a valid output directory."
            )
            return result

        ext_dir = os.path.join(self.output_base_dir, extension_name)
        if os.path.isdir(ext_dir) and not force_overwrite:
            result["error"] = f"CLI for '{extension_name}' already exists. Use force_overwrite=True."
            return result

        try:
            # Set up debug directory (lazily created on first LLM call)
            self._debug_dir = os.path.join(ext_dir, "debug")

            # ── Stage 1: AST Scanning (no LLM) ──
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

            # Extract Widget button→logic-method connections for post-classification verification
            self._widget_connections = []
            widget_info = scan_result.get("widget_class")
            if widget_info:
                widget_source = self._extract_class_source(
                    widget_info.get("file", ""), widget_info.get("class_name", "")
                )
                if widget_source:
                    self._widget_connections = self._extract_widget_connections(widget_source)
            scan_result["ui_widgets"] = self._extract_ui_widget_inventory(
                scan_result.get("ui_files", [])
            )
            entry_source = self._read_entry_source(scan_result)
            self._ui_parameter_bindings = self._extract_ui_parameter_bindings(
                entry_source,
                scan_result.get("ui_files", []),
                self._widget_connections,
            )
            scan_result["ui_parameter_bindings"] = self._ui_parameter_bindings

            # ── Stage 2: Cookbook Detection & Parsing (no LLM, REQUIRED) ──
            self._current_stage_label = "2"
            cookbook_path = self._find_cookbook(extension_name)
            if not cookbook_path:
                result["error"] = (
                    f"No cookbook found for '{extension_name}'. "
                    "A cookbook (.md) in Resources/extensions_cookbook/ is required "
                    "for pipeline generation. "
                    "Expected: Resources/extensions_cookbook/{extension_name}.md "
                    "or Resources/extensions_cookbook/Slicer{extension_name}.md"
                )
                return result

            try:
                from ..CookbookParser import CookbookParser
                parser = CookbookParser()
                self._cookbook_def = parser.parse(cookbook_path)
            except Exception as e:
                result["error"] = f"Cookbook parse error: {e}"
                return result

            if not self._cookbook_def or not self._cookbook_def.steps:
                result["error"] = (
                    f"Cookbook found at {cookbook_path} but failed to parse "
                    "or contains no steps."
                )
                return result

            self.on_progress(
                2, "Cookbook detection & parsing",
                f"Parsed cookbook: {cookbook_path} "
                f"({len(self._cookbook_def.steps)} steps)"
            )
            result["stages_completed"].append(2)

            # ── Stage 3: Logic Class Analysis (LLM) ──
            self._current_stage_label = "3"
            logic_analysis = self._stage3_analyze_logic(scan_result)
            self._last_logic_analysis = logic_analysis
            result["logic_analysis"] = logic_analysis
            result["stages_completed"].append(3)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 3"
                return result

            # ── Stage 3.5: AST Signature Verification (no LLM) ──
            self._current_stage_label = "3.5"
            self._verify_signatures_ast(logic_analysis, scan_result)
            result["stages_completed"].append("3.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 3.5"
                return result

            # ── Stage 4: Validated LLM semantic decomposition ──
            self._current_stage_label = "4"
            stage_map = self._stage4_cookbook_decomposition(
                self._cookbook_def, logic_analysis, scan_result
            )
            result["stages_completed"].append(4)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 4"
                return result

            # ── Stage 4.5: Cross-Stage Parameter Mapping (LLM) ──
            self._current_stage_label = "4.5"
            cross_stage_map = self._stage4_5_cross_stage_mapping(
                stage_map, logic_analysis, extension_name
            )
            result["stages_completed"].append("4.5")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 4.5"
                return result

            # ── Stage 5C: Workflow Graph Building ──
            self._current_stage_label = "5C"
            workflow_graph = self._build_workflow_from_cookbook(
                self._cookbook_def, logic_analysis, stage_map
            )
            self._workflow_metadata = self._build_workflow_metadata(
                scan_result, logic_analysis, workflow_graph
            )
            self._enrich_workflow_with_metadata(workflow_graph, self._workflow_metadata)
            result["stages_completed"].append("5C")
            if self._cancelled:
                result["error"] = "Cancelled during Stage 5C"
                return result

            # ── Stage 5T: Slicer Op Template Generation (KB search) ──
            self._current_stage_label = "5T"
            self._slicer_op_templates = self._generate_slicer_op_templates(
                stage_map
            )
            result["stages_completed"].append("5T")

            # ── Stage 6: Tool Schema Generation (LLM) ──
            self._current_stage_label = "6"
            tool_schemas = self._stage6_generate_schemas(
                extension_name, stage_map, logic_analysis,
                cross_stage_map=cross_stage_map,
                workflow_graph=workflow_graph,
            )
            result["stages_completed"].append(6)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 6"
                return result

            # ── Stage 7: Code Template Generation (LLM + internal review) ──
            self._current_stage_label = "7"
            node_lifecycle = self._compute_node_lifecycle(scan_result, logic_analysis)
            self._placement_starter_methods = self._classify_placement_starter_methods(
                logic_analysis
            )
            self._normalize_workflow_contracts(
                workflow_graph, self._workflow_metadata, scan_result, logic_analysis
            )
            self._synthesize_workflow_ui_guidance(
                workflow_graph, self._workflow_metadata, scan_result, logic_analysis
            )
            templates = self._stage7_generate_templates(
                extension_name, stage_map, node_lifecycle, scan_result, logic_analysis,
                cross_stage_map=cross_stage_map,
                workflow_graph=workflow_graph,
            )
            # Internal LLM review (not a separate numbered stage)
            templates = self._review_templates(templates, logic_analysis, node_lifecycle)  # internal LLM review
            templates = self._sanitize_templates(templates)
            result["stages_completed"].append(7)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 7"
                return result

            # ── Stage 7.5: Live API Probing ──
            self._current_stage_label = "7.5"
            probe_result = self._stage7c_live_api_probe(templates)
            self._last_api_probe_result = probe_result
            result["stages_completed"].append("7.5")
            result["api_probe_result"] = probe_result
            if probe_result.get("revised", 0) > 0:
                templates = self._sanitize_templates(templates)
            if self._cancelled:
                result["error"] = "Cancelled during Stage 7.5"
                return result

            # ── Stage 9: Validation + Save ──
            self._current_stage_label = "9"
            manifest, generators = self._build_manifest_and_generators(
                extension_name, scan_result, stage_map,
                workflow_graph=workflow_graph,
            )
            self._sync_template_contracts(
                templates,
                generators,
                workflow_graph=workflow_graph,
            )
            result["workflow_metadata"] = self._workflow_metadata
            validation_result = self._stage9_validate(
                templates, generators, logic_analysis=logic_analysis,
                api_probe_result=probe_result,
                extension_name=extension_name,
            )
            result["stages_completed"].append(9)
            result["validation_result"] = validation_result

            if validation_result.get("valid"):
                manifest["status"] = "validated"
                manifest["validation_state"] = self._workflow_metadata.get("validation_state", {})
                # ── Stage 8: Prompt Fragment Generation (LLM) ──
                # Generate the runtime prompt only after templates are known to
                # be valid, so prompt instructions match the final package.
                self._current_stage_label = "8"
                prompt_fragment = self._stage8_generate_prompt(
                    extension_name, tool_schemas, stage_map, logic_analysis,
                    workflow_graph=workflow_graph,
                )
                result["stages_completed"].append(8)
                if self._cancelled:
                    result["error"] = "Cancelled during Stage 8"
                    return result
            else:
                manifest["status"] = "validation_failed"
                manifest["validation_state"] = self._workflow_metadata.get("validation_state", {})
                prompt_fragment = (
                    f"### {extension_name}\n\n"
                    "Generation failed validation. This CLI package is saved "
                    "only for debugging/revision and is not loaded as a runtime tool.\n"
                )
            templates["workflow_metadata.json"] = json.dumps(self._workflow_metadata or {}, indent=2)

            # Save CLI package
            from ..ExtensionCLILoader import save_cli_package
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
                    "api_probe_result": probe_result,
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



def _patch_mixin_globals():
    from . import scan, logic_analysis, stage4_decomposition, cross_stage, workflow_detection, schemas, workflow_templates, template_helpers, api_probe, template_generation, prompt_validation, validation_contracts, validation_semantics, live_revision, cookbook_mapping, parameter_metadata, workflow_contracts, slicer_op_manifest
    for _module in [scan, logic_analysis, stage4_decomposition, cross_stage, workflow_detection, schemas, workflow_templates, template_helpers, api_probe, template_generation, prompt_validation, validation_contracts, validation_semantics, live_revision, cookbook_mapping, parameter_metadata, workflow_contracts, slicer_op_manifest]:
        _module.ExtensionCLIAnalyzer = ExtensionCLIAnalyzer


_patch_mixin_globals()
