from .common import *


class AnalyzerLiveRevisionMixin:
    def live_validate_templates(
        cli_dir: str,
        executor,
        on_progress=None,
    ) -> Dict[str, Dict]:
        """Validate generated templates by executing them in Slicer's Python console.

        Runs each .py.tpl template with safe default placeholder values via
        SafeExecutor.execute(always_rollback=True). The MRML scene is always
        rolled back after each execution regardless of success or failure.

        This method MUST be called on the Qt main thread (where SafeExecutor
        has access to slicer.mrmlScene).

        Args:
            cli_dir: Path to the extension CLI directory (containing templates/).
            executor: A SafeExecutor instance.
            on_progress: Optional callback(idx, total, key, result) called
                after each template is tested.

        Returns:
            Dict mapping template key → {
                "live_valid": bool,
                "error": str or None,
                "output": str,
                "execution_time": float,
            }
        """
        import glob as _glob

        templates_dir = os.path.join(cli_dir, "templates")
        if not os.path.isdir(templates_dir):
            return {}

        # Collect all .py.tpl files
        tpl_files = sorted(_glob.glob(os.path.join(templates_dir, "*.py.tpl")))
        if not tpl_files:
            return {}

        # ── Capture full scene state before validation ──
        # SafeExecutor's always_rollback handles MRML node add/remove,
        # but does NOT restore layout, module, or interaction state.
        # We capture these separately and restore after all templates run.
        _pre_layout = None
        _pre_module = None
        _pre_interaction_node_mode = None
        try:
            lm = slicer.app.layoutManager()
            if lm:
                _pre_layout = lm.layout
        except Exception:
            pass
        try:
            _pre_module = slicer.util.getSelectedModule()
        except Exception:
            pass
        try:
            interactionNode = slicer.mrmlScene.GetNodeByID(
                "vtkMRMLInteractionNodeSingleton"
            )
            if interactionNode:
                _pre_interaction_node_mode = interactionNode.GetCurrentInteractionMode()
        except Exception:
            pass

        results = {}
        total = len(tpl_files)

        for idx, tpl_path in enumerate(tpl_files):
            tpl_key = os.path.relpath(tpl_path, cli_dir)  # e.g. "templates/cb_step_4_pre.py.tpl"
            tpl_name = os.path.basename(tpl_path)

            try:
                with open(tpl_path, "r", encoding="utf-8") as f:
                    raw_code = f.read()
            except Exception as e:
                results[tpl_key] = {
                    "live_valid": False,
                    "error": f"Failed to read template: {e}",
                    "output": "",
                    "execution_time": 0,
                }
                if on_progress:
                    on_progress(idx, total, tpl_key, results[tpl_key])
                continue

            if not raw_code.strip():
                results[tpl_key] = {
                    "live_valid": True,
                    "error": None,
                    "output": "(empty template)",
                    "execution_time": 0,
                }
                if on_progress:
                    on_progress(idx, total, tpl_key, results[tpl_key])
                continue

            # Fill placeholders with safe defaults
            filled_code = ExtensionCLIAnalyzer._fill_remaining_placeholders(raw_code)

            # Wrap in try/except to catch runtime errors cleanly
            wrapped_code = textwrap.dedent(f"""\
                _tpl_validation_error = None
                try:
                    exec({repr(filled_code)})
                except SystemExit:
                    pass  # SystemExit is ok (e.g. sys.exit in tested code)
                except Exception as _e:
                    _tpl_validation_error = f"{{type(_e).__name__}}: {{_e}}"
                """).strip()

            exec_result = executor.execute(wrapped_code, always_rollback=True)

            # Check for execution errors
            error = exec_result.get("error")
            output = exec_result.get("output", "")
            traceback_str = exec_result.get("traceback", "")

            # Also check the captured validation error from the wrapper
            # (runtime errors like AttributeError, ImportError are caught by our wrapper)
            if not error and "_tpl_validation_error" in output:
                # Parse the validation error from stdout
                import re
                m = re.search(r"_tpl_validation_error\s*=\s*(.+)", output)
                if m:
                    val_err = m.group(1).strip().strip('"').strip("'")
                    if val_err and val_err != "None":
                        error = val_err

            # If executor itself reported an error (syntax error, etc.), use that
            if exec_result.get("error") and not error:
                error = exec_result["error"]

            # Also check traceback for useful error info
            if not error and traceback_str:
                # Extract the last line of traceback which has the error type
                tb_lines = traceback_str.strip().split("\n")
                for line in reversed(tb_lines):
                    line = line.strip()
                    if line and not line.startswith("File ") and not line.startswith("Traceback"):
                        error = line
                        break

            results[tpl_key] = {
                "live_valid": error is None,
                "error": error,
                "output": output[:500] if output else "",
                "execution_time": exec_result.get("execution_time", 0),
            }

            if on_progress:
                on_progress(idx, total, tpl_key, results[tpl_key])

        # ── Restore scene state after all templates validated ──
        try:
            if _pre_layout is not None:
                lm = slicer.app.layoutManager()
                if lm and lm.layout != _pre_layout:
                    lm.setLayout(_pre_layout)
        except Exception:
            pass
        try:
            if _pre_module is not None:
                slicer.util.selectModule(_pre_module)
        except Exception:
            pass
        try:
            if _pre_interaction_node_mode is not None:
                interactionNode = slicer.mrmlScene.GetNodeByID(
                    "vtkMRMLInteractionNodeSingleton"
                )
                if interactionNode:
                    interactionNode.SetCurrentInteractionMode(
                        _pre_interaction_node_mode
                    )
        except Exception:
            pass

        return results

    # ================================================================
    # Revision System
    # ================================================================

    _MAX_SOURCE_CONTEXT_CHARS = 400_000

    def _build_revision_source_context(
        self,
        source_path: Optional[str],
        manifest: Dict,
        generators: List[Dict],
    ) -> str:
        """Build source code context for the revision prompt.

        Extracts the logic class method sources and UI file content from
        the extension's source directory.  Method sources are extracted via
        AST so only the relevant class body is included (not the full file).

        Returns a formatted string for the revision prompt, or empty string
        if source_path is not available.
        """
        if not source_path or not os.path.isdir(source_path):
            return ""

        logic_class_name = manifest.get("logic_class_name", "")
        module_name = manifest.get("extension_module_name", "")

        # Find .py and .ui files in the extension source tree
        py_files = []
        ui_files = []
        for root, dirs, files in os.walk(source_path):
            dirs[:] = [d for d in dirs if not d.startswith((".", "__")) and d != "build"]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
                elif f.endswith(".ui"):
                    ui_files.append(os.path.join(root, f))

        # Extract logic class method sources
        parts = []
        for py_file in py_files:
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                tree = ast.parse(source)
            except Exception:
                continue

            lines = source.split("\n")
            for node in ast.iter_child_nodes(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                # Match the logic class
                if node.name != logic_class_name and not (
                    logic_class_name and node.name.endswith("Logic")
                ):
                    continue

                method_sources = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        start = item.lineno - 1
                        end = (
                            item.end_lineno
                            if hasattr(item, "end_lineno") and item.end_lineno
                            else start + 60
                        )
                        method_src = "\n".join(lines[start:end])
                        method_sources.append(
                            f"  def {item.name}(...):\n"
                            + "\n".join("    " + l for l in method_src.split("\n")[1:])
                        )

                if method_sources:
                    parts.append(
                        f"--- {os.path.basename(py_file)}: class {node.name} ---\n"
                        + "\n\n".join(method_sources)
                    )
                break

        # Include UI file (has parameter node property names, widget names)
        for ui_file in ui_files:
            try:
                with open(ui_file, "r", encoding="utf-8", errors="ignore") as f:
                    ui_content = f.read()
                # Trim UI to just property names and node references (remove layout boilerplate)
                if len(ui_content) > 5000:
                    # Extract lines with objectName, property, node references
                    import re
                    key_lines = []
                    for line in ui_content.split("\n"):
                        stripped = line.strip()
                        if any(kw in stripped for kw in (
                            "objectName", "property", "nodeReference",
                            "MRMLNode", "parameterName", "SetNodeReferenceID",
                            "<property", "<string>", "ctkMRMLNodeComboBox",
                        )):
                            key_lines.append(stripped)
                    if key_lines:
                        ui_content = (
                            "<!-- UI key properties (truncated) -->\n"
                            + "\n".join(key_lines[:200])
                        )
                    else:
                        ui_content = "<!-- UI file present but no key properties found -->"
                parts.append(f"--- {os.path.basename(ui_file)} (UI) ---\n{ui_content}")
            except Exception:
                pass

        combined = "\n\n".join(parts)

        # Truncate if too large
        if len(combined) > self._MAX_SOURCE_CONTEXT_CHARS:
            combined = combined[:self._MAX_SOURCE_CONTEXT_CHARS] + "\n# ... [truncated]"

        return combined

    def _build_revision_prompt_fragment(
        self,
        extension_name: str,
        tool_schemas: List[Dict],
        generators: List[Dict],
    ) -> str:
        """Build a deterministic prompt fragment after successful revision."""
        tool_names = []
        for schema in tool_schemas or []:
            fn = schema.get("function", {}) if isinstance(schema, dict) else {}
            if fn.get("name"):
                tool_names.append(fn["name"])
        if not tool_names:
            tool_names = [extension_name]

        step_lines = []
        for gen in generators or []:
            step = gen.get("param_signature", {}).get("workflow_step", "")
            desc = gen.get("description", "")
            step_type = gen.get("step_type", "automated")
            if step:
                step_lines.append(f"- `{step}` [{step_type}]: {desc}")

        return (
            f"### {extension_name}\n\n"
            f"Generated CLI package status: validated.\n\n"
            f"Available tool: `{tool_names[0]}`.\n\n"
            "Execute cookbook workflow steps in order. For automated steps, run "
            "the returned code. For interactive or mixed steps, run the pre-code, "
            "wait for the user to finish the requested interaction, then run the "
            "post-code.\n\n"
            "Workflow steps:\n"
            + "\n".join(step_lines)
            + "\n"
        )

    def revise(
        self,
        extension_name: str,
        errors: List[str],
        max_attempts: int = _MAX_REVISION_ATTEMPTS,
        source_path: Optional[str] = None,
        logic_analysis: Optional[Dict] = None,
        api_probe_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Revise failed templates using LLM feedback.

        Args:
            extension_name: Name of the CLI to revise.
            errors: List of error messages from validation or testing.
            source_path: Path to the extension's source directory. If provided,
                the logic class source code and UI file are included in the
                revision prompt so the LLM can verify API calls against actual
                method signatures.
            logic_analysis: Optional in-memory logic analysis from the failed
                generation run. When present, semantic validation remains active
                during revision.
            api_probe_result: Optional live API probe result from the failed
                generation run. When present, unresolved live API failures remain
                blocking during revision.

        Returns:
            Dict with 'success', 'validation_result', 'attempts' keys.
        """
        extension_name = _validate_extension_name(extension_name)
        from ..ExtensionCLILoader import get_cli_base_dir

        cli_dir = os.path.join(get_cli_base_dir(), extension_name)
        if not os.path.isdir(cli_dir):
            return {"success": False, "error": f"No CLI found for {extension_name}"}

        # Load existing CLI data
        manifest_path = os.path.join(cli_dir, "manifest.json")
        generators_path = os.path.join(cli_dir, "code_generators.json")
        tool_schemas_path = os.path.join(cli_dir, "tool_schemas.json")
        workflow_metadata_path = os.path.join(cli_dir, "workflow_metadata.json")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(generators_path, "r", encoding="utf-8") as f:
            generators = json.load(f)

        workflow_metadata = {}
        if os.path.isfile(workflow_metadata_path):
            try:
                with open(workflow_metadata_path, "r", encoding="utf-8") as f:
                    workflow_metadata = json.load(f)
            except Exception:
                workflow_metadata = {}
        self._workflow_metadata = workflow_metadata

        # Load workflow.json to provide semantic context for revision
        workflow_path = os.path.join(cli_dir, manifest.get("workflow_graph_file", "workflow.json"))
        workflow_steps = {}
        workflow_data = None
        if os.path.isfile(workflow_path):
            try:
                with open(workflow_path, "r", encoding="utf-8") as f:
                    workflow_data = json.load(f)
                for ws in workflow_data.get("steps", []):
                    workflow_steps[ws.get("step_id", "")] = ws
            except Exception:
                pass

        tool_schemas = []
        if os.path.isfile(tool_schemas_path):
            with open(tool_schemas_path, "r", encoding="utf-8") as f:
                tool_schemas = json.load(f)

        # Collect source code context (logic class methods + UI file)
        source_context = self._build_revision_source_context(
            source_path, manifest, generators
        )

        # Repair deterministic workflow contracts before template-only LLM
        # revision.  This lets revision fix stale graph/generator metadata from
        # older failed packages without extension-specific rules.
        if source_path and workflow_data and logic_analysis:
            try:
                scan_result = self._stage1_scan(source_path)
                self._placement_starter_methods = self._classify_placement_starter_methods(
                    logic_analysis
                )
                self._normalize_workflow_contracts(
                    workflow_data, self._workflow_metadata, scan_result, logic_analysis
                )
                self._synthesize_workflow_ui_guidance(
                    workflow_data, self._workflow_metadata, scan_result, logic_analysis
                )
                _, generators = self._build_workflow_manifest_and_generators(
                    extension_name, scan_result, workflow_data
                )
                with open(workflow_path, "w", encoding="utf-8") as f:
                    json.dump(workflow_data, f, indent=2)
                with open(generators_path, "w", encoding="utf-8") as f:
                    json.dump(generators, f, indent=2)
                with open(workflow_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self._workflow_metadata, f, indent=2)
                workflow_steps = {
                    ws.get("step_id", ""): ws
                    for ws in workflow_data.get("steps", [])
                }
            except Exception:
                logger.debug("Revision workflow normalization failed", exc_info=True)

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

            source_section = ""
            if source_context:
                source_section = f"\nEXTENSION SOURCE CODE (use to verify correct API calls):\n{source_context}\n"

            # Build semantic context from workflow.json so the LLM knows what
            # each template is SUPPOSED to do (not just what the stub looks like).
            semantic_section = ""
            if workflow_steps:
                semantic_lines = []
                for tpl_file, tpl_code in templates.items():
                    # Find matching workflow step(s) by template file reference
                    step_id = None
                    # Extract step ID from template filename (e.g., cb_step_5 from templates/cb_step_5.py.tpl)
                    import re as _re_for_tpl
                    m = _re_for_tpl.search(r"(cb_step_\d+)", tpl_file)
                    if m:
                        step_id = m.group(1)
                    ws = workflow_steps.get(step_id) if step_id else None
                    if ws:
                        desc = ws.get("description", "")
                        sub_ops = ws.get("sub_operations", [])
                        sub_ops_text = "\n".join(
                            f"    - [{so.get('op_type')}] {so.get('description')}"
                            for so in sub_ops
                        )
                        semantic_lines.append(
                            f"Template '{tpl_file}' (step {step_id}):\n"
                            f"  Cookbook description: {desc}\n"
                            f"  Required sub-operations:\n{sub_ops_text}"
                        )
                if semantic_lines:
                    semantic_section = (
                        "\nSEMANTIC CONTEXT (what each template should implement):\n"
                        + "\n".join(semantic_lines) + "\n"
                    )

            prompt = textwrap.dedent(f"""\
The following code templates for the "{extension_name}" extension failed validation.
Please fix ALL errors while maintaining the template format (use {{placeholder}} for dynamic values, {{{{ }}}} for literal braces).

ERRORS:
{chr(10).join(f'- {e}' for e in errors)}
{source_section}
{semantic_section}
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
                templates[tpl_name] = tpl_content

            templates = self._sanitize_templates(templates)
            for tpl_name, tpl_content in templates.items():
                if not tpl_name.endswith(".py.tpl"):
                    continue
                tpl_path = os.path.join(cli_dir, tpl_name)
                os.makedirs(os.path.dirname(tpl_path), exist_ok=True)
                with open(tpl_path, "w", encoding="utf-8") as f:
                    f.write(tpl_content)

            # Re-validate
            if not self.code_validator:
                from ..CodeValidator import CodeValidator
                self.code_validator = CodeValidator()

            self._sync_template_contracts(
                templates,
                generators,
                workflow_graph=workflow_data,
            )
            with open(generators_path, "w", encoding="utf-8") as f:
                json.dump(generators, f, indent=2)
            if workflow_data is not None:
                with open(workflow_path, "w", encoding="utf-8") as f:
                    json.dump(workflow_data, f, indent=2)
            with open(workflow_metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._workflow_metadata, f, indent=2)

            fresh_probe_result = self._stage7c_live_api_probe(templates)
            for tpl_name, tpl_content in templates.items():
                tpl_path = os.path.join(cli_dir, tpl_name)
                if tpl_name.endswith(".py.tpl") and os.path.isfile(tpl_path):
                    with open(tpl_path, "w", encoding="utf-8") as f:
                        f.write(tpl_content)

            self._sync_template_contracts(
                templates,
                generators,
                workflow_graph=workflow_data,
            )
            with open(generators_path, "w", encoding="utf-8") as f:
                json.dump(generators, f, indent=2)
            if workflow_data is not None:
                with open(workflow_path, "w", encoding="utf-8") as f:
                    json.dump(workflow_data, f, indent=2)
            with open(workflow_metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._workflow_metadata, f, indent=2)

            validation_result = self._stage9_validate(
                templates, generators,
                logic_analysis=logic_analysis,
                api_probe_result=fresh_probe_result,
                extension_name=extension_name,
            )

            if validation_result.get("valid"):
                if isinstance(self._workflow_metadata, dict):
                    resolved_syntax_issues = self._workflow_metadata.pop(
                        "stage7_syntax_issues", None
                    )
                    if resolved_syntax_issues:
                        self._workflow_metadata["resolved_stage7_syntax_issues"] = (
                            resolved_syntax_issues
                        )
                    self._workflow_metadata["revision_validation_status"] = "passed"

                # Update manifest status
                manifest["status"] = "validated"
                manifest["validation_state"] = self._workflow_metadata.get("validation_state", {})
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                if os.path.isfile(workflow_metadata_path):
                    with open(workflow_metadata_path, "w", encoding="utf-8") as f:
                        json.dump(self._workflow_metadata, f, indent=2)

                prompt_fragment = self._build_revision_prompt_fragment(
                    extension_name, tool_schemas, generators
                )
                with open(
                    os.path.join(cli_dir, "prompt_fragment.md"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(prompt_fragment)

                # Append to generation log
                log_path = os.path.join(cli_dir, "generation_log.json")
                log_entries = []
                if os.path.isfile(log_path):
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_entries = json.load(f)
                log_entries.append({
                    "attempt": len(log_entries) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "stage": "revision",
                    "trigger": "validation_failure",
                    "error": "; ".join(errors),
                    "fix": fixed.get("fix_description", ""),
                    "api_probe_result": fresh_probe_result,
                    "validation_result": validation_result,
                })
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log_entries, f, indent=2)

                from ..ExtensionCLILoader import invalidate_cache
                invalidate_cache()

                result["success"] = True
                result["validation_result"] = validation_result
                return result

            manifest["status"] = "validation_failed"
            manifest["validation_state"] = self._workflow_metadata.get("validation_state", {})
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            if os.path.isfile(workflow_metadata_path):
                with open(workflow_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self._workflow_metadata, f, indent=2)
            log_path = os.path.join(cli_dir, "generation_log.json")
            log_entries = []
            if os.path.isfile(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_entries = json.load(f)
                except Exception:
                    log_entries = []
            log_entries.append({
                "attempt": len(log_entries) + 1,
                "timestamp": datetime.now().isoformat(),
                "stage": "revision",
                "trigger": "validation_failure",
                "status": "validation_failed",
                "error": "; ".join(errors),
                "fix": fixed.get("fix_description", ""),
                "api_probe_result": fresh_probe_result,
                "validation_result": validation_result,
            })
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2)
            errors = validation_result.get("errors", [])

        manifest["status"] = "validation_failed"
        manifest["validation_state"] = self._workflow_metadata.get("validation_state", {})
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(
            os.path.join(cli_dir, "prompt_fragment.md"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                f"### {extension_name}\n\n"
                "Revision failed validation. This CLI package is saved only "
                "for debugging and is not loaded as a runtime tool.\n"
            )

        result["error"] = f"Revision failed after {max_attempts} attempts"
        return result

    # ================================================================
