from .common import *


class AnalyzerPromptValidationMixin:
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
            op_type = _operation_type_for_step(step)
            desc = step.get("description", step["step_id"])
            if op_type == "extension_op":
                marker = "[extension_op]"
            elif op_type == "slicer_op":
                marker = "[slicer_op]"
            elif op_type == "user_interaction":
                marker = "[user_interaction]"
            elif op_type == "user_choice":
                marker = "[user_choice]"
            else:
                marker = f"[unsupported: {op_type or 'missing'}]"
            # Truncate long descriptions for readability
            short_desc = desc.split("\n")[0][:150] if len(desc) > 150 else desc.split("\n")[0]
            lines.append(f"{i+1}. `{step['step_id']}` {marker} — {short_desc}")
            if op_type == "user_interaction":
                lines.append(f"   - Interaction: {step.get('interaction_type', 'unknown')}")
                if step.get("placement_instructions"):
                    lines.append(f"   - Tell user: {step['placement_instructions'][:200]}")
            elif op_type == "user_choice":
                choice = step.get("choice_info") or {}
                if choice.get("question"):
                    lines.append(f"   - Ask user: {choice['question'][:200]}")

        lines.extend([
            "",
            "**Protocol:**",
            f"1. Call `{tool_name}` with `workflow_step='{first_step_id}'` and `user_action='start'` to begin",
            "2. For **extension_op** and **slicer_op** steps: output the returned `code` verbatim in a ```python block. Then call the next step.",
            "3. For **user_interaction** steps: output the returned `pre_code` verbatim in a ```python block. Relay instructions to the user. Wait for them to click 'Done'.",
            "4. For **user_choice** steps: ask the returned question. After the user answers, call the same step with `user_action='choice_made'` and `choice_value`.",
            "5. After each step completes, call the tool with the NEXT step's `step_id` and `user_action='start'`.",
            "6. Continue until all steps are done.",
            "",
            "**CRITICAL RULES:**",
            "- Execute ONE step per turn. Do NOT call multiple steps in a single turn.",
            "- Do NOT skip extension_op or slicer_op steps. Their code MUST be output and executed.",
            "- Always start from step 1 (`" + first_step_id + "`) and proceed in order.",
        ])

        fragment = "\n".join(lines)
        self.on_progress(8, "Generating prompt fragment", "Generated workflow prompt")
        return fragment

    def _stage8_generate_prompt(
        self,
        extension_name: str,
        tool_schemas: List[Dict],
        stage_map: Dict,
        logic_analysis: Dict,
        workflow_graph: Optional[Dict] = None,
    ) -> str:
        """Generate markdown prompt fragment for system prompt injection."""
        self.on_progress(8, "Generating prompt fragment", "Building usage instructions...")

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

        self.on_progress(8, "Generating prompt fragment", "Prompt fragment generated")
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
    # Stage 9: Validation + Save
    # ================================================================

    def _stage9_validate(
        self,
        templates: Dict[str, str],
        generators: List[Dict],
        logic_analysis: Optional[Dict] = None,
        api_probe_result: Optional[Dict] = None,
        extension_name: str = "",
    ) -> Dict:
        """Validate all templates with CodeValidator + semantic checks."""
        self.on_progress(9, "Validating templates", "Running CodeValidator...")

        if not self.code_validator:
            from ..CodeValidator import CodeValidator
            self.code_validator = CodeValidator()

        # Stash extension name for the prelude dry-validation pass (used by
        # build_assembled_code_for_validation to derive the per-extension
        # logic variable name).
        self._prelude_validation_ext_name = extension_name or ""

        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "per_template": {},
        }

        template_context = self._build_template_validation_context(generators)

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
            if validation.get("destructive_ops"):
                gen = template_context.get(tpl_name, {}).get("generator", {})
                destructive_contract = self._destructive_ops_contract(
                    sample_code,
                    tpl_content,
                    gen,
                    validation.get("destructive_ops", []) or [],
                )
                if destructive_contract.get("allowed"):
                    validation["requires_confirmation"] = bool(
                        destructive_contract.get("scope") != "display_view_scope_reset"
                    )
                    gen["allow_destructive_ops"] = True
                    gen["destructive_ops_contract"] = destructive_contract
                else:
                    validation["valid"] = False
                    reason = (
                        "Template contains destructive operations without an explicit "
                        f"allow_destructive_ops contract: {validation.get('destructive_ops')}"
                    )
                    validation["reason"] = (
                        f"{validation.get('reason')}; {reason}"
                        if validation.get("reason") else reason
                    )

            if "SLICER_OP_GENERATION_FAILED" in sample_code:
                validation["valid"] = False
                reason = "Slicer-op template generation failed due to insufficient retrieval evidence"
                validation["reason"] = (
                    f"{validation.get('reason')}; {reason}"
                    if validation.get("reason") else reason
                )
            if "UNKNOWN_OP_GENERATION_FAILED" in sample_code:
                validation["valid"] = False
                reason = "Operation type could not be proven from extension or Slicer-core evidence"
                validation["reason"] = (
                    f"{validation.get('reason')}; {reason}"
                    if validation.get("reason") else reason
                )

            # Semantic validation (undefined vars, arg count)
            if logic_analysis:
                semantic = self._semantic_validate(sample_code, logic_analysis, api_probe_result=api_probe_result)
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

            contract = self._validate_template_contract(
                tpl_name, sample_code, template_context.get(tpl_name), templates,
                raw_code=tpl_content,
            )
            if isinstance(self._workflow_metadata, dict):
                gen_for_debug = template_context.get(tpl_name, {}).get("generator", {})
                self._workflow_metadata.setdefault("semantic_contracts", {})[tpl_name] = {
                    "operation_intents": self._operation_intents_for_generator(gen_for_debug),
                    "errors": list(contract.get("errors") or []),
                    "warnings": list(contract.get("warnings") or []),
                }
            final_state_contract = self._validate_final_state_contract(
                tpl_name, sample_code, template_context.get(tpl_name)
            )
            if final_state_contract.get("intent") and isinstance(self._workflow_metadata, dict):
                self._workflow_metadata.setdefault("final_state_intents", {})[tpl_name] = (
                    final_state_contract["intent"]
                )
            if isinstance(self._workflow_metadata, dict):
                chains = self._extract_api_chains(sample_code)
                if chains:
                    self._workflow_metadata.setdefault("api_evidence", {})[tpl_name] = {
                        "api_chains": chains,
                        "probe_status": (
                            "not_run" if api_probe_result is None
                            else "passed" if not api_probe_result.get("unresolved_failures") else "failed"
                        ),
                    }
            if contract.get("errors"):
                validation["valid"] = False
                existing_reason = validation.get("reason") or ""
                new_reasons = "; ".join(contract["errors"])
                validation["reason"] = (
                    f"{existing_reason}; {new_reasons}" if existing_reason
                    else new_reasons
                )
            if contract.get("warnings"):
                validation.setdefault("warnings", []).extend(contract["warnings"])
            if final_state_contract.get("errors"):
                validation["valid"] = False
                existing_reason = validation.get("reason") or ""
                new_reasons = "; ".join(final_state_contract["errors"])
                validation["reason"] = (
                    f"{existing_reason}; {new_reasons}" if existing_reason
                    else new_reasons
                )
            if final_state_contract.get("warnings"):
                validation.setdefault("warnings", []).extend(final_state_contract["warnings"])

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

        generator_contract = self._validate_generator_contracts(generators)
        if generator_contract.get("errors"):
            results["valid"] = False
            results["errors"].extend(generator_contract["errors"])
        if generator_contract.get("warnings"):
            results["warnings"].extend(generator_contract["warnings"])

        if api_probe_result:
            unresolved = api_probe_result.get("unresolved_failures")
            if unresolved is None and api_probe_result.get("failures") and not api_probe_result.get("revised"):
                unresolved = api_probe_result.get("failures", [])
            unresolved = unresolved or []
            if unresolved:
                results["valid"] = False
                for failure in unresolved:
                    template = failure.get("template", "unknown template")
                    chain = failure.get("chain", "unknown API")
                    error = failure.get("error") or (
                        "API call does not exist"
                        if not failure.get("receiver_is_none")
                        else "API receiver resolved to None"
                    )
                    results["errors"].append(
                        f"{template}: Unresolved live API probe failure for '{chain}': {error}"
                    )

        # Prelude dry-validation: assemble prelude + filled template for each
        # generator and run ast.parse + CodeValidator. This catches generator-
        # side prelude bugs (e.g., bad f-string formatting, missing imports,
        # json.dumps-vs-Python-literal issues) at generation time rather than
        # at every runtime step.
        prelude_results = self._validate_assembled_preludes(generators, templates)
        for entry in prelude_results.get("failures", []):
            results["valid"] = False
            results["errors"].append(
                f"{entry['template']}: {entry['error']}"
            )
        for entry in prelude_results.get("warnings", []):
            results["warnings"].append(
                f"{entry['template']}: {entry['warning']}"
            )

        if isinstance(self._workflow_metadata, dict):
            static_valid = all(
                item.get("valid", True)
                for item in results.get("per_template", {}).values()
            )
            probe_failures = bool(
                api_probe_result
                and (
                    api_probe_result.get("unresolved_failures")
                    or (
                        api_probe_result.get("failures")
                        and not api_probe_result.get("revised")
                    )
                )
            )
            self._workflow_metadata["validation_state"] = {
                "static_valid": static_valid,
                "api_probe_valid": None if api_probe_result is None else not probe_failures,
                "contract_valid": not bool(
                    generator_contract.get("errors")
                    or any(
                        "Required operation" in e
                        or "operation_model" in e
                        or "extension_op without" in e
                        or "Interaction is owned" in e
                        for e in results.get("errors", [])
                    )
                ),
                "prelude_valid": prelude_results.get("valid", True),
                "prelude_bytes_max": prelude_results.get("bytes_max", 0),
                "overall_valid": bool(results.get("valid")),
            }

        self.on_progress(
            9, "Validating templates",
            "PASS" if results["valid"] else f"FAIL: {results['errors']}"
        )

        return results

    # Soft size budget for assembled prelude + template. Anything above this
    # suggests the generator is re-emitting constant metadata as source text
    # (the regression Revision A is designed to prevent).
    _PRELUDE_SIZE_BUDGET_BYTES = 5120

    def _validate_assembled_preludes(
        self,
        generators: List[Dict],
        templates: Dict[str, str],
    ) -> Dict:
        """Dry-validate prelude + filled template for each generator.

        For each generator's template_file / pre_template_file / post_template_file,
        fill placeholders, prepend the assembled prelude (via the loader's
        ``build_assembled_code_for_validation`` helper), and run ``ast.parse``
        plus ``CodeValidator.validate``. This catches generator-side prelude
        bugs (bad f-string formatting, missing imports, source-text metadata
        serialization) at generation time instead of at every runtime step.

        Returns:
            Dict with ``valid`` (bool), ``failures`` (List[{template, error}]),
            ``warnings`` (List[{template, warning}]), and ``bytes_max`` (int).
        """
        result = {"valid": True, "failures": [], "warnings": [], "bytes_max": 0}
        if not generators or not templates:
            return result

        # Lazy import so the analyzer can run without the loader's runtime deps.
        try:
            from ..extension_cli_loader.choice_helpers import (
                build_assembled_code_for_validation,
            )
        except ImportError:
            # Loader unavailable; skip prelude validation. ``prelude_valid``
            # stays True so the pipeline isn't blocked when the loader can't
            # be loaded (e.g., running the analyzer outside Slicer without
            # the extension_cli_loader package importable).
            return result

        if not self.code_validator:
            try:
                from ..CodeValidator import CodeValidator
                self.code_validator = CodeValidator()
            except ImportError:
                return result

        ext_name = getattr(self, "_prelude_validation_ext_name", "") or ""
        metadata = self._workflow_metadata if isinstance(self._workflow_metadata, dict) else {}

        for gen in generators or []:
            for key in ("template_file", "pre_template_file", "post_template_file"):
                tpl_name = gen.get(key)
                if not tpl_name:
                    continue
                tpl_content = templates.get(tpl_name)
                if tpl_content is None:
                    # Templates may be keyed by basename when the analyzer
                    # stored them without the leading directory prefix.
                    base = os.path.basename(tpl_name)
                    for k, v in templates.items():
                        if os.path.basename(k) == base:
                            tpl_content = v
                            break
                if tpl_content is None:
                    continue

                sample_code = self._fill_remaining_placeholders(
                    tpl_content.replace(
                        "{vol_lookup}",
                        "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')",
                    )
                )

                try:
                    assembled = build_assembled_code_for_validation(
                        ext_name, metadata, gen, sample_code,
                    )
                except Exception as exc:
                    result["valid"] = False
                    result["failures"].append({
                        "template": tpl_name,
                        "error": f"Prelude assembly failed: {exc}",
                    })
                    continue

                code_bytes = len(assembled.encode("utf-8"))
                if code_bytes > result["bytes_max"]:
                    result["bytes_max"] = code_bytes

                # Revision E: soft size warning for assembled prelude+template.
                if code_bytes > self._PRELUDE_SIZE_BUDGET_BYTES:
                    result["warnings"].append({
                        "template": tpl_name,
                        "warning": (
                            f"Assembled prelude+template is {code_bytes}B — exceeds "
                            f"{self._PRELUDE_SIZE_BUDGET_BYTES // 1024}KB budget. The "
                            f"generator may be re-emitting constant metadata as source text."
                        ),
                    })

                try:
                    ast.parse(assembled)
                except SyntaxError as exc:
                    result["valid"] = False
                    result["failures"].append({
                        "template": tpl_name,
                        "error": f"Assembled prelude+template syntax error: {exc}",
                    })
                    continue

                try:
                    validation = self.code_validator.validate(assembled)
                except Exception as exc:
                    result["valid"] = False
                    result["failures"].append({
                        "template": tpl_name,
                        "error": f"CodeValidator raised on assembled code: {exc}",
                    })
                    continue
                if not validation.get("valid", True):
                    result["valid"] = False
                    result["failures"].append({
                        "template": tpl_name,
                        "error": (
                            "CodeValidator rejected assembled code: "
                            f"{validation.get('reason', 'unknown')}"
                        ),
                    })

        return result

    def _validate_final_state_contract(
        self,
        tpl_name: str,
        code: str,
        context: Optional[Dict],
    ) -> Dict:
        """Reject final-state slicer operations that invert current state."""
        result = {"errors": [], "warnings": [], "intent": None}
        gen = (context or {}).get("generator", {}) or {}
        if not gen:
            return result
        has_slicer_op = gen.get("op_type") == "slicer_op" or any(
            so.get("op_type") == "slicer_op"
            for so in (gen.get("sub_operations", []) or [])
        )
        if not has_slicer_op:
            return result

        text_parts = [_text_or_empty(gen.get("description", ""))]
        if gen.get("op_type") == "slicer_op":
            text_parts.append(_text_or_empty(gen.get("description", "")))
        for so in gen.get("sub_operations", []) or []:
            if so.get("op_type") != "slicer_op":
                continue
            text_parts.append(_text_or_empty(so.get("description", "")))
            text_parts.extend(_text_or_empty(k) for k in (so.get("slicer_api_keywords") or []))
        evidence = gen.get("api_evidence") or {}
        text_parts.extend(
            _text_or_empty(desc)
            for desc in (evidence.get("operation_descriptions") or [])
        )

        intent = _infer_final_state_intent(" ".join(text_parts))
        if intent.get("mode") == "unspecified":
            return result
        result["intent"] = intent
        if intent.get("mode") != "set":
            return result

        inversion_patterns = (
            r"=\s*not\s+current[A-Za-z0-9_]*\b",
            r"[,(\[]\s*not\s+current[A-Za-z0-9_]*\b",
            r"=\s*not\s+[A-Za-z0-9_]*(Visibility|Enabled|Checked|Selected)\b",
            r"[,(\[]\s*not\s+[A-Za-z0-9_]*(Visibility|Enabled|Checked|Selected)\b",
        )
        if any(_re.search(pattern, code) for pattern in inversion_patterns):
            desired = "ON/TRUE" if intent.get("state") is True else "OFF/FALSE"
            result["errors"].append(
                f"Final-state operation requests {desired}, but template inverts current state"
            )
        return result

    def _validate_generator_contracts(self, generators: List[Dict]) -> Dict:
        """Validate workflow generator metadata that is not tied to a template."""
        result = {"errors": [], "warnings": []}
        by_step = {
            (gen.get("param_signature", {}) or {}).get("workflow_step", ""): gen
            for gen in generators or []
        }
        for gen in generators or []:
            step_id = gen.get("param_signature", {}).get("workflow_step", "")
            step_type = gen.get("operation_type") or gen.get("step_type", "")
            operation_model = gen.get("operation_model") or {}
            if step_type and operation_model and operation_model.get("step_type") != step_type:
                result["errors"].append(
                    f"{step_id}: operation_model step_type does not match generator step_type"
                )
            if step_type in ("unknown_op", "mixed", "branch"):
                result["errors"].append(
                    f"{step_id}: unsupported generated operation type '{step_type}'"
                )

            if step_type == "user_choice":
                choice_desc = gen.get("choice_descriptor", {}) or {}
                parameter_name = choice_desc.get("parameter_name", "")
                binding = (
                    choice_desc.get("binding")
                    or self._workflow_metadata.get("choice_bindings", {}).get(step_id, {})
                )
                is_closed_form = self._choice_is_closed_form(choice_desc)
                is_count_like = self._choice_is_count_like(
                    choice_desc,
                    {"description": gen.get("description", "")},
                )
                if parameter_name and not binding and not is_closed_form and not is_count_like:
                    result["warnings"].append(
                        f"{step_id}: user choice '{parameter_name}' has no source-derived parameter binding"
                    )
            if step_type in ("user_interaction", "interactive", "mixed"):
                interaction_desc = gen.get("interaction_descriptor", {}) or {}
                node_class = interaction_desc.get("node_class", "")
                if node_class and self._is_markup_node_class(node_class):
                    owner = interaction_desc.get("interaction_owner", "")
                    starter = interaction_desc.get("placement_starter_method", "")
                    if owner == "extension_method" and not starter:
                        result["errors"].append(
                            f"{step_id}: interaction owned by extension method but no placement_starter_method is recorded"
                        )
            repeat_group = (
                gen.get("repeat_group")
                or (gen.get("choice_descriptor") or {}).get("repeat_group")
                or (gen.get("interaction_descriptor") or {}).get("repeat_group")
            )
            if repeat_group and repeat_group.get("interaction_step") == step_id:
                start_gen = by_step.get(repeat_group.get("start_step", ""))
                start_starter = self._generator_placement_starter(start_gen)
                interaction_starter = self._generator_placement_starter(gen)
                interaction_owner = (
                    (gen.get("interaction_descriptor") or {}).get("interaction_owner", "")
                )
                if (
                    interaction_owner != "previous_extension_method"
                    and start_starter
                    and interaction_starter
                    and start_starter == interaction_starter
                ):
                    result["errors"].append(
                        f"{step_id}: repeat interaction and start step both call placement starter '{interaction_starter}'"
                    )
                instruction_text = " ".join([
                    _text_or_empty(interaction_desc.get("placement_instructions", "")),
                    _text_or_empty(gen.get("placement_instructions", "")),
                ]).lower()
                if _re.search(r"\b(repeat|for each|each requested|continue placing)\b", instruction_text):
                    result["errors"].append(
                        f"{step_id}: repeat interaction instructions must describe one item per Done"
                    )
        return result

    @staticmethod
    def _generator_placement_starter(gen: Optional[Dict]) -> str:
        """Return the extension placement starter method recorded on a generator."""
        if not gen:
            return ""
        desc = gen.get("interaction_descriptor") or {}
        if desc.get("placement_starter_method"):
            return desc.get("placement_starter_method", "")
        for so in gen.get("sub_operations", []) or []:
            if so.get("op_type") == "extension_op" and so.get("extension_method_hint"):
                return so.get("extension_method_hint", "")
        return ""

    @staticmethod
    def _build_template_validation_context(generators: List[Dict]) -> Dict[str, Dict]:
        """Map each template file to its workflow generator and role."""
        context = {}
        for gen in generators or []:
            for role, key in (
                ("template", "template_file"),
                ("pre", "pre_template_file"),
                ("post", "post_template_file"),
            ):
                tpl_name = gen.get(key)
                if tpl_name:
                    context[tpl_name] = {"generator": gen, "role": role}
        return context
