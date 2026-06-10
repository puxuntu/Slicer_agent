from .common import *


class AnalyzerWorkflowTemplatesMixin:
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

        def _matching_slicer_template_items(step_id: str) -> List[Tuple[str, str]]:
            matches = []
            prefix = f"{step_id}_"
            for tpl_key, tpl_code in self._slicer_op_templates.items():
                if tpl_key == step_id or tpl_key.startswith(prefix):
                    matches.append((tpl_key, tpl_code))
            matches.sort(key=lambda item: item[0])
            return matches

        def _matching_slicer_templates(step_id: str) -> List[str]:
            return [code for _, code in _matching_slicer_template_items(step_id)]

        def _attach_slicer_evidence(step: Dict, template_file: str, source_keys: List[str]):
            evidence_items = [
                self._slicer_op_evidence.get(key, {})
                for key in source_keys
                if self._slicer_op_evidence.get(key)
            ]
            if not evidence_items:
                return
            merged = self._merge_api_evidence(evidence_items)
            step["api_evidence"] = merged
            if isinstance(self._workflow_metadata, dict):
                self._workflow_metadata.setdefault("api_evidence", {})[template_file] = merged

        for step_index, step in enumerate(steps):
            step_id = step["step_id"]
            op_type = _operation_type_for_step(step)
            step["operation_type"] = op_type
            step["op_type"] = op_type
            step.setdefault("step_type", op_type)
            step_type = _legacy_step_type_for_operation(op_type)

            if self._parameter_update_ops_for_step(step):
                key = f"templates/{step_id}.py.tpl"
                step["code_template"] = key
                templates[key] = self._generate_parameter_update_template(
                    extension_name, step, logic_class_name, module_name,
                )

            elif step_type == "automated" and op_type == "slicer_op":
                # Slicer_op templates are pre-generated in Stage 5T
                key = f"templates/{step_id}_slicer.py.tpl"
                step["code_template"] = key
                pregen_items = _matching_slicer_template_items(step_id)
                if pregen_items:
                    templates[key] = "\n\n".join(code for _, code in pregen_items)
                    _attach_slicer_evidence(step, key, [item_key for item_key, _ in pregen_items])
                else:
                    # Fallback: generate via LLM with generic slicer prompt
                    templates[key] = self._generate_slicer_fallback_template(step)

            elif step_type == "automated" and op_type == "mixed":
                # Automated step with mixed sub-operations: compose ALL sub-ops
                # (extension_op + slicer_op) into a single template.
                sub_ops = step.get("sub_operations", [])
                auto_parts = []
                # Consume Stage 5T templates once for the whole step.
                consumed_5t = False
                for so in sub_ops:
                    if so.get("operation_intent") == "extension_parameter_update":
                        ext_step = dict(step)
                        ext_step["sub_operations"] = [so]
                        auto_parts.append(
                            f"# Extension parameter update: {so.get('description', '')}\n"
                            + self._generate_parameter_update_template(
                                extension_name, ext_step, logic_class_name, module_name,
                            )
                        )
                    elif so["op_type"] == "extension_op" and so.get("extension_method_hint"):
                        ext_step = dict(step)
                        ext_step["method_name"] = so["extension_method_hint"]
                        ext_step["description"] = so["description"]
                        if so["extension_method_hint"] in self._placement_starter_methods:
                            ext_tpl = self._generate_placement_starter_pre_template(
                                extension_name, ext_step, logic_class_name, module_name,
                            )
                        else:
                            ext_tpl = self._generate_automated_workflow_template(
                                extension_name, ext_step, logic_class_name, module_name,
                                logic_analysis,
                            )
                        auto_parts.append(f"# Extension op: {so['description']}\n{ext_tpl}")
                    elif so["op_type"] == "slicer_op":
                        if not consumed_5t:
                            pregen_items = _matching_slicer_template_items(step_id)
                            if pregen_items:
                                auto_parts.append(
                                    "# Slicer ops (Stage 5T)\n"
                                    + "\n\n".join(code for _, code in pregen_items)
                                )
                                _attach_slicer_evidence(
                                    step,
                                    f"templates/{step_id}.py.tpl",
                                    [item_key for item_key, _ in pregen_items],
                                )
                            else:
                                so_keywords = so.get("slicer_api_keywords", [])
                                so_desc = so.get("description", "")
                                llm_code = self._generate_slicer_api_template_llm(
                                    step_id, so_desc, so_keywords,
                                )
                                if llm_code:
                                    auto_parts.append(f"# Slicer op: {so_desc}\n{llm_code}")
                                else:
                                    auto_parts.append(f"# Slicer op: {so_desc}\n# TODO: generate slicer code\npass")
                            consumed_5t = True
                    elif so["op_type"] in ("extension_op", "unknown_op") and not so.get("extension_method_hint"):
                        if so.get("extension_function_hint"):
                            ext_step = dict(step)
                            ext_step["extension_function_name"] = so["extension_function_hint"]
                            ext_step["description"] = so.get("description", step.get("description", ""))
                            auto_parts.append(
                                f"# Extension function op: {so['description']}\n"
                                + self._generate_extension_function_template(
                                    extension_name, ext_step, module_name,
                                )
                            )
                            continue
                        # No method hint — generate targeted code for this sub-op only.
                        so_desc = so.get("description", "")
                        so_keywords = so.get("slicer_api_keywords", [])
                        category = so.get("slicer_op_category")
                        if category or so_keywords:
                            llm_code = self._generate_slicer_api_template_llm(
                                step_id, so_desc, so_keywords,
                            )
                            if llm_code:
                                auto_parts.append(f"# Auto op: {so_desc}\n{llm_code}")
                            else:
                                auto_parts.append(
                                    f"# Auto op: {so_desc}\n"
                                    f"# TODO: implement this operation\npass"
                                )
                key = f"templates/{step_id}.py.tpl"
                templates[key] = "\n\n".join(auto_parts) if auto_parts else "# No sub-operations\npass"
                step["code_template"] = key

            elif step_type == "automated":
                # Single code template for automated steps
                if step.get("extension_function_name"):
                    tpl = self._generate_extension_function_template(
                        extension_name, step, module_name,
                    )
                elif step.get("method_name") in self._placement_starter_methods:
                    step["interaction_owner"] = "extension_method"
                    step["placement_starter_method"] = step.get("method_name")
                    step["created_node_source"] = "extension_method"
                    tpl = self._generate_placement_starter_pre_template(
                        extension_name, step, logic_class_name, module_name,
                    )
                else:
                    tpl = self._generate_automated_workflow_template(
                        extension_name, step, logic_class_name, module_name, logic_analysis,
                    )
                templates[f"templates/{step_id}.py.tpl"] = tpl
                step["code_template"] = f"templates/{step_id}.py.tpl"

            elif step_type == "interactive":
                node_class = step.get("node_class", "")
                # Revision A: dispatch on interaction_kind (semantic intent)
                # rather than node_class (data shape). Previously the guard
                # `bool(node_class) and not _is_markup_node_class(node_class)`
                # required a non-empty node_class to reach the view-adjustment
                # branch, so view_adjustment steps with empty node_class fell
                # into _generate_pre_interaction_template which always creates
                # a Markups node and enters placement mode — hijacking the
                # mouse for any drag-existing-handle or viewport-adjustment
                # interaction.
                interaction_kind = (step.get("interaction_kind") or "").strip()
                # Infer interaction_kind conservatively from node_class when
                # Stage 4 didn't set it (preserves legacy behavior for steps
                # classified before Revision B's descriptor enrichment).
                if not interaction_kind:
                    if self._is_markup_node_class(node_class):
                        interaction_kind = "markup_placement"
                    elif node_class:
                        interaction_kind = "view_adjustment"
                    step["interaction_kind"] = interaction_kind

                starter_binding = self._find_recent_placement_starter_for_interaction(
                    steps, step_index
                )
                prev_starter_method = (
                    step.get("placement_starter_method")
                    or starter_binding.get("method")
                )
                if prev_starter_method and interaction_kind == "markup_placement":
                    step["interaction_owner"] = "previous_extension_method"
                    step["placement_starter_method"] = prev_starter_method
                    step["created_node_source"] = "previous_extension_method"
                    if starter_binding:
                        step["placement_starter_step_id"] = starter_binding.get("step_id", "")
                        step["placement_binding_reason"] = starter_binding.get("reason", "")

                is_view_adjustment = interaction_kind == "view_adjustment"

                # Pre-interaction template
                if interaction_kind == "markup_placement" and prev_starter_method:
                    pre_tpl = self._generate_existing_placement_pre_template(
                        extension_name, step, prev_starter_method,
                    )
                elif is_view_adjustment:
                    pre_tpl = self._generate_view_adjustment_pre_template(
                        extension_name, step,
                    )
                else:
                    # markup_placement without prev_starter_method, or
                    # unknown/missing interaction_kind falling back to the
                    # legacy markups-creation default.
                    if interaction_kind == "markup_placement" or not interaction_kind:
                        if self._is_markup_node_class(node_class) or not node_class:
                            step.setdefault("interaction_owner", "runtime_template")
                            step.setdefault("created_node_source", "template")
                    pre_tpl = self._generate_pre_interaction_template(
                        extension_name, step, logic_class_name, module_name,
                    )
                templates[f"templates/{step_id}_pre.py.tpl"] = pre_tpl
                step["pre_template"] = f"templates/{step_id}_pre.py.tpl"

                # Post-interaction template — mirror the same dispatch.
                if is_view_adjustment:
                    post_tpl = self._generate_view_adjustment_post_template(
                        extension_name, step,
                    )
                else:
                    post_tpl = self._generate_post_interaction_template(
                        extension_name, step, logic_class_name, module_name, logic_analysis,
                    )
                templates[f"templates/{step_id}_post.py.tpl"] = post_tpl
                step["post_template"] = f"templates/{step_id}_post.py.tpl"

            elif step_type == "mixed":
                # Mixed step: pre_template contains all automated sub-ops,
                # then user interaction follows.
                sub_ops = step.get("sub_operations", [])
                auto_parts = []
                placement_starter_method = None
                slicer_templates_appended = False
                pre_key = f"templates/{step_id}_pre.py.tpl"
                for so in sub_ops:
                    if so.get("operation_intent") == "extension_parameter_update":
                        ext_step = dict(step)
                        ext_step["sub_operations"] = [so]
                        auto_parts.append(
                            f"# Extension parameter update: {so.get('description', '')}\n"
                            + self._generate_parameter_update_template(
                                extension_name, ext_step, logic_class_name, module_name,
                            )
                        )
                    elif so["op_type"] == "extension_op" and so.get("extension_method_hint"):
                        # Generate extension_op code
                        ext_step = dict(step)
                        ext_step["method_name"] = so["extension_method_hint"]
                        ext_step["description"] = so["description"]
                        if so["extension_method_hint"] in self._placement_starter_methods:
                            placement_starter_method = so["extension_method_hint"]
                            step["interaction_owner"] = "extension_method"
                            step["placement_starter_method"] = placement_starter_method
                            step["created_node_source"] = "extension_method"
                            ext_tpl = self._generate_placement_starter_pre_template(
                                extension_name, ext_step, logic_class_name, module_name,
                            )
                        else:
                            ext_tpl = self._generate_automated_workflow_template(
                                extension_name, ext_step, logic_class_name, module_name,
                                logic_analysis,
                            )
                        auto_parts.append(f"# Extension op: {so['description']}\n{ext_tpl}")
                    elif so["op_type"] == "extension_op" and so.get("extension_function_hint"):
                        ext_step = dict(step)
                        ext_step["extension_function_name"] = so["extension_function_hint"]
                        ext_step["description"] = so.get("description", step.get("description", ""))
                        auto_parts.append(
                            f"# Extension function op: {so['description']}\n"
                            + self._generate_extension_function_template(
                                extension_name, ext_step, module_name,
                            )
                        )
                    elif so["op_type"] == "slicer_op":
                        # Use pre-generated slicer_op template
                        if slicer_templates_appended:
                            continue
                        pregen_parts = _matching_slicer_templates(step_id)
                        if pregen_parts:
                            auto_parts.append(
                                f"# Slicer op: {so['description']}\n"
                                + "\n\n".join(pregen_parts)
                            )
                            _attach_slicer_evidence(
                                step,
                                pre_key,
                                [item_key for item_key, _ in _matching_slicer_template_items(step_id)],
                            )
                            slicer_templates_appended = True
                        else:
                            # No pre-generated template — try LLM with keywords
                            so_keywords = so.get("slicer_api_keywords", [])
                            so_desc = so.get("description", "")
                            llm_code = self._generate_slicer_api_template_llm(
                                step_id, so_desc, so_keywords,
                            )
                            if llm_code:
                                auto_parts.append(f"# Slicer op: {so_desc}\n{llm_code}")
                            else:
                                auto_parts.append(f"# Slicer op: {so_desc}\n# TODO: generate slicer code\npass")

                step["pre_template"] = pre_key

                # Append node creation + ID storage for the interaction sub-op
                # so the post-template can retrieve the node.
                interaction_sub_ops = [
                    so for so in sub_ops if so.get("op_type") == "user_interaction"
                ]
                if interaction_sub_ops:
                    iso = interaction_sub_ops[0]
                    node_class = iso.get("node_class") or step.get("node_class", "")
                    # Revision A (mixed-step mirror): dispatch on interaction_kind
                    # rather than requiring a non-empty node_class. Same fix as
                    # the interactive branch above.
                    interaction_kind = (iso.get("interaction_kind") or step.get("interaction_kind") or "").strip()
                    if not interaction_kind:
                        if self._is_markup_node_class(node_class):
                            interaction_kind = "markup_placement"
                        elif node_class:
                            interaction_kind = "view_adjustment"
                        step["interaction_kind"] = interaction_kind

                    is_view_adjustment = interaction_kind == "view_adjustment"
                    is_markup_placement = interaction_kind == "markup_placement"

                    if (
                        is_markup_placement
                        and node_class
                        and self._is_markup_node_class(node_class)
                        and not placement_starter_method
                    ):
                        step["interaction_owner"] = "runtime_template"
                        step["created_node_source"] = "template"
                        node_name = step_id.replace("_", " ").title()
                        node_var = f"_{extension_name.lower()}_{step_id}_id"
                        instructions = self._interaction_instructions_for_template(step)
                        policy = self._placement_mode_policy(step)
                        block_lines = [
                            "",
                            "",
                            "# --- Setup interaction node ---",
                            "import slicer",
                            "from SlicerAIAgentLib.workflow_state import remember_interaction_node",
                            f"node = slicer.mrmlScene.AddNewNodeByClass(\"{node_class}\", \"{node_name}\")",
                            "displayNode = node.GetDisplayNode()",
                            "if displayNode is not None:",
                            "    displayNode.SetVisibility(True)",
                        ]
                        if policy.get("should_set_active_list"):
                            block_lines.append("slicer.modules.markups.logic().SetActiveListID(node)")
                        if policy.get("should_enter_placement_mode"):
                            block_lines.extend([
                                "interactionNode = slicer.mrmlScene.GetNodeByID(\"vtkMRMLInteractionNodeSingleton\")",
                                "if interactionNode is not None:",
                                *self._placement_mode_code(policy),
                            ])
                        block_lines.extend([
                            f"{node_var} = node.GetID()",
                            (
                                "remember_interaction_node("
                                f"_workflow_runtime_extension, _workflow_runtime_id, \"{step_id}\", "
                                f"{node_var}, _workflow_runtime_repeat_index)"
                            ),
                            f"print(\"[{extension_name}] Please {instructions}\")",
                        ])
                        interaction_block = "\n".join(block_lines)
                        auto_parts.append(interaction_block)
                    elif is_view_adjustment:
                        step["interaction_kind"] = "view_adjustment"
                        instructions = iso.get(
                            "placement_instructions",
                            step.get("description", ""),
                        )
                        auto_parts.append(
                            "\n\n# --- View adjustment interaction ---\n"
                            f"print(\"[{extension_name}] Please {instructions}\")\n"
                            "print(\"When finished, press the 'Done' button in the workflow panel.\")\n"
                        )

                templates[pre_key] = (
                    "\n\n".join(auto_parts) if auto_parts
                    else "# No automated sub-operations\npass"
                )

                # Post-interaction template for the user_interaction part
                post_key = f"templates/{step_id}_post.py.tpl"
                step["post_template"] = post_key
                if placement_starter_method:
                    templates[post_key] = self._generate_placement_starter_post_template(
                        extension_name, step, placement_starter_method,
                    )
                elif interaction_sub_ops and not self._is_markup_node_class(
                    interaction_sub_ops[0].get("node_class") or step.get("node_class", "")
                ):
                    templates[post_key] = self._generate_view_adjustment_post_template(
                        extension_name, step,
                    )
                else:
                    templates[post_key] = self._generate_post_interaction_template(
                        extension_name, step, logic_class_name, module_name, logic_analysis,
                    )

            elif step_type == "branch":
                # Branch steps don't need templates — handled by the orchestrator
                pass

            elif step_type == "user_choice":
                # user_choice steps don't need code templates — handled by the
                # orchestrator which presents the question and collects the answer.
                pass

        # Post-generation consistency check: verify pre/post templates agree
        # on node ID variables (_ext_step_id).
        consistency_fixes = 0
        for step in steps:
            exec_type = _legacy_step_type_for_operation(_operation_type_for_step(step))
            if exec_type != "interactive":
                continue
            s_id = step["step_id"]
            node_var = f"_{extension_name.lower()}_{s_id}_id"

            # Check post-template references the var
            post_key = step.get("post_template", "")
            if not post_key or post_key not in templates:
                continue
            if node_var not in templates[post_key]:
                continue

            # Verify pre-template defines it
            pre_key = step.get("pre_template", "")
            if not pre_key or pre_key not in templates:
                continue
            if node_var in templates[pre_key]:
                continue

            # Missing — inject node ID storage at end of pre-template.
            # Only inject when node_class is set (a markup node was created).
            # For steps with empty node_class (e.g. slice crosshair adjustment),
            # no node exists to store — skip injection.
            node_class = step.get("node_class", "")
            if not node_class:
                continue
            injection = (
                f"\n# [Auto-injected] Store node ID for post-step\n"
                f"try:\n"
                f"    {node_var} = node.GetID()\n"
                f"except NameError:\n"
                f"    {node_var} = ''\n"
            )
            templates[pre_key] = templates[pre_key].rstrip() + "\n" + injection
            consistency_fixes += 1
            logger.info(
                "[Stage 7] Injected missing node ID '%s' into %s",
                node_var, pre_key,
            )

        if consistency_fixes:
            logger.info(
                "[Stage 7] Fixed %d pre/post node ID consistency issues",
                consistency_fixes,
            )

        # Store workflow graph as JSON template (only valid steps)
        valid_types = CANONICAL_OPERATION_TYPES
        clean_graph = {k: v for k, v in workflow_graph.items() if k != "steps"}
        clean_graph["steps"] = [
            s for s in steps
            if _operation_type_for_step(s) in valid_types
        ]
        templates["workflow.json"] = json.dumps(clean_graph, indent=2)
        templates["workflow_metadata.json"] = json.dumps(self._workflow_metadata or {}, indent=2)

        self.on_progress(
            7, "Generating code templates",
            f"Generated {len(templates)} workflow templates"
        )
        return templates

    @staticmethod
    def _parameter_update_ops_for_step(step: Dict) -> List[Dict]:
        operations = list(step.get("sub_operations") or [])
        operations.extend(step.get("atomic_operations") or [])
        return [
            so for so in operations
            if so.get("operation_intent") in (
                "extension_parameter_update",
                "extension_node_reference_update",
            )
            and so.get("parameter_name")
        ]

    @staticmethod
    def _parameter_placeholder_name(role: str) -> str:
        text = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", role or "")
        text = _re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
        return text or "parameter_value"

    @staticmethod
    def _placeholder_default_literal(value: Any, value_property: str) -> str:
        if value is None:
            return "0.0" if value_property == "value" else "''"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        return repr(str(value))

    @staticmethod
    def _node_class_for_reference_role(role: str, binding: Dict[str, Any]) -> str:
        text = " ".join([
            role or "",
            (binding or {}).get("widget_name", ""),
            (binding or {}).get("ui_text", ""),
        ]).lower()
        if "segmentation" in text:
            return "vtkMRMLSegmentationNode"
        if "volume" in text:
            return "vtkMRMLScalarVolumeNode"
        if "curve" in text:
            return "vtkMRMLMarkupsCurveNode"
        if "line" in text:
            return "vtkMRMLMarkupsLineNode"
        if "plane" in text:
            return "vtkMRMLMarkupsPlaneNode"
        if "model" in text:
            return "vtkMRMLModelNode"
        return "vtkMRMLNode"

    @staticmethod
    def _node_reference_keywords(role: str, step: Dict) -> List[str]:
        text = " ".join([
            role or "",
            _text_or_empty(step.get("description", "")),
            " ".join(
                _text_or_empty(so.get("description", ""))
                for so in (step.get("sub_operations") or [])
            ),
        ]).lower()
        tokens = []
        for token in _re.findall(r"[a-z0-9]+", text):
            if len(token) < 4:
                continue
            if token in {
                "select", "choose", "current", "scalar", "volume",
                "segmentation", "node", "option", "section",
            }:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:5]

    def _generate_parameter_update_template(
        self,
        extension_name: str,
        step: Dict,
        logic_class_name: str,
        module_name: str,
    ) -> str:
        """Generate deterministic code for extension parameter-node UI updates."""
        step_id = step.get("step_id", "")
        ops = self._parameter_update_ops_for_step(step)
        if not ops:
            return self._generate_unknown_op_template(step)

        logic_var = f"_{extension_name.lower()}_logic"
        lines = [
            *self._template_header_lines(extension_name, step, ""),
            "import slicer",
            f"from {module_name} import {logic_class_name}",
            "",
            *self._emit_module_enter_precondition(module_name),
            "try:",
            f"    logic = {logic_var}",
            "except NameError:",
            f"    logic = {logic_class_name}()",
            f"    {logic_var} = logic",
            "",
            "parameterNode = logic.getParameterNode()",
        ]

        for so in ops:
            role = so.get("parameter_name", "")
            mode = so.get("target_value_mode", "")
            target = so.get("target_value")
            binding = so.get("ui_parameter_binding") or {}
            role_info = binding.get("role") or {}
            value_property = so.get("value_property") or role_info.get("value_property", "")
            if so.get("operation_intent") == "extension_node_reference_update":
                placeholder = self._parameter_placeholder_name(role)
                node_class = self._node_class_for_reference_role(role, binding)
                keywords = self._node_reference_keywords(role, step)
                lines.extend([
                    f"{placeholder}_node_name = {{{placeholder}_node_name: ''}}",
                    f"{placeholder}_node = None",
                    f"if {placeholder}_node_name:",
                    "    try:",
                    f"        {placeholder}_node = slicer.util.getNode({placeholder}_node_name)",
                    "    except Exception:",
                    "        pass",
                    f"if {placeholder}_node is None:",
                    f"    _nodes = slicer.mrmlScene.GetNodesByClass({node_class!r})",
                    f"    _keywords = {keywords!r}",
                    "    for _i in range(_nodes.GetNumberOfItems()):",
                    "        _candidate = _nodes.GetItemAsObject(_i)",
                    "        _name = (_candidate.GetName() or '').lower()",
                    "        if not _keywords or any(_kw in _name for _kw in _keywords):",
                    f"            {placeholder}_node = _candidate",
                    "            break",
                    f"if {placeholder}_node is None:",
                    f"    raise RuntimeError('Could not find node for parameter reference {role}')",
                    f"parameterNode.SetNodeReferenceID({role!r}, {placeholder}_node.GetID())",
                ])
            elif target is True:
                lines.append(f"parameterNode.SetParameter({role!r}, 'True')")
            elif target is False:
                lines.append(f"parameterNode.SetParameter({role!r}, 'False')")
            elif mode == "invert":
                lines.extend([
                    f"_current_{role} = parameterNode.GetParameter({role!r}) == 'True'",
                    f"parameterNode.SetParameter({role!r}, 'False' if _current_{role} else 'True')",
                ])
            elif value_property in ("value", "currentText", "currentIndex"):
                placeholder = self._parameter_placeholder_name(role)
                default = (binding.get("properties") or {}).get(value_property)
                default_literal = self._placeholder_default_literal(default, value_property)
                lines.append(f"{placeholder} = {{{placeholder}: {default_literal}}}")
                lines.append(f"parameterNode.SetParameter({role!r}, str({placeholder}))")
            else:
                default = (
                    role_info.get("true_value")
                )
                if default is None:
                    default = "True"
                if isinstance(default, bool):
                    default_text = "True" if default else "False"
                else:
                    default_text = str(default)
                lines.append(
                    f"# Final state was not explicit; apply source-derived/default truthy state for {role}"
                )
                lines.append(f"parameterNode.SetParameter({role!r}, {default_text!r})")

        lines.extend([
            "try:",
            "    parameterNode.Modified()",
            "except Exception:",
            "    pass",
            f"{logic_var} = logic",
            f"print(\"[{extension_name}] Step '{step_id}' completed.\")",
        ])
        return "\n".join(lines) + "\n"

    def _generate_extension_function_template(
        self, extension_name: str, step: Dict, module_name: str,
    ) -> str:
        """Generate deterministic code for an extension-owned module function."""
        function_name = step.get("extension_function_name", "")
        step_id = step.get("step_id", "")
        if not function_name:
            return self._generate_unknown_op_template(step)
        lines = [
            *self._template_header_lines(extension_name, step, ""),
            "import slicer",
            f"from {module_name} import {function_name}",
            "",
            *self._emit_module_enter_precondition(module_name),
            f"{function_name}()",
            "",
            f"print(\"[{extension_name}] Step '{step_id}' completed.\")",
        ]
        return "\n".join(lines) + "\n"

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
        if method_name:
            method_call_lines = [
                "# Execute the automated step",
                f"if hasattr(logic, '{method_name}'):",
                f"    result = logic.{method_name}()",
                "else:",
                "    result = None",
            ]
        else:
            # No extension method — try to generate code from sub-operations via LLM
            sub_ops = step.get("sub_operations", [])
            sub_op_parts = []
            for so in sub_ops:
                so_desc = so.get("description", "")
                so_keywords = so.get("slicer_api_keywords", [])
                so_method = so.get("extension_method_hint")
                if so.get("op_type") == "slicer_op" and so_keywords:
                    code = self._generate_slicer_api_template_llm(
                        step_id, so_desc, so_keywords,
                    )
                    if code:
                        sub_op_parts.append(f"# {so_desc}\n{code}")
                elif so.get("op_type") == "extension_op" and so_method:
                    sub_op_parts.append(
                        f"# {so_desc}\n"
                        f"if hasattr(logic, '{so_method}'):\n"
                        f"    logic.{so_method}()"
                    )
            if sub_op_parts:
                method_call_lines = ["\n\n".join(sub_op_parts)]
            else:
                method_call_lines = [
                    "# No specific method mapped to this step",
                    "# TODO: Determine the correct extension method to call",
                    "pass",
                ]

        lines = self._template_header_lines(extension_name, step, "")
        # Only include extension import boilerplate if we actually call an extension method
        if method_name or any(
            so.get("op_type") == "extension_op" and so.get("extension_method_hint")
            for so in step.get("sub_operations", [])
        ):
            lines.extend([
                "try:",
                f"    from {module_name} import {logic_class_name}",
                "except ImportError:",
                f"    raise RuntimeError(\"{extension_name} extension is not installed.\")",
                "",
                *self._emit_module_enter_precondition(module_name),
                "try:",
                f"    logic = _{extension_name.lower()}_logic",
                "except NameError:",
                f"    logic = {logic_class_name}()",
                "",
            ])
        lines.extend(method_call_lines)
        lines.append("")
        # Only store logic instance if we actually created one
        if method_name or any(
            so.get("op_type") == "extension_op" and so.get("extension_method_hint")
            for so in step.get("sub_operations", [])
        ):
            lines.append(f"_{extension_name.lower()}_logic = logic")
        lines.append(f"print(\"[{extension_name}] Step '{step_id}' completed.\")")

        return "\n".join(lines) + "\n"

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

        # Truncate source if needed (no limit — full method source for template generation)
        # if len(method_source) > 5000:
        #     method_source = method_source[:5000] + "\n# ... [truncated]"

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

        parameter_context = ""
        if isinstance(self._workflow_metadata, dict):
            defaults = self._workflow_metadata.get("parameter_defaults", {}) or {}
            deps = self._workflow_metadata.get("parameter_method_dependencies", {}).get(method_name, {})
            roles = deps.get("parameter_roles", []) or []
            if roles:
                rows = []
                for role in roles:
                    default = defaults.get(role)
                    if default:
                        rows.append(
                            f"  - {role}: default {default.get('value')!r} "
                            f"({default.get('source')}, {default.get('confidence')} confidence)"
                        )
                    else:
                        rows.append(f"  - {role}: no inferred default")
                parameter_context = "Parameter-node dependencies:\n" + "\n".join(rows)

        prompt = textwrap.dedent(f"""\
            Generate a Python code snippet for a 3D Slicer extension workflow step.

            Extension: {extension_name}
            Logic class: `{logic_class_name}` (import from `{module_name}`)
            Step: {step.get('step_id', '')}
            Method to call: `{method_name}()`
            {ui_context}

            {params_desc}
            {parameter_context}

            Method source code:
            ```python
            {method_source}
            ```

            The code must:
            1. Import the logic class from `{module_name}`
            2. Ensure the extension module is active so module.enter() has run (extension methods may depend on parameter-node init, observers, and UI bindings set up by the widget's enter() method). Emit (include the marker comments exactly):
               ```
               # precondition:begin
               _active_module = slicer.app.moduleManager().activeModule()
               if _active_module is None or _active_module.name != "{module_name}":
                   try:
                       slicer.util.selectModule("{module_name}")
                   except Exception as _module_enter_error:
                       print(f"Warning: could not activate module '{module_name}': {{_module_enter_error}}")
               # precondition:end
               ```
            3. Reuse the existing logic instance `_{extension_name.lower()}_logic` if it exists, otherwise create a new `{logic_class_name}()`
            4. Set up any required state on the logic instance BEFORE calling the method (e.g., if the method reads `self.mandibleSegmentationNode`, find the node in the scene and assign it)
               If the method or its helper methods read scalar values from `parameterNode.GetParameter(...)`, initialize missing values using the provided source-derived defaults with `parameterNode.SetParameter(...)` before calling the method. Never overwrite a non-empty parameter value.
            5. To find scene nodes, use robust fuzzy matching — NEVER rely on exact node names. Use this pattern:
               ```python
               nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
               for i in range(nodes.GetNumberOfItems()):
                   n = nodes.GetItemAsObject(i)
                   if "fibula" in n.GetName().lower():
                       fibulaNode = n
                       break
               ```
               Or for parameter node references: first check `parameterNode.GetNodeReference("refName")`, and if None, search the scene by class + name substring, then set via `parameterNode.SetNodeReferenceID("refName", node.GetID())`.
            6. Call the method with correct arguments
            7. Store the logic instance as `_{extension_name.lower()}_logic` for subsequent steps
            8. Print a completion message

            IMPORTANT restrictions:
            - Do NOT use `dir()`, `eval()`, `exec()`, `globals()`, or `locals()` — these are blocked in the execution sandbox.
            - Use `try/except NameError` to check if a variable exists, NOT `if 'var' in dir()`.
            - Do NOT use curly brace template placeholders. Write actual Python values (strings, numbers, etc.). If you need a node name, hardcode a reasonable default like `slicer.util.getNode('FibulaModel')`.
            - Escape all braces in f-strings and .format() calls by doubling them: use doubled-braces for literal braces in output strings.
            - Return ONLY raw Python code. Do NOT wrap it in markdown fences (```python ... ```).""")

        try:
            for _attempt in range(2):
                response = self._call_llm(prompt)
                response = self._strip_markdown_fences(response) if response else None
                if not response:
                    break
                # Validate syntax immediately — retry once on failure
                import ast as _ast
                try:
                    _ast.parse(response)
                    return response
                except (SyntaxError, IndentationError) as e:
                    if _attempt == 0:
                        logger.info(
                            "LLM automated template for step %s had syntax error: %s. Retrying...",
                            step.get("step_id", "?"), e,
                        )
                        # Add error context for retry
                        prompt += (
                            f"\n\nYour previous output had a syntax error: {e}\n"
                            "Output ONLY the corrected Python code, no explanation."
                        )
                    else:
                        logger.warning(
                            "LLM automated template for step %s still has syntax error after retry: %s",
                            step.get("step_id", "?"), e,
                        )
                        return response  # Return as-is, stage 9 / revision will catch it
        except Exception:
            logger.debug("LLM automated template generation failed", exc_info=True)
        return None

    def _generate_slicer_api_template_llm(
        self, step_id: str, description: str, slicer_api_keywords: List[str],
    ) -> Optional[str]:
        """Generate pure Slicer API code via LLM for steps with no extension method.

        Called when a step has slicer_op sub-operations but no method_name on the
        extension Logic class.  Uses the LLM with the step description and API
        keyword hints to produce Slicer core API code (no extension imports).
        """
        keywords_str = ", ".join(slicer_api_keywords) if slicer_api_keywords else "none"
        prompt = textwrap.dedent(f"""\
            Generate a Python code snippet for a 3D Slicer operation.

            Step ID: {step_id}
            Description: {description}
            API keyword hints: [{keywords_str}]

            The code must:
            1. Use Slicer's built-in Python API only (slicer.mrmlScene,
               slicer.app.layoutManager(), slicer.modules, etc.)
            2. Be a complete, self-contained snippet that performs the described operation
            3. Use robust patterns (check for None, handle missing nodes gracefully)
            4. Print a short completion message

            IMPORTANT restrictions:
            - Do NOT use `dir()`, `eval()`, `exec()`, `globals()`, or `locals()`
            - Do NOT import the extension module — use only Slicer core APIs
            - Do NOT use curly brace template placeholders — write actual Python values
            - Escape all braces in f-strings by doubling them
            - Return ONLY raw Python code, no markdown fences""")

        try:
            for _attempt in range(2):
                response = self._call_llm(prompt)
                response = self._strip_markdown_fences(response) if response else None
                if not response or "slicer" not in response.lower():
                    break
                import ast as _ast
                try:
                    _ast.parse(response)
                    return response
                except (SyntaxError, IndentationError) as e:
                    if _attempt == 0:
                        prompt += (
                            f"\n\nPrevious output had syntax error: {e}\n"
                            "Output ONLY the corrected Python code, no explanation."
                        )
                    else:
                        return response
        except Exception:
            logger.debug("Slicer API LLM template generation failed", exc_info=True)
        return None

    def _generate_placement_starter_pre_template(
        self, extension_name, step, logic_class_name, module_name,
    ) -> str:
        """Generate deterministic setup code for extension-driven placement.

        The extension method itself creates the markup node and enters
        placement mode, so this template only calls that method.
        """
        method_name = step.get("method_name", "")
        step_id = step.get("step_id", "")
        repeat_group = step.get("repeat_group") or {}
        interaction_step_id = repeat_group.get("interaction_step") or step_id
        starter_info = self._placement_starter_info(method_name)
        node_classes = starter_info.get("node_classes") or []
        node_class = node_classes[0] if node_classes else step.get("node_class", "vtkMRMLMarkupsFiducialNode")

        lines = [
            *self._template_header_lines(extension_name, step, "Setup"),
            "import slicer",
            "from SlicerAIAgentLib.workflow_state import remember_interaction_node",
            f"from {module_name} import {logic_class_name}",
            "",
            *self._emit_module_enter_precondition(module_name),
            "try:",
            f"    logic = _{extension_name.lower()}_logic",
            "except NameError:",
            f"    logic = {logic_class_name}()",
            "",
            f"_workflow_before_ids = set()",
            f"_workflow_nodes = slicer.mrmlScene.GetNodesByClass(\"{node_class}\")",
            "_workflow_before_count = _workflow_nodes.GetNumberOfItems()",
            "for _workflow_i in range(_workflow_before_count):",
            "    _workflow_n = _workflow_nodes.GetItemAsObject(_workflow_i)",
            "    if _workflow_n is not None:",
            "        _workflow_before_ids.add(_workflow_n.GetID())",
            "",
            f"logic.{method_name}()",
            "",
            "_workflow_created_node = None",
            f"_workflow_nodes = slicer.mrmlScene.GetNodesByClass(\"{node_class}\")",
            "for _workflow_i in range(_workflow_nodes.GetNumberOfItems() - 1, -1, -1):",
            "    _workflow_n = _workflow_nodes.GetItemAsObject(_workflow_i)",
            "    if _workflow_n is not None and _workflow_n.GetID() not in _workflow_before_ids:",
            "        _workflow_created_node = _workflow_n",
            "        break",
            "if _workflow_created_node is not None:",
            f"    remember_interaction_node(_workflow_runtime_extension, _workflow_runtime_id, \"{interaction_step_id}\", _workflow_created_node.GetID(), _workflow_runtime_repeat_index)",
            f"_{extension_name.lower()}_logic = logic",
            "",
            f"print(\"[{extension_name}] Placement started for step '{step_id}'.\")",
        ]
        return "\n".join(lines) + "\n"

    def _generate_placement_starter_post_template(
        self, extension_name, step, method_name,
    ) -> str:
        """Generate post-interaction code for extension-driven placement.

        Placement-starter extension methods usually attach their own observers
        and process control points as the user places them.  The post step only
        exits placement mode and reports completion.
        """
        step_id = step.get("step_id", "")
        lines = [
            *self._template_header_lines(extension_name, step, "Done"),
            "import slicer",
            "",
            "interactionNode = slicer.mrmlScene.GetNodeByID(\"vtkMRMLInteractionNodeSingleton\")",
            "if interactionNode is not None:",
            "    interactionNode.SwitchToViewTransformMode()",
            "",
            f"print(\"[{extension_name}] Step '{step_id}' interaction completed after {method_name}().\")",
        ]
        return "\n".join(lines) + "\n"
