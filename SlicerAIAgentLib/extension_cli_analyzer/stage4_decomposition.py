from .common import *


class AnalyzerStage4DecompositionMixin:
    def _collect_step_evidence(
        self, step_description: str, method_names: List[str], extension_name: str = ""
    ) -> Dict[str, Any]:
        """Collect deterministic classification evidence for one cookbook step."""
        desc = _text_or_empty(step_description)
        desc_lower = desc.lower()
        evidence = {
            "logic_method_candidates": [],
            "widget_candidates": [],
            "ui_parameter_candidates": [],
            "ui_control_candidates": [],
            "slicer_core_candidates": [],
            "interaction_candidates": [],
            "choice_candidates": [],
        }

        matched_method = self._match_description_to_method(desc_lower, method_names)
        if matched_method:
            evidence["logic_method_candidates"].append({
                "method": matched_method,
                "reason": "method name/purpose token overlap",
            })

        for conn in getattr(self, "_widget_connections", []) or []:
            btn_name = _text_or_empty(conn.get("button_widget_name"))
            btn_text = btn_name.lower().replace("_", " ").replace(".", " ")
            btn_words = [w for w in btn_text.split() if len(w) > 3]
            if not btn_words:
                continue
            hits = [w for w in btn_words if w in desc_lower]
            if len(hits) >= 2 or (len(btn_words) == 1 and hits):
                evidence["widget_candidates"].append({
                    "button_widget_name": btn_name,
                    "logic_methods": conn.get("logic_methods", []),
                    "matched_words": hits,
                })

        evidence["ui_parameter_candidates"] = self._match_ui_parameter_bindings(desc)

        choice_patterns = {
            "left/right": ("left" in desc_lower and "right" in desc_lower),
            "which side": "which side" in desc_lower,
            "which type": "which type" in desc_lower,
            "select scene node": any(
                p in desc_lower for p in (
                    "select the", "choose the", "current scalar volume",
                    "select segmentation", "select volume", "select model",
                )
            ),
            "number requested": any(p in desc_lower for p in ("how many", "number of")),
        }
        for name, present in choice_patterns.items():
            if present:
                evidence["choice_candidates"].append(name)

        # Revision B: previously this was a 4-entry table baking in
        # `node_class = "vtkMRMLCrosshairNode"` for view_adjustment, which
        # misled the LLM and downstream dispatch for every drag-handle or
        # viewport-adjustment interaction that wasn't literally a crosshair.
        # The table is now split into two general candidate lists keyed on
        # semantic intent only — `node_class` is left empty for
        # view_adjustment and the LLM (Stage 4) decides whether to fill it.
        placement_words = (
            "draw", "drawn", "place", "placing", "place a", "place the",
            "click to add", "control point", "control points",
            "position", "positioning", "create", "creating",
        )
        view_adjustment_words = (
            "crosshair", "slice intersection", "drag", "dragging",
            "rotate", "rotating", "translate", "translating",
            "adjust", "adjusting", "handles", "handle",
            "axis", "axes", "viewport", "view adjust",
            "switch to", "turn on", "turn off", "enable", "disable",
            "set interaction", "interaction options",
        )
        action_words = (
            "manual", "manually", "draw", "drag", "click", "place",
            "position", "adjust", "rotate", "translate", "switch",
        )
        if any(w in desc_lower for w in placement_words) and any(
            v in desc_lower for v in action_words
        ):
            evidence["interaction_candidates"].append({
                "interaction_kind": "markup_placement",
                "matched_terms": [w for w in placement_words if w in desc_lower],
            })
        if any(w in desc_lower for w in view_adjustment_words) and any(
            v in desc_lower for v in action_words
        ):
            # Don't double-suggest if the same description matched placement.
            existing_kinds = {
                c.get("interaction_kind") for c in evidence.get("interaction_candidates", [])
            }
            if "view_adjustment" not in existing_kinds:
                evidence["interaction_candidates"].append({
                    "interaction_kind": "view_adjustment",
                    "matched_terms": [w for w in view_adjustment_words if w in desc_lower],
                })

        slicer_concepts = [
            ("layout_slice_view", ("layout", "conventional", "slice visibility", "red view", "fov", "spacing")),
            ("markups_display", ("display panel", "advanced panel", "view 1", "markups")),
            ("module_switching", ("switch to", "open module", "open the markups module", "select module", "activate module")),
            ("crosshair", ("crosshair", "slice intersection", "enable interaction")),
            ("subject_hierarchy", ("subject hierarchy", "folder")),
            ("node_display", ("display node", "visibility", "view node")),
        ]
        for category, terms in slicer_concepts:
            matched = [t for t in terms if t in desc_lower]
            if matched:
                evidence["slicer_core_candidates"].append({
                    "category": category,
                    "matched_terms": matched,
                })

        extension_tokens = []
        if extension_name:
            extension_tokens.append(extension_name.lower())
            parts = _re.sub(r'([a-z])([A-Z])', r'\1 \2', extension_name).split()
            acronym = "".join(p[0].lower() for p in parts if p)
            if len(acronym) >= 2:
                extension_tokens.append(acronym)
        for token in extension_tokens:
            if token and token in desc_lower:
                evidence["ui_control_candidates"].append({
                    "control": token,
                    "reason": "extension-specific name/acronym appears in step",
                })

        return evidence

    def _match_ui_parameter_bindings(self, text: str) -> List[Dict[str, Any]]:
        """Match cookbook text to source-derived UI parameter bindings."""
        desc = _text_or_empty(text)
        desc_lower = desc.lower()
        if not desc_lower:
            return []
        text_tokens = set(self._role_keywords(desc))
        # Generic icon aliases. These describe common UI semantics, not an
        # extension-specific mapping.
        if "eye" in desc_lower:
            text_tokens.update({"show", "visible", "visibility"})
        if "axes" in desc_lower or "axis" in desc_lower:
            text_tokens.update({"axes", "axis", "interaction", "handles", "handle"})
        if "checkbox" in desc_lower:
            text_tokens.update({"checked", "check", "checkbox"})
        if "tool button" in desc_lower or "toolbutton" in desc_lower:
            text_tokens.update({"tool", "button"})

        candidates = []
        bindings = getattr(self, "_ui_parameter_bindings", {}) or {}
        for widget_name, binding in bindings.items():
            roles = binding.get("roles") or []
            if not roles:
                continue
            binding_tokens = set(binding.get("keywords") or [])
            binding_tokens.update(self._role_keywords(widget_name))
            binding_tokens.update(self._role_keywords(binding.get("ui_text", "")))
            overlap = text_tokens & binding_tokens
            if not overlap:
                continue
            role_scores = []
            for role in roles:
                role_name = role.get("parameter_name", "")
                role_tokens = set(self._role_keywords(role_name))
                role_overlap = text_tokens & role_tokens
                score = len(overlap) + (2 * len(role_overlap))
                # Visible label exact text is strong evidence when present.
                ui_text = _text_or_empty(binding.get("ui_text", "")).lower()
                if ui_text and ui_text in desc_lower:
                    score += 4
                # Widget suffixes such as CheckBox/ToolButton should not
                # dominate, but they help distinguish controls.
                widget_tokens = set(self._role_keywords(widget_name))
                score += min(len(text_tokens & widget_tokens), 3)
                if "eye" in desc_lower and role_tokens & {"handle", "handles", "interaction", "intera"}:
                    score -= 4
                if ("axes" in desc_lower or "axis" in desc_lower) and not (
                    role_tokens & {"handle", "handles", "interaction", "intera", "axes", "axis"}
                ):
                    score -= 4
                role_scores.append((score, role, sorted(role_overlap)))
            role_scores.sort(key=lambda item: item[0], reverse=True)
            if not role_scores:
                continue
            score, role, role_overlap = role_scores[0]
            if score < 3:
                continue
            candidates.append({
                "widget_name": widget_name,
                "widget_class": binding.get("widget_class", ""),
                "ui_text": binding.get("ui_text", ""),
                "properties": binding.get("properties", {}),
                "role": role,
                "score": score,
                "matched_tokens": sorted(overlap),
                "matched_role_tokens": role_overlap,
            })
        candidates.sort(key=lambda item: item.get("score", 0), reverse=True)
        return candidates[:5]

    @staticmethod
    def _evidence_has(evidence: Dict[str, Any], key: str) -> bool:
        value = evidence.get(key, [])
        return isinstance(value, list) and bool(value)

    @staticmethod
    def _has_explicit_ui_parameter_state(final_state: Dict[str, Any]) -> bool:
        """Return true when cookbook text gives a state safe to automate."""
        mode = (final_state or {}).get("mode")
        return mode in ("set", "invert")

    @staticmethod
    def _has_ui_parameter_value_input_intent(text: str) -> bool:
        """Return true when cookbook text asks the user to enter/set a value."""
        normalized = _re.sub(r"[^a-z0-9]+", " ", _text_or_empty(text).lower()).strip()
        padded = f" {normalized} "
        value_patterns = (
            " enter ",
            " input ",
            " type ",
            " fill ",
            " set value ",
            " set the value ",
            " desired value ",
            " value in ",
            " value for ",
        )
        return any(pattern in padded for pattern in value_patterns)

    @staticmethod
    def _has_ui_node_reference_selection_intent(text: str) -> bool:
        """Return true when cookbook text asks to choose/select a MRML node."""
        normalized = _re.sub(r"[^a-z0-9]+", " ", _text_or_empty(text).lower()).strip()
        padded = f" {normalized} "
        selection_patterns = (
            " select ",
            " choose ",
            " current scalar volume ",
            " current volume ",
            " node ",
            " segmentation ",
            " volume ",
            " model ",
        )
        return any(pattern in padded for pattern in selection_patterns)

    @staticmethod
    def _ui_binding_operation_intent(candidate: Dict[str, Any]) -> str:
        role = (candidate or {}).get("role", {}) or {}
        if (
            role.get("access") == "node_reference_write"
            or role.get("value_property") == "currentNodeID"
        ):
            return "extension_node_reference_update"
        return "extension_parameter_update"

    @staticmethod
    def _clear_ui_parameter_fields(so: Dict[str, Any]) -> None:
        for key in (
            "operation_intent",
            "parameter_name",
            "value_property",
            "target_value",
            "target_value_mode",
            "ui_parameter_binding",
        ):
            so.pop(key, None)

    def _should_apply_ui_parameter_candidate(
        self,
        candidate: Dict[str, Any],
        final_state: Dict[str, Any],
        text: str,
    ) -> bool:
        """Decide whether UI-parameter evidence is strong enough to automate."""
        operation_intent = self._ui_binding_operation_intent(candidate)
        if operation_intent == "extension_node_reference_update":
            return self._has_ui_node_reference_selection_intent(text)
        if self._has_explicit_ui_parameter_state(final_state):
            return True
        role = (candidate or {}).get("role", {}) or {}
        value_property = role.get("value_property", "")
        return (
            value_property in ("value", "currentText", "currentIndex")
            and self._has_ui_parameter_value_input_intent(text)
        )

    def _infer_interaction_kind_from_evidence(self, evidence: Dict[str, Any], node_class: str = "") -> str:
        for item in evidence.get("interaction_candidates", []) or []:
            kind = item.get("interaction_kind")
            if kind:
                return kind
        if self._is_markup_node_class(node_class):
            return "markup_placement"
        if node_class:
            return "view_adjustment"
        return "none"

    def _infer_slicer_op_category(self, text: str, evidence: Optional[Dict[str, Any]] = None) -> Optional[str]:
        desc = _text_or_empty(text).lower()
        if evidence:
            cats = evidence.get("slicer_core_candidates", []) or []
            if self._is_ui_location_context_text(desc) and not self._is_explicit_module_switch_text(desc):
                for cat in cats:
                    if isinstance(cat, dict) and cat.get("category") == "markups_display":
                        return "markups_display"
            if cats and isinstance(cats[0], dict) and cats[0].get("category"):
                return cats[0]["category"]
        if any(t in desc for t in ("layout", "red view", "slice visibility", "fov", "spacing")):
            return "layout_slice_view"
        if "crosshair" in desc or "slice intersection" in desc:
            return "crosshair"
        if self._is_explicit_module_switch_text(desc):
            return "module_switching"
        if "display panel" in desc or "advanced panel" in desc or (
            "markups module" in desc and any(t in desc for t in ("display", "view", "advanced", "configure", "set", "show"))
        ):
            return "markups_display"
        if "subject hierarchy" in desc or "folder" in desc:
            return "subject_hierarchy"
        if "display" in desc or "visibility" in desc:
            return "node_display"
        return None

    @staticmethod
    def _is_explicit_module_switch_text(text: str) -> bool:
        """Return true only when switching modules is the requested action."""
        desc = _text_or_empty(text).lower()
        if not desc or "module" not in desc:
            return False
        explicit_patterns = (
            r"\b(switch|change|go|navigate|open|select|activate|enter)\s+(?:to\s+|the\s+)?[a-z0-9 _-]*\bmodule\b",
            r"\bmodule\s+selector\b",
            r"\bselectmodule\b",
        )
        return any(_re.search(pattern, desc) for pattern in explicit_patterns)

    @staticmethod
    def _is_ui_location_context_text(text: str) -> bool:
        """Return true when a module/panel phrase describes UI location context."""
        desc = _text_or_empty(text).lower()
        if not desc:
            return False
        has_location = bool(_re.search(r"\b(in|under|inside|from)\s+(?:the\s+)?[a-z0-9 _'-]*\b(module|panel|tab|section)\b", desc))
        has_setting_action = any(
            term in desc
            for term in (
                "configure", "set ", "show ", "display", "view", "advanced",
                "turn on", "turn off", "enable", "disable", "select "
            )
        )
        return has_location and has_setting_action

    def _stage4_cookbook_decomposition(
        self, cookbook_def, logic_analysis: Dict, scan_result: Optional[Dict] = None
    ) -> Dict:
        """Use one validated LLM call to interpret all cookbook step semantics."""
        self.on_progress(
            "contract", "Build Workflow Contract",
            "LLM interpreting cookbook steps and repeat intent..."
        )
        context = self._stage4_semantic_context(cookbook_def, logic_analysis, scan_result)
        # Closed-loop re-entry: structured downstream failure feedback (from
        # verify_repair) is injected so the re-derived contract addresses the
        # diagnosed dataflow/contract gaps.
        if getattr(self, "_upstream_feedback", None):
            context["upstream_feedback"] = self._upstream_feedback

        def _validate(candidate, raw):
            if not isinstance(candidate, dict):
                return None, ["expected a JSON object"]
            normalized, normalization_notes = self._normalize_stage4_semantic_result(
                candidate, context
            )
            if normalization_notes:
                logger.info(
                    "[Stage 4] Applied deterministic contract normalization: %s",
                    "; ".join(normalization_notes[:20]),
                )
            return normalized, self._validate_stage4_semantic_result(normalized, context)

        result = self._call_llm_structured(
            prompt=lambda prior, errors: self._stage4_semantic_prompt(context, prior, errors),
            validator=_validate,
            call_class="contract",
            failure_label="Stage 4 semantic decomposition",
        )
        stage_map = self._stage_map_from_semantic_result(
            result, cookbook_def, logic_analysis
        )
        self.on_progress(
            "contract", "Build Workflow Contract",
            f"LLM interpreted {stage_map['stage_count']} cookbook steps"
        )
        return stage_map

    def _stage4_semantic_context(
        self, cookbook_def, logic_analysis: Dict, scan_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build source-derived candidate allowlists for semantic interpretation."""
        methods = []
        node_classes = {
            "vtkMRMLMarkupsCurveNode",
            "vtkMRMLMarkupsPlaneNode",
            "vtkMRMLMarkupsLineNode",
            "vtkMRMLMarkupsFiducialNode",
            "vtkMRMLMarkupsROINode",
            "vtkMRMLCrosshairNode",
        }
        for method in logic_analysis.get("methods", []) or []:
            if not isinstance(method, dict) or not method.get("name"):
                continue
            for parameter in method.get("parameters", []) or []:
                if isinstance(parameter, dict):
                    parameter_type = _text_or_empty(
                        parameter.get("type") or parameter.get("node_class")
                    )
                    if parameter_type.startswith("vtkMRML"):
                        node_classes.add(parameter_type)
            methods.append({
                "name": method["name"],
                "parameters": method.get("parameters", []),
                "return_value": method.get("return_value"),
                "state_reads": method.get("state_reads", []),
                "state_writes": method.get("state_writes", []),
                "side_effects": method.get("side_effects", []),
            })

        widgets = []
        seen = set()
        for connection in getattr(self, "_widget_connections", []) or []:
            if not isinstance(connection, dict):
                continue
            name = _text_or_empty(connection.get("button_widget_name"))
            if name and name not in seen:
                widgets.append({
                    "widget_name": name,
                    "logic_methods": connection.get("logic_methods", []),
                })
                seen.add(name)
        ui_widgets = (scan_result or {}).get("ui_widgets", {}) or {}
        for name in ui_widgets:
            if name and name not in seen:
                widgets.append({
                    "widget_name": name,
                    "logic_methods": [],
                })
                seen.add(name)
        extension_functions = []
        if scan_result and hasattr(self, "_build_extension_callable_inventory"):
            inventory = self._build_extension_callable_inventory(
                scan_result, logic_analysis
            ).get("module_functions", {})
            extension_functions = [
                {
                    "name": name,
                    "param_count": info.get("param_count", 0),
                    "effects": info.get("effects", []),
                }
                for name, info in sorted(inventory.items())
            ]

        ui_bindings = []
        for widget_name, binding in (getattr(self, "_ui_parameter_bindings", {}) or {}).items():
            for role in binding.get("roles", []) or []:
                if not isinstance(role, dict) or not role.get("parameter_name"):
                    continue
                node_class = _text_or_empty(role.get("node_class"))
                if node_class:
                    node_classes.add(node_class)
                ui_bindings.append({
                    "widget_name": widget_name,
                    "parameter_name": role["parameter_name"],
                    "value_property": role.get("value_property", ""),
                    "access": role.get("access", ""),
                    "node_class": node_class,
                    "ui_text": binding.get("ui_text", ""),
                })

        parameter_roles = []
        logic_source = _text_or_empty(logic_analysis.get("_logic_source"))
        if logic_source and hasattr(self, "_extract_parameter_roles_from_source"):
            parameter_roles = list(
                self._extract_parameter_roles_from_source(logic_source).values()
            )
            for role in parameter_roles:
                node_class = _text_or_empty(role.get("node_class"))
                if node_class:
                    node_classes.add(node_class)

        steps = [{
            "step_number": step.step_number,
            "operation_type": step.operation_type,
            "description": step.description,
            "depends_on": step.depends_on,
        } for step in cookbook_def.steps]
        return {
            "extension_name": cookbook_def.extension_name,
            "steps": steps,
            "logic_methods": methods,
            "extension_functions": extension_functions,
            "widgets": widgets,
            "ui_parameter_bindings": ui_bindings,
            "parameter_roles": parameter_roles,
            "allowed_slicer_op_categories": [
                "layout_slice_view", "markups_display", "module_switching",
                "crosshair", "subject_hierarchy", "node_display",
                "generic_slicer_api",
            ],
            "allowed_interaction_kinds": [
                "none", "markup_placement", "view_adjustment",
            ],
            "allowed_operation_intents": [
                "extension_parameter_update", "extension_node_reference_update",
                "layout_activate", "layout_register",
                "slice_intersection_visibility", "slice_intersection_interaction",
                "view_display_scope", "module_switch",
            ],
            "allowed_node_role_kinds": [
                "choice_input", "interaction_output",
                "extension_input", "extension_output",
            ],
            "allowed_node_classes": sorted(node_classes),
        }

    @staticmethod
    def _stage4_semantic_prompt(
        context: Dict[str, Any],
        prior_result: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
    ) -> str:
        repair = ""
        if errors:
            repair = (
                "\nYour previous result was rejected. Correct every validation error.\n"
                f"Validation errors:\n{json.dumps(errors, indent=2)}\n"
                f"Previous result:\n{json.dumps(prior_result, indent=2)}\n"
            )
        upstream = context.get("upstream_feedback") or []
        if upstream:
            repair += (
                "\nA PREVIOUS GENERATION FAILED DOWNSTREAM VALIDATION. The failures "
                "below were diagnosed as contract/dataflow root causes (not template "
                "bugs). Re-derive the affected steps so that every required input "
                "has a producing step or user interaction, and every referenced "
                "parameter is bound to a source-backed candidate:\n"
                + json.dumps(upstream, indent=2) + "\n"
            )
        return textwrap.dedent(f"""\
        Interpret a user-authored 3D Slicer extension cookbook as a generic workflow.
        The cookbook operation_type annotations are authoritative and MUST NOT change.
        Identify semantic bindings and repeat intent from meaning, not fixed phrases.
        Select methods, extension functions, widgets, UI bindings, categories,
        interaction kinds, and node classes only from the supplied candidate lists.
        Use null when no candidate is semantically justified. Do not invent references.

        Return strict JSON only:
        {{
          "steps": [{{
            "step_number": 1,
            "operation_type": "extension_op|slicer_op|user_interaction|user_choice",
            "semantic_intent": "concise meaning",
            "extension_method_hint": null,
            "extension_function_hint": null,
            "widget_name": null,
            "ui_parameter_binding": null,
            "target_value": null,
            "target_value_mode": null,
            "slicer_op_category": null,
            "slicer_api_keywords": [],
            "interaction_kind": "none|markup_placement|view_adjustment",
            "interaction_type": null,
            "node_class": null,
            "creates_node": false,
            "requires_place_mode": false,
            "setup_dependencies": [],
            "placement_instructions": null,
            "choice": null,
            "is_optional": false,
            "operation_intents": [],
            "node_roles": [],
            "confidence": "high|medium|low",
            "evidence_ids": []
          }}],
          "repeat_blocks": [{{
            "repeat_id": "stable_generic_id",
            "body_steps": [2, 3],
            "controller": {{
              "kind": "count|until_choice|while_choice",
              "source_step": 1,
              "prompt": "",
              "exit_value": null
            }},
            "evidence_step_ids": [1, 3],
            "confidence": "high|medium"
          }}]
        }}

        For choice, use null or:
        {{"question":"", "choices":[{{"label":"","value":null}}],
          "parameter_name":"", "default_value":null, "value_kind":""}}
        ui_parameter_binding must be null or exactly:
        {{"widget_name":"candidate widget", "parameter_name":"candidate parameter"}}
        node_roles must contain only explicit records shaped as
        {{"role_kind":"candidate kind","node_class":"candidate class or empty",
          "parameter_name":"candidate parameter or empty"}}; use [] if unknown.
        node_roles describe MRML-node flow only. Do not create a node role for a
        scalar, boolean, count, text, or other non-node user choice. A choice's
        parameter_name is independent from node_roles and may be a stable runtime
        name when no source-derived UI or method parameter exists.
        A non-empty node-role parameter_name must come from ui_parameter_bindings,
        a logic method parameter, or parameter_roles in the candidate context.
        A repeat body must be a contiguous range of existing steps. A count repeat must
        reference a preceding user_choice source step. until_choice/while_choice must
        include a clear prompt and boolean exit_value. Repeat blocks may not overlap.
        Every step requires medium or high confidence. Include exactly one output step
        for every input step, even when most optional semantic fields are null.

        For user_interaction steps, fill in these descriptor fields precisely — they
        drive template dispatch and KB retrieval downstream:
        - creates_node: true if the user's interaction creates a NEW MRML node
          (e.g. drawing a new curve/plane/fiducial). false if the user is
          interacting with EXISTING geometry or viewport state (e.g. dragging
          slice intersection handles, rotating an already-placed plane, toggling
          visibility). When in doubt, false.
        - requires_place_mode: true only when the user must enter Slicer's
          persistent place mode (SwitchToPersistentPlaceMode) to add new control
          points to a fresh Markups node. false for view adjustments, drag-handle
          interactions, and any case where creates_node is false.
        - slicer_api_keywords: 3-8 short API hints that a KB search would use to
          find relevant Slicer code. For markups placement: include the node class
          and verbs (e.g. ["vtkMRMLMarkupsCurveNode", "place", "SetActiveListID"]).
          For view adjustments: include the relevant node/widget and verbs
          (e.g. ["vtkMRMLCrosshairNode", "slice intersection", "SetInteractiveMode"],
          or ["vtkMRMLMarkupsPlaneNode", "handles", "Translate", "Rotate"]).
        - setup_dependencies: list of prior step_numbers whose outputs this step's
          setup depends on (e.g. if this step drags handles on a plane created in
          step 5, list [5]). Empty list when there is no such dependency.

        Example for a view_adjustment step (drag existing slice intersection handles):
        "interaction_kind": "view_adjustment", "node_class": null,
        "creates_node": false, "requires_place_mode": false,
        "slicer_api_keywords": ["slice intersection", "Translate", "Rotate"],
        "setup_dependencies": []

        Example for a markup_placement step (draw a new curve):
        "interaction_kind": "markup_placement", "node_class": "vtkMRMLMarkupsCurveNode",
        "creates_node": true, "requires_place_mode": true,
        "slicer_api_keywords": ["vtkMRMLMarkupsCurveNode", "place", "draw"],
        "setup_dependencies": []
        {repair}
        Candidate context:
        {json.dumps(context, indent=2)}
        """)

    @staticmethod
    def _stage4_allowed_parameter_names(context: Dict[str, Any]) -> set:
        """Return every source-backed parameter role exposed to Stage 4."""
        allowed = {
            item.get("parameter_name")
            for item in context.get("ui_parameter_bindings", [])
            if isinstance(item, dict) and item.get("parameter_name")
        }
        allowed.update(
            role.get("role")
            for role in context.get("parameter_roles", [])
            if isinstance(role, dict) and role.get("role")
        )
        for method in context.get("logic_methods", []):
            if not isinstance(method, dict):
                continue
            for parameter in method.get("parameters", []) or []:
                if isinstance(parameter, dict) and parameter.get("name"):
                    allowed.add(parameter["name"])
        return allowed

    @staticmethod
    def _normalize_stage4_semantic_result(
        result: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Repair deterministic Stage 4 contract invariants before validation.

        The LLM remains responsible for semantic interpretation. This pass only
        restores cookbook-authoritative fields and removes references that the
        source-derived allowlists prove cannot be valid.
        """
        normalized = json.loads(json.dumps(result))
        notes = []
        expected = {
            step["step_number"]: step
            for step in context.get("steps", [])
            if isinstance(step, dict) and _int_or_none(step.get("step_number")) is not None
        }
        allowed_parameters = AnalyzerStage4DecompositionMixin._stage4_allowed_parameter_names(
            context
        )
        # The source-derived set of valid (widget, parameter) bindings. Used to
        # detect a UI binding the LLM invented for a widget the binding
        # inference could not ground (e.g. an action checkbox wired to a handler
        # method, or a parameterNodeWrapper field), so the step can degrade
        # instead of dead-ending the gate in _validate_stage4_semantic_result.
        binding_pairs = {
            (b.get("widget_name"), b.get("parameter_name"))
            for b in context.get("ui_parameter_bindings", []) or []
            if isinstance(b, dict)
        }

        for item in normalized.get("steps", []) or []:
            if not isinstance(item, dict):
                continue
            number = _int_or_none(item.get("step_number"))
            expected_step = expected.get(number)
            if expected_step is None:
                continue

            authoritative_type = expected_step.get("operation_type")
            if item.get("operation_type") != authoritative_type:
                item["operation_type"] = authoritative_type

            # target_value_mode names HOW a value is applied; the contract allows
            # only a small set of modes. The LLM sometimes emits a value TYPE
            # instead (e.g. "boolean", "exact") — especially for a checkbox step.
            # Coerce an out-of-set mode deterministically rather than dead-ending
            # the gate: "set" when a concrete value is supplied, else clear it.
            if item.get("target_value_mode") not in (
                None, "set", "invert", "node_reference", "runtime_value"
            ):
                item["target_value_mode"] = (
                    "set" if item.get("target_value") is not None else None
                )
                notes.append(f"step {number}: coerced invalid target_value_mode")

            # Parameter-update intents only carry contract meaning when the
            # step provides a value to bind (target_value / choice) or a UI
            # binding. With the complete method universe in context the LLM
            # truthfully tags plain button-click steps whose methods update
            # parameter-node state INTERNALLY; demanding a UI binding for
            # those dead-ends the contract. Drop the inferred intents instead
            # — the method call itself performs the internal updates. Steps
            # that DO carry a value still reach the validator's binding rule.
            intents = item.get("operation_intents")
            if isinstance(intents, list) and any(
                intent in ("extension_parameter_update", "extension_node_reference_update")
                for intent in intents
            ):
                # Strip a UI binding the LLM invented for a widget the source
                # could not ground (its (widget, parameter) pair is not in the
                # source-derived bindings). The validator would hard-reject it;
                # dropping it lets the step degrade to a plain step.
                binding = item.get("ui_parameter_binding")
                if isinstance(binding, dict) and (
                    binding.get("widget_name"), binding.get("parameter_name")
                ) not in binding_pairs:
                    item["ui_parameter_binding"] = None
                    notes.append(
                        f"step {number}: stripped ungrounded UI parameter binding "
                        f"{binding.get('parameter_name')!r}"
                    )
                has_binding = isinstance(item.get("ui_parameter_binding"), dict)
                # A parameter/node-reference update can only be honored when it
                # binds to a source-derived UI parameter. Without one — e.g. an
                # action checkbox wired to a handler method, or a
                # parameterNodeWrapper field the binding inference cannot model —
                # drop the intent so the step degrades gracefully instead of
                # dead-ending the gate in _validate_stage4_semantic_result. Any
                # internal parameter writes are still performed by the step's own
                # method call. (Previously kept when a target_value was present,
                # which is what let such steps reach the gate and hard-fail.)
                if not has_binding:
                    item["operation_intents"] = [
                        intent for intent in intents
                        if intent not in (
                            "extension_parameter_update",
                            "extension_node_reference_update",
                        )
                    ]
                    notes.append(
                        f"step {number}: dropped parameter-update intent(s) with "
                        "no source-derived UI binding"
                    )
                notes.append(f"step {number} restored authoritative operation_type")

            if authoritative_type == "user_choice":
                choice = item.get("choice")
                if not isinstance(choice, dict):
                    choice = {}
                    item["choice"] = choice
                if not _text_or_empty(choice.get("question")):
                    choice["question"] = _text_or_empty(expected_step.get("description"))
                    notes.append(f"step {number} restored choice question")
                if not _text_or_empty(choice.get("parameter_name")):
                    binding = item.get("ui_parameter_binding")
                    binding_parameter = (
                        binding.get("parameter_name")
                        if isinstance(binding, dict) else ""
                    )
                    choice["parameter_name"] = (
                        binding_parameter or f"choice_step_{number}"
                    )
                    notes.append(f"step {number} restored choice parameter_name")
                if not isinstance(choice.get("choices"), list):
                    choice["choices"] = []
                    notes.append(f"step {number} restored choice choices list")
                elif choice.get("value_kind") == "node" and choice.get("choices"):
                    # Node picks resolve from the live scene (node tree), so any
                    # literal cookbook-label choices are spurious; clear them so
                    # the persisted artifact matches the bound node-selection shape.
                    choice["choices"] = []
                    notes.append(
                        f"step {number} cleared literal choices for node-pick value_kind"
                    )

            roles = item.get("node_roles")
            if not isinstance(roles, list):
                continue
            normalized_roles = []
            for role in roles:
                if not isinstance(role, dict):
                    normalized_roles.append(role)
                    continue
                normalized_role = dict(role)
                parameter_name = _text_or_empty(normalized_role.get("parameter_name"))
                if parameter_name and parameter_name not in allowed_parameters:
                    normalized_role["parameter_name"] = ""
                    notes.append(
                        f"step {number} cleared ungrounded node-role parameter_name"
                    )
                if (
                    not _text_or_empty(normalized_role.get("node_class"))
                    and not _text_or_empty(normalized_role.get("parameter_name"))
                ):
                    notes.append(f"step {number} removed non-node node role")
                    continue
                normalized_roles.append(normalized_role)
            item["node_roles"] = normalized_roles

        return normalized, notes

    def _validate_stage4_semantic_result(
        self, result: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Validate all LLM semantics against deterministic source allowlists."""
        errors = []
        expected = {step["step_number"]: step for step in context["steps"]}
        raw_steps = result.get("steps")
        if not isinstance(raw_steps, list):
            return ["steps must be a list"]
        by_number = {}
        for item in raw_steps:
            if not isinstance(item, dict):
                errors.append("every steps item must be an object")
                continue
            number = _int_or_none(item.get("step_number"))
            if number is None or number not in expected:
                errors.append(f"invalid step_number: {item.get('step_number')!r}")
                continue
            if number in by_number:
                errors.append(f"duplicate step_number: {number}")
            by_number[number] = item
        missing = sorted(set(expected) - set(by_number))
        if missing:
            errors.append(f"missing step numbers: {missing}")

        method_names = {item["name"] for item in context["logic_methods"]}
        function_names = {item["name"] for item in context["extension_functions"]}
        widget_names = {item["widget_name"] for item in context["widgets"]}
        binding_pairs = {
            (item["widget_name"], item["parameter_name"])
            for item in context["ui_parameter_bindings"]
        }
        allowed_categories = set(context["allowed_slicer_op_categories"])
        allowed_kinds = set(context["allowed_interaction_kinds"])
        allowed_classes = set(context["allowed_node_classes"])
        allowed_intents = set(context["allowed_operation_intents"])
        allowed_role_kinds = set(context["allowed_node_role_kinds"])
        allowed_parameters = self._stage4_allowed_parameter_names(context)
        for number, item in by_number.items():
            if item.get("operation_type") != expected[number]["operation_type"]:
                errors.append(f"step {number} changed authoritative operation_type")
            if item.get("confidence") not in ("high", "medium"):
                errors.append(f"step {number} requires medium or high confidence")
            method = item.get("extension_method_hint")
            if method is not None and method not in method_names:
                errors.append(f"step {number} references unknown method {method!r}")
            function = item.get("extension_function_hint")
            if function is not None and function not in function_names:
                errors.append(f"step {number} references unknown extension function {function!r}")
            if method is not None and function is not None:
                errors.append(f"step {number} cannot select both a method and extension function")
            widget = item.get("widget_name")
            if widget is not None and widget not in widget_names:
                errors.append(f"step {number} references unknown widget {widget!r}")
            binding = item.get("ui_parameter_binding")
            if binding is not None:
                if not isinstance(binding, dict) or (
                    binding.get("widget_name"), binding.get("parameter_name")
                ) not in binding_pairs:
                    errors.append(f"step {number} references unknown UI parameter binding")
            target_mode = item.get("target_value_mode")
            if target_mode not in (None, "set", "invert", "node_reference", "runtime_value"):
                errors.append(f"step {number} has invalid target_value_mode {target_mode!r}")
            category = item.get("slicer_op_category")
            if category is not None and category not in allowed_categories:
                errors.append(f"step {number} references unknown Slicer category {category!r}")
            kind = item.get("interaction_kind", "none")
            if kind not in allowed_kinds:
                errors.append(f"step {number} has invalid interaction_kind {kind!r}")
            node_class = item.get("node_class")
            if node_class is not None and node_class not in allowed_classes:
                errors.append(f"step {number} references unknown node_class {node_class!r}")
            # Revision B: validate the new descriptor fields. These are
            # advisory (consumed by template dispatch and verify_repair semantic
            # checks), so downgrade type problems to soft rewrites instead
            # of hard validation errors — the rest of the pipeline still
            # functions with defaulted values.
            if "creates_node" in item and not isinstance(item["creates_node"], bool):
                errors.append(f"step {number} creates_node must be a boolean")
            if "requires_place_mode" in item and not isinstance(item["requires_place_mode"], bool):
                errors.append(f"step {number} requires_place_mode must be a boolean")
            deps = item.get("setup_dependencies")
            if deps is not None and (not isinstance(deps, list) or any(
                not isinstance(d, int) for d in deps
            )):
                errors.append(f"step {number} setup_dependencies must be a list of step numbers")
            intents = item.get("operation_intents", [])
            if not isinstance(intents, list) or any(
                intent not in allowed_intents for intent in intents
            ):
                errors.append(f"step {number} contains an unknown operation intent")
            elif any(
                intent in ("extension_parameter_update", "extension_node_reference_update")
                for intent in intents
            ) and not isinstance(binding, dict):
                errors.append(f"step {number} parameter update intent requires a UI binding")
            node_roles = item.get("node_roles", [])
            if not isinstance(node_roles, list):
                errors.append(f"step {number} node_roles must be a list")
            else:
                for role in node_roles:
                    if not isinstance(role, dict):
                        errors.append(f"step {number} contains an invalid node role")
                        continue
                    role_class = role.get("node_class")
                    role_parameter = role.get("parameter_name")
                    if role.get("role_kind") not in allowed_role_kinds:
                        errors.append(f"step {number} node role uses unknown role_kind")
                    if role_class and role_class not in allowed_classes:
                        errors.append(f"step {number} node role uses unknown node_class")
                    if role_parameter and role_parameter not in allowed_parameters:
                        errors.append(
                            f"step {number} node role parameter_name "
                            f"{role_parameter!r} is not source-backed; use an allowed "
                            "parameter or leave it empty"
                        )
            if expected[number]["operation_type"] == "user_choice":
                choice = item.get("choice")
                if not isinstance(choice, dict) or not _text_or_empty(choice.get("question")):
                    errors.append(f"step {number} user_choice requires a choice question")
                elif not _text_or_empty(choice.get("parameter_name")):
                    errors.append(f"step {number} user_choice requires parameter_name")
                elif not isinstance(choice.get("choices", []), list):
                    errors.append(f"step {number} user_choice choices must be a list")

        repeats = result.get("repeat_blocks", [])
        if not isinstance(repeats, list):
            errors.append("repeat_blocks must be a list")
            repeats = []
        used = set()
        repeat_ids = set()
        ordered_numbers = sorted(expected)
        for index, block in enumerate(repeats):
            label = f"repeat_blocks[{index}]"
            if not isinstance(block, dict):
                errors.append(f"{label} must be an object")
                continue
            repeat_id = _text_or_empty(block.get("repeat_id"))
            if not repeat_id or repeat_id in repeat_ids:
                errors.append(f"{label} requires a unique repeat_id")
            repeat_ids.add(repeat_id)
            body = block.get("body_steps")
            if not isinstance(body, list) or not body:
                errors.append(f"{label} requires body_steps")
                continue
            body_numbers = [_int_or_none(value) for value in body]
            if any(value not in expected for value in body_numbers):
                errors.append(f"{label} references unknown body step")
                continue
            positions = [ordered_numbers.index(value) for value in body_numbers]
            if positions != list(range(min(positions), max(positions) + 1)):
                errors.append(f"{label} body_steps must be ordered and contiguous")
            if used.intersection(body_numbers):
                errors.append(f"{label} overlaps another repeat block")
            used.update(body_numbers)
            controller = block.get("controller")
            if not isinstance(controller, dict):
                errors.append(f"{label} requires controller")
                continue
            kind = controller.get("kind")
            if kind == "count":
                source = _int_or_none(controller.get("source_step"))
                if (
                    source not in expected
                    or expected[source]["operation_type"] != "user_choice"
                    or source >= body_numbers[0]
                ):
                    errors.append(f"{label} count source_step must be a preceding user_choice")
            elif kind in ("until_choice", "while_choice"):
                if not _text_or_empty(controller.get("prompt")):
                    errors.append(f"{label} condition controller requires prompt")
                if not isinstance(controller.get("exit_value"), bool):
                    errors.append(f"{label} condition controller requires boolean exit_value")
            else:
                errors.append(f"{label} has invalid controller kind {kind!r}")
            if block.get("confidence") not in ("high", "medium"):
                errors.append(f"{label} requires medium or high confidence")
        return errors

    def _stage_map_from_semantic_result(
        self, result: Dict[str, Any], cookbook_def, logic_analysis: Dict
    ) -> Dict[str, Any]:
        """Convert validated semantic output to the existing stage-map contract."""
        raw_methods = logic_analysis.get("methods", []) or []
        all_methods = {
            item["name"]: item for item in raw_methods
            if isinstance(item, dict) and item.get("name")
        }
        semantic_steps = {
            int(item["step_number"]): item for item in result["steps"]
        }
        stages = []
        for cb_step in cookbook_def.steps:
            semantic = semantic_steps[cb_step.step_number]
            op_type = cb_step.operation_type
            choice = semantic.get("choice") or {}
            semantic_binding = semantic.get("ui_parameter_binding")
            resolved_binding = None
            if isinstance(semantic_binding, dict):
                source_binding = (
                    getattr(self, "_ui_parameter_bindings", {}) or {}
                ).get(semantic_binding.get("widget_name"), {})
                role = next((
                    item for item in source_binding.get("roles", []) or []
                    if item.get("parameter_name") == semantic_binding.get("parameter_name")
                ), {})
                resolved_binding = {
                    "widget_name": semantic_binding.get("widget_name"),
                    "widget_class": source_binding.get("widget_class", ""),
                    "ui_text": source_binding.get("ui_text", ""),
                    "properties": source_binding.get("properties", {}),
                    "role": role,
                }
            operation_intents = semantic.get("operation_intents", [])
            node_roles = [
                {**role, "step_id": f"cb_step_{cb_step.step_number}"}
                for role in semantic.get("node_roles", [])
            ]
            sub_op = {
                "op_type": op_type,
                "operation_type": op_type,
                "description": _text_or_empty(semantic.get("semantic_intent")) or cb_step.description,
                "extension_method_hint": semantic.get("extension_method_hint"),
                "extension_function_hint": semantic.get("extension_function_hint"),
                "widget_name": semantic.get("widget_name"),
                "ui_parameter_binding": resolved_binding,
                "target_value": semantic.get("target_value"),
                "target_value_mode": semantic.get("target_value_mode"),
                "slicer_op_category": semantic.get("slicer_op_category"),
                "slicer_api_keywords": _text_list(semantic.get("slicer_api_keywords", [])),
                "interaction_kind": semantic.get("interaction_kind") or "none",
                "interaction_type": semantic.get("interaction_type"),
                "node_class": semantic.get("node_class"),
                "creates_node": bool(semantic.get("creates_node", False)),
                "requires_place_mode": bool(semantic.get("requires_place_mode", False)),
                "setup_dependencies": [
                    int(d) for d in (semantic.get("setup_dependencies") or [])
                    if isinstance(d, (int, float)) and not isinstance(d, bool)
                ],
                "placement_instructions": semantic.get("placement_instructions"),
                "min_control_points": 0,
                "is_optional": bool(semantic.get("is_optional")),
                "operation_intents": operation_intents,
                "node_roles": node_roles,
                "confidence": semantic.get("confidence"),
                "evidence_type": "llm_semantic_decomposition",
                "evidence_id": ",".join(_text_list(semantic.get("evidence_ids", []))),
            }
            if len(operation_intents) == 1:
                sub_op["operation_intent"] = operation_intents[0]
            if resolved_binding:
                role = resolved_binding.get("role", {})
                sub_op["parameter_name"] = role.get("parameter_name", "")
                sub_op["value_property"] = role.get("value_property", "")
            if op_type == "user_choice":
                sub_op.update({
                    "question": choice.get("question", cb_step.description),
                    "choices": choice.get("choices", []),
                    "parameter_name": choice.get(
                        "parameter_name", f"choice_step_{cb_step.step_number}"
                    ),
                    "default_value": choice.get("default_value"),
                    "value_kind": choice.get("value_kind", ""),
                })
                # A node-pick choice is resolved from the live scene via the
                # node tree, not from literal cookbook labels; clear any choices
                # the LLM emitted alongside value_kind == "node" so the artifact
                # matches the bound (BRP) node-selection shape.
                if sub_op.get("value_kind") == "node":
                    sub_op["choices"] = []
            method_name = semantic.get("extension_method_hint")
            method_details = [all_methods[method_name]] if method_name else []
            stages.append({
                "stage_index": cb_step.step_number - 1,
                "stage_name": _text_or_empty(semantic.get("semantic_intent"))[:80]
                or f"Cookbook step {cb_step.step_number}",
                "methods": [method_name] if method_name else [],
                "method_details": method_details,
                "depends_on": (
                    [f"cb_step_{number}" for number in cb_step.depends_on]
                    if cb_step.depends_on
                    else ([f"cb_step_{cb_step.step_number - 1}"] if cb_step.step_number > 1 else [])
                ),
                "input_nodes": [],
                "output_nodes": [],
                "op_type": op_type,
                "operation_type": op_type,
                "cookbook_step": cb_step,
                "sub_operations": [sub_op],
                "is_optional": bool(semantic.get("is_optional")),
            })

        step_numbers = [step.step_number for step in cookbook_def.steps]
        repeat_blocks = []
        for block in result.get("repeat_blocks", []):
            body_numbers = [int(value) for value in block["body_steps"]]
            terminal_position = step_numbers.index(body_numbers[-1])
            controller = dict(block["controller"])
            source = _int_or_none(controller.get("source_step"))
            controller["source_step"] = f"cb_step_{source}" if source is not None else ""
            repeat_blocks.append({
                "repeat_id": block["repeat_id"],
                "body_steps": [f"cb_step_{number}" for number in body_numbers],
                "entry_step": f"cb_step_{body_numbers[0]}",
                "terminal_step": f"cb_step_{body_numbers[-1]}",
                "exit_step": (
                    f"cb_step_{step_numbers[terminal_position + 1]}"
                    if terminal_position + 1 < len(step_numbers) else ""
                ),
                "controller": controller,
                "max_iterations": 100 if controller["kind"] == "count" else 20,
                "inference": {
                    "source": "llm_semantic_decomposition",
                    "evidence_step_ids": block.get("evidence_step_ids", []),
                    "confidence": block.get("confidence"),
                },
            })
        return {
            "stages": stages,
            "stage_count": len(stages),
            "source": "llm_semantic_decomposition",
            "repeat_blocks": repeat_blocks,
        }

    def _build_stage_map_from_typed_cookbook(
        self, cookbook_def, logic_analysis: Dict
    ) -> Dict:
        """Build stage map from explicit cookbook operation_type annotations."""
        raw_methods = logic_analysis.get("methods", [])
        if isinstance(raw_methods, list):
            all_methods = {
                m["name"]: m
                for m in raw_methods
                if isinstance(m, dict) and m.get("name")
            }
        else:
            all_methods = raw_methods if isinstance(raw_methods, dict) else {}
        method_names = list(all_methods.keys())
        stages = []
        for cb_step in cookbook_def.steps:
            step_num = cb_step.step_number
            op_type = _operation_type_for_step({"operation_type": cb_step.operation_type})
            if op_type not in CANONICAL_OPERATION_TYPES:
                raise RuntimeError(
                    f"Cookbook step {step_num} has unsupported operation type '{op_type}'."
                )
            desc = _text_or_empty(cb_step.description)
            desc_lower = desc.lower()
            evidence = self._collect_step_evidence(
                desc, method_names, cookbook_def.extension_name
            )
            sub_op = {
                "op_type": op_type,
                "operation_type": op_type,
                "description": desc[:300],
                "extension_method_hint": None,
                "slicer_api_keywords": [],
                "interaction_type": None,
                "node_class": None,
                "placement_instructions": None,
                "min_control_points": 0,
                "evidence_type": "cookbook_annotation",
                "evidence_id": op_type,
                "confidence": "high",
                "interaction_kind": "none",
                "slicer_op_category": None,
                "is_optional": self._is_optional_cookbook_step(desc),
            }

            final_state = _infer_final_state_intent(desc)
            ui_param_candidates = evidence.get("ui_parameter_candidates") or []
            if op_type == "extension_op":
                self._enrich_typed_extension_op(
                    sub_op, desc_lower, method_names, evidence, final_state, desc
                )
            elif op_type == "slicer_op":
                category = self._infer_slicer_op_category(desc, evidence) or "generic_slicer_api"
                sub_op["slicer_op_category"] = category
                sub_op["slicer_api_keywords"] = [category]
                sub_op["evidence_type"] = "slicer_core"
                sub_op["evidence_id"] = category
            elif op_type == "user_interaction":
                self._enrich_typed_user_interaction(sub_op, desc, evidence)
            elif op_type == "user_choice":
                self._enrich_typed_user_choice(
                    sub_op, cb_step, evidence, ui_param_candidates, final_state
                )

            stage_methods = []
            method_hint = sub_op.get("extension_method_hint")
            if op_type == "extension_op" and method_hint:
                m_info = all_methods.get(method_hint, {})
                stage_methods.append({
                    "name": method_hint,
                    "purpose": sub_op["description"],
                    "parameters": m_info.get("parameters", []),
                    "return_value": m_info.get("return_value"),
                    "state_reads": m_info.get("state_reads", []),
                    "state_writes": m_info.get("state_writes", []),
                    "calls_addnode": m_info.get("calls_addnode", False),
                    "adds_output_to_scene": m_info.get("adds_output_to_scene", False),
                    "side_effects": m_info.get("side_effects", []),
                })
            stage_method_names = [m["name"] for m in stage_methods]
            stages.append({
                "stage_index": step_num - 1,
                "stage_name": self._infer_stage_name(
                    stage_method_names, step_num - 1, len(cookbook_def.steps)
                ),
                "methods": stage_method_names,
                "method_details": stage_methods,
                "depends_on": (
                    [f"cb_step_{d}" for d in cb_step.depends_on]
                    if cb_step.depends_on
                    else ([f"cb_step_{step_num - 1}"] if step_num > 1 else [])
                ),
                "input_nodes": [],
                "output_nodes": [],
                "op_type": op_type,
                "operation_type": op_type,
                "cookbook_step": cb_step,
                "sub_operations": [sub_op],
                "is_optional": bool(sub_op.get("is_optional")),
            })

        self.on_progress(
            "contract", "Build Workflow Contract",
            f"Built {len(stages)} typed cookbook steps"
        )
        return {
            "stages": stages,
            "stage_count": len(stages),
            "source": "typed_cookbook",
        }

    @staticmethod
    def _is_optional_cookbook_step(description: str) -> bool:
        desc = _text_or_empty(description).lower()
        return any(token in desc for token in ("optional", "if desired", "if needed"))

    def _enrich_typed_extension_op(
        self,
        sub_op: Dict[str, Any],
        desc_lower: str,
        method_names: List[str],
        evidence: Dict[str, Any],
        final_state: Dict[str, Any],
        state_text: str,
    ) -> None:
        ui_candidates = evidence.get("ui_parameter_candidates") or []
        if ui_candidates and self._should_apply_ui_parameter_candidate(
            ui_candidates[0], final_state, state_text,
        ):
            candidate = ui_candidates[0]
            role = candidate.get("role", {}) or {}
            sub_op["operation_intent"] = self._ui_binding_operation_intent(candidate)
            sub_op["parameter_name"] = role.get("parameter_name", "")
            sub_op["value_property"] = role.get("value_property", "")
            if sub_op["operation_intent"] == "extension_node_reference_update":
                sub_op["target_value"] = None
                sub_op["target_value_mode"] = "node_reference"
            else:
                sub_op["target_value"] = final_state.get("state")
                sub_op["target_value_mode"] = final_state.get("mode")
            sub_op["ui_parameter_binding"] = candidate
            sub_op["evidence_type"] = "ui_parameter_binding"
            sub_op["evidence_id"] = candidate.get("widget_name")
            return
        if evidence.get("widget_candidates"):
            widget = evidence["widget_candidates"][0]
            logic_methods = widget.get("logic_methods") or []
            if logic_methods:
                sub_op["extension_method_hint"] = logic_methods[0]
            sub_op["evidence_type"] = "widget_connection"
            sub_op["evidence_id"] = widget.get("button_widget_name")
            return
        matched = self._match_description_to_method(desc_lower, method_names)
        if matched:
            sub_op["extension_method_hint"] = matched
            sub_op["evidence_type"] = "logic_method"
            sub_op["evidence_id"] = matched
            return
        if evidence.get("ui_control_candidates"):
            sub_op["evidence_type"] = "ui_control"
            sub_op["evidence_id"] = evidence["ui_control_candidates"][0].get("control")
            sub_op["confidence"] = "medium"

    def _enrich_typed_user_interaction(
        self,
        sub_op: Dict[str, Any],
        description: str,
        evidence: Dict[str, Any],
    ) -> None:
        interaction = (evidence.get("interaction_candidates") or [{}])[0]
        text = _text_or_empty(description).lower()
        node_class = interaction.get("node_class", "")
        interaction_type = interaction.get("interaction_type", "")
        adjusts_existing = (
            any(word in text for word in ("drag", "adjust", "rotate", "translate"))
            and not any(word in text for word in ("add", "create", "draw", "place"))
        )
        if adjusts_existing:
            interaction = {}
            node_class = ""
            interaction_type = "generic"
        if not node_class and not adjusts_existing:
            if "curve" in text:
                node_class, interaction_type = "vtkMRMLMarkupsCurveNode", "curve"
            elif "plane" in text:
                node_class, interaction_type = "vtkMRMLMarkupsPlaneNode", "plane"
            elif "line" in text:
                node_class, interaction_type = "vtkMRMLMarkupsLineNode", "line"
            elif "point" in text or "fiducial" in text:
                node_class, interaction_type = "vtkMRMLMarkupsFiducialNode", "fiducial"
            # Revision B: previously this else-block defaulted
            # `node_class = "vtkMRMLCrosshairNode"` for any drag/adjust/
            # rotate/translate action without an explicit markup keyword.
            # That conflated view_adjustment with crosshair and broke the
            # generate dispatch for non-crosshair viewport adjustments.
            # node_class now stays empty for view_adjustment fallbacks.
        interaction_kind = (
            interaction.get("interaction_kind")
            or ("view_adjustment" if adjusts_existing else "markup_placement")
        )
        sub_op["interaction_kind"] = interaction_kind
        sub_op["interaction_type"] = interaction_type or _derive_interaction_type(node_class)
        sub_op["node_class"] = node_class
        # Revision B: populate the new descriptor fields so the heuristic
        # path matches the LLM semantic path's contract.
        is_markup = self._is_markup_node_class(node_class)
        sub_op["creates_node"] = bool(
            interaction_kind == "markup_placement" and is_markup and not adjusts_existing
        )
        sub_op["requires_place_mode"] = bool(
            interaction_kind == "markup_placement"
            and is_markup
            and not adjusts_existing
        )
        sub_op.setdefault("setup_dependencies", [])
        sub_op["placement_instructions"] = description
        sub_op["evidence_type"] = "viewport_action"
        sub_op["evidence_id"] = sub_op["interaction_type"]

    def _enrich_typed_user_choice(
        self,
        sub_op: Dict[str, Any],
        cb_step,
        evidence: Dict[str, Any],
        ui_param_candidates: List[Dict[str, Any]],
        final_state: Dict[str, Any],
    ) -> None:
        desc = _text_or_empty(cb_step.description)
        text = desc.lower()
        sub_op["question"] = desc
        sub_op["choices"] = []
        sub_op["parameter_name"] = f"choice_step_{cb_step.step_number}"
        sub_op["default_value"] = None
        if ui_param_candidates:
            candidate = ui_param_candidates[0]
            role = candidate.get("role", {}) or {}
            parameter_name = role.get("parameter_name", "")
            if parameter_name:
                sub_op["parameter_name"] = parameter_name
            sub_op["value_property"] = role.get("value_property", "")
            sub_op["ui_parameter_binding"] = candidate
            sub_op["operation_intent"] = self._ui_binding_operation_intent(candidate)
            if sub_op.get("value_property") == "checked" or "right" in text or "left" in text:
                sub_op["choices"] = [
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ]
                sub_op["default_value"] = final_state.get("state")
        if not sub_op["choices"] and ("left" in text and "right" in text):
            sub_op["choices"] = [
                {"label": "Left", "value": "left"},
                {"label": "Right", "value": "right"},
            ]
            sub_op["parameter_name"] = (
                sub_op["parameter_name"]
                if sub_op["parameter_name"] != f"choice_step_{cb_step.step_number}"
                else "side"
            )
        if any(token in text for token in ("how many", "number of", "count")):
            if sub_op["parameter_name"].startswith("choice_step_"):
                sub_op["parameter_name"] = f"number_step_{cb_step.step_number}"
        sub_op["evidence_type"] = "user_context"
        sub_op["evidence_id"] = sub_op["parameter_name"]

    def _build_stage_map_from_decomposition(
        self, decomposition: Dict, cookbook_def, logic_analysis: Dict
    ) -> Dict:
        """Convert LLM decomposition output into the stage_map dict format."""
        raw_methods = logic_analysis.get("methods", [])
        if isinstance(raw_methods, list):
            all_methods = {}
            for m in raw_methods:
                if isinstance(m, dict) and m.get("name"):
                    all_methods[m["name"]] = m
        else:
            all_methods = raw_methods if isinstance(raw_methods, dict) else {}

        stages = []
        llm_steps = {}
        for raw_step in decomposition.get("steps", []):
            if not isinstance(raw_step, dict):
                continue
            step_number = _int_or_none(raw_step.get("step_number"))
            if step_number is not None:
                llm_steps[step_number] = raw_step

        for cb_step in cookbook_def.steps:
            step_num = cb_step.step_number
            step_id = f"cb_step_{step_num}"
            llm_step = llm_steps.get(step_num, {})
            llm_sub_ops = llm_step.get("sub_operations", [])
            method_names = list(all_methods.keys())
            step_evidence = self._collect_step_evidence(
                cb_step.description, method_names, cookbook_def.extension_name
            )

            # If LLM returned nothing for this step, fall back to heuristic
            if not llm_sub_ops:
                llm_sub_ops = [{
                    "op_type": "extension_op",
                    "description": cb_step.description[:200],
                    "extension_method_hint": None,
                    "slicer_api_keywords": [],
                    "interaction_type": None,
                    "node_class": None,
                    "placement_instructions": None,
                }]

            # Normalize sub-operations: ensure required fields
            sub_ops = []
            for so in llm_sub_ops:
                if not isinstance(so, dict):
                    so = {"description": so}
                op_type = _optional_text(so.get("op_type")) or "extension_op"
                if op_type not in ("extension_op", "slicer_op", "user_interaction", "user_choice", "unknown_op"):
                    op_type = "extension_op"
                description = _text_or_empty(
                    so.get("description", cb_step.description[:200])
                )
                method_hint = _optional_text(so.get("extension_method_hint"))
                node_class = _optional_text(so.get("node_class"))
                evidence_type = _optional_text(so.get("evidence_type")) or "unknown"
                evidence_id = _optional_text(so.get("evidence_id"))
                confidence = (_optional_text(so.get("confidence")) or "medium").lower()
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                interaction_kind = (
                    _optional_text(so.get("interaction_kind"))
                    or self._infer_interaction_kind_from_evidence(step_evidence, node_class or "")
                )
                slicer_op_category = (
                    _optional_text(so.get("slicer_op_category"))
                    or self._infer_slicer_op_category(description, step_evidence)
                )
                normalized = {
                    "op_type": op_type,
                    "description": description,
                    "extension_method_hint": method_hint,
                    "slicer_api_keywords": _text_list(so.get("slicer_api_keywords", [])),
                    "interaction_type": _optional_text(so.get("interaction_type")),
                    "node_class": node_class,
                    "creates_node": bool(so.get("creates_node", False)),
                    "requires_place_mode": bool(so.get("requires_place_mode", False)),
                    "setup_dependencies": [
                        int(d) for d in (so.get("setup_dependencies") or [])
                        if isinstance(d, (int, float)) and not isinstance(d, bool)
                    ],
                    "placement_instructions": _optional_text(so.get("placement_instructions")),
                    "min_control_points": so.get("min_control_points", 0),
                    "evidence_type": evidence_type,
                    "evidence_id": evidence_id,
                    "confidence": confidence,
                    "interaction_kind": interaction_kind,
                    "slicer_op_category": slicer_op_category,
                    "is_optional": so.get("is_optional", False),
                }
                # user_choice-specific fields
                if op_type == "user_choice":
                    normalized["question"] = _text_or_empty(so.get("question", cb_step.description))
                    choices = so.get("choices", [])
                    normalized["choices"] = choices if isinstance(choices, list) else []
                    normalized["parameter_name"] = (
                        _optional_text(so.get("parameter_name")) or f"choice_step_{step_num}"
                    )
                    normalized["default_value"] = so.get("default_value")
                sub_ops.append(normalized)

            original_step_text = _text_or_empty(cb_step.description)
            is_location_context = (
                self._is_ui_location_context_text(original_step_text)
                and not self._is_explicit_module_switch_text(original_step_text)
            )
            if is_location_context:
                kept_sub_ops = []
                removed_module_switch = False
                for so in sub_ops:
                    is_module_switch = (
                        so.get("op_type") == "slicer_op"
                        and so.get("slicer_op_category") == "module_switching"
                    )
                    if is_module_switch:
                        removed_module_switch = True
                        continue
                    kept_sub_ops.append(so)
                if removed_module_switch and kept_sub_ops:
                    sub_ops = kept_sub_ops
                    logger.info(
                        "[Stage 4] Removed invented module_switching sub-operation "
                        "from step %d because the cookbook text is UI-location context",
                        step_num,
                    )
                elif removed_module_switch and not kept_sub_ops:
                    sub_ops = [{
                        "op_type": "slicer_op",
                        "description": original_step_text[:300],
                        "extension_method_hint": None,
                        "slicer_api_keywords": ["display", "view"],
                        "interaction_type": None,
                        "node_class": None,
                        "placement_instructions": None,
                        "min_control_points": 0,
                        "evidence_type": "slicer_core",
                        "evidence_id": "markups_display",
                        "confidence": "medium",
                        "interaction_kind": "none",
                        "slicer_op_category": "markups_display",
                        "is_optional": False,
                    }]
                    logger.info(
                        "[Stage 4] Reinterpreted module location context as markups_display "
                        "for step %d",
                        step_num,
                    )

            # Resolve extension_method_hint to actual method names
            for so in sub_ops:
                hint = so.get("extension_method_hint")
                if hint and hint not in all_methods:
                    matched = self._match_method_name(hint, method_names)
                    if matched:
                        so["extension_method_hint"] = matched

            # Deterministic evidence validation. The LLM may propose op_type,
            # but source/UI/Slicer-core evidence decides whether it can stand.
            for so in sub_ops:
                desc_lower = _text_or_empty(so.get("description")).lower()
                ui_param_candidates = step_evidence.get("ui_parameter_candidates") or []
                final_state = _infer_final_state_intent(
                    " ".join([
                        _text_or_empty(cb_step.description),
                        _text_or_empty(so.get("description")),
                    ])
                )
                state_text = " ".join([
                    _text_or_empty(cb_step.description),
                    _text_or_empty(so.get("description")),
                ])
                if (
                    ui_param_candidates
                    and not (
                        so["op_type"] == "slicer_op"
                        and self._evidence_has(step_evidence, "slicer_core_candidates")
                    )
                    and self._should_apply_ui_parameter_candidate(
                        ui_param_candidates[0], final_state, state_text
                    )
                    and so["op_type"] not in ("user_choice", "user_interaction")
                ):
                    candidate = ui_param_candidates[0]
                    role = candidate.get("role", {}) or {}
                    so["op_type"] = "extension_op"
                    so["extension_method_hint"] = None
                    so["slicer_api_keywords"] = []
                    so["evidence_type"] = "ui_parameter_binding"
                    so["evidence_id"] = candidate.get("widget_name")
                    so["confidence"] = "high"
                    so["operation_intent"] = self._ui_binding_operation_intent(candidate)
                    so["parameter_name"] = role.get("parameter_name", "")
                    so["value_property"] = role.get("value_property", "")
                    if so["operation_intent"] == "extension_node_reference_update":
                        so["target_value"] = None
                        so["target_value_mode"] = "node_reference"
                    else:
                        so["target_value"] = final_state.get("state")
                        so["target_value_mode"] = final_state.get("mode")
                    so["ui_parameter_binding"] = candidate

                if (
                    so["op_type"] not in ("user_choice", "user_interaction")
                    and self._evidence_has(step_evidence, "choice_candidates")
                    and not self._evidence_has(step_evidence, "widget_candidates")
                    and not self._evidence_has(step_evidence, "ui_parameter_candidates")
                    and not so.get("extension_method_hint")
                ):
                    so["op_type"] = "user_choice"
                    so["question"] = so.get("question") or so.get("description", cb_step.description)
                    so["choices"] = so.get("choices", [])
                    so["parameter_name"] = so.get("parameter_name") or f"choice_step_{step_num}"
                    so["slicer_api_keywords"] = []
                    so["evidence_type"] = "user_context"
                    so["confidence"] = "high"
                elif (
                    so["op_type"] == "unknown_op"
                    and self._evidence_has(step_evidence, "interaction_candidates")
                ):
                    interaction = step_evidence["interaction_candidates"][0]
                    so["op_type"] = "user_interaction"
                    so["interaction_kind"] = interaction.get("interaction_kind", "markup_placement")
                    so["interaction_type"] = interaction.get("interaction_type")
                    so["node_class"] = interaction.get("node_class")
                    # Revision B: populate the new descriptor fields on the
                    # heuristic-reclassification path too, so all user_interaction
                    # sub_ops carry a consistent contract.
                    kind = so["interaction_kind"]
                    is_markup = self._is_markup_node_class(so.get("node_class") or "")
                    so["creates_node"] = bool(kind == "markup_placement" and is_markup)
                    so["requires_place_mode"] = bool(kind == "markup_placement" and is_markup)
                    so.setdefault("setup_dependencies", [])
                    so["placement_instructions"] = so.get("placement_instructions") or so.get("description")
                    so["evidence_type"] = "viewport_action"
                    so["confidence"] = "high"
                elif so["op_type"] == "unknown_op" and self._evidence_has(
                    step_evidence, "logic_method_candidates"
                ):
                    method = step_evidence["logic_method_candidates"][0].get("method")
                    so["op_type"] = "extension_op"
                    so["extension_method_hint"] = method
                    so["evidence_type"] = "logic_method"
                    so["evidence_id"] = method
                    so["confidence"] = "high"
                elif so["op_type"] == "unknown_op" and self._evidence_has(
                    step_evidence, "slicer_core_candidates"
                ):
                    category = self._infer_slicer_op_category(so.get("description", ""), step_evidence)
                    if category:
                        so["op_type"] = "slicer_op"
                        so["slicer_op_category"] = category
                        so["evidence_type"] = "slicer_core"
                        so["evidence_id"] = category
                        so["confidence"] = "medium"

                if so["op_type"] == "extension_op":
                    if so.get("operation_intent") == "extension_parameter_update":
                        so["evidence_type"] = so.get("evidence_type") or "ui_parameter_binding"
                        so["confidence"] = "high"
                    elif (
                        self._evidence_has(step_evidence, "ui_parameter_candidates")
                        and self._should_apply_ui_parameter_candidate(
                            step_evidence["ui_parameter_candidates"][0],
                            final_state,
                            state_text,
                        )
                    ):
                        candidate = step_evidence["ui_parameter_candidates"][0]
                        role = candidate.get("role", {}) or {}
                        so["extension_method_hint"] = None
                        so["operation_intent"] = self._ui_binding_operation_intent(candidate)
                        so["parameter_name"] = role.get("parameter_name", "")
                        so["value_property"] = role.get("value_property", "")
                        if so["operation_intent"] == "extension_node_reference_update":
                            so["target_value"] = None
                            so["target_value_mode"] = "node_reference"
                        else:
                            so["target_value"] = final_state.get("state")
                            so["target_value_mode"] = final_state.get("mode")
                        so["ui_parameter_binding"] = candidate
                        so["evidence_type"] = "ui_parameter_binding"
                        so["evidence_id"] = candidate.get("widget_name")
                        so["confidence"] = "high"
                    elif so.get("extension_method_hint"):
                        so["evidence_type"] = "logic_method"
                        so["evidence_id"] = so["extension_method_hint"]
                        so["confidence"] = "high"
                    elif self._evidence_has(step_evidence, "widget_candidates"):
                        widget = step_evidence["widget_candidates"][0]
                        logic_methods = widget.get("logic_methods") or []
                        if logic_methods:
                            so["extension_method_hint"] = logic_methods[0]
                        so["evidence_type"] = "widget_connection"
                        so["evidence_id"] = widget.get("button_widget_name")
                        so["confidence"] = "high"
                    elif self._evidence_has(step_evidence, "ui_control_candidates"):
                        so["evidence_type"] = "ui_control"
                        so["evidence_id"] = step_evidence["ui_control_candidates"][0].get("control")
                        so["confidence"] = "medium"
                    else:
                        matched = self._match_description_to_method(desc_lower, method_names)
                        if matched:
                            so["extension_method_hint"] = matched
                            so["evidence_type"] = "logic_method"
                            so["evidence_id"] = matched
                            so["confidence"] = "high"
                        else:
                            so["op_type"] = "unknown_op"
                            so["evidence_type"] = "unknown"
                            so["confidence"] = "low"
                elif so["op_type"] == "slicer_op":
                    category = so.get("slicer_op_category") or self._infer_slicer_op_category(
                        so.get("description", ""), step_evidence
                    )
                    if self._evidence_has(step_evidence, "slicer_core_candidates") and category:
                        self._clear_ui_parameter_fields(so)
                        so["slicer_op_category"] = category
                        so["evidence_type"] = "slicer_core"
                        so["evidence_id"] = category
                        if so.get("confidence") == "low":
                            so["confidence"] = "medium"
                    else:
                        so["op_type"] = "unknown_op"
                        so["slicer_api_keywords"] = []
                        so["evidence_type"] = "unknown"
                        so["confidence"] = "low"
                elif so["op_type"] == "user_interaction":
                    so["evidence_type"] = "viewport_action"
                    so["interaction_kind"] = (
                        so.get("interaction_kind")
                        or self._infer_interaction_kind_from_evidence(
                            step_evidence, so.get("node_class", "")
                        )
                    )
                    # Revision B: previously this block forcibly injected
                    # `node_class = "vtkMRMLCrosshairNode"` whenever a
                    # view_adjustment step had empty node_class. That hack
                    # conflated "no node created" with "crosshair node" and
                    # broke the generate dispatch for every drag-existing-handle
                    # or viewport-adjustment interaction. With Revision A
                    # dispatching on interaction_kind, node_class is allowed
                    # to be empty for view_adjustment steps, so the injection
                    # is gone. PR2's retrieval-grounded generator will pick
                    # the right node class (or none) from KB search results.
                elif so["op_type"] == "user_choice":
                    so["evidence_type"] = "user_context"
                elif so["op_type"] == "unknown_op":
                    so["confidence"] = "low"

            # ── Reclassify extension-specific slicer_ops → extension_op ──
            # slicer_ops that reference the extension's own resources (custom
            # layouts, extension-defined node types, extension module names)
            # cannot be resolved via Slicer KB search. Reclassify them as
            # extension_op so they use the extension source for code generation.
            ext_name = _text_or_empty(cookbook_def.extension_name)
            # Common patterns indicating extension-specific knowledge:
            # - The extension's own name in layout/view descriptions
            # - Custom layout IDs registered by the extension
            # - Extension module names not in Slicer core
            # Build the indicator set from ALL authoritative identifiers, not just
            # the repo/cookbook name: the cookbook name is often the repo name
            # ("Slicer<ModuleName>") while steps reference the bare module name
            # ("<ModuleName>"). The module name derives from the Logic class
            # name minus the trailing "Logic".
            logic_class_name = _text_or_empty(logic_analysis.get("class_name"))
            module_name = (
                logic_class_name[: -len("Logic")]
                if logic_class_name.endswith("Logic") else ""
            )
            _extension_specific_indicators = set()
            for _id_name in (ext_name, logic_class_name, module_name):
                if not _id_name:
                    continue
                _extension_specific_indicators.add(_id_name.lower())
                # Also add a shortened acronym derived from the camelCase name.
                parts = _re.sub(r'([a-z])([A-Z])', r'\1 \2', _id_name).split()
                if parts:
                    acronym = "".join(p[0].lower() for p in parts if p)
                    if len(acronym) >= 2:
                        _extension_specific_indicators.add(acronym)

            for so in sub_ops:
                if so["op_type"] != "slicer_op":
                    continue
                desc_lower = _text_or_empty(so.get("description")).lower()
                keywords_lower = [
                    _text_or_empty(k).lower()
                    for k in _text_list(so.get("slicer_api_keywords", []))
                ]
                combined_text = desc_lower + " " + " ".join(keywords_lower)

                # Check if the description/keywords reference the extension itself
                is_extension_specific = False
                for indicator in _extension_specific_indicators:
                    if indicator in combined_text:
                        is_extension_specific = True
                        break

                if is_extension_specific:
                    # Try to find a matching extension method
                    matched_method = self._match_description_to_method(desc_lower, method_names)
                    so["op_type"] = "extension_op"
                    so["extension_method_hint"] = matched_method
                    so["slicer_api_keywords"] = []
                    so["evidence_type"] = "ui_control"
                    so["evidence_id"] = ext_name
                    so["confidence"] = "medium"
                    logger.info(
                        "[Stage 4] Reclassified step %d slicer_op → extension_op "
                        "(extension-specific: '%s'): %s",
                        step_num, ext_name, desc_lower[:60],
                    )

            # ── Widget connection verification: slicer_ops matching extension buttons → extension_op ──
            if self._widget_connections:
                for so in sub_ops:
                    if so["op_type"] != "slicer_op":
                        continue
                    desc_lower = _text_or_empty(so.get("description")).lower()
                    for conn in self._widget_connections:
                        btn_name = _text_or_empty(conn.get("button_widget_name")).lower()
                        logic_methods = conn.get("logic_methods", [])
                        # Extract significant words from button name for matching
                        btn_words = [w for w in btn_name.replace("_", " ").replace(".", " ").split() if len(w) > 3]
                        if not btn_words:
                            continue
                        match_count = sum(1 for w in btn_words if w in desc_lower)
                        if match_count >= 2 or (len(btn_words) == 1 and match_count == 1):
                            so["op_type"] = "extension_op"
                            if logic_methods:
                                so["extension_method_hint"] = logic_methods[0]
                            so["slicer_api_keywords"] = []
                            so["evidence_type"] = "widget_connection"
                            so["evidence_id"] = conn.get("button_widget_name", "")
                            so["confidence"] = "high"
                            logger.info(
                                "[Stage 4] Reclassified step %d slicer_op → extension_op "
                                "(widget button '%s' match): %s",
                                step_num, conn.get("button_widget_name", ""), desc_lower[:60],
                            )
                            break

            # ── "Select/choose/set" without method hint → user_choice ──
            for so in sub_ops:
                if so["op_type"] != "slicer_op":
                    continue
                desc = _text_or_empty(so.get("description")).lower()
                has_hint = bool(so.get("extension_method_hint"))
                if not has_hint and any(
                    desc.startswith(w) for w in ("select ", "choose ", "set the ")
                ):
                    so["op_type"] = "user_choice"
                    so["question"] = so.get("description", "")
                    so["slicer_api_keywords"] = []
                    so["parameter_name"] = f"choice_step_{step_num}"
                    so["evidence_type"] = "user_context"
                    so["confidence"] = "high"
                    logger.info(
                        "[Stage 4] Reclassified step %d slicer_op → user_choice "
                        "(selection without method hint): %s",
                        step_num, desc[:60],
                    )

            # ── "Tick/enable/toggle" extension UI controls without method hint → extension_op ──
            # Extension UI checkboxes/toggles set extension parameter node values,
            # not Slicer core API. If there's no Logic method, it's still extension_op
            # because the parameter names are extension-specific.
            for so in sub_ops:
                if so["op_type"] != "slicer_op":
                    continue
                desc = _text_or_empty(so.get("description")).lower()
                has_hint = bool(so.get("extension_method_hint"))
                if not has_hint and any(
                    desc.startswith(w) for w in ("tick ", "toggle ", "enable ", "check ", "uncheck ")
                ):
                    so["op_type"] = "extension_op"
                    so["slicer_api_keywords"] = []
                    so["evidence_type"] = "ui_control"
                    so["evidence_id"] = "extension_ui_toggle"
                    so["confidence"] = "medium"
                    logger.info(
                        "[Stage 4] Reclassified step %d slicer_op → extension_op "
                        "(extension UI control without method hint): %s",
                        step_num, desc[:60],
                    )

            # ── Reclassify extension_op/unknown_op → slicer_op when no method hint and matches Slicer core ──
            # Sub-ops classified as extension_op (or downgraded to unknown_op due to
            # no matching method) that describe Slicer core UI operations should be
            # slicer_op so the ground/generate phases can produce proper Slicer API code.
            _SLICER_CORE_PATTERNS = {
                "layout_slice_view": [
                    "layout", "conventional", "four-up", "slice view",
                    "fov", "spacing match", "field of view",
                ],
                "module_switching": [
                    "switch to", "open module", "open the",
                    "select module", "activate module", "go to module",
                ],
                "crosshair": [
                    "crosshair", "slice intersection",
                    "intersection visibility", "enable interaction",
                ],
                "node_display": [
                    "slice visibility", "slice visible",
                    "set slice visible", "toggle slice",
                ],
                "markups_display": [
                    "display view", "view node", "display node",
                    "set view", "view 1", "advanced panel",
                ],
            }
            for so in sub_ops:
                if so["op_type"] not in ("extension_op", "unknown_op"):
                    continue
                if so.get("extension_method_hint"):
                    continue
                desc_lower = _text_or_empty(so.get("description")).lower()
                # Skip if description references the extension name
                is_extension_specific = False
                for indicator in _extension_specific_indicators:
                    if indicator in desc_lower:
                        is_extension_specific = True
                        break
                if is_extension_specific:
                    continue
                category = so.get("slicer_op_category", "")
                patterns = _SLICER_CORE_PATTERNS.get(category, [])
                if not patterns or not any(kw in desc_lower for kw in patterns):
                    # Also check all categories for keyword matches.  The LLM may
                    # provide a broad category such as layout_slice_view for a
                    # more specific Slicer core operation such as slice visibility.
                    for cat, kws in _SLICER_CORE_PATTERNS.items():
                        if any(kw in desc_lower for kw in kws):
                            category = cat
                            patterns = kws
                            break
                if category == "module_switching" and not self._is_explicit_module_switch_text(desc_lower):
                    continue
                if patterns and any(kw in desc_lower for kw in patterns):
                    so["op_type"] = "slicer_op"
                    so["slicer_op_category"] = category
                    so["slicer_api_keywords"] = so.get("slicer_api_keywords") or [category]
                    so["evidence_type"] = "slicer_core"
                    so["evidence_id"] = category
                    so["confidence"] = "high"
                    self._clear_ui_parameter_fields(so)
                    logger.info(
                        "[Stage 4] Reclassified step %d %s → slicer_op "
                        "(Slicer core UI pattern '%s'): %s",
                        step_num, so.get("op_type", "?"), category, desc_lower[:60],
                    )

            # Build method_details from matched extension_op sub-operations
            stage_methods = []
            for so in sub_ops:
                if so["op_type"] == "extension_op" and so.get("extension_method_hint"):
                    matched = so["extension_method_hint"]
                    m_info = all_methods.get(matched, {})
                    stage_methods.append({
                        "name": matched,
                        "purpose": so["description"],
                        "parameters": m_info.get("parameters", []),
                        "return_value": m_info.get("return_value"),
                        "state_reads": m_info.get("state_reads", []),
                        "state_writes": m_info.get("state_writes", []),
                        "calls_addnode": m_info.get("calls_addnode", False),
                        "adds_output_to_scene": m_info.get("adds_output_to_scene", False),
                        "side_effects": m_info.get("side_effects", []),
                    })

            # Determine overall step op_type
            op_types = {so["op_type"] for so in sub_ops}
            if len(op_types) > 1:
                stage_op_type = "mixed"
            elif op_types:
                stage_op_type = op_types.pop()
            else:
                stage_op_type = "extension_op"

            # Determine stage name
            stage_method_names = [m["name"] for m in stage_methods] if stage_methods else []
            stage_name = self._infer_stage_name(
                stage_method_names, step_num - 1, len(cookbook_def.steps)
            )

            # Determine if the step is optional
            is_optional = any(so.get("is_optional") for so in sub_ops)

            stage = {
                "stage_index": step_num - 1,
                "stage_name": stage_name,
                "methods": stage_method_names,
                "method_details": stage_methods,
                "depends_on": (
                    [f"cb_step_{d}" for d in cb_step.depends_on]
                    if cb_step.depends_on
                    else ([f"cb_step_{step_num - 1}"] if step_num > 1 else [])
                ),
                "input_nodes": [],
                "output_nodes": [],
                "op_type": stage_op_type,
                "cookbook_step": cb_step,
                "sub_operations": sub_ops,
                "is_optional": is_optional,
            }
            stages.append(stage)

        self.on_progress(
            "contract", "Build Workflow Contract",
            f"Decomposed {len(stages)} cookbook steps via LLM"
        )

        return {
            "stages": stages,
            "stage_count": len(stages),
            "source": "cookbook_llm_decomposition",
        }

    # ================================================================
