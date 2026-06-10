from .common import *


class AnalyzerValidationContractsMixin:
    def _sync_template_contracts(
        self,
        templates: Dict[str, str],
        generators: List[Dict],
        workflow_graph: Optional[Dict] = None,
    ) -> Dict[str, Dict]:
        """Synchronize deterministic template evidence back into contracts.

        Template revision can repair code, but validation consumes generator and
        workflow metadata.  This pass keeps those representations aligned
        without extension-specific rules.
        """
        context = self._build_template_validation_context(generators)
        workflow_steps = {
            step.get("step_id", ""): step
            for step in (workflow_graph or {}).get("steps", [])
            if isinstance(step, dict)
        }
        sync_report = {
            "extension_functions": [],
            "destructive_contracts": [],
        }
        final_api_evidence = {}

        for tpl_name, raw_code in templates.items():
            if not tpl_name.endswith((".py.tpl", ".py")):
                continue
            ctx = context.get(tpl_name)
            if not ctx:
                continue
            gen = ctx.get("generator", {})
            step_id = (gen.get("param_signature") or {}).get("workflow_step", "")
            workflow_step = workflow_steps.get(step_id)
            sample_code = self._fill_remaining_placeholders(
                raw_code.replace(
                    "{vol_lookup}",
                    "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')",
                )
            )

            function_name = self._detect_extension_function_call(sample_code)
            if function_name:
                self._record_extension_function_contract(
                    gen, workflow_step, function_name
                )
                sync_report["extension_functions"].append({
                    "template": tpl_name,
                    "function": function_name,
                })

            destructive_ops = self._template_destructive_ops(sample_code)
            if destructive_ops:
                contract = self._destructive_ops_contract(
                    sample_code, raw_code, gen, destructive_ops
                )
                if contract.get("allowed"):
                    gen["allow_destructive_ops"] = True
                    gen["destructive_ops_contract"] = contract
                    if workflow_step is not None:
                        workflow_step["allow_destructive_ops"] = True
                        workflow_step["destructive_ops_contract"] = contract
                    sync_report["destructive_contracts"].append({
                        "template": tpl_name,
                        "ops": destructive_ops,
                        "scope": contract.get("scope", ""),
                    })

            if gen.get("op_type") == "slicer_op" or any(
                so.get("op_type") == "slicer_op"
                for so in (gen.get("sub_operations", []) or [])
            ):
                existing_evidence = gen.get("api_evidence") or {}
                current_evidence = self._build_template_api_evidence(
                    sample_code,
                    gen,
                    source="template_contract_sync",
                )
                if existing_evidence.get("accepted_footprints"):
                    existing_evidence = self._merge_api_evidence([
                        existing_evidence,
                        current_evidence,
                    ])
                else:
                    existing_evidence = current_evidence
                gen["api_evidence"] = existing_evidence
                if workflow_step is not None:
                    workflow_step["api_evidence"] = existing_evidence
                if isinstance(self._workflow_metadata, dict):
                    self._workflow_metadata.setdefault("api_evidence", {})[tpl_name] = existing_evidence
                final_api_evidence[tpl_name] = existing_evidence

            self._refresh_generator_operation_model(gen, workflow_step)

        if isinstance(self._workflow_metadata, dict):
            self._workflow_metadata["template_contract_sync"] = sync_report
            if final_api_evidence:
                self._workflow_metadata["final_api_evidence"] = final_api_evidence
                self._workflow_metadata["stage5T_api_evidence_note"] = (
                    "stage5T_api_evidence is pre-revision retrieval evidence; "
                    "final_api_evidence reflects the current validated templates."
                )
            self._workflow_metadata["operation_model"] = {
                (gen.get("param_signature") or {}).get("workflow_step", ""): gen.get("operation_model", {})
                for gen in generators or []
                if (gen.get("param_signature") or {}).get("workflow_step")
            }
        return sync_report

    @staticmethod
    def _merge_api_evidence(evidence_items: List[Dict]) -> Dict:
        """Merge API evidence records from multiple generated sub-templates."""
        merged = {
            "source": "stage5T",
            "accepted_footprints": [],
            "api_chains": [],
            "operation_descriptions": [],
            "slicer_op_categories": [],
            "slicer_api_keywords": [],
        }
        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue
            for key in (
                "accepted_footprints", "api_chains", "operation_descriptions",
                "slicer_op_categories", "slicer_api_keywords",
            ):
                values = evidence.get(key) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    if value and value not in merged[key]:
                        merged[key].append(value)
        return merged

    def _build_template_api_evidence(
        self,
        code: str,
        op_context: Optional[Any] = None,
        source: str = "template",
    ) -> Dict:
        """Build per-template API evidence from generated code and op metadata."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None

        footprints = []
        local_chains = []
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr not in footprints:
                        footprints.append(attr)
                    try:
                        chain = ast.unparse(node.func)
                    except Exception:
                        chain = attr
                    if chain and chain not in local_chains:
                        local_chains.append(chain)
                elif isinstance(node, ast.Attribute):
                    attr = node.attr
                    if attr and attr not in footprints:
                        footprints.append(attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith("vtkMRML") and value not in footprints:
                        footprints.append(value)

        api_chains = self._extract_api_chains(code)
        for chain in local_chains:
            if chain not in api_chains:
                api_chains.append(chain)

        def _ctx_get(name: str, default=None):
            if isinstance(op_context, dict):
                return op_context.get(name, default)
            return getattr(op_context, name, default)

        sub_ops = _ctx_get("sub_operations", []) or []
        descriptions = []
        categories = []
        keywords = []
        if sub_ops:
            for so in sub_ops:
                if not isinstance(so, dict):
                    continue
                desc = so.get("description")
                category = so.get("slicer_op_category")
                if desc and desc not in descriptions:
                    descriptions.append(desc)
                if category and category not in categories:
                    categories.append(category)
                for keyword in so.get("slicer_api_keywords", []) or []:
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
        else:
            desc = _ctx_get("description", "")
            category = _ctx_get("slicer_op_category", "")
            if desc:
                descriptions.append(desc)
            if category:
                categories.append(category)
            for keyword in (_ctx_get("slicer_api_keywords", []) or []):
                if keyword and keyword not in keywords:
                    keywords.append(keyword)

        return {
            "source": source,
            "accepted_footprints": sorted(set(footprints)),
            "api_chains": sorted(set(api_chains)),
            "operation_descriptions": descriptions,
            "slicer_op_categories": categories,
            "slicer_api_keywords": keywords,
        }

    def _record_extension_function_contract(
        self,
        gen: Dict,
        workflow_step: Optional[Dict],
        function_name: str,
    ) -> None:
        """Record a top-level extension function in generator/workflow contracts."""
        gen["extension_function_name"] = function_name
        if workflow_step is not None:
            workflow_step["extension_function_name"] = function_name

        for target in (gen, workflow_step):
            if not isinstance(target, dict):
                continue
            for so in target.get("sub_operations", []) or []:
                if (
                    so.get("op_type") == "extension_op"
                    and not so.get("extension_method_hint")
                ):
                    so["extension_function_hint"] = function_name
                    so["evidence_type"] = "module_function"
                    so["evidence_id"] = function_name
                    so["confidence"] = "high"

    def _refresh_generator_operation_model(
        self,
        gen: Dict,
        workflow_step: Optional[Dict],
    ) -> None:
        """Recompute operation models after contract synchronization."""
        source = workflow_step if workflow_step is not None else gen
        operation_model = self._build_step_operation_model(source)
        gen["operation_model"] = operation_model
        if workflow_step is not None:
            workflow_step["operation_model"] = operation_model
        step_id = (gen.get("param_signature") or {}).get("workflow_step", "")
        if step_id and isinstance(self._workflow_metadata, dict):
            self._workflow_metadata.setdefault("operation_model", {})[step_id] = operation_model

    def _detect_extension_function_call(self, code: str) -> str:
        """Return an imported extension module function that is called in code."""
        inventory = (
            self._workflow_metadata.get("extension_callable_inventory", {})
            if isinstance(self._workflow_metadata, dict) else {}
        )
        function_names = set(inventory.get("module_functions", []) or [])
        if not function_names:
            return ""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""

        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_name = alias.name
                    local_name = alias.asname or alias.name
                    if imported_name in function_names:
                        imported[local_name] = imported_name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    if local_name in function_names:
                        imported[local_name] = local_name

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in imported:
                    return imported[func.id]
                if isinstance(func, ast.Attribute) and func.attr in function_names:
                    return func.attr
        return ""

    def _template_destructive_ops(self, code: str) -> List[str]:
        """Extract destructive operations using CodeValidator when available."""
        if not self.code_validator:
            from ..CodeValidator import CodeValidator
            self.code_validator = CodeValidator()
        try:
            validation = self.code_validator.validate(code)
            return validation.get("destructive_ops", []) or []
        except Exception:
            return []

    def _syntax_check_templates(self, templates: Dict[str, str]) -> List[Dict[str, str]]:
        """Return syntax issues for Python templates after placeholder filling."""
        issues = []
        for tpl_name, tpl_content in templates.items():
            if not tpl_name.endswith((".py.tpl", ".py")):
                continue
            sample_code = self._fill_remaining_placeholders(
                tpl_content.replace(
                    "{vol_lookup}",
                    "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')",
                )
            )
            try:
                ast.parse(sample_code)
            except SyntaxError as exc:
                issues.append({
                    "template": tpl_name,
                    "error": str(exc),
                })
        return issues

    @staticmethod
    def _template_has_destructive_allow_comment(raw_code: str) -> bool:
        """Return True when a template declares an explicit destructive contract."""
        for line in raw_code.splitlines()[:10]:
            stripped = line.strip().lower()
            if not stripped.startswith("#"):
                continue
            if "allow_destructive_ops" in stripped and "true" in stripped:
                return True
        return False

    @staticmethod
    def _is_display_view_scope_reset(code: str, destructive_ops: List[str], gen: Dict) -> bool:
        """Return True for display-node view-list resets followed by scoped adds."""
        if not destructive_ops:
            return False
        if any("RemoveAllViewNodeIDs" not in op for op in destructive_ops):
            return False
        if "AddViewNodeID" not in code:
            return False
        categories = {
            so.get("slicer_op_category", "")
            for so in (gen.get("sub_operations", []) or [])
            if isinstance(so, dict)
        }
        if categories and not (categories & {"markups_display", "node_display", "layout_slice_view"}):
            return False
        return True

    def _destructive_ops_contract(
        self,
        code: str,
        raw_code: str,
        gen: Dict,
        destructive_ops: List[str],
    ) -> Dict:
        """Build a typed destructive-operation policy decision."""
        explicit = bool(
            gen.get("allow_destructive_ops")
            or self._template_has_destructive_allow_comment(raw_code)
        )
        display_scope_reset = self._is_display_view_scope_reset(code, destructive_ops, gen)
        allowed = explicit or display_scope_reset
        scope = "display_view_scope_reset" if display_scope_reset else "destructive_operation"
        return {
            "allowed": allowed,
            "explicit": explicit,
            "scope": scope,
            "operations": destructive_ops,
            "reason": (
                "Display node view restrictions are cleared before assigning explicit view IDs."
                if display_scope_reset
                else "Template declares allow_destructive_ops."
                if explicit
                else "No destructive operation contract was found."
            ),
        }

    @staticmethod
    def _template_has_meaningful_code(code: str) -> bool:
        """Return False for required templates that only pass/print/comment."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
                continue
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "print"
            ):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            return True
        return False

    @staticmethod
    def _template_assigns_name(code: str, name: str) -> bool:
        """Return True if code assigns a variable name directly."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        return False

    @staticmethod
    def _template_matches_api_evidence(code: str, evidence: Dict) -> bool:
        """Return True if code contains any accepted API evidence footprint."""
        if not isinstance(evidence, dict):
            return False
        code_lower = code.lower()
        footprints = evidence.get("accepted_footprints") or []
        chains = evidence.get("api_chains") or []
        for item in list(footprints) + list(chains):
            if not item:
                continue
            if str(item).lower() in code_lower:
                return True
        return False

    @staticmethod
    def _invalid_slicer_api_aliases(code: str) -> List[str]:
        """Return known invalid Slicer API spellings found in code."""
        aliases = {
            "SetSliceVisibility": "SetSliceVisible",
            "GetSliceVisibility": "GetSliceVisible",
        }
        found = []
        for invalid, replacement in aliases.items():
            if _re.search(rf"\.{_re.escape(invalid)}\s*\(", code):
                found.append(f"{invalid} does not exist; use {replacement}")
        return found

    def _validate_template_contract(
        self,
        tpl_name: str,
        code: str,
        context: Optional[Dict],
        templates: Dict[str, str],
        raw_code: Optional[str] = None,
    ) -> Dict:
        """Validate workflow-level contracts that CodeValidator cannot know."""
        result = {"errors": [], "warnings": []}
        if not context:
            return result

        gen = context.get("generator", {})
        role = context.get("role")
        raw_code = raw_code if raw_code is not None else code
        sub_ops = gen.get("sub_operations", []) or []
        operation_code = self._strip_precondition_regions(code)
        api_chains = self._extract_api_chains(operation_code)
        code_has_slicer_api = bool(api_chains)
        invalid_aliases = self._invalid_slicer_api_aliases(code)
        if invalid_aliases:
            result["errors"].extend(invalid_aliases)
        extension_call_contract = self._validate_extension_callable_contract(code)
        result["errors"].extend(extension_call_contract["errors"])
        for so in sub_ops:
            if so.get("op_type") == "unknown_op":
                result["errors"].append("Required operation has unknown_op classification")
            if so.get("op_type") == "slicer_op" and so.get("confidence") == "low":
                result["errors"].append("Required slicer_op has low classification confidence")
            if (
                so.get("op_type") == "extension_op"
                and not so.get("extension_method_hint")
                and not so.get("extension_function_hint")
                and so.get("operation_intent") not in (
                    "extension_parameter_update",
                    "extension_node_reference_update",
                )
                and code_has_slicer_api
            ):
                result["errors"].append(
                    "extension_op without an extension method contains Slicer API calls; "
                    "classify it as slicer_op or bind it to an extension parameter role"
                )

        operation_model = gen.get("operation_model") or {}
        if code_has_slicer_api and not (
            operation_model.get("invokes_slicer_api")
            or operation_model.get("implementation_uses_slicer_api")
        ):
            has_extension_callable = bool(
                gen.get("method_name")
                or gen.get("extension_function_name")
                or any(so.get("extension_method_hint") for so in sub_ops)
                or any(so.get("extension_function_hint") for so in sub_ops)
            )
            if not has_extension_callable:
                result["errors"].append(
                    "Template uses Slicer API calls but operation_model.invokes_slicer_api is false"
                )

        if self._template_calls_select_module(code) and not self._module_switch_allowed_by_contract(gen, code):
            result["errors"].append(
                "Template switches the active Slicer module without an explicit module_switching contract; "
                "module/panel phrases in cookbook steps are UI-location context and should be implemented "
                "without slicer.util.selectModule"
            )

        semantic_contract = self._validate_slicer_operation_semantics(code, gen)
        result["errors"].extend(semantic_contract["errors"])
        result["warnings"].extend(semantic_contract["warnings"])

        # Revision D: interaction_kind contract — catch view_adjustment
        # templates that misroute into Markups node creation or placement mode.
        interaction_contract = self._validate_interaction_kind_contract(code, gen)
        result["errors"].extend(interaction_contract["errors"])
        result["warnings"].extend(interaction_contract["warnings"])

        # Module-enter precondition contract — catch templates that call
        # extension logic methods without ensuring the module widget is alive.
        module_enter_contract = self._validate_module_enter_precondition(code, gen)
        result["errors"].extend(module_enter_contract["errors"])
        result["warnings"].extend(module_enter_contract["warnings"])

        # ── Sub-operation coverage check ──
        # Verify every non-optional, code-generating sub-operation has a code
        # footprint in the template.  user_interaction and user_choice don't
        # generate code in the same template, so skip them.
        if sub_ops and self._template_has_meaningful_code(code):
            for so in sub_ops:
                if so.get("is_optional"):
                    continue
                so_type = so.get("op_type", "")
                # Skip sub-ops that don't contribute code to this template
                if so_type in ("user_interaction", "user_choice"):
                    continue
                so_desc = (so.get("description") or "").lower()
                so_keywords = [k.lower() for k in (so.get("slicer_api_keywords") or [])]
                so_method = so.get("extension_method_hint") or ""
                # Check if the template references this sub-operation
                found = False
                # Check for extension method call
                if so_method and so_method in code:
                    found = True
                so_function = so.get("extension_function_hint") or ""
                if not found and so_function and so_function in code:
                    found = True
                # Check for comment header referencing the description
                if not found:
                    # Extract significant words from description (>4 chars)
                    desc_words = [w for w in so_desc.split() if len(w) > 4]
                    if desc_words:
                        match_count = sum(1 for w in desc_words if w in code.lower())
                        if match_count >= min(2, len(desc_words)):
                            found = True
                # Check for slicer API keywords
                if not found and so_keywords:
                    for kw in so_keywords:
                        if kw and kw in code.lower():
                            found = True
                            break
                # Prefer per-template API evidence discovered during Stage 5T
                # or contract synchronization over broad category fallbacks.
                if not found and so_type == "slicer_op":
                    evidence = gen.get("api_evidence") or {}
                    if self._template_matches_api_evidence(code, evidence):
                        found = True
                # Check for slicer_op_category-specific API patterns
                if not found and so_type == "slicer_op":
                    category = so.get("slicer_op_category", "")
                    _CATEGORY_API_HINTS = {
                        "layout_slice_view": ["setLayout", "SliceVisible", "SetSliceResolutionMode", "SliceResolutionMatch", "SetViewArrangement", "AddLayoutDescription", "GetLayoutByName", "layoutManager"],
                        "module_switching": ["selectModule", "moduleManager"],
                        "markups_display": ["GetDisplayNode", "SetViewNodeID", "AddViewNodeID"],
                        "crosshair": [
                            "Crosshair", "SetCrosshairMode", "ShowIntersection",
                            "SetCrosshairBehavior", "OffsetJumpSlice", "NoCrosshair",
                            "vtkMRMLSliceDisplayNode", "SliceDisplayNode",
                            "SetIntersectingSlicesVisibility",
                            "GetIntersectingSlicesVisibility",
                            "IntersectingSlicesVisibility",
                        ],
                        "node_display": ["SetSliceVisible", "SetVisibility", "GetDisplayNode"],
                    }
                    hints = _CATEGORY_API_HINTS.get(category, [])
                    code_lower = code.lower()
                    for hint in hints:
                        if hint.lower() in code_lower:
                            found = True
                            break
                if not found:
                    result["errors"].append(
                        f"Sub-operation '{so_desc[:60]}' ({so_type}) has no code in template"
                    )
        node_class = (
            gen.get("interaction_descriptor", {}).get("node_class", "")
            if isinstance(gen.get("interaction_descriptor"), dict)
            else ""
        )
        is_markup = self._is_markup_node_class(node_class)

        if "TODO" in code:
            result["errors"].append("Required template contains TODO")
        unresolved_placeholders = [
            p["name"] for p in self._find_template_placeholders(raw_code)
            if p["name"] != "vol_lookup" and not p["has_default"]
        ]
        if unresolved_placeholders:
            result["errors"].append(
                "Required template contains unresolved placeholders: "
                + ", ".join(unresolved_placeholders)
            )
        allow_instruction_only = role == "pre" and node_class and not is_markup
        if not allow_instruction_only and not self._template_has_meaningful_code(code):
            result["errors"].append("Required template is a stub (only pass/comments/prints)")

        if gen.get("step_type") == "user_choice":
            choice_desc = gen.get("choice_descriptor", {}) or {}
            parameter_name = choice_desc.get("parameter_name", "")
            binding = choice_desc.get("binding")
            metadata_binding = self._workflow_metadata.get("choice_bindings", {}).get(
                gen.get("param_signature", {}).get("workflow_step", ""),
                {},
            )
            choices = choice_desc.get("choices", [])
            is_closed_form = self._choice_is_closed_form(choice_desc)
            is_count_like = self._choice_is_count_like(
                choice_desc,
                {"description": gen.get("description", "")},
            )
            if parameter_name and not binding and not metadata_binding and not is_closed_form and not is_count_like:
                result["warnings"].append(
                    f"User choice '{parameter_name}' has no source-derived parameter binding"
                )

        if "GetNumberOfControlPoints(" in code and node_class and not is_markup:
            result["errors"].append(
                f"Template uses Markups control-point API on non-Markups node class '{node_class}'"
            )
        if "SetActiveListID(" in code and node_class and not is_markup:
            result["errors"].append(
                f"Template enters Markups placement mode for non-Markups node class '{node_class}'"
            )

        if role == "post":
            pre_name = gen.get("pre_template_file")
            pre_code = templates.get(pre_name, "") if pre_name else ""
            for var_name in sorted(set(_re.findall(r"\b(_[A-Za-z0-9]+_cb_step_\d+_id)\b", code))):
                if not pre_code:
                    result["errors"].append(
                        f"Post-template references '{var_name}' but has no pre-template"
                    )
                elif not self._template_assigns_name(pre_code, var_name):
                    result["errors"].append(
                        f"Post-template references '{var_name}' but pre-template does not assign it"
                    )

        interaction_desc = gen.get("interaction_descriptor", {}) or {}
        owner = interaction_desc.get("interaction_owner", "")
        if role == "pre" and owner in ("extension_method", "previous_extension_method"):
            if self._template_creates_markup_node(code):
                result["errors"].append(
                    "Interaction is owned by an extension placement method but pre-template creates a new Markups node"
                )
            if self._template_enters_markup_placement_mode(code):
                result["errors"].append(
                    "Interaction is owned by an extension placement method but pre-template enters Markups placement mode"
                )

        repeat_group = (
            gen.get("repeat_group")
            or interaction_desc.get("repeat_group")
            or (gen.get("choice_descriptor") or {}).get("repeat_group")
            or {}
        )
        step_id = (
            gen.get("param_signature", {}).get("workflow_step", "")
            or gen.get("step_id", "")
        )
        if role == "pre" and repeat_group.get("interaction_step") == step_id:
            if "SwitchToPersistentPlaceMode" in code:
                result["errors"].append(
                    "Repeated interaction pre-template uses persistent placement mode; repeat groups must advance one item per Done"
                )
            if self._template_print_text_has_repeat_instruction(code):
                result["errors"].append(
                    "Repeated interaction pre-template tells the user to repeat inside one wait step"
                )

        scalar_contract = self._validate_scalar_parameter_contract(code)
        result["errors"].extend(scalar_contract["errors"])
        result["warnings"].extend(scalar_contract["warnings"])

        node_class_contract = self._validate_node_class_contract(code)
        result["errors"].extend(node_class_contract["errors"])

        node_requirement_contract = self._validate_node_requirement_contract(code)
        result["errors"].extend(node_requirement_contract["errors"])
        result["warnings"].extend(node_requirement_contract["warnings"])

        return result

    def _validate_extension_callable_contract(self, code: str) -> Dict[str, List[str]]:
        """Validate extension-owned function/method calls against metadata."""
        result = {"errors": []}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return result
        if not isinstance(self._workflow_metadata, dict):
            return result
        inventory = self._workflow_metadata.get("extension_callable_inventory", {}) or {}
        logic_methods = set(inventory.get("logic_methods", []) or [])
        module_function_counts = inventory.get("module_function_param_counts", {}) or {}

        imported_names = set()
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.asname or alias.name.split(".")[0])
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in imported_names and fname in module_function_counts:
                    expected = int(module_function_counts.get(fname, 0) or 0)
                    actual = len(node.args)
                    if actual != expected:
                        result["errors"].append(
                            f"Extension function {fname}() called with {actual} args, expected {expected}"
                        )
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    module_alias = node.func.value.id
                    fname = node.func.attr
                    if module_alias in imported_modules and fname in module_function_counts:
                        expected = int(module_function_counts.get(fname, 0) or 0)
                        actual = len(node.args)
                        if actual != expected:
                            result["errors"].append(
                                f"Extension function {fname}() called with {actual} args, expected {expected}"
                            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (
                isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == "hasattr"
                and len(test.args) >= 2
                and isinstance(test.args[1], ast.Constant)
                and isinstance(test.args[1].value, str)
            ):
                continue
            method_name = test.args[1].value
            if method_name in logic_methods:
                continue
            has_noop_else = False
            for stmt in node.orelse:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                    has_noop_else = True
                elif isinstance(stmt, ast.Pass):
                    has_noop_else = True
            if has_noop_else:
                result["errors"].append(
                    f"Required step silently skips missing logic method '{method_name}'"
                )
        return result
