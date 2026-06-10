from .common import *

_PRECONDITION_BEGIN = "# precondition:begin"
_PRECONDITION_END = "# precondition:end"


class AnalyzerValidationSemanticsMixin:
    @staticmethod
    def _template_calls_select_module(code: str) -> bool:
        return bool(_re.search(r"\bslicer\s*\.\s*util\s*\.\s*selectModule\s*\(", code))

    @staticmethod
    def _template_enters_markup_placement_mode(code: str) -> bool:
        return bool(_re.search(
            r"\b(SwitchToSinglePlaceMode|SwitchToPersistentPlaceMode|StartPlaceMode|SetPlaceModeEnabled)\s*\(",
            code or "",
        ))

    @staticmethod
    def _template_print_text_has_repeat_instruction(code: str) -> bool:
        printed = []
        for match in _re.finditer(r"\bprint\s*\(\s*([rubfRUBF]*)(['\"])(.*?)\2\s*\)", code or "", _re.DOTALL):
            printed.append(match.group(3))
        text = "\n".join(printed).lower()
        return bool(_re.search(
            r"\b(repeat|for each|each requested|requested .* times|continue placing)\b",
            text,
        ))

    def _operation_intents_for_generator(self, gen: Dict) -> List[str]:
        operation_model = gen.get("operation_model") or {}
        intents = list(operation_model.get("operation_intents") or [])
        if intents:
            return sorted(set(intents))
        return self._infer_step_operation_intents(gen)

    @staticmethod
    def _display_scope_categories(gen: Dict) -> List[str]:
        categories = []
        for so in gen.get("sub_operations", []) or []:
            if so.get("op_type") == "slicer_op" and so.get("slicer_op_category"):
                categories.append(_text_or_empty(so.get("slicer_op_category")))
        if gen.get("slicer_op_category"):
            categories.append(_text_or_empty(gen.get("slicer_op_category")))
        return sorted(set(c for c in categories if c))

    @staticmethod
    def _display_scope_text(gen: Dict) -> str:
        parts = [_text_or_empty(gen.get("description"))]
        for so in gen.get("sub_operations", []) or []:
            parts.append(_text_or_empty(so.get("description")))
            parts.extend(_text_list(so.get("slicer_api_keywords", [])))
            parts.append(_text_or_empty(so.get("node_class")))
            parts.append(_text_or_empty(so.get("slicer_op_category")))
        operation_model = gen.get("operation_model") or {}
        parts.extend(_text_list(operation_model.get("operation_intents", [])))
        return " ".join(parts).lower()

    @staticmethod
    def _display_scope_node_class(gen: Dict) -> str:
        for so in gen.get("sub_operations", []) or []:
            node_class = _text_or_empty(so.get("node_class"))
            if node_class:
                return node_class
        descriptor = gen.get("interaction_descriptor") or {}
        if isinstance(descriptor, dict):
            node_class = _text_or_empty(descriptor.get("node_class"))
            if node_class:
                return node_class
        return ""

    @staticmethod
    def _display_scope_targets_slice_view(code: str, gen: Dict) -> bool:
        text = AnalyzerValidationSemanticsMixin._display_scope_text(gen)
        code_text = (code or "").lower()
        return bool(
            _re.search(r"\b(red|green|yellow)\b", text)
            or "slice view" in text
            or "vtkmrmlslicenode" in text
            or _re.search(r"vtkmrmlslicenode(red|green|yellow)?", code_text)
            or _re.search(r"GetSingletonNode\s*\(\s*['\"](Red|Green|Yellow)['\"]", code or "")
        )

    @staticmethod
    def _code_enables_markups_slice_visibility(code: str) -> bool:
        return bool(
            _re.search(r"\b(SetVisibility2D|Visibility2DOn)\s*\(", code or "")
            or _re.search(r"\b(SetSliceProjection|SliceProjectionOn)\s*\(", code or "")
        )

    @staticmethod
    def _code_enables_model_slice_visibility(code: str) -> bool:
        return bool(_re.search(r"\b(SetVisibility2D|Visibility2DOn)\s*\(", code or ""))

    @staticmethod
    def _code_enables_segmentation_slice_visibility(code: str) -> bool:
        return bool(
            _re.search(
                r"\b(SetVisibility2D|Visibility2DOn|SetVisibility2DFill|"
                r"SetVisibility2DOutline|SetSegmentVisibility2D)",
                code or "",
            )
        )

    def _validate_display_view_scope_semantics(self, code: str, gen: Dict, intents: set) -> Dict:
        """Validate that display view filters also enable slice/2D visibility."""
        result = {"errors": [], "warnings": []}
        categories = set(self._display_scope_categories(gen))
        if not (
            "view_display_scope" in intents
            or categories & {"markups_display", "node_display"}
        ):
            return result
        if not self._display_scope_targets_slice_view(code, gen):
            return result

        text = self._display_scope_text(gen)
        node_class = self._display_scope_node_class(gen)
        node_class_l = node_class.lower()
        code_l = (code or "").lower()
        is_markups = (
            "markups_display" in categories
            or "vtkmrmlmarkups" in node_class_l
            or "markups" in text
            or "markups" in code_l
        )
        is_segmentation = "segmentation" in node_class_l or "segmentation" in text
        is_model = (
            "vtkmrmlmodel" in node_class_l
            or "model" in text
            or "modeldisplaynode" in code_l
        )

        if is_markups:
            if not self._code_enables_markups_slice_visibility(code):
                result["errors"].append(
                    "Markups display step targets a slice view but only restricts view IDs; "
                    "expected SetVisibility2D/Visibility2DOn or SetSliceProjection/"
                    "SliceProjectionOn so the markup is actually visible in slice views"
                )
        elif is_segmentation:
            if not self._code_enables_segmentation_slice_visibility(code):
                result["errors"].append(
                    "Segmentation display step targets a slice view but does not enable "
                    "2D fill/outline visibility"
                )
        elif is_model:
            if not self._code_enables_model_slice_visibility(code):
                result["errors"].append(
                    "Model display step targets a slice view but does not enable "
                    "2D visibility"
                )
        else:
            if "AddViewNodeID" in code or "SetViewNodeIDs" in code:
                result["warnings"].append(
                    "Display step targets a slice view; verify the display node class also "
                    "enables the required 2D/slice visibility API"
                )
        return result

    def _extension_function_effects(self, function_name: str) -> List[str]:
        if not function_name or not isinstance(self._workflow_metadata, dict):
            return []
        inventory = self._workflow_metadata.get("extension_callable_inventory", {}) or {}
        effects_by_name = inventory.get("module_function_effects", {}) or {}
        return list(effects_by_name.get(function_name, []) or [])

    @staticmethod
    def _function_name_suggests_layout_activation(function_name: str) -> bool:
        name_l = _text_or_empty(function_name).lower()
        return (
            "layout" in name_l
            and name_l.startswith(("set", "switch", "activate", "restore", "show"))
        )

    def _validate_slicer_operation_semantics(self, code: str, gen: Dict) -> Dict:
        """Validate generic Slicer UI-operation effects beyond API existence."""
        result = {"errors": [], "warnings": []}
        intents = set(self._operation_intents_for_generator(gen))
        if not intents:
            return result

        if "layout_activate" in intents:
            called_function = self._detect_extension_function_call(code)
            called_effects = set(self._extension_function_effects(called_function))
            code_activates_layout = bool(
                _re.search(r"\.\s*setLayout\s*\(", code)
                or _re.search(r"\blayoutManager\s*\([^)]*\)\s*\.\s*setLayout\s*\(", code)
                or "layout_activate" in called_effects
                or (
                    called_function
                    and not called_effects
                    and self._function_name_suggests_layout_activation(called_function)
                )
            )
            code_only_registers_layout = bool(
                "AddLayoutDescription" in code
                or "layout_register" in called_effects
                or (
                    called_function
                    and "layout" in called_function.lower()
                    and called_function.lower().startswith("add")
                    and "layout_activate" not in called_effects
                )
            )
            if not code_activates_layout:
                result["errors"].append(
                    "Layout activation step does not switch the active layout; expected "
                    "layoutManager.setLayout(...) or an extension function with layout_activate effect"
                )
            elif (
                code_only_registers_layout
                and "layout_activate" not in called_effects
                and not _re.search(r"\.\s*setLayout\s*\(", code)
            ):
                result["errors"].append(
                    "Layout activation step only registers a layout; expected active layout switch"
                )

        if "slice_intersection_visibility" in intents:
            uses_app_logic = "SetIntersectingSlicesEnabled" in code
            uses_display_setter = "SetIntersectingSlicesVisibility" in code
            refreshes_slice_nodes = bool("Modified(" in code and "vtkMRMLSliceNode" in code)
            uses_crosshair_only = bool(
                ("vtkMRMLCrosshairNode" in code or "SetCrosshairMode" in code)
                and not uses_app_logic
                and not uses_display_setter
            )
            if uses_crosshair_only:
                result["errors"].append(
                    "Slice intersection visibility step uses crosshair visibility APIs; "
                    "expected slice-intersection state APIs"
                )
            elif not uses_app_logic and not (uses_display_setter and refreshes_slice_nodes):
                result["errors"].append(
                    "Slice intersection visibility step must use applicationLogic()."
                    "SetIntersectingSlicesEnabled(...) or refresh vtkMRMLSliceNode.Modified() "
                    "after direct slice display-node setters"
                )

        display_scope_contract = self._validate_display_view_scope_semantics(code, gen, intents)
        result["errors"].extend(display_scope_contract["errors"])
        result["warnings"].extend(display_scope_contract["warnings"])

        return result

    def _module_switch_allowed_by_contract(self, gen: Dict, code: str = "") -> bool:
        operation_model = gen.get("operation_model") or {}
        if operation_model.get("allow_module_switch"):
            return True
        intents = set(_text_list(gen.get("operation_intents", [])))
        for so in gen.get("sub_operations", []) or []:
            intents.update(_text_list(so.get("operation_intents", [])))
            if so.get("operation_intent"):
                intents.add(so["operation_intent"])
        if "module_switch" in intents:
            return True
        # Precondition exception: a template that imports an extension module
        # is making a logic-method call, and any slicer.util.selectModule() in
        # it is for widget-lifecycle setup (so module.enter() has run), not
        # UI navigation. The original rule targets UI-navigation cookbooks
        # ("go to module X") which don't import the extension's code.
        if code and self._template_imports_extension_module(code):
            return True
        return False

    @staticmethod
    def _template_imports_extension_module(code: str) -> bool:
        """Return True if the template imports a non-standard (extension) module.

        Used to distinguish widget-lifecycle selectModule (allowed) from
        UI-navigation selectModule (blocked). Standard imports (slicer, vtk,
        qt, SlicerAIAgentLib, python stdlib) don't count — only imports of
        third-party / extension modules.
        """
        if not code:
            return False
        _NON_EXTENSION_IMPORTS = {
            "slicer", "SlicerAIAgentLib", "vtk", "qt", "ctk", "logging",
            "os", "sys", "json", "re", "abc", "typing", "functools",
            "collections", "itertools", "math", "time", "copy",
            "pathlib", "io", "traceback", "warnings", "numpy",
        }
        for m in _re.finditer(r'^\s*from\s+([\w.]+)\s+import\s+', code, _re.MULTILINE):
            top = m.group(1).split(".")[0]
            if top not in _NON_EXTENSION_IMPORTS:
                return True
        for m in _re.finditer(r'^\s*import\s+([\w.]+)', code, _re.MULTILINE):
            top = m.group(1).split(".")[0]
            if top not in _NON_EXTENSION_IMPORTS:
                return True
        return False

    @staticmethod
    def _strip_precondition_regions(code: str) -> str:
        """Remove emitted precondition blocks from code for operation-level checks.

        Preconditions (widget-lifecycle setup wrapped in `# precondition:begin` /
        `# precondition:end` marker comments) are not part of the template's
        Slicer API surface. Operation-level checks (e.g. code_has_slicer_api)
        should evaluate only the operation itself, not the emitted setup.
        """
        if not code or _PRECONDITION_BEGIN not in code:
            return code
        out = []
        in_prec = False
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped == _PRECONDITION_BEGIN:
                in_prec = True
                continue
            if stripped == _PRECONDITION_END:
                in_prec = False
                continue
            if not in_prec:
                out.append(line)
        return "\n".join(out)

    @staticmethod
    def _extension_methods_called_by_template(code: str) -> List[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        methods = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "logic"
            ):
                methods.append(func.attr)
        return sorted(set(methods))

    @staticmethod
    def _template_sets_parameter(code: str, role: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = ExtensionCLIAnalyzer._get_call_name(node)
            if not func_name.endswith("SetParameter") or not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and arg0.value == role:
                return True
        return False

    def _validate_scalar_parameter_contract(self, code: str) -> Dict[str, List[str]]:
        """Ensure automated extension method calls have scalar parameter defaults."""
        result = {"errors": [], "warnings": []}
        if not isinstance(self._workflow_metadata, dict):
            return result
        methods = self._extension_methods_called_by_template(code)
        if not methods:
            return result

        bindings = self._workflow_metadata.get("parameter_bindings", {}) or {}
        defaults = self._workflow_metadata.get("parameter_defaults", {}) or {}
        dependencies = self._workflow_metadata.get("parameter_method_dependencies", {}) or {}
        choice_bound_roles = {
            binding.get("parameter_name")
            for binding in (self._workflow_metadata.get("choice_bindings", {}) or {}).values()
            if isinstance(binding, dict)
        }

        for method in methods:
            dep = dependencies.get(method, {}) or {}
            for role in dep.get("parameter_roles", []) or []:
                info = bindings.get(role, {}) or {}
                if info.get("node_class"):
                    continue
                value_types = set(info.get("value_types") or [])
                if not (value_types & {"float", "int", "bool"}):
                    continue
                if (
                    role in defaults
                    or role in choice_bound_roles
                    or self._template_sets_parameter(code, role)
                ):
                    default_info = defaults.get(role, {}) or {}
                    if default_info.get("confidence") == "low":
                        result["warnings"].append(
                            f"Parameter '{role}' for logic.{method}() uses low-confidence "
                            f"{default_info.get('source', 'default')} default {default_info.get('value')!r}"
                        )
                    continue
                result["errors"].append(
                    f"logic.{method}() depends on scalar parameter '{role}' "
                    f"({', '.join(sorted(value_types))}) but no user binding, template assignment, "
                    "or source-derived default is available"
                )
        return result

    def _validate_node_class_contract(self, code: str) -> Dict[str, List[str]]:
        """Detect fallback code that resolves a parameter role with the wrong MRML class."""
        result = {"errors": []}
        if not isinstance(self._workflow_metadata, dict):
            return result
        bindings = self._workflow_metadata.get("parameter_bindings", {}) or {}
        for role, info in bindings.items():
            expected = info.get("node_class", "")
            if not expected or "vtkMRMLMarkups" not in expected or role not in code:
                continue
            classes = set(_re.findall(r"['\"](vtkMRMLMarkups[A-Za-z0-9_]+Node)['\"]", code))
            mismatches = sorted(cls for cls in classes if cls != expected)
            if mismatches and expected not in classes:
                result["errors"].append(
                    f"Template references parameter role '{role}' with node class "
                    f"{', '.join(mismatches)} but metadata expects {expected}"
                )
        return result

    def _validate_node_requirement_contract(self, code: str) -> Dict[str, List[str]]:
        """Validate generated pre-call node-reference requirement metadata."""
        result = {"errors": [], "warnings": []}
        if not isinstance(self._workflow_metadata, dict):
            return result
        dependencies = self._workflow_metadata.get("parameter_method_dependencies", {}) or {}
        defaults = self._workflow_metadata.get("parameter_defaults", {}) or {}
        valid_kinds = {"required", "conditional", "produced_by_method", "optional_unknown"}
        for method in self._extension_methods_called_by_template(code):
            requirements = (dependencies.get(method, {}) or {}).get("node_requirements", {}) or {}
            for role, requirement in requirements.items():
                kind = requirement.get("requirement", "optional_unknown")
                conditions = requirement.get("conditions") or []
                if kind not in valid_kinds:
                    result["errors"].append(
                        f"Node reference '{role}' for logic.{method}() has invalid requirement kind '{kind}'"
                    )
                    continue
                if kind == "conditional" and not conditions:
                    result["warnings"].append(
                        f"Node reference '{role}' for logic.{method}() is conditional but has no "
                        "runtime-evaluable condition; missing-reference validation will be skipped"
                    )
                for condition in conditions:
                    parameter = condition.get("parameter", "")
                    if not parameter:
                        result["warnings"].append(
                            f"Node reference '{role}' for logic.{method}() has a condition without a parameter"
                        )
                    elif parameter not in defaults:
                        result["warnings"].append(
                            f"Conditional node reference '{role}' for logic.{method}() depends on "
                            f"parameter '{parameter}' without a source-derived default"
                        )
        return result

    @staticmethod
    def _template_creates_markup_node(code: str) -> bool:
        """Return True when code creates a vtkMRMLMarkups* node directly."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name not in ("CreateNodeByClass", "AddNewNodeByClass"):
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                if arg0.value.startswith("vtkMRMLMarkups"):
                    return True
        return False

    @staticmethod
    def _template_creates_any_node(code: str) -> bool:
        """Return True when code creates any MRML node via CreateNodeByClass/AddNewNodeByClass."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in ("CreateNodeByClass", "AddNewNodeByClass"):
                return True
        return False

    @staticmethod
    def _template_enters_place_mode(code: str) -> bool:
        """Return True when code calls SwitchToPersistentPlaceMode / StartPlaceMode / SetCurrentInteractionMode."""
        return any(
            call in code
            for call in (
                "SwitchToPersistentPlaceMode",
                "StartPlaceMode",
                "SetCurrentInteractionMode",
            )
        )

    @staticmethod
    def _template_references_markups_class(code: str) -> bool:
        """Return True when code references any vtkMRMLMarkups*Node class string."""
        return bool(_re.search(r"['\"]vtkMRMLMarkups[A-Za-z0-9_]*Node['\"]", code))

    def _validate_interaction_kind_contract(self, code: str, gen: Dict) -> Dict[str, List[str]]:
        """Revision D: catch the original bug class (view_adjustment descriptor
        but template creates a Markups node and enters placement mode) at
        generation time, not at every runtime step.

        Rules (errors block manifest validation):
          - interaction_kind == "view_adjustment" AND template creates any
            MRML node → misrouted dispatch.
          - interaction_kind == "view_adjustment" AND template references a
            vtkMRMLMarkups*Node string → misrouted dispatch.
          - creates_node == False (explicit) AND template creates any node
            → descriptor/template mismatch.
          - requires_place_mode == False (explicit) AND template enters
            placement mode → descriptor/template mismatch.
        """
        result: Dict[str, List[str]] = {"errors": [], "warnings": []}
        if not isinstance(gen, dict):
            return result
        sub_ops = gen.get("sub_operations") or []
        # Find the user_interaction sub-op driving this template, if any.
        interaction_so = None
        for so in sub_ops:
            if so.get("op_type") == "user_interaction":
                interaction_so = so
                break
        # If no explicit user_interaction sub-op, fall back to top-level
        # generator fields (some generators carry interaction metadata there).
        interaction_kind = (
            (interaction_so or {}).get("interaction_kind")
            or gen.get("interaction_kind")
            or ""
        )
        creates_node = (
            (interaction_so or {}).get("creates_node")
            if interaction_so is not None
            else gen.get("creates_node")
        )
        requires_place_mode = (
            (interaction_so or {}).get("requires_place_mode")
            if interaction_so is not None
            else gen.get("requires_place_mode")
        )
        if not interaction_kind and creates_node is None and requires_place_mode is None:
            return result  # not an interaction template

        tpl_label = gen.get("template_file") or gen.get("pre_template_file") or gen.get("post_template_file") or "<unknown>"
        creates_any = self._template_creates_any_node(code)
        enters_place = self._template_enters_place_mode(code)
        refs_markups = self._template_references_markups_class(code)

        if interaction_kind == "view_adjustment":
            if creates_any:
                result["errors"].append(
                    f"view_adjustment template '{tpl_label}' creates an MRML node; "
                    "dispatch misrouted a viewport/handle-drag interaction into node creation"
                )
            if refs_markups:
                result["errors"].append(
                    f"view_adjustment template '{tpl_label}' references a vtkMRMLMarkups*Node; "
                    "dispatch misrouted a viewport/handle-drag interaction into Markups placement"
                )
            if enters_place:
                result["errors"].append(
                    f"view_adjustment template '{tpl_label}' enters placement mode; "
                    "this hijacks the mouse for viewport/handle-drag interactions"
                )
        # creates_node/requires_place_mode checks fire only when the descriptor
        # explicitly said False — missing values (None) are advisory, not contracts.
        if creates_node is False and creates_any:
            result["errors"].append(
                f"Template '{tpl_label}' creates a node but the step descriptor "
                "declared creates_node=false"
            )
        if requires_place_mode is False and enters_place:
            result["errors"].append(
                f"Template '{tpl_label}' enters placement mode but the step descriptor "
                "declared requires_place_mode=false"
            )
        return result

    def _validate_module_enter_precondition(self, code: str, gen: Dict) -> Dict[str, List[str]]:
        """Catch templates that call extension logic methods without ensuring
        the module is active. Extension methods assume module.enter() has run
        (parameter node init, observers, UI bindings); without
        slicer.util.selectModule(), the widget lifecycle never fires and the
        method silently misbehaves (missing UI, missing observers, etc.).

        Detection: the template imports an extension module (anything outside
        the standard Slicer/python import set) AND does not contain a
        slicer.util.selectModule() call. The check is intentionally general —
        it does not encode any specific module name.
        """
        result: Dict[str, List[str]] = {"errors": [], "warnings": []}
        if not isinstance(gen, dict) or not isinstance(code, str) or not code:
            return result

        _NON_EXTENSION_IMPORTS = {
            "slicer", "SlicerAIAgentLib", "vtk", "qt", "ctk", "logging",
            "os", "sys", "json", "re", "abc", "typing", "functools",
            "collections", "itertools", "math", "time", "copy",
            "pathlib", "io", "traceback", "warnings", "numpy",
        }
        extension_modules = set()
        for m in _re.finditer(r'^\s*from\s+([\w.]+)\s+import\s+', code, _re.MULTILINE):
            top = m.group(1).split(".")[0]
            if top not in _NON_EXTENSION_IMPORTS:
                extension_modules.add(top)
        for m in _re.finditer(r'^\s*import\s+([\w.]+)', code, _re.MULTILINE):
            top = m.group(1).split(".")[0]
            if top not in _NON_EXTENSION_IMPORTS:
                extension_modules.add(top)

        if not extension_modules:
            return result  # no extension import — not a logic-calling template

        if "slicer.util.selectModule(" in code:
            return result  # precondition present

        tpl_label = (
            gen.get("template_file")
            or gen.get("pre_template_file")
            or gen.get("post_template_file")
            or "<unknown>"
        )
        imported = ", ".join(sorted(extension_modules))
        result["errors"].append(
            f"Template '{tpl_label}' imports extension module(s) {imported} "
            "but does not call slicer.util.selectModule() to ensure the module is active; "
            "extension logic methods require module.enter() to have run (parameter node init, "
            "observers, UI bindings)"
        )
        return result

    @staticmethod
    def _find_template_placeholders(template_str: str) -> List[Dict[str, Any]]:
        """Find single-brace template placeholders outside Python strings."""
        string_ranges = []
        for m in _re.finditer(
            r'(?:[fFrRbBuU]{0,2})("""|\'\'\'|"|\')(.*?)\1',
            template_str,
            _re.DOTALL,
        ):
            string_ranges.append((m.start(), m.end()))

        def _in_string(pos: int) -> bool:
            return any(start <= pos < end for start, end in string_ranges)

        placeholders = []
        i = 0
        while i < len(template_str):
            if template_str.startswith("{{", i):
                i += 2
                continue
            if template_str[i] != "{" or _in_string(i):
                i += 1
                continue
            depth = 0
            j = i
            found = False
            while j < len(template_str):
                if template_str[j] == "{":
                    depth += 1
                elif template_str[j] == "}":
                    depth -= 1
                    if depth == 0:
                        found = True
                        break
                j += 1
            if found:
                inner = template_str[i + 1:j]
                has_default = ":" in inner
                name = inner.split(":", 1)[0].strip()
                if name.isidentifier():
                    placeholders.append({"name": name, "has_default": has_default})
                i = j + 1
            else:
                i += 1
        deduped = {}
        for placeholder in placeholders:
            name = placeholder["name"]
            deduped[name] = {
                "name": name,
                "has_default": deduped.get(name, {}).get("has_default", False)
                or placeholder["has_default"],
            }
        return [deduped[name] for name in sorted(deduped)]

    def _semantic_validate(self, code: str, logic_analysis: Dict,
                           api_probe_result: Optional[Dict] = None) -> Dict:
        """Check for undefined variables, wrong arg counts, invalid node types,
        and cross-reference API chains against live probe failures."""
        result = {"errors": [], "warnings": []}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result["errors"].append("Syntax error in generated code")
            return result

        # Collect defined names (assignments, imports, function/class defs, for-loop targets)
        defined = set()
        # All Python builtins (functions, constants, exceptions, types)
        import builtins as _builtins
        defined.update(name for name in dir(_builtins) if not name.startswith("_"))
        # Slicer-runtime names that are always available but not in builtins
        defined.update({
            "slicer", "qt", "vtk", "ctk", "inputVolume", "logic",
            "json", "math", "time", "path",
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
                elif isinstance(node.target, (ast.Tuple, ast.List)):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
            elif isinstance(node, ast.comprehension):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
                elif isinstance(node.target, (ast.Tuple, ast.List)):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
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
            "vtkMRMLCrosshair", "vtkMRMLScriptedModule",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in ("CreateNodeByClass", "AddNewNodeByClass"):
                    for arg in node.args[:1]:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            cls = arg.value
                            if not cls.startswith(valid_prefixes):
                                result["warnings"].append(
                                    f"Unknown MRML node class: '{cls}'"
                                )

        # Cross-check API chains against live probe failures
        if api_probe_result and api_probe_result.get("failures"):
            failed_chains = {f.get("chain", "") for f in api_probe_result["failures"]}
            template_chains = ExtensionCLIAnalyzer._extract_api_chains(code)
            for chain in template_chains:
                for failed in failed_chains:
                    if chain == failed or chain.startswith(failed + "."):
                        result["warnings"].append(
                            f"API chain '{chain}' was flagged by live probe as potentially invalid"
                        )
                        break

        return result

    @staticmethod
    def _fill_remaining_placeholders(code: str) -> str:
        """Fill remaining template placeholders outside Python strings."""
        string_ranges = []
        for match in _re.finditer(
            r'(?:[fFrRbBuU]{0,2})("""|\'\'\'|"|\')(.*?)\1',
            code,
            _re.DOTALL,
        ):
            string_ranges.append((match.start(), match.end()))

        def _string_end_at(pos: int) -> Optional[int]:
            for start, end in string_ranges:
                if start <= pos < end:
                    return end
            return None

        def _sample_value(name: str) -> str:
            lower = name.lower()
            if "name" in lower:
                return '"SampleNode"'
            if "radius" in lower or "size" in lower or "distance" in lower:
                return "1.5"
            if "path" in lower:
                return '"/tmp/sample"'
            # Boolean-like placeholders — emit True so the assembled code parses
            # even when the template expects a bool/checked value.
            if any(k in lower for k in (
                "checked", "enabled", "visible", "active", "flag",
                "show_", "_show", "use_", "_use", "is_", "_is",
            )):
                return "True"
            # Integer counts/indices
            if any(k in lower for k in ("count", "number", "index", "num_")):
                return "1"
            # Choice/selection values
            if any(k in lower for k in ("choice", "selection", "option", "value", "selected")):
                return '"sample"'
            # MRML node IDs
            if lower.endswith("_id") or lower.endswith("id") and "node" in lower:
                return '"vtkMRMLSampleNode1"'
            return '""'

        result = []
        i = 0
        n = len(code)
        while i < n:
            string_end = _string_end_at(i)
            if string_end is not None:
                result.append(code[i:string_end])
                i = string_end
                continue

            if code.startswith("{{", i) or code.startswith("}}", i):
                result.append(code[i:i + 2])
                i += 2
                continue

            if code[i] != "{":
                result.append(code[i])
                i += 1
                continue

            depth = 0
            j = i
            found = False
            while j < n:
                if code[j] == "{":
                    depth += 1
                elif code[j] == "}":
                    depth -= 1
                    if depth == 0:
                        found = True
                        break
                j += 1

            if not found:
                result.append(code[i])
                i += 1
                continue

            inner = code[i + 1:j]
            colon_pos = inner.find(":")
            if colon_pos >= 0 and inner[:colon_pos].strip().isidentifier():
                name = inner[:colon_pos].strip()
                default = inner[colon_pos + 1:]
                if default.startswith(" "):
                    default = default[1:]
                result.append(default)
            elif inner.strip().isidentifier():
                result.append(_sample_value(inner.strip()))
            else:
                result.append(code[i:j + 1])
            i = j + 1

        return "".join(result)

    # ================================================================
