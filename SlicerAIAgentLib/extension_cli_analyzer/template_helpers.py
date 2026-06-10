from .common import *

# CJK fullwidth ASCII range (U+FF01–U+FF5E) → ASCII (U+0021–U+007E)
_FULLWIDTH_TO_ASCII = str.maketrans(
    {chr(i): chr(i - 0xFEE0) for i in range(0xFF01, 0xFF5F)}
)

_GETATTR_LITERAL_RE = _re.compile(
    r'\bgetattr\s*\(\s*([^,]+?)\s*,\s*([\'"])([A-Za-z_]\w*)\2\s*\)'
)
# Matches setattr(obj, 'attr', <simple_value>) where the value has no
# nested parentheses.  Complex expressions like setattr(o, 'a', f(x))
# are left for the auto-revise LLM to handle.
_SETATTR_LITERAL_RE = _re.compile(
    r'\bsetattr\s*\(\s*([^,]+?)\s*,\s*([\'"])([A-Za-z_]\w*)\2\s*,\s*([^)]+?)\s*\)'
)


class AnalyzerTemplateHelpersMixin:
    @staticmethod
    def _is_markup_node_class(node_class: str) -> bool:
        """Return True for MRML Markups nodes that support placement/control points."""
        return _text_or_empty(node_class).startswith("vtkMRMLMarkups")

    @staticmethod
    def _python_comment_block(text: str, prefix: str = "# ") -> List[str]:
        """Convert arbitrary multi-line text into valid Python comment lines."""
        lines = []
        for raw_line in _text_or_empty(text).splitlines() or [""]:
            line = raw_line.strip()
            lines.append(f"{prefix}{line}" if line else prefix.rstrip())
        return lines

    def _template_header_lines(self, extension_name: str, step: Dict, phase: str = "") -> List[str]:
        """Return a Python-safe multi-line template header for a workflow step."""
        step_id = step.get("step_id", "")
        description = step.get("description", step_id)
        suffix = f" ({phase})" if phase else ""
        header_text = f"--- {extension_name}: {description}{suffix} ---"
        return self._python_comment_block(header_text)

    @staticmethod
    def _emit_module_enter_precondition(module_name: str) -> List[str]:
        """Emit Python lines that ensure the extension's Slicer module is active.

        Why: extension logic methods assume module.enter() has run (parameter
        node init, observers, UI bindings). The agent runtime imports the
        Logic class and calls methods directly, bypassing the module widget
        lifecycle, so enter() never fires. slicer.util.selectModule() triggers
        the full lifecycle (instantiate widget, call enter()). The guard is
        idempotent — cheap name check skips the call when already active.
        """
        if not module_name:
            return []
        return [
            "# precondition:begin",
            "# Ensure the extension module is active so module.enter() has run.",
            "_active_module = slicer.app.moduleManager().activeModule()",
            f"if _active_module is None or _active_module.name != {module_name!r}:",
            "    try:",
            f"        slicer.util.selectModule({module_name!r})",
            "    except Exception as _module_enter_error:",
            f"        print(f\"Warning: could not activate module {module_name!r}: {{_module_enter_error}}\")",
            "# precondition:end",
            "",
        ]

    @staticmethod
    def _repair_multiline_comment_headers(code: str) -> str:
        """Prefix accidentally split template-header continuation lines as comments."""
        repaired = []
        in_header = False
        for line in _text_or_empty(code).splitlines():
            stripped = line.strip()
            if stripped.startswith("# ---") and not stripped.endswith("---"):
                in_header = True
                repaired.append(line)
                continue
            if in_header:
                if stripped and not stripped.startswith("#"):
                    indent = line[:len(line) - len(line.lstrip())]
                    repaired.append(f"{indent}# {stripped}")
                else:
                    repaired.append(line)
                if stripped.endswith("---"):
                    in_header = False
                continue
            repaired.append(line)
        if code.endswith("\n"):
            return "\n".join(repaired) + "\n"
        return "\n".join(repaired)

    @staticmethod
    def _placement_mode_from_source(source: str) -> str:
        """Infer the placement mode a source snippet explicitly enters."""
        source = source or ""
        if "SwitchToSinglePlaceMode" in source:
            return "single"
        if "SwitchToPersistentPlaceMode" in source:
            return "persistent"
        if "StartPlaceMode" in source or "SetPlaceModeEnabled" in source:
            return "unknown"
        return "none"

    def _placement_starter_info(self, method_name: str) -> Dict:
        """Return source-derived placement-starter metadata for a method."""
        if not method_name:
            return {}
        return self._placement_starter_methods.get(method_name) or {}

    @staticmethod
    def _is_repeat_interaction_step(step: Dict) -> bool:
        """Return True when this interaction step is controlled by a repeat group."""
        repeat_group = step.get("repeat_group") or {}
        return bool(
            repeat_group
            and repeat_group.get("interaction_step")
            and repeat_group.get("interaction_step") == step.get("step_id")
        )

    def _interaction_object_name(self, step: Dict) -> str:
        """Return a short generic object name for user-facing placement text."""
        interaction_type = _text_or_empty(step.get("interaction_type", "")).lower()
        if interaction_type and interaction_type != "unknown":
            return interaction_type
        node_class = self._step_interaction_node_class(step)
        mapping = {
            "vtkMRMLMarkupsPlaneNode": "plane",
            "vtkMRMLMarkupsCurveNode": "curve",
            "vtkMRMLMarkupsLineNode": "line",
            "vtkMRMLMarkupsFiducialNode": "point",
        }
        return mapping.get(node_class, "placement")

    def _normalize_repeat_interaction_instructions(self, step: Dict) -> None:
        """Make repeated interaction steps describe one runtime iteration only."""
        if not self._is_repeat_interaction_step(step):
            return
        object_name = self._interaction_object_name(step)
        if object_name == "placement":
            instruction = "Complete this placement, then click Done."
        else:
            instruction = f"Place this {object_name}, then click Done."
        step["placement_instructions"] = instruction
        for so in step.get("sub_operations", []) or []:
            if so.get("op_type") == "user_interaction":
                so["placement_instructions"] = instruction

    def _interaction_instructions_for_template(self, step: Dict) -> str:
        """Return user-facing instructions with repeat semantics applied."""
        if self._is_repeat_interaction_step(step):
            object_name = self._interaction_object_name(step)
            if object_name == "placement":
                return "Complete this placement, then click Done."
            return f"Place this {object_name}, then click Done."
        return step.get("placement_instructions", step.get("description", ""))

    def _placement_mode_policy(self, step: Dict, starter_info: Optional[Dict] = None) -> Dict:
        """Decide how generated code should enter Markups placement mode."""
        owner = step.get("interaction_owner", "")
        starter_info = starter_info or {}
        if owner in ("extension_method", "previous_extension_method"):
            if starter_info.get("starts_markup_placement"):
                return {
                    "should_set_active_list": False,
                    "should_enter_placement_mode": False,
                    "placement_mode": None,
                    "reason": "extension_starter_already_controls_placement",
                }
            return {
                "should_set_active_list": True,
                "should_enter_placement_mode": True,
                "placement_mode": "single" if self._is_repeat_interaction_step(step) else "persistent",
                "reason": "extension_starter_did_not_enter_placement",
            }
        if self._is_repeat_interaction_step(step):
            return {
                "should_set_active_list": True,
                "should_enter_placement_mode": True,
                "placement_mode": "single",
                "reason": "repeat_group_one_item_per_runtime_iteration",
            }
        return {
            "should_set_active_list": True,
            "should_enter_placement_mode": True,
            "placement_mode": "persistent",
            "reason": "runtime_template_controls_placement",
        }

    @staticmethod
    def _placement_mode_code(policy: Dict) -> List[str]:
        """Return Python lines that enter the policy-selected placement mode."""
        if not policy.get("should_enter_placement_mode"):
            return []
        mode = policy.get("placement_mode")
        if mode == "single":
            return ["    interactionNode.SwitchToSinglePlaceMode()"]
        return ["    interactionNode.SwitchToPersistentPlaceMode()"]

    def _generate_existing_placement_pre_template(
        self, extension_name, step, starter_method,
    ) -> str:
        """Reuse the markup node created by the previous placement-starter step."""
        step_id = step.get("step_id", "")
        node_class = step.get("node_class", "vtkMRMLMarkupsFiducialNode")
        instructions = self._interaction_instructions_for_template(step)
        node_var = f"_{extension_name.lower()}_{step_id}_id"
        starter_info = self._placement_starter_info(starter_method)
        policy = self._placement_mode_policy(step, starter_info)

        lines = [
            *self._template_header_lines(extension_name, step, "Setup"),
            "import slicer",
            "from SlicerAIAgentLib.workflow_state import remember_interaction_node",
            "",
            f"# Reuse the markup node created by {starter_method}() in the previous step.",
            f"nodes = slicer.mrmlScene.GetNodesByClass(\"{node_class}\")",
            "node = None",
            "for i in range(nodes.GetNumberOfItems() - 1, -1, -1):",
            "    candidate = nodes.GetItemAsObject(i)",
            "    if candidate is not None:",
            "        node = candidate",
            "        break",
            "if node is None:",
            f"    raise RuntimeError(\"No {node_class} found from previous placement step.\")",
            "",
            "displayNode = node.GetDisplayNode()",
            "if displayNode is not None:",
            "    displayNode.SetVisibility(True)",
        ]
        if policy.get("should_set_active_list"):
            lines.append("slicer.modules.markups.logic().SetActiveListID(node)")
        if policy.get("should_enter_placement_mode"):
            lines.extend([
                "interactionNode = slicer.mrmlScene.GetNodeByID(\"vtkMRMLInteractionNodeSingleton\")",
                "if interactionNode is not None:",
                *self._placement_mode_code(policy),
            ])
        lines.extend([
            f"{node_var} = node.GetID()",
            (
                "remember_interaction_node("
                f"_workflow_runtime_extension, _workflow_runtime_id, \"{step_id}\", "
                f"{node_var}, _workflow_runtime_repeat_index)"
            ),
            "",
            f"print(\"[{extension_name}] Please {instructions}\")",
            "print(\"When finished, press the 'Done' button in the workflow panel.\")",
        ])
        return "\n".join(lines) + "\n"

    def _generate_view_adjustment_pre_template(self, extension_name, step) -> str:
        """Generate setup for interactions that do not create markups nodes."""
        step_id = step.get("step_id", "")
        instructions = step.get("placement_instructions", step.get("description", ""))
        lines = [
            *self._template_header_lines(extension_name, step, "Setup"),
            "import slicer",
            "",
            "# This step is a view adjustment, not a Markups placement.",
            f"print(\"[{extension_name}] Please {instructions}\")",
            "print(\"When finished, press the 'Done' button in the workflow panel.\")",
        ]
        return "\n".join(lines) + "\n"

    def _generate_view_adjustment_post_template(self, extension_name, step) -> str:
        """Generate completion code for non-markup interactions."""
        step_id = step.get("step_id", "")
        lines = [
            *self._template_header_lines(extension_name, step, "Done"),
            "import slicer",
            "",
            "interactionNode = slicer.mrmlScene.GetNodeByID(\"vtkMRMLInteractionNodeSingleton\")",
            "if interactionNode is not None:",
            "    interactionNode.SwitchToViewTransformMode()",
            "",
            f"print(\"[{extension_name}] Step '{step_id}' view adjustment completed.\")",
        ]
        return "\n".join(lines) + "\n"

    def _generate_pre_interaction_template(
        self, extension_name, step, logic_class_name, module_name,
    ) -> str:
        """Generate the pre-interaction template for an interactive step."""
        interaction_type = step.get("interaction_type", "unknown")
        node_class = step.get("node_class", "vtkMRMLMarkupsFiducialNode")
        instructions = self._interaction_instructions_for_template(step)
        node_name = step["step_id"].replace("_", " ").title()
        min_points = step.get("min_control_points", 0)
        policy = self._placement_mode_policy(step)

        lines = [
            *self._template_header_lines(extension_name, step, "Setup"),
            "import slicer",
            "from SlicerAIAgentLib.workflow_state import remember_interaction_node",
            "",
            "# Create the markup node for user interaction",
            f"node = slicer.mrmlScene.AddNewNodeByClass(\"{node_class}\", \"{node_name}\")",
            "displayNode = node.GetDisplayNode()",
            "if displayNode is not None:",
            "    displayNode.SetVisibility(True)",
            "",
            f"print(\"[{extension_name}] Please {instructions}\")",
            "print(\"When finished, press the 'Done' button in the workflow panel.\")",
            "",
            "# Enter placement mode",
        ]
        if policy.get("should_set_active_list"):
            lines.append("slicer.modules.markups.logic().SetActiveListID(node)")
        if policy.get("should_enter_placement_mode"):
            lines.extend([
                "interactionNode = slicer.mrmlScene.GetNodeByID(\"vtkMRMLInteractionNodeSingleton\")",
                "if interactionNode is not None:",
                *self._placement_mode_code(policy),
            ])
        lines.extend([
            "",
            f"_{extension_name.lower()}_{step['step_id']}_id = node.GetID()",
            (
                "remember_interaction_node("
                f"_workflow_runtime_extension, _workflow_runtime_id, \"{step['step_id']}\", "
                f"_{extension_name.lower()}_{step['step_id']}_id, _workflow_runtime_repeat_index)"
            ),
        ])

        return "\n".join(lines) + "\n"

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

        lines = [
            *self._template_header_lines(extension_name, step, "Process"),
            "import slicer",
            "from SlicerAIAgentLib.workflow_state import resolve_interaction_node",
            "",
            (
                "node = resolve_interaction_node("
                f"_workflow_runtime_extension, _workflow_runtime_id, \"{step['step_id']}\", "
                f"\"{step.get('node_class', '')}\", _workflow_runtime_repeat_index)"
            ),
            "if node is None:",
            f"    node = slicer.mrmlScene.GetNodeByID({node_var})",
            "if node is None:",
            f"    raise RuntimeError(\"Node not found for step '{step['step_id']}'\")",
            "",
        ]

        if min_points > 0:
            lines += [
                "# Validate user input",
                "numPoints = node.GetNumberOfControlPoints()",
                f"if numPoints < {min_points}:",
                f"    raise RuntimeError(\"Need at least {min_points} control points, got %d. Please add more.\" % numPoints)",
                "",
            ]

        if step.get("reactive_chains"):
            for chain in step["reactive_chains"]:
                lines.append(f"# Reactive chain: {chain.get('recompute_description', '')}")

        parameter_role = (
            step.get("parameter_role")
            or (step.get("interaction_binding") or {}).get("parameter_name", "")
        )
        if parameter_role:
            lines += [
                "# Store the placed node on the extension parameter node for later steps",
                f"from {module_name} import {logic_class_name}",
                *self._emit_module_enter_precondition(module_name),
                "try:",
                f"    logic = _{extension_name.lower()}_logic",
                "except NameError:",
                f"    logic = {logic_class_name}()",
                "parameterNode = logic.getParameterNode()",
                f"parameterNode.SetNodeReferenceID(\"{parameter_role}\", node.GetID())",
                f"_{extension_name.lower()}_logic = logic",
                "",
            ]

        lines += [
            "# Exit placement mode",
            "interactionNode = slicer.mrmlScene.GetNodeByID(\"vtkMRMLInteractionNodeSingleton\")",
            "interactionNode.SwitchToViewTransformMode()",
            "",
            f"print(\"[{extension_name}] Step '{step['step_id']}' processed with %d control points.\" % node.GetNumberOfControlPoints())",
        ]

        return "\n".join(lines) + "\n"

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

        # Method source sent in full (no truncation)
        # if len(method_source) > 5000:
        #     method_source = method_source[:5000] + "\n# ... [truncated]"

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

            Context: The user just placed control points on a markup node. The node ID is stored in workflow runtime state and may also be available in variable `{node_var}` as a fallback.
            Parameter-node role for this interaction, if any: `{step.get('parameter_role') or (step.get('interaction_binding') or {}).get('parameter_name', '')}`.

            The code must:
            1. Import the logic class from `{module_name}`
            2. Import `resolve_interaction_node` from `SlicerAIAgentLib.workflow_state`, then retrieve the markup node with:
               `node = resolve_interaction_node(_workflow_runtime_extension, _workflow_runtime_id, "{step.get('step_id', '')}", "{node_class}", _workflow_runtime_repeat_index)`.
               If that returns None, fall back to `slicer.mrmlScene.GetNodeByID({node_var})`.
            3. Validate the user placed enough control points ({min_points} minimum)
            4. Ensure the extension module is active so module.enter() has run (extension methods may depend on parameter-node init, observers, and UI bindings set up by the widget's enter() method). Emit (include the marker comments exactly):
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
            5. Reuse the existing logic instance `_{extension_name.lower()}_logic` if it exists in `dir()`, otherwise create a new `{logic_class_name}()`
            6. Set up any required state on the logic instance BEFORE calling the method (e.g., if the method reads `self.inputMarkupNode`, assign the retrieved node to it)
            7. Call the method `{method_name}()` with correct arguments — pass the markup node if the method expects it
            8. Exit placement mode: `interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")` then `interactionNode.SwitchToViewTransformMode()`
            9. Store the logic instance as `_{extension_name.lower()}_logic` for subsequent steps
            10. Print a completion message with the number of control points
            11. If a non-empty parameter-node role is listed above, call `logic.getParameterNode().SetNodeReferenceID(role, node.GetID())` before later steps need that node.

            IMPORTANT restrictions:
            - Do NOT use `dir()`, `eval()`, `exec()`, `globals()`, or `locals()` — these are blocked in the execution sandbox.
            - Use `try/except NameError` to check if a variable exists, NOT `if 'var' in dir()`.
            - Do NOT use curly brace template placeholders. Write actual Python values (strings, numbers, etc.). If you need a node name, use a hardcoded lookup like `slicer.util.getNode('NodeName')`.
            - Escape all braces in f-strings and .format() calls by doubling them: use doubled-braces for literal braces in output strings.
            - Return ONLY raw Python code. Do NOT wrap it in markdown fences (```python ... ```).""")

        try:
            for _attempt in range(2):
                response = self._call_llm(prompt)
                response = self._strip_markdown_fences(response) if response else None
                if not response or "import" not in response:
                    break
                # Validate syntax immediately — retry once on failure
                import ast as _ast
                try:
                    _ast.parse(response)
                    return response
                except (SyntaxError, IndentationError) as e:
                    if _attempt == 0:
                        logger.info(
                            "LLM post-interaction template for step %s had syntax error: %s. Retrying...",
                            step.get("step_id", "?"), e,
                        )
                        prompt += (
                            f"\n\nYour previous output had a syntax error: {e}\n"
                            "Output ONLY the corrected Python code, no explanation."
                        )
                    else:
                        logger.warning(
                            "LLM post-interaction template for step %s still has syntax error after retry: %s",
                            step.get("step_id", "?"), e,
                        )
                        return response
        except Exception:
            logger.debug("LLM post-interaction template generation failed", exc_info=True)
        return None

    @staticmethod
    def _sanitize_templates(templates: Dict[str, str]) -> Dict[str, str]:
        """Post-generation sanitization of code templates.

        Fixes common LLM output issues that would cause Stage 9 validation
        failures:
        1. Null bytes in generated code
        2. Blocked module imports (sys, os, subprocess, etc.)
        3. Trailing whitespace / mixed line endings
        4. Unexpected indentation (LLM returns indented blocks)
        5. Empty method calls like ``logic.()`` from null method hints
        6. Multi-line Python comment headers accidentally split by raw descriptions

        This runs before Stage 9 validation and is also applied to LLM review
        and revision outputs, so cheap syntax repairs do not require another
        LLM pass.
        """
        import ast as _ast
        import textwrap as _textwrap

        # Blocked imports — mirror CodeValidator's list
        _BLOCKED_MODULES = {
            "os", "sys", "subprocess", "socket", "shutil",
            "pathlib", "signal", "ctypes", "multiprocessing",
        }
        _BLOCKED_IMPORT_RE = _re.compile(
            r'^(\s*)import\s+(' + "|".join(_BLOCKED_MODULES) + r')\b.*$',
            _re.MULTILINE,
        )
        _BLOCKED_FROM_IMPORT_RE = _re.compile(
            r'^(\s*)from\s+(' + "|".join(_BLOCKED_MODULES) + r')\s+import\b.*$',
            _re.MULTILINE,
        )
        # Empty method call pattern: logic.() or result = logic.()
        _EMPTY_METHOD_CALL_RE = _re.compile(
            r'(\w+)\.\(\)',
        )

        sanitized = {}
        fixes_applied = 0
        for key, code in templates.items():
            if not isinstance(code, str) or not code.strip():
                sanitized[key] = code
                continue

            # Skip non-code entries (workflow.json, etc.)
            if not key.endswith(".py.tpl"):
                sanitized[key] = code
                continue

            original = code

            # 1. Strip null bytes
            code = code.replace("\x00", "")

            # 1b. Normalize CJK fullwidth ASCII characters (U+FF01–U+FF5E)
            # Multilingual LLMs occasionally emit fullwidth variants like
            # ｜ (U+FF5C) instead of |, （ instead of (, etc.
            code = code.translate(_FULLWIDTH_TO_ASCII)

            # 2. Normalize line endings
            code = code.replace("\r\n", "\n").replace("\r", "\n")

            # 3. Repair comment headers split by multi-line descriptions
            code = ExtensionCLIAnalyzer._repair_multiline_comment_headers(code)

            # 4. Remove blocked module imports
            code = _BLOCKED_FROM_IMPORT_RE.sub(
                lambda m: f"{m.group(1)}# [removed blocked import: {m.group(0).strip()}]",
                code,
            )
            code = _BLOCKED_IMPORT_RE.sub(
                lambda m: f"{m.group(1)}# [removed blocked import: {m.group(0).strip()}]",
                code,
            )

            # 4b. Replace getattr(obj, 'attr') / setattr(obj, 'attr', value)
            # with obj.attr / obj.attr = value when the attribute is a string
            # literal containing a valid Python identifier.  The CodeValidator
            # blocks getattr/setattr entirely; this rewrite is safe only when
            # the attribute name is a compile-time constant.
            code = _GETATTR_LITERAL_RE.sub(lambda m: f"{m.group(1).strip()}.{m.group(3)}", code)
            code = _SETATTR_LITERAL_RE.sub(
                lambda m: f"{m.group(1).strip()}.{m.group(3)} = {m.group(4).strip()}", code,
            )

            # 5. Fix indentation: try ast.parse, on failure try dedent
            try:
                _ast.parse(code)
            except (SyntaxError, IndentationError) as e:
                if "indent" in str(e).lower() or "unexpected" in str(e).lower():
                    dedented = _textwrap.dedent(code)
                    try:
                        _ast.parse(dedented)
                        code = dedented
                        logger.info(
                            "[Stage 7] Fixed indentation in '%s' via dedent",
                            key,
                        )
                    except (SyntaxError, IndentationError):
                        # dedent didn't help — leave for revision
                        pass

            # 6. Fix empty method calls: logic.() → # logic.<no method available>()
            if _EMPTY_METHOD_CALL_RE.search(code):
                def _fix_empty_call(m):
                    var = m.group(1)
                    # Only fix if it looks like a method call on a logic/object var
                    if var in ("logic", "_logic", "result"):
                        return f"# {var}.<method>()  # method name not available"
                    return m.group(0)
                code = _EMPTY_METHOD_CALL_RE.sub(_fix_empty_call, code)

            # 7. Detect stub templates (only pass + comments/print)
            _stripped = [
                l.strip() for l in code.split('\n')
                if l.strip() and not l.strip().startswith('#')
            ]
            _non_trivial = [
                l for l in _stripped
                if l != 'pass' and not l.startswith('print(')
            ]
            if not _non_trivial:
                fixes_applied += 1
                logger.warning(
                    "[Stage 7] Template '%s' appears to be a stub "
                    "(only pass/comments). Consider regenerating.",
                    key,
                )

            if code != original:
                fixes_applied += 1
                logger.info(
                    "[Stage 7] Sanitized template '%s'",
                    key,
                )

            sanitized[key] = code

        if fixes_applied:
            logger.info(
                "[Stage 7] Sanitization fixed %d/%d templates",
                fixes_applied, len(templates),
            )

        return sanitized

    # ================================================================
    # Stage 7.5: Live API Probing
    # ================================================================
