from .common import *


class AnalyzerParameterMetadataMixin:
    def _extract_parameter_roles_from_source(self, source: str) -> Dict[str, Dict]:
        """Extract parameter-node role reads/writes from extension Python source."""
        roles: Dict[str, Dict] = {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return roles

        def _record(role: str, access: str, method: str = ""):
            if not role:
                return
            entry = roles.setdefault(role, {
                "role": role,
                "accesses": [],
                "methods": [],
                "method_accesses": {},
                "keywords": self._role_keywords(role),
                "node_class": self._guess_node_class_for_role(role),
                "value_types": [],
            })
            if access not in entry["accesses"]:
                entry["accesses"].append(access)
            if method and method not in entry["methods"]:
                entry["methods"].append(method)
            if method:
                method_accesses = entry.setdefault("method_accesses", {}).setdefault(method, [])
                if access not in method_accesses:
                    method_accesses.append(access)
            value_type = self._parameter_access_value_type(node) if access == "parameter_read" else ""
            if value_type and value_type not in entry["value_types"]:
                entry["value_types"].append(value_type)

        parent_stack = []
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, "_parent", node)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = self._get_call_name(node)
            if not func_name:
                continue
            if not any(
                suffix in func_name for suffix in (
                    "GetNodeReference", "SetNodeReferenceID",
                    "GetParameter", "SetParameter",
                )
            ):
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
                continue
            method = ""
            parent = getattr(node, "_parent", None)
            while parent is not None:
                if isinstance(parent, ast.FunctionDef):
                    method = parent.name
                    break
                parent = getattr(parent, "_parent", None)
            if "NodeReference" in func_name:
                access = "node_reference_write" if "Set" in func_name else "node_reference_read"
            else:
                access = "parameter_write" if "Set" in func_name else "parameter_read"
            _record(arg0.value, access, method)
        return roles

    @staticmethod
    def _parameter_access_value_type(get_parameter_call: ast.Call) -> str:
        """Infer the expected scalar type around a GetParameter(...) call."""
        parent = getattr(get_parameter_call, "_parent", None)
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
            if parent.func.id in ("float", "int", "bool", "str"):
                return parent.func.id
        if isinstance(parent, ast.Compare):
            constants = [
                c.value for c in parent.comparators
                if isinstance(c, ast.Constant)
            ]
            if any(str(v).lower() in ("true", "false") for v in constants):
                return "bool"
            if any(isinstance(v, (int, float)) for v in constants):
                return "float" if any(isinstance(v, float) for v in constants) else "int"
        return ""

    @staticmethod
    def _parameter_default_value_type(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        text = str(value)
        lowered = text.strip().lower()
        if lowered in ("true", "false"):
            return "bool"
        try:
            int(text)
            return "int"
        except Exception:
            pass
        try:
            float(text)
            return "float"
        except Exception:
            pass
        return "str"

    @staticmethod
    def _parameter_default_to_string(value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def _record_parameter_default(
        self,
        defaults: Dict[str, Dict],
        role: str,
        value: Any,
        source: str,
        method: str = "",
        widget: str = "",
        property_name: str = "",
        confidence: str = "high",
        precedence: int = 50,
    ) -> None:
        if not role or value is None:
            return
        existing = defaults.get(role)
        if existing and existing.get("_precedence", 0) > precedence:
            return
        defaults[role] = {
            "value": self._parameter_default_to_string(value),
            "value_type": self._parameter_default_value_type(value),
            "source": source,
            "method": method,
            "widget": widget,
            "property": property_name,
            "confidence": confidence,
            "_precedence": precedence,
        }

    @staticmethod
    def _literal_parameter_value(node: ast.AST) -> Optional[Any]:
        """Extract constants commonly passed to SetParameter(...)."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "str" and node.args:
                return AnalyzerParameterMetadataMixin._literal_parameter_value(node.args[0])
            if node.func.id in ("float", "int", "bool") and node.args:
                raw = AnalyzerParameterMetadataMixin._literal_parameter_value(node.args[0])
                if raw is None:
                    return None
                try:
                    return {"float": float, "int": int, "bool": bool}[node.func.id](raw)
                except Exception:
                    return None
        return None

    @staticmethod
    def _widget_reference_from_expr(node: ast.AST) -> Tuple[str, str]:
        """Return (widget_name, property_name) from expressions like self.ui.foo.value."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            if node.func.id in ("str", "float", "int", "bool"):
                return AnalyzerParameterMetadataMixin._widget_reference_from_expr(node.args[0])
        attrs = []
        current = node
        while isinstance(current, ast.Attribute):
            attrs.append(current.attr)
            current = current.value
        attrs = list(reversed(attrs))
        if len(attrs) >= 2 and attrs[0] == "ui":
            return attrs[1], attrs[-1]
        if len(attrs) >= 3 and attrs[0] == "self" and attrs[1] == "ui":
            return attrs[2], attrs[-1]
        return "", ""

    @staticmethod
    def _default_property_for_widget(qt_class: str, expr_property: str) -> str:
        prop = (expr_property or "").lower()
        cls = qt_class or ""
        if prop in ("checked", "ischecked"):
            return "checked"
        if prop in ("value", "number"):
            return "value"
        if prop in ("currenttext", "currentindex"):
            return "currentIndex" if prop == "currentindex" else "currentText"
        if "CheckBox" in cls or "Checkable" in cls or "RadioButton" in cls:
            return "checked"
        if "SpinBox" in cls or "Slider" in cls:
            return "value"
        if "ComboBox" in cls:
            return "currentIndex"
        return expr_property

    def _extract_parameter_defaults(
        self,
        source: str,
        ui_files: List[str],
        roles: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Extract source-derived scalar defaults for parameter-node roles."""
        defaults: Dict[str, Dict] = {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        widgets = self._extract_ui_widget_inventory(ui_files)
        if tree is not None:
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    setattr(child, "_parent", node)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func_name = self._get_call_name(node)
                if not func_name.endswith("SetParameter") or len(node.args) < 2:
                    continue
                role_arg = node.args[0]
                if not isinstance(role_arg, ast.Constant) or not isinstance(role_arg.value, str):
                    continue
                role = role_arg.value
                method = ""
                parent = getattr(node, "_parent", None)
                while parent is not None:
                    if isinstance(parent, ast.FunctionDef):
                        method = parent.name
                        break
                    parent = getattr(parent, "_parent", None)

                literal = self._literal_parameter_value(node.args[1])
                if literal is not None:
                    self._record_parameter_default(
                        defaults, role, literal,
                        source="python_setparameter_literal",
                        method=method, confidence="high", precedence=90,
                    )
                    continue

                widget_name, expr_property = self._widget_reference_from_expr(node.args[1])
                widget_info = widgets.get(widget_name, {})
                prop_name = self._default_property_for_widget(
                    widget_info.get("class", ""), expr_property
                )
                if widget_info and prop_name in widget_info.get("properties", {}):
                    self._record_parameter_default(
                        defaults, role, widget_info["properties"][prop_name],
                        source="ui_widget_default",
                        method=method, widget=widget_name,
                        property_name=prop_name, confidence="high", precedence=80,
                    )

        for role, info in (roles or {}).items():
            if info.get("node_class") or role in defaults:
                continue
            value_types = set(info.get("value_types") or [])
            if "float" in value_types:
                self._record_parameter_default(
                    defaults, role, "0.0", source="typed_read_safe_fallback",
                    confidence="low", precedence=10,
                )
            elif "int" in value_types:
                self._record_parameter_default(
                    defaults, role, "0", source="typed_read_safe_fallback",
                    confidence="low", precedence=10,
                )
            elif "bool" in value_types:
                self._record_parameter_default(
                    defaults, role, "False", source="typed_read_safe_fallback",
                    confidence="low", precedence=10,
                )

        for entry in defaults.values():
            entry.pop("_precedence", None)
        return defaults

    def _extract_ui_parameter_bindings(
        self,
        source: str,
        ui_files: List[str],
        widget_connections: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict]:
        """Map extension UI widgets to parameter-node roles using source AST.

        The mapping is generic and source-derived.  It recognizes common
        scripted-extension patterns such as:

        - parameterNode.SetParameter("role", str(self.ui.spinBox.value))
        - parameterNode.SetNodeReferenceID("role", self.ui.selector.currentNodeID)
        - if self.ui.checkBox.checked:
              parameterNode.SetParameter("role", "True")
          else:
              parameterNode.SetParameter("role", "False")

        The result is keyed by widget object name, with enough metadata for
        Stage 4 to classify cookbook UI-control steps and Stage 7 to generate
        deterministic parameter-update templates.
        """
        if not source:
            return {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}

        widgets = self._extract_ui_widget_inventory(ui_files)
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, "_parent", node)

        connections_by_widget: Dict[str, List[Dict]] = {}
        for conn in widget_connections or []:
            raw_name = _text_or_empty(conn.get("button_widget_name", ""))
            widget_name = raw_name.split(".")[-1] if raw_name else ""
            if widget_name:
                connections_by_widget.setdefault(widget_name, []).append(conn)

        bindings: Dict[str, Dict] = {}

        def _record(
            widget_name: str,
            role: str,
            access: str,
            value_property: str = "",
            true_value: Any = None,
            false_value: Any = None,
        ) -> None:
            if not widget_name or not role:
                return
            widget_info = widgets.get(widget_name, {})
            entry = bindings.setdefault(widget_name, {
                "widget_name": widget_name,
                "widget_class": widget_info.get("class", ""),
                "ui_text": _text_or_empty(
                    (widget_info.get("properties") or {}).get("text", "")
                ),
                "properties": widget_info.get("properties", {}),
                "roles": [],
                "connections": connections_by_widget.get(widget_name, []),
                "keywords": sorted(set(
                    self._role_keywords(widget_name)
                    + self._role_keywords(role)
                    + self._role_keywords(
                        _text_or_empty((widget_info.get("properties") or {}).get("text", ""))
                    )
                )),
            })
            role_entry = {
                "parameter_name": role,
                "access": access,
                "value_property": value_property,
            }
            if true_value is not None or false_value is not None:
                role_entry["true_value"] = true_value
                role_entry["false_value"] = false_value
            if role_entry not in entry["roles"]:
                entry["roles"].append(role_entry)
            entry["keywords"] = sorted(set(
                entry.get("keywords", [])
                + self._role_keywords(role)
                + self._role_keywords(widget_name)
            ))

        def _literal(node: ast.AST) -> Optional[Any]:
            if isinstance(node, ast.Constant):
                return node.value
            return self._literal_parameter_value(node)

        def _find_if_widget_test(node: ast.AST) -> Tuple[str, str]:
            parent = getattr(node, "_parent", None)
            while parent is not None:
                if isinstance(parent, ast.If):
                    widget_name, prop = self._widget_reference_from_expr(parent.test)
                    if widget_name:
                        return widget_name, prop
                parent = getattr(parent, "_parent", None)
            return "", ""

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = self._get_call_name(node)
            if not func_name or len(node.args) < 2:
                continue
            if not (func_name.endswith("SetParameter") or func_name.endswith("SetNodeReferenceID")):
                continue
            role_arg = node.args[0]
            if not isinstance(role_arg, ast.Constant) or not isinstance(role_arg.value, str):
                continue
            role = role_arg.value
            access = "parameter_write" if func_name.endswith("SetParameter") else "node_reference_write"

            widget_name, prop = self._widget_reference_from_expr(node.args[1])
            if widget_name:
                _record(widget_name, role, access, prop)
                continue

            # Checkbox/toolbutton boolean pattern: the value argument is a
            # literal, while the widget is in the surrounding if condition.
            literal_value = _literal(node.args[1])
            if literal_value is not None:
                test_widget, test_prop = _find_if_widget_test(node)
                if test_widget:
                    parent_if = getattr(node, "_parent", None)
                    while parent_if is not None and not isinstance(parent_if, ast.If):
                        parent_if = getattr(parent_if, "_parent", None)
                    true_value = literal_value
                    false_value = None
                    if parent_if is not None:
                        current_stmt = getattr(node, "_parent", None)
                        while current_stmt is not None and getattr(current_stmt, "_parent", None) is not parent_if:
                            current_stmt = getattr(current_stmt, "_parent", None)
                        if current_stmt in getattr(parent_if, "orelse", []):
                            false_value = literal_value
                            true_value = None
                    _record(
                        test_widget, role, access, test_prop,
                        true_value=true_value,
                        false_value=false_value,
                    )

        return bindings

    def _read_entry_source(self, scan_result: Dict) -> str:
        entry_module = scan_result.get("entry_module", "") if scan_result else ""
        if entry_module and os.path.isfile(entry_module):
            try:
                with open(entry_module, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""
        return ""

    def _extract_parameter_method_dependencies(
        self,
        source: str,
        roles: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Map logic methods to direct/transitive parameter-node roles they read."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}

        direct_parameter_roles: Dict[str, set] = {}
        direct_node_roles: Dict[str, set] = {}
        direct_node_requirements = self._extract_direct_node_requirements(tree)
        calls: Dict[str, set] = {}
        method_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        for role, info in (roles or {}).items():
            method_access_map = info.get("method_accesses") or {}
            for method in info.get("methods") or []:
                accesses = method_access_map.get(method) or []
                if "parameter_read" in accesses:
                    direct_parameter_roles.setdefault(method, set()).add(role)
                if "node_reference_read" in accesses:
                    direct_node_roles.setdefault(method, set()).add(role)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                    and func.attr in method_names
                ):
                    calls.setdefault(node.name, set()).add(func.attr)

        dependencies: Dict[str, Dict] = {}
        for method in method_names:
            seen = set()
            stack = [method]
            parameter_roles_for_method = set()
            node_roles_for_method = set()
            node_requirements_for_method: Dict[str, Dict] = {}
            callees = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                if current != method:
                    callees.add(current)
                parameter_roles_for_method.update(direct_parameter_roles.get(current, set()))
                node_roles_for_method.update(direct_node_roles.get(current, set()))
                for role, requirement in direct_node_requirements.get(current, {}).items():
                    node_requirements_for_method[role] = self._merge_node_requirements(
                        node_requirements_for_method.get(role),
                        requirement,
                    )
                for callee in calls.get(current, set()):
                    if callee not in seen:
                        stack.append(callee)
            if parameter_roles_for_method or node_roles_for_method:
                entry = {"transitive_methods": sorted(callees)}
                if parameter_roles_for_method:
                    entry["parameter_roles"] = sorted(parameter_roles_for_method)
                if node_roles_for_method:
                    entry["node_roles"] = sorted(node_roles_for_method)
                    entry["node_requirements"] = {
                        role: node_requirements_for_method.get(
                            role, {
                                "requirement": "optional_unknown",
                                "conditions": [],
                                "condition_groups": [],
                            }
                        )
                        for role in sorted(node_roles_for_method)
                    }
                dependencies[method] = entry
        return dependencies

    @staticmethod
    def _merge_node_requirements(existing: Optional[Dict], incoming: Dict) -> Dict:
        """Merge node requirements, preserving the strongest pre-call contract."""
        if not existing:
            return {
                "requirement": incoming.get("requirement", "optional_unknown"),
                "conditions": list(incoming.get("conditions") or []),
                "condition_groups": list(incoming.get("condition_groups") or []),
            }
        rank = {
            "produced_by_method": 0,
            "optional_unknown": 1,
            "conditional": 2,
            "required": 3,
        }
        existing_kind = existing.get("requirement", "optional_unknown")
        incoming_kind = incoming.get("requirement", "optional_unknown")
        strongest = existing_kind if rank.get(existing_kind, 1) >= rank.get(incoming_kind, 1) else incoming_kind
        conditions = []
        for condition in list(existing.get("conditions") or []) + list(incoming.get("conditions") or []):
            if condition not in conditions:
                conditions.append(condition)
        condition_groups = []
        for group in list(existing.get("condition_groups") or []) + list(incoming.get("condition_groups") or []):
            if group not in condition_groups:
                condition_groups.append(group)
        return {
            "requirement": strongest,
            "conditions": conditions if strongest == "conditional" else [],
            "condition_groups": condition_groups if strongest == "conditional" else [],
        }

    def _extract_direct_node_requirements(self, tree: ast.AST) -> Dict[str, Dict[str, Dict]]:
        """Classify direct node-reference reads by their actual use sites."""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, "_parent", node)

        result: Dict[str, Dict[str, Dict]] = {}
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            assignments: Dict[str, ast.AST] = {}
            reference_vars: Dict[str, List[Tuple[str, int]]] = {}
            read_roles = set()
            write_calls: Dict[str, List[ast.Call]] = {}

            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            assignments[target.id] = value
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                call_name = self._get_call_name(node)
                arg0 = node.args[0]
                if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
                    continue
                role = arg0.value
                if call_name.endswith("GetNodeReference"):
                    read_roles.add(role)
                    parent = getattr(node, "_parent", None)
                    if isinstance(parent, ast.Assign):
                        for target in parent.targets:
                            if isinstance(target, ast.Name):
                                reference_vars.setdefault(target.id, []).append((role, node.lineno))
                    elif isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
                        reference_vars.setdefault(parent.target.id, []).append((role, node.lineno))
                elif call_name.endswith("SetNodeReferenceID"):
                    write_calls.setdefault(role, []).append(node)

            method_requirements: Dict[str, Dict] = {}
            for role in read_roles:
                uses: List[Tuple[int, List[Dict], bool]] = []
                role_vars = {
                    var_name
                    for var_name, refs in reference_vars.items()
                    if any(ref_role == role for ref_role, _ in refs)
                }
                for node in ast.walk(function):
                    if not isinstance(node, ast.Name) or node.id not in role_vars or not isinstance(node.ctx, ast.Load):
                        continue
                    parent = getattr(node, "_parent", None)
                    if isinstance(parent, ast.Assign) and parent.value is node:
                        continue
                    conditions, presence_guarded = self._enclosing_node_conditions(
                        node, assignments, role_vars
                    )
                    uses.append((getattr(node, "lineno", 0), conditions, presence_guarded))

                if not uses:
                    method_requirements[role] = {
                        "requirement": "optional_unknown",
                        "conditions": [],
                        "condition_groups": [],
                    }
                    continue

                first_use_line = min(line for line, _, _ in uses)
                unconditional_write_lines = [
                    call.lineno for call in write_calls.get(role, [])
                    if not self._has_enclosing_control_flow(call)
                ]
                if unconditional_write_lines and min(unconditional_write_lines) < first_use_line:
                    method_requirements[role] = {
                        "requirement": "produced_by_method",
                        "conditions": [],
                        "condition_groups": [],
                    }
                    continue

                if any(not conditions and not presence_guarded for _, conditions, presence_guarded in uses):
                    method_requirements[role] = {
                        "requirement": "required",
                        "conditions": [],
                        "condition_groups": [],
                    }
                    continue

                conditions = []
                condition_groups = []
                for _, use_conditions, presence_guarded in uses:
                    if presence_guarded:
                        continue
                    if use_conditions and use_conditions not in condition_groups:
                        condition_groups.append(use_conditions)
                    for condition in use_conditions:
                        if condition not in conditions:
                            conditions.append(condition)
                method_requirements[role] = {
                    "requirement": "conditional" if conditions else "optional_unknown",
                    "conditions": conditions,
                    "condition_groups": condition_groups,
                }

            if method_requirements:
                result[function.name] = method_requirements
        return result

    @staticmethod
    def _has_enclosing_control_flow(node: ast.AST) -> bool:
        control_flow_types = (
            ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
            ast.With, ast.AsyncWith,
        )
        if hasattr(ast, "Match"):
            control_flow_types += (ast.Match,)
        parent = getattr(node, "_parent", None)
        while parent is not None and not isinstance(parent, ast.FunctionDef):
            if isinstance(parent, control_flow_types):
                return True
            parent = getattr(parent, "_parent", None)
        return False

    def _enclosing_node_conditions(
        self,
        node: ast.AST,
        assignments: Dict[str, ast.AST],
        reference_vars: set,
    ) -> Tuple[List[Dict], bool]:
        """Return supported parameter guards and whether a node-presence guard applies."""
        conditions: List[Dict] = []
        presence_guarded = False
        parent = getattr(node, "_parent", None)
        while parent is not None and not isinstance(parent, ast.FunctionDef):
            if isinstance(parent, ast.If):
                condition, checks_presence = self._condition_from_expression(
                    parent.test, assignments, reference_vars
                )
                if checks_presence:
                    presence_guarded = True
                elif condition:
                    if any(
                        any(descendant is node for descendant in ast.walk(statement))
                        for statement in parent.orelse
                    ):
                        condition = self._invert_condition(condition)
                    if condition not in conditions:
                        conditions.append(condition)
            parent = getattr(parent, "_parent", None)
        return conditions, presence_guarded

    @staticmethod
    def _invert_condition(condition: Dict) -> Dict:
        inverted = dict(condition)
        inverted["operator"] = (
            "not_equals" if condition.get("operator") == "equals" else "equals"
        )
        return inverted

    def _condition_from_expression(
        self,
        expression: ast.AST,
        assignments: Dict[str, ast.AST],
        reference_vars: set,
    ) -> Tuple[Optional[Dict], bool]:
        """Extract a simple GetParameter comparison used as an if guard."""
        if isinstance(expression, ast.Name):
            if expression.id in reference_vars:
                return None, True
            assigned = assignments.get(expression.id)
            if assigned is not None and assigned is not expression:
                return self._condition_from_expression(assigned, assignments, reference_vars)
            return None, False
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            condition, presence = self._condition_from_expression(
                expression.operand, assignments, reference_vars
            )
            if condition:
                condition = dict(condition)
                condition["operator"] = (
                    "not_equals" if condition.get("operator") == "equals" else "equals"
                )
            return condition, presence
        if not isinstance(expression, ast.Compare) or len(expression.ops) != 1 or len(expression.comparators) != 1:
            return None, False
        left, right = expression.left, expression.comparators[0]
        if isinstance(left, ast.Constant) and not isinstance(right, ast.Constant):
            left, right = right, left
        if not isinstance(right, ast.Constant):
            return None, False
        if isinstance(left, ast.Name) and left.id in assignments:
            left = assignments[left.id]
        if not isinstance(left, ast.Call) or not left.args:
            return None, False
        call_name = self._get_call_name(left)
        role_arg = left.args[0]
        if not call_name.endswith("GetParameter"):
            return None, False
        if not isinstance(role_arg, ast.Constant) or not isinstance(role_arg.value, str):
            return None, False
        operator = "equals" if isinstance(expression.ops[0], (ast.Eq, ast.Is)) else (
            "not_equals" if isinstance(expression.ops[0], (ast.NotEq, ast.IsNot)) else ""
        )
        if not operator:
            return None, False
        return {
            "parameter": role_arg.value,
            "operator": operator,
            "value": self._parameter_default_to_string(right.value),
        }, False

    def _build_workflow_metadata(
        self, scan_result: Dict, logic_analysis: Dict, workflow_graph: Dict,
    ) -> Dict:
        """Build generic workflow metadata used by templates and runtime dispatch."""
        source = ""
        entry_module = scan_result.get("entry_module", "")
        if entry_module and os.path.isfile(entry_module):
            try:
                with open(entry_module, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except Exception:
                source = ""

        bindings = self._extract_parameter_roles_from_source(source)
        parameter_defaults = self._extract_parameter_defaults(
            source, scan_result.get("ui_files", []) or [], bindings
        )
        for role, default in parameter_defaults.items():
            if role in bindings:
                bindings[role]["default"] = default
        parameter_dependencies = self._extract_parameter_method_dependencies(
            source, bindings
        )
        ui_parameter_bindings = (
            getattr(self, "_ui_parameter_bindings", None)
            or scan_result.get("ui_parameter_bindings", {})
            or self._extract_ui_parameter_bindings(
                source,
                scan_result.get("ui_files", []) or [],
                getattr(self, "_widget_connections", []),
            )
        )
        metadata = {
            "extension_module_name": os.path.splitext(os.path.basename(entry_module))[0],
            "logic_class_name": scan_result.get("logic_class", {}).get("class_name", ""),
            "metadata_version": 5,
            "parameter_bindings": bindings,
            "parameter_defaults": parameter_defaults,
            "ui_parameter_bindings": ui_parameter_bindings,
            "parameter_method_dependencies": parameter_dependencies,
            "choice_bindings": {},
            "interaction_bindings": {},
            "operation_model": {},
            "node_roles": {},
            "repeat_groups": {},
            "repeat_blocks": {},
            "validation_state": {
                "static_valid": None,
                "api_probe_valid": None,
                "contract_valid": None,
                "prelude_valid": None,
            },
            "api_evidence": {},
        }

        for step in workflow_graph.get("steps", []):
            step_id = step.get("step_id", "")
            operation_model = self._build_step_operation_model(step)
            metadata["operation_model"][step_id] = operation_model
            step["operation_model"] = operation_model

            choice = step.get("choice_info") or {}
            pname = choice.get("parameter_name")
            if pname and pname in bindings:
                metadata["choice_bindings"][step_id] = {
                    "parameter_name": pname,
                    "choice_parameter_name": pname,
                    **bindings[pname],
                }

            node_roles = self._infer_step_node_roles(step, metadata)
            if node_roles:
                metadata["node_roles"][step_id] = node_roles
                step["node_roles"] = node_roles
                interaction_role = next((
                    role for role in node_roles
                    if role.get("role_kind") == "interaction_output"
                    and role.get("parameter_name") in bindings
                ), None)
                if interaction_role:
                    parameter_name = interaction_role["parameter_name"]
                    metadata["interaction_bindings"][step_id] = {
                        "parameter_name": parameter_name,
                        **bindings[parameter_name],
                    }
                    step["parameter_role"] = parameter_name

        # Attach extension-agnostic repeat blocks identified by Stage 4.
        for block in workflow_graph.get("repeat_blocks", []) or []:
            repeat_id = block.get("repeat_id", "")
            body_steps = block.get("body_steps", []) or []
            if not repeat_id or not body_steps:
                continue
            metadata["repeat_blocks"][repeat_id] = block
            for step in workflow_graph.get("steps", []):
                if step.get("step_id") in body_steps:
                    step["repeat_block"] = block

            # Preserve the established count-placement template contract while
            # runtime control moves to generic repeat_blocks.
            controller = block.get("controller", {}) or {}
            body = [
                step for step in workflow_graph.get("steps", [])
                if step.get("step_id") in body_steps
            ]
            if (
                controller.get("kind") == "count"
                and len(body) == 2
                and _operation_type_for_step(body[0]) in ("extension_op", "slicer_op")
                and _operation_type_for_step(body[1]) == "user_interaction"
            ):
                source_step = controller.get("source_step", "")
                source = next(
                    (step for step in workflow_graph.get("steps", [])
                     if step.get("step_id") == source_step),
                    {},
                )
                count_parameter = (source.get("choice_info") or {}).get("parameter_name", "")
                group = {
                    "group_id": repeat_id,
                    "count_parameter": count_parameter,
                    "count_step": source_step,
                    "start_step": body_steps[0],
                    "interaction_step": body_steps[-1],
                }
                metadata["repeat_groups"][repeat_id] = group
                source["repeat_group"] = group
                for step in body:
                    step["repeat_group"] = group

        return metadata
