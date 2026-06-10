from .common import *


class AnalyzerScanMixin:
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

    def _extract_ui_widget_inventory(self, ui_files: List[str]) -> Dict[str, Dict]:
        """Extract a lightweight widget inventory from Qt Designer .ui files.

        This is intentionally generic: it records widget names, Qt classes, and
        simple string/list properties such as qMRMLNodeComboBox nodeTypes.  Later
        stages can use this as evidence for parameter-node bindings without the
        cookbook author knowing API details.
        """
        import xml.etree.ElementTree as ET

        widgets: Dict[str, Dict] = {}
        for ui_path in ui_files or []:
            try:
                root = ET.parse(ui_path).getroot()
            except Exception:
                continue
            for widget in root.iter("widget"):
                name = widget.get("name", "")
                if not name:
                    continue
                info = {
                    "class": widget.get("class", ""),
                    "properties": {},
                    "ui_file": os.path.basename(ui_path),
                }
                for prop in widget.findall("property"):
                    pname = prop.get("name", "")
                    if not pname:
                        continue
                    value = self._parse_ui_property_value(prop)
                    if value is not None:
                        info["properties"][pname] = value
                widgets[name] = info
        return widgets

    @staticmethod
    def _parse_ui_property_value(prop) -> Optional[Any]:
        """Return a simple Python value for common Qt Designer property tags."""
        for tag in ("string", "cstring", "enum", "set"):
            el = prop.find(tag)
            if el is not None:
                return el.text or ""
        bool_el = prop.find("bool")
        if bool_el is not None:
            return str(bool_el.text or "").strip().lower() == "true"
        for tag in ("number", "int", "uint"):
            el = prop.find(tag)
            if el is not None:
                try:
                    return int(str(el.text or "0").strip())
                except ValueError:
                    return el.text or ""
        for tag in ("double", "float"):
            el = prop.find(tag)
            if el is not None:
                try:
                    return float(str(el.text or "0").strip())
                except ValueError:
                    return el.text or ""
        return None

    # ================================================================
    # Widget Signal Connection Extraction (AST)
    # ================================================================

    def _extract_widget_connections(self, widget_source: str) -> List[Dict]:
        """Extract widget signal→handler→logic_method mappings from Widget source.

        This intentionally covers more than push-button clicks.  Many Slicer
        scripted extensions persist checkbox/toolbutton/spinbox changes through
        a shared updateParameterNodeFromGUI handler, and those controls are
        workflow operations even when no Logic method is called directly.
        """
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
        """Find common Qt signal connect(self.XXX) patterns in setup()."""
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
            signal_name = ""

            # Pattern 1: something.clicked.connect(self.handlerMethod)
            # Also supports stateChanged/valueChanged/currentTextChanged/etc.
            receiver = func.value
            _SUPPORTED_SIGNALS = {
                "clicked", "toggled", "stateChanged", "valueChanged",
                "currentTextChanged", "currentIndexChanged", "textActivated",
                "checkBoxToggled", "currentNodeChanged",
            }
            if isinstance(receiver, ast.Attribute) and receiver.attr in _SUPPORTED_SIGNALS:
                button_name = self._get_attribute_chain(receiver.value)
                signal_name = receiver.attr
                if stmt.args:
                    arg = stmt.args[0]
                    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                        if arg.value.id == "self":
                            handler_name = arg.attr

            # Pattern 2: something.connect('clicked(bool)', self.handlerMethod)
            #            something.connect("stateChanged(int)", self.handlerMethod)
            if not button_name and stmt.args:
                first_arg = stmt.args[0]
                if (isinstance(first_arg, ast.Constant)
                        and isinstance(first_arg.value, str)
                        and any(sig in first_arg.value for sig in _SUPPORTED_SIGNALS)):
                    button_name = self._get_attribute_chain(func.value)
                    signal_name = first_arg.value
                    if len(stmt.args) > 1:
                        second_arg = stmt.args[1]
                        if isinstance(second_arg, ast.Attribute) and isinstance(second_arg.value, ast.Name):
                            if second_arg.value.id == "self":
                                handler_name = second_arg.attr

            if button_name and handler_name:
                logic_methods = handler_logic_map.get(handler_name, [])
                bare_name = button_name.split(".")[-1] if button_name else ""
                connections.append({
                    "button_widget_name": bare_name,
                    "signal": signal_name,
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
    # Stage 3: Logic Class Analysis (LLM)
    # ================================================================
