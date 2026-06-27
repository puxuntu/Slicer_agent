from .common import *
import json


class AnalyzerContractsMixin:
    def test_Stage4UnboundParameterStepDegrades(self):
        """A parameter-update step the source cannot bind degrades, not dead-ends.

        Regression for SlicerOrbitSurgerySim's 'Enable 3D interaction transform
        handle' checkbox: it is wired to a handler method, not a parameter, so no
        source-derived UI binding exists. The LLM oscillates between omitting the
        binding ('parameter update intent requires a UI binding') and inventing
        one ('references unknown UI parameter binding'); both used to hard-fail
        the run. Normalize now strips an ungrounded binding and drops the
        parameter-update intent so the step validates as a plain step, while a
        source-grounded binding is preserved.
        """
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())

        context = {
            "steps": [
                {"step_number": 7, "operation_type": "user_choice", "description": "Tick the checkbox"},
                {"step_number": 8, "operation_type": "user_choice", "description": "Pick the segmentation"},
            ],
            "logic_methods": [],
            "extension_functions": [],
            "widgets": [
                {"widget_name": "interactionTransformCheckbox"},
                {"widget_name": "segSelector"},
            ],
            # Only step 8's binding is source-derived.
            "ui_parameter_bindings": [{"widget_name": "segSelector", "parameter_name": "seg"}],
            "parameter_roles": [],
            "allowed_slicer_op_categories": [],
            "allowed_interaction_kinds": ["none"],
            "allowed_node_classes": [],
            "allowed_operation_intents": [
                "extension_parameter_update", "extension_node_reference_update",
            ],
            "allowed_node_role_kinds": ["choice_input"],
        }
        base = {"confidence": "high", "interaction_kind": "none", "choice": None, "node_roles": []}
        grounded_step8 = {
            **base, "step_number": 8, "operation_type": "user_choice", "widget_name": "segSelector",
            "operation_intents": ["extension_node_reference_update"],
            "ui_parameter_binding": {"widget_name": "segSelector", "parameter_name": "seg"},
        }

        def _run(step7):
            normalized, _notes = analyzer._normalize_stage4_semantic_result(
                {"steps": [step7, dict(grounded_step8)], "repeat_blocks": []}, context
            )
            errors = analyzer._validate_stage4_semantic_result(normalized, context)
            by_num = {s["step_number"]: s for s in normalized["steps"]}
            return by_num, errors

        # Case 1: intent + INVENTED binding (the 'unknown UI parameter binding' path).
        by_num, errors = _run({
            **base, "step_number": 7, "operation_type": "user_choice",
            "widget_name": "interactionTransformCheckbox", "target_value": True,
            "operation_intents": ["extension_parameter_update"],
            "ui_parameter_binding": {
                "widget_name": "interactionTransformCheckbox",
                "parameter_name": "InteractionTransformEnabled",
            },
        })
        self.assertIsNone(by_num[7]["ui_parameter_binding"])
        self.assertNotIn("extension_parameter_update", by_num[7]["operation_intents"])
        self.assertFalse(any("binding" in e for e in errors), errors)
        # The grounded step keeps its binding and intent (no regression).
        self.assertEqual(by_num[8]["ui_parameter_binding"]["parameter_name"], "seg")
        self.assertIn("extension_node_reference_update", by_num[8]["operation_intents"])

        # Case 2: intent + OMITTED binding (the 'requires a UI binding' gate).
        by_num2, errors2 = _run({
            **base, "step_number": 7, "operation_type": "user_choice",
            "widget_name": "interactionTransformCheckbox", "target_value": True,
            "operation_intents": ["extension_parameter_update"],
            "ui_parameter_binding": None,
        })
        self.assertEqual(by_num2[7]["operation_intents"], [])
        self.assertFalse(any("requires a UI binding" in e for e in errors2), errors2)

        self.delayDisplay("Stage 4 unbound parameter step degrades gracefully")

    def test_Stage4InvalidTargetValueModeCoerced(self):
        """An invalid target_value_mode (a value TYPE like 'boolean') is coerced.

        Regression for SlicerOrbitSurgerySim's 'Enable 3D interaction transform
        handle' checkbox: the LLM emitted target_value_mode 'boolean'/'exact'
        (value types, not the allowed modes), hard-failing Stage 4. Normalize now
        coerces an out-of-set mode to 'set' (value present) or None.
        """
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())
        context = {
            "steps": [{"step_number": 7, "operation_type": "extension_op", "description": "Tick the checkbox"}],
            "logic_methods": [], "extension_functions": [],
            "widgets": [{"widget_name": "interactionTransformCheckbox"}],
            "ui_parameter_bindings": [], "parameter_roles": [],
            "allowed_slicer_op_categories": [], "allowed_interaction_kinds": ["none"],
            "allowed_node_classes": [], "allowed_operation_intents": [],
            "allowed_node_role_kinds": ["choice_input"],
        }
        result = {"steps": [{
            "step_number": 7, "operation_type": "extension_op", "confidence": "high",
            "interaction_kind": "none", "operation_intents": [], "node_roles": [],
            "widget_name": "interactionTransformCheckbox",
            "target_value": True, "target_value_mode": "boolean",
            "ui_parameter_binding": None,
        }], "repeat_blocks": []}
        normalized, _notes = analyzer._normalize_stage4_semantic_result(result, context)
        self.assertEqual(normalized["steps"][0]["target_value_mode"], "set")
        errors = analyzer._validate_stage4_semantic_result(normalized, context)
        self.assertFalse(any("target_value_mode" in e for e in errors), errors)
        self.delayDisplay("Invalid target_value_mode coerced, not rejected")

    def test_Stage4NodePickChoicesCleared(self):
        """A value_kind==node user_choice has its literal choices stripped.

        Regression for PelvicFracturePlanning cb_step_1: the LLM emitted a node
        pick (value_kind=node) but ALSO a literal choices=[{"label":"Pelvic
        Volume"}] echoed from the cookbook text. Those labels shadow the scene
        node tree at runtime, so normalization now clears them for node picks
        while genuine boolean choices are preserved (matching the bound BRP
        node-selection shape: empty choices).
        """
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())
        context = {
            "steps": [
                {"step_number": 1, "operation_type": "user_choice", "description": "Choose the input volume"},
                {"step_number": 2, "operation_type": "user_choice", "description": "Manually adjust a fragment?"},
            ],
            "logic_methods": [], "extension_functions": [],
            "widgets": [], "ui_parameter_bindings": [], "parameter_roles": [],
            "allowed_slicer_op_categories": [], "allowed_interaction_kinds": ["none"],
            "allowed_node_classes": ["vtkMRMLScalarVolumeNode"], "allowed_operation_intents": [],
            "allowed_node_role_kinds": ["choice_input"],
        }
        base = {"confidence": "high", "interaction_kind": "none", "node_roles": []}
        result = {"steps": [
            {**base, "step_number": 1, "operation_type": "user_choice",
             "choice": {"question": "Choose the input volume", "parameter_name": "inputVolume",
                        "value_kind": "node",
                        "choices": [{"label": "Pelvic Volume", "value": "inputVolume"}]}},
            {**base, "step_number": 2, "operation_type": "user_choice",
             "choice": {"question": "Manually adjust a fragment?", "parameter_name": "manualAdjust",
                        "value_kind": "bool",
                        "choices": [{"label": "Yes", "value": True}, {"label": "No", "value": False}]}},
        ], "repeat_blocks": []}

        normalized, _notes = analyzer._normalize_stage4_semantic_result(result, context)
        by_num = {s["step_number"]: s for s in normalized["steps"]}
        # Node pick: literal choices stripped, value_kind retained.
        self.assertEqual(by_num[1]["choice"]["choices"], [])
        self.assertEqual(by_num[1]["choice"]["value_kind"], "node")
        # Boolean choice: choices preserved.
        self.assertEqual(len(by_num[2]["choice"]["choices"]), 2)

        self.delayDisplay("Stage 4 node-pick choices cleared, boolean preserved")

    def test_ModuleLevelSelfRejected(self):
        """Generated templates run at module level — a module-level `self` is a bug.

        Regression for the SlicerOrbitSurgerySim auto-correction that copied
        `self._parameterNode.orbitLm` from the widget source and crashed with
        'name self is not defined'. _semantic_validate now flags module-level
        self/cls but still allows them inside a function/method definition.
        """
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())

        flagged = analyzer._semantic_validate(
            "logic = something()\nx = self._parameterNode.orbitLm\n", {"methods": []}
        )
        self.assertTrue(
            any("self" in e for e in flagged["errors"]),
            f"module-level self should be flagged: {flagged['errors']}",
        )
        allowed = analyzer._semantic_validate(
            "def _helper(self):\n    return self.value\ny = _helper\n", {"methods": []}
        )
        self.assertFalse(
            any("self" in e for e in allowed["errors"]),
            f"self inside a def must NOT be flagged: {allowed['errors']}",
        )
        self.delayDisplay("Module-level self rejected; in-function self allowed")

    def test_ParameterNodeWrapperDetectionAndTyping(self):
        """Detect @parameterNodeWrapper classes and type getParameterNode() to them.

        Regression for SlicerOrbitSurgerySim crashing with
        '<ParameterNode> object has no attribute GetNodeReference': the param node
        is a parameterNodeWrapper (typed properties), not the classic VTK API.
        """
        import os
        import tempfile
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator
        from SlicerAIAgentLib.extension_cli_analyzer.api_proof import TypeProvenanceGraph

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())

        src = (
            "from slicer.parameterNodeWrapper import parameterNodeWrapper\n"
            "from slicer import vtkMRMLModelNode, vtkMRMLMarkupsFiducialNode\n"
            "from typing import Optional\n"
            "@parameterNodeWrapper\n"
            "class WidgetParameterNode:\n"
            "    orbitModel: vtkMRMLModelNode\n"
            "    orbitLm: Optional[vtkMRMLMarkupsFiducialNode]\n"
            "    rightSide: bool\n"
            "class WidgetLogic(ScriptedLoadableModuleLogic):\n"
            "    def process(self):\n"
            "        pass\n"
        )
        tmp = tempfile.mkdtemp(prefix="cli_wrapper_test_")
        with open(os.path.join(tmp, "Widget.py"), "w", encoding="utf-8") as f:
            f.write(src)
        scan = analyzer._stage1_scan(tmp)
        wrapper = scan.get("parameter_node_wrapper")
        self.assertIsNotNone(wrapper, "parameterNodeWrapper class not detected")
        self.assertEqual(wrapper["class_name"], "WidgetParameterNode")
        self.assertEqual(wrapper["fields"].get("orbitModel"), "vtkMRMLModelNode")
        # Optional[...] is unwrapped to the node class.
        self.assertEqual(wrapper["fields"].get("orbitLm"), "vtkMRMLMarkupsFiducialNode")

        # getParameterNode() now types to the wrapper class, not vtkMRMLScriptedModuleNode.
        code = (
            "logic = WidgetLogic()\n"
            "paramNode = logic.getParameterNode()\n"
            "n = paramNode.orbitModel\n"
        )
        graph = TypeProvenanceGraph(
            code, logic_class_name="WidgetLogic",
            parameter_node_wrapper_class="WidgetParameterNode",
        )
        cands = graph.receiver_candidates("paramNode")
        self.assertTrue(cands and cands[0]["type"] == "WidgetParameterNode", cands)
        self.delayDisplay("parameterNodeWrapper detected and typed")

    def test_WidgetClassMatchesLogicModule(self):
        """Widget class is selected from the same module file as the logic class.

        Regression: a repo bundling two modules (PlateRegistration +
        MirrorOrbitRecon) picked the sibling module's widget, so the scanned
        button->handler connections were wrong and handler-invocation generation
        silently could not fire.
        """
        import os
        import tempfile
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())
        tmp = tempfile.mkdtemp(prefix="cli_widgetsel_")
        # Sibling module with a larger widget (wins the old size-based heuristic).
        with open(os.path.join(tmp, "Sibling.py"), "w", encoding="utf-8") as f:
            f.write(
                "class SiblingWidget(ScriptedLoadableModuleWidget):\n"
                "    def setup(self): pass\n"
                + "".join(f"    def m{i}(self): pass\n" for i in range(25))
            )
        # Main module: logic + its widget in ONE file.
        with open(os.path.join(tmp, "Main.py"), "w", encoding="utf-8") as f:
            f.write(
                "class MainLogic(ScriptedLoadableModuleLogic):\n"
                "    def process(self): pass\n"
                "class MainWidget(ScriptedLoadableModuleWidget):\n"
                "    def setup(self): pass\n"
            )
        scan = analyzer._stage1_scan(tmp)
        self.assertEqual(scan["logic_class"]["class_name"], "MainLogic")
        self.assertEqual(scan["widget_class"]["class_name"], "MainWidget")
        self.delayDisplay("Widget class matches the logic class's module")

    def test_ExtensionMethodExistenceGuard(self):
        """Pre-execution guard flags a method absent from the runtime class.

        Regression: the runtime PlateRegistrationLogic lacked rotation_p_stop
        (present only in the newer knowledge-base copy), so the correction loop
        repeatedly called a non-existent method. check_extension_methods now
        catches it before execution; check_code only covered slicer/vtk/qt/ctk.
        """
        import sys
        import types
        from SlicerAIAgentLib.ApiSanityChecker import check_extension_methods

        mod = types.ModuleType("FakeExtMod")

        class FakeLogic:
            def real_method(self, a):
                pass

        mod.FakeLogic = FakeLogic
        sys.modules["FakeExtMod"] = mod
        try:
            bad = (
                "from FakeExtMod import FakeLogic\n"
                "logic = FakeLogic()\n"
                "logic.real_method(1)\n"
                "logic.hallucinated_method(1)\n"
            )
            res = check_extension_methods(bad)
            self.assertFalse(res["ok"])
            self.assertTrue(any(m["missing_attr"] == "hallucinated_method" for m in res["missing"]))
            self.assertFalse(any(m["missing_attr"] == "real_method" for m in res["missing"]))
            ok = "from FakeExtMod import FakeLogic\nlogic = FakeLogic()\nlogic.real_method(1)\n"
            self.assertTrue(check_extension_methods(ok)["ok"])
        finally:
            sys.modules.pop("FakeExtMod", None)
        self.delayDisplay("Extension method existence guard works")

    def test_ButtonClickStepInvokesHandler(self):
        """A button-click step drives the widget handler, not the low-level logic method."""
        import ast as _ast
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())
        # This case represents SlicerOrbitSurgerySim, a @parameterNodeWrapper /
        # handler-driven extension: the handler is the correct entry point even
        # though the step has a method hint. The paradigm gate keys on this flag.
        analyzer._parameter_node_wrapper = {
            "class_name": "PlateRegistrationParameterNode", "fields": {},
        }
        analyzer._widget_connections = [{
            "button_widget_name": "initialRegistrationPushButton",
            "signal": "clicked(bool)",
            "handler_method": "onInitialRegistrationPushButton",
            "logic_methods": ["rigid_transform"],
        }]
        step = {
            "step_id": "cb_step_5",
            "description": 'Click "Initial registration" button.',
            "sub_operations": [{
                "op_type": "extension_op",
                "widget_name": "initialRegistrationPushButton",
                "extension_method_hint": "rigid_transform",
            }],
        }
        tpl = analyzer._maybe_generate_button_handler_template(
            "SlicerOrbitSurgerySim", step, "PlateRegistrationLogic", "PlateRegistration"
        )
        self.assertIsNotNone(tpl)
        self.assertIn("onInitialRegistrationPushButton()", tpl)
        self.assertNotIn("rigid_transform", tpl)   # not the low-level helper
        self.assertNotIn("getattr(", tpl)          # CodeValidator-blocked
        _ast.parse(tpl)                             # valid Python

        # A step whose button has no scanned connection -> no override (falls back).
        plain = {
            "step_id": "cb_step_9",
            "description": "do x",
            "sub_operations": [{
                "op_type": "extension_op", "widget_name": "unconnectedButton",
                "extension_method_hint": "foo",
            }],
        }
        self.assertIsNone(
            analyzer._maybe_generate_button_handler_template("X", plain, "XLogic", "X")
        )

        # A toggled CHECKBOX step (no method hint) drives the handler and sets the
        # control — instead of becoming a stub the LLM fills with a bogus
        # destructive call (the cb_step_7 writeParameterDict failure).
        analyzer._widget_connections.append({
            "button_widget_name": "interactionTransformCheckbox",
            "signal": "toggled(bool)",
            "handler_method": "onInteractionTransform",
            "logic_methods": [],
        })
        cb_step = {
            "step_id": "cb_step_7",
            "description": "Tick the Enable 3D interaction transform handle checkbox.",
            "sub_operations": [{
                "op_type": "extension_op",
                "widget_name": "interactionTransformCheckbox",
                "target_value": True,
            }],
        }
        cb_tpl = analyzer._maybe_generate_button_handler_template(
            "SlicerOrbitSurgerySim", cb_step, "PlateRegistrationLogic", "PlateRegistration"
        )
        self.assertIsNotNone(cb_tpl)
        self.assertIn("onInteractionTransform", cb_tpl)
        self.assertIn("interactionTransformCheckbox", cb_tpl)  # sets the control state
        self.assertIn("checked = True", cb_tpl)
        self.assertNotIn("writeParameterDict", cb_tpl)
        self.assertNotIn("getattr(", cb_tpl)
        _ast.parse(cb_tpl)

        self.delayDisplay("Button-click and checkbox-toggle steps invoke the widget handler")

    def test_ClassicStepKeepsLogicMethodPath(self):
        """A classic extension's button step with a logic method is NOT handler-driven.

        Regression for BoneReconstructionPlanner: the handler-invocation feature
        (added for the wrapper-based Orbit extension) was firing on EVERY button
        step, so classic BRP steps like centerFibulaLine flipped from
        `logic.centerFibulaLine()` to `_widget.onCenterFibulaLineButton()`,
        dropping the param-node reference plumbing the logic method depends on
        (fibulaLine reference -> None -> crash). The paradigm gate keeps classic
        extensions on the logic-method path, while still driving the handler for
        UI-only toggles that have no logic method.
        """
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())
        # Classic extension: no @parameterNodeWrapper.
        analyzer._parameter_node_wrapper = None
        analyzer._widget_connections = [
            {
                "button_widget_name": "centerFibulaLineButton",
                "signal": "clicked(bool)",
                "handler_method": "onCenterFibulaLineButton",
                "logic_methods": ["centerFibulaLine"],
            },
            {
                "button_widget_name": "showHideButton",
                "signal": "toggled(bool)",
                "handler_method": "onShowHide",
                "logic_methods": [],
            },
        ]
        # Step WITH a resolvable logic method -> keep the logic-method path (None).
        method_step = {
            "step_id": "cb_step_22",
            "description": "Click 'Center fibula line' button.",
            "method_name": "centerFibulaLine",
            "sub_operations": [{
                "op_type": "extension_op",
                "widget_name": "centerFibulaLineButton",
                "extension_method_hint": "centerFibulaLine",
            }],
        }
        self.assertIsNone(
            analyzer._maybe_generate_button_handler_template(
                "BoneReconstructionPlanner", method_step,
                "BoneReconstructionPlannerLogic", "BoneReconstructionPlanner",
            )
        )
        # UI-only toggle with NO logic method -> the handler is the only entry
        # point, so it IS driven even for a classic extension.
        toggle_step = {
            "step_id": "cb_step_25",
            "description": "Toggle visibility.",
            "sub_operations": [{
                "op_type": "extension_op",
                "widget_name": "showHideButton",
                "target_value": True,
            }],
        }
        toggle_tpl = analyzer._maybe_generate_button_handler_template(
            "BoneReconstructionPlanner", toggle_step,
            "BoneReconstructionPlannerLogic", "BoneReconstructionPlanner",
        )
        self.assertIsNotNone(toggle_tpl)
        self.assertIn("onShowHide", toggle_tpl)
        self.delayDisplay("Classic logic-method step keeps the logic path; UI-only toggle drives the handler")

    def test_WidgetHandlerStepPassesContractAndProof(self):
        """Driving the widget handler is a first-class extension op in both validators.

        Regression for cb_step_7 (a no-logic-method checkbox step whose template
        drives _widget.onInteractionTransform()): the contract validator rejected
        it ('extension_op without an extension method contains Slicer API calls')
        and api_proof flagged the _widget.<handler>() call as an unproven receiver,
        so the repair loop kept rewriting the correct handler step into invented
        APIs (writeParameterDict / an 'InteractionFlag' role).
        """
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator
        from SlicerAIAgentLib.extension_cli_analyzer.api_proof import TypeProvenanceGraph

        analyzer = ExtensionCLIAnalyzer(llm_client=None, code_validator=CodeValidator())

        # Contract validator: a no-method extension_op that drives the widget
        # handler is NOT flagged as 'Slicer API without an extension method'.
        context = {"generator": {"sub_operations": [{"op_type": "extension_op"}], "operation_model": {}}}
        handler_code = (
            "import slicer\n"
            "_widget = slicer.util.getModuleWidget('PlateRegistration')\n"
            "_widget.onInteractionTransform()\n"
        )
        res = analyzer._validate_template_contract(
            "templates/cb_step_7.py.tpl", handler_code, context, {}
        )
        self.assertFalse(
            any("without an extension method" in e for e in res["errors"]), res["errors"]
        )
        # Sanity: raw Slicer API with no handler-drive is still flagged.
        raw = "import slicer\nslicer.mrmlScene.GetNodeByID('x')\n"
        res2 = analyzer._validate_template_contract("t.py.tpl", raw, context, {})
        self.assertTrue(
            any("without an extension method" in e for e in res2["errors"]), res2["errors"]
        )

        # api_proof: getModuleWidget() types to the module-widget marker, so a
        # handler call on it is provable (not an unproven receiver).
        graph = TypeProvenanceGraph(handler_code)
        cands = graph.receiver_candidates("_widget")
        self.assertTrue(cands and cands[0]["type"] == "_module_widget", cands)
        self.delayDisplay("Widget-handler step passes contract + api_proof")

    def test_GetNodesByClassReceiverTyping(self):
        """A receiver bound by a `GetNodesByClass` loop is typed and provable.

        Regression for the BoneReconstructionPlanner cb_step_8 failure: the
        generator's own slice-node idiom — iterate `scene.GetNodesByClass(<cls>)`
        and keep the match — left the receiver untyped, so every state-changing
        call on it was permanently `method_unproven` and the package failed even
        though the method was real. Both the static proof graph and the live-probe
        receiver inference must resolve the element class generically.
        """
        from SlicerAIAgentLib.extension_cli_analyzer.api_proof import (
            ApiProofValidator,
            TemplateApiAnalyzer,
            TypeProvenanceGraph,
        )
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer

        # The exact failing shape: None-init + direct iteration + alias + break.
        code = (
            "import slicer\n"
            "sliceNode = None\n"
            "for node in slicer.mrmlScene.GetNodesByClass('vtkMRMLSliceNode'):\n"
            "    if 'Red' in node.GetName():\n"
            "        sliceNode = node\n"
            "        break\n"
            "sliceNode.SetSliceResolutionMode(2)\n"
        )
        contracts = {
            "vtkMRMLSliceNode": {
                "methods": {
                    "GetName": {"exists": True, "effect": "read_only", "source": "syn"},
                    "SetSliceResolutionMode": {
                        "exists": True, "effect": "state_change", "source": "syn",
                    },
                    "SetBogusMethod": {
                        "exists": False, "effect": "state_change", "source": "syn",
                    },
                },
            },
        }

        # 1) Static graph: the loop var and its alias both resolve to the class.
        graph = TypeProvenanceGraph(code, return_contracts=contracts)
        self.assertIn(
            "vtkMRMLSliceNode",
            {c["type"] for c in graph.receiver_candidates("node")},
        )
        self.assertIn(
            "vtkMRMLSliceNode",
            {c["type"] for c in graph.receiver_candidates("sliceNode")},
        )

        # 2) End-to-end proof: the real state-changing method now clears (no
        #    blocking issues) instead of being a fatal method_unproven.
        proof = ApiProofValidator(type_contracts=contracts).validate_inventory(
            TemplateApiAnalyzer().analyze("templates/cb_step_8.py.tpl", code),
            graph,
        )
        self.assertFalse(
            [i for i in proof["issues"] if i.get("blocking")],
            proof["issues"],
        )

        # 3) Typing also enables correct rejection of a genuinely-missing method.
        bogus = code.replace("SetSliceResolutionMode(2)", "SetBogusMethod(2)")
        bogus_graph = TypeProvenanceGraph(bogus, return_contracts=contracts)
        bogus_proof = ApiProofValidator(type_contracts=contracts).validate_inventory(
            TemplateApiAnalyzer().analyze("templates/bogus.py.tpl", bogus),
            bogus_graph,
        )
        self.assertTrue(any(
            i["issue_type"] == "InvalidApiCall" and i["blocking"]
            for i in bogus_proof["issues"]
        ), bogus_proof["issues"])

        # 4) Assign-then-iterate form types the element too (no regression to the
        #    lowercase slicer.util.getNodesByClass path).
        assign_iter = (
            "import slicer\n"
            "coll = slicer.mrmlScene.GetNodesByClass('vtkMRMLModelNode')\n"
            "for m in coll:\n"
            "    chosen = m\n"
        )
        ai_graph = TypeProvenanceGraph(assign_iter)
        self.assertIn(
            "vtkMRMLModelNode",
            {c["type"] for c in ai_graph.receiver_candidates("chosen")},
        )

        # 5) Live-probe receiver inference resolves the same receiver and emits a
        #    class-targeted probe spec for the call.
        inferred = ExtensionCLIAnalyzer._infer_probe_receiver_types(code)
        self.assertEqual(inferred.get("node"), "vtkMRMLSliceNode")
        self.assertEqual(inferred.get("sliceNode"), "vtkMRMLSliceNode")
        specs = ExtensionCLIAnalyzer._extract_api_probe_specs(code)
        self.assertTrue(any(
            s.get("attr") == "SetSliceResolutionMode"
            and s.get("receiver_expr") == "slicer.vtkMRMLSliceNode"
            for s in specs
        ), specs)

    def test_ExtensionCLIAnalyzerContracts(self):
        """Test generic workflow-generation validation contracts."""
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
        from SlicerAIAgentLib.CodeValidator import CodeValidator

        analyzer = ExtensionCLIAnalyzer(
            llm_client=None,
            code_validator=CodeValidator(),
        )

        # Active generation/repair sources must not accumulate
        # extension-specific domain vocabulary.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        guarded_sources = [
            "SlicerAIAgentLib/extension_cli_analyzer/repair_loop.py",
            "SlicerAIAgentLib/extension_cli_analyzer/workflow_templates.py",
            "SlicerAIAgentLib/extension_cli_analyzer/template_helpers.py",
            "SlicerAIAgentLib/extension_cli_analyzer/stage4_decomposition.py",
            "SlicerAIAgentLib/extension_cli_analyzer/cookbook_mapping.py",
            "SlicerAIAgentLib/extension_cli_analyzer/cross_stage.py",
            "SlicerAIAgentLib/extension_cli_analyzer/schemas.py",
            "SlicerAIAgentLib/extension_cli_analyzer/prompt_validation.py",
            "SlicerAIAgentLib/extension_cli_analyzer/contract_audit.py",
            "SlicerAIAgentLib/extension_cli_analyzer/api_proof.py",
            "SlicerAIAgentLib/slicer_op_generator/common.py",
        ]
        forbidden_domain_terms = {
            "bonereconstructionplanner", "fibula", "mandible", "mandibular",
            "hemimandibulectomy", "pelvicfractureplanning",
        }
        for relative_path in guarded_sources:
            with open(os.path.join(project_root, relative_path), "r", encoding="utf-8") as source_file:
                source_text = source_file.read().lower()
            for forbidden_term in forbidden_domain_terms:
                self.assertNotIn(forbidden_term, source_text, relative_path)

        from SlicerAIAgentLib.extension_cli_analyzer.repair_loop import (
            _SEMANTIC_REPAIR_RECIPES,
        )
        for recipe_id, recipe in _SEMANTIC_REPAIR_RECIPES.items():
            self.assertEqual(recipe.get("knowledge_level"), "behavioral_contract", recipe_id)
            self.assertTrue(recipe.get("behavior_requirements"), recipe_id)
            self.assertNotIn("required_api_patterns", recipe, recipe_id)
            self.assertNotIn("forbidden_api_patterns", recipe, recipe_id)
            self.assertEqual(
                recipe.get("retrieval_query_provenance"),
                "generic_slicer_search_hint",
                recipe_id,
            )

        # v2 issue typing and workflow contracts are extension-agnostic.
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/step.py.tpl: Unresolved live API probe failure for 'slicer.app.bad': API call does not exist"
            ),
            "InvalidSlicerAPI",
        )
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/step.py.tpl: Template references parameter role 'line' with node class vtkMRMLMarkupsCurveNode but metadata expects vtkMRMLMarkupsLineNode"
            ),
            "NodeClassMismatch",
        )
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/step.py.tpl: CallableReferenceMisuse: used slicer.util.selectedModule without calling it"
            ),
            "CallableReferenceMisuse",
        )
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/step.py.tpl: BadInstructionText: invalid user-facing instruction 'Please None'"
            ),
            "BadInstructionText",
        )
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/step.py.tpl: Parameter 'x' for logic.run() uses low-confidence typed_read_safe_fallback default '0.0'"
            ),
            "LowConfidenceFallback",
        )
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/cb_step_1_slicer.py.tpl: Slice intersection visibility step does not provide evidence-backed intersection visibility behavior"
            ),
            "SlicerSemanticError",
        )
        self.assertFalse(analyzer._is_extension_module_import("vtkMRMLApplicationLogic"))
        self.assertTrue(analyzer._is_extension_module_import("FakeExtension"))

        contract_audit = analyzer._audit_workflow_contract({
            "steps": [{
                "step_id": "cb_step_1",
                "description": "Toggle on slice visibility in 3D view for the Red slice",
                "operation_model": {
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "operations": [{
                    "description": "Toggle on slice visibility in 3D view",
                    "intent": "slice_intersection_visibility",
                }],
            }]
        })
        self.assertFalse(contract_audit["valid"])
        self.assertTrue(any("ContractFidelity" in e for e in contract_audit["errors"]))

        contract_audit = analyzer._audit_workflow_contract({
            "steps": [{
                "step_id": "cb_step_1",
                "description": "Show slice intersections and enable interaction",
                "operation_model": {
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "operations": [{
                    "description": "Show slice intersections",
                    "intent": "slice_intersection_visibility",
                }],
            }]
        })
        self.assertTrue(contract_audit["valid"])

        self.assertEqual(
            analyzer._classify_validation_issue(
                "Workflow contract audit failed: ContractFidelity: cb_step_1 intent mismatch"
            ),
            "ContractFidelity",
        )
        routed_issues = analyzer._validation_issues_from_result(
            {
                "errors": [
                    "workflow_contract.json: ContractFidelity: cb_step_1 intent mismatch"
                ]
            },
            generators=[],
            workflow_contract={"steps": []},
        )
        self.assertEqual(routed_issues[0]["issue_class"], "contract")
        self.assertEqual(routed_issues[0]["repair_route"], "rebuild_contract")

        manifest = {"status": "validation_failed"}
        analyzer._workflow_metadata = {
            "revision_validation_status": "passed",
            "verify_repair": {"status": "failed", "used_outer_revision": False},
            "validation_state": {"contract_audit_valid": True},
            "api_proof_coverage": {
                "inventory_complete": True,
                "invalid_calls": 0,
                "blocking_unproven_calls": 0,
                "warning_unproven_reads": 0,
            },
        }
        analyzer._finalize_package_validation_state(manifest, {
            "valid": True,
            "api_proof_coverage": analyzer._workflow_metadata["api_proof_coverage"],
        })
        self.assertEqual(manifest["status"], "validated")
        self.assertTrue(manifest["validation_state"]["overall_valid"])
        self.assertEqual(
            analyzer._workflow_metadata["verify_repair"]["status"],
            "passed_after_revision",
        )

        from SlicerAIAgentLib.skill_indexer.retriever_builder import _SOURCE_TYPE_WEIGHTS
        self.assertGreater(_SOURCE_TYPE_WEIGHTS.get("doc_example", 0), 1.0)

        semantic_issues = analyzer._validation_issues_from_result(
            {
                "errors": [
                    "templates/cb_step_1_slicer.py.tpl: Slice intersection visibility step does not provide evidence-backed intersection visibility behavior"
                ]
            },
            generators=[{
                "template_file": "templates/cb_step_1_slicer.py.tpl",
                "operation_type": "slicer_op",
                "operation_model": {
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Show slice intersections",
                    "slicer_op_category": "crosshair",
                }],
            }],
            workflow_contract={
                "steps": [{
                    "step_id": "cb_step_1",
                    "description": "Show slice intersections",
                }]
            },
        )
        self.assertEqual(semantic_issues[0]["step_id"], "cb_step_1")
        self.assertEqual(semantic_issues[0]["repair_strategy"], "reground_slicer_op")
        self.assertEqual(
            semantic_issues[0]["semantic_recipe_id"],
            "slice_intersection_visibility",
        )
        self.assertTrue(
            semantic_issues[0]["semantic_recipe"]["behavior_requirements"]
        )
        self.assertEqual(
            semantic_issues[0]["semantic_recipe"]["knowledge_level"],
            "behavioral_contract",
        )
        self.assertNotIn("required_api_patterns", semantic_issues[0]["semantic_recipe"])
        self.assertNotIn("forbidden_api_patterns", semantic_issues[0]["semantic_recipe"])
        original_reground = analyzer._reground_slicer_template
        try:
            analyzer._reground_slicer_template = lambda issue, templates, generators: (
                issue["template_key"],
                "apply_source_backed_intersection_state()\n",
                {"source": "mock_reground"},
            )
            repaired, strategy_results = analyzer._execute_repair_plan(
                "FakeExt",
                {"templates/cb_step_1_slicer.py.tpl": "bad_code()\n"},
                [{
                    "template_file": "templates/cb_step_1_slicer.py.tpl",
                    "operation_type": "slicer_op",
                    "sub_operations": [{"op_type": "slicer_op"}],
                }],
                {"steps": []},
                semantic_issues,
                None,
            )
            self.assertIn(
                "apply_source_backed_intersection_state",
                repaired["templates/cb_step_1_slicer.py.tpl"],
            )
            self.assertTrue(strategy_results[0]["changed"])
        finally:
            analyzer._reground_slicer_template = original_reground

        callable_contract = analyzer._validate_callable_reference_misuse(
            "_active_module = slicer.util.selectedModule\n"
        )
        self.assertTrue(any("CallableReferenceMisuse" in e for e in callable_contract["errors"]))
        callable_contract = analyzer._validate_callable_reference_misuse(
            "_active_module = slicer.util.selectedModule()\n"
        )
        self.assertFalse(callable_contract["errors"])

        instruction_contract = analyzer._validate_user_instruction_text(
            "print('[FakeExt] Please None')\n"
        )
        self.assertTrue(any("BadInstructionText" in e for e in instruction_contract["errors"]))
        self.assertEqual(
            analyzer._sanitize_interaction_instruction(None, fallback="Adjust the existing plane."),
            "Adjust the existing plane.",
        )

        analyzer._workflow_metadata = {
            "parameter_bindings": {
                "targetLine": {
                    "node_class": "vtkMRMLMarkupsLineNode",
                    "methods": ["run"],
                    "accesses": ["node_reference_read"],
                }
            },
            "operation_model": {
                "cb_step_1": {"step_type": "extension_op", "operation_intents": ["extension_call"]}
            },
        }
        v2_contract = analyzer._build_workflow_contract_v2(
            "FakeExt",
            {
                "entry_module": "/tmp/FakeExt/FakeExt.py",
                "source_path": "/tmp/FakeExt",
                "logic_class": {"class_name": "FakeExtLogic"},
            },
            "/tmp/FakeExt.md",
            {"methods": []},
            {
                "steps": [{
                    "step_id": "cb_step_1",
                    "operation_type": "extension_op",
                    "description": "Run extension method",
                    "method_name": "run",
                    "depends_on": [],
                    "code_template": "templates/cb_step_1.py.tpl",
                    "sub_operations": [{
                        "op_type": "extension_op",
                        "description": "Run extension method",
                        "extension_method_hint": "run",
                        "confidence": "high",
                    }],
                }]
            },
        )
        self.assertEqual(v2_contract["schema_version"], 2)
        self.assertEqual(v2_contract["pipeline_version"], "agentic-cli-v2")
        self.assertEqual(v2_contract["steps"][0]["operation_type"], "extension_op")
        self.assertEqual(
            v2_contract["steps"][0]["parameter_roles"]["targetLine"]["node_class"],
            "vtkMRMLMarkupsLineNode",
        )

        analyzer._workflow_metadata = {
            "parameter_bindings": {
                "initialSpace": {
                    "value_types": ["float"],
                    "node_class": "",
                },
            },
            "parameter_defaults": {
                "initialSpace": {
                    "value": "0.0",
                    "source": "typed_read_safe_fallback",
                    "confidence": "low",
                },
            },
            "parameter_method_dependencies": {
                "run": {
                    "parameter_roles": ["initialSpace"],
                },
            },
            "choice_bindings": {},
        }
        scalar_contract = analyzer._validate_scalar_parameter_contract("logic.run()\n")
        self.assertTrue(any("low-confidence" in e for e in scalar_contract["errors"]))
        analyzer._workflow_metadata["choice_bindings"] = {
            "cb_step_1": {"parameter_name": "initialSpace"}
        }
        scalar_contract = analyzer._validate_scalar_parameter_contract("logic.run()\n")
        self.assertFalse(scalar_contract["errors"])
        self.assertTrue(analyzer._template_sets_parameter(
            (
                "roles = ['initialSpace', 'targetSpacing']\n"
                "for role in roles:\n"
                "    parameterNode.SetParameter(role, '1.0')\n"
            ),
            "targetSpacing",
        ))
        self.assertFalse(analyzer._template_sets_parameter(
            (
                "roles = get_runtime_roles()\n"
                "for role in roles:\n"
                "    parameterNode.SetParameter(role, '1.0')\n"
            ),
            "targetSpacing",
        ))

        analyzer._workflow_metadata = {
            "parameter_bindings": {
                "initialSpace": {
                    "value_types": ["float"],
                    "node_class": "",
                },
            },
            "parameter_defaults": {
                "initialSpace": {
                    "value": "0.0",
                    "source": "typed_read_safe_fallback",
                    "confidence": "low",
                },
            },
            "parameter_method_dependencies": {
                "consume": {
                    "parameter_roles": ["initialSpace"],
                },
            },
            "method_parameter_effects": {
                "produce": {
                    "writes": ["initialSpace"],
                    "reads": [],
                },
                "consume": {
                    "writes": [],
                    "reads": ["initialSpace"],
                },
            },
            "choice_bindings": {},
            "interaction_bindings": {},
            "workflow_steps": [
                {"step_id": "cb_step_1"},
                {"step_id": "cb_step_2"},
            ],
        }
        contract_audit = analyzer._audit_workflow_contract({
            "steps": [
                {
                    "step_id": "cb_step_1",
                    "description": "Prepare required value",
                    "operations": [{"extension_method_hint": "produce"}],
                },
                {
                    "step_id": "cb_step_2",
                    "description": "Use required value",
                    "operations": [{"extension_method_hint": "consume"}],
                },
            ]
        })
        self.assertTrue(contract_audit["valid"], contract_audit.get("errors"))
        scalar_contract = analyzer._validate_scalar_parameter_contract(
            "logic.consume()\n",
            step_id="cb_step_2",
        )
        self.assertFalse(scalar_contract["errors"])

        analyzer._workflow_metadata["workflow_steps"] = [
            {"step_id": "cb_step_1"},
            {"step_id": "cb_step_2"},
        ]
        contract_audit = analyzer._audit_workflow_contract({
            "steps": [
                {
                    "step_id": "cb_step_1",
                    "description": "Use required value",
                    "operations": [{"extension_method_hint": "consume"}],
                },
                {
                    "step_id": "cb_step_2",
                    "description": "Prepare required value",
                    "operations": [{"extension_method_hint": "produce"}],
                },
            ]
        })
        self.assertFalse(contract_audit["valid"])
        self.assertTrue(any("WorkflowState" in e for e in contract_audit["errors"]))

        resolved_dynamic_code = (
            "redSliceNode = slicer.vtkMRMLSliceNode.SafeDownCast("
            "slicer.mrmlScene.GetSingletonNode('Red', 'vtkMRMLSliceNode'))\n"
            "if redSliceNode is not None:\n"
            "    redSliceNode.SetSliceVisible(True)\n"
        )
        self.assertTrue(any(
            spec.get("attr") == "SetSliceVisible"
            for spec in analyzer._extract_api_probe_specs(resolved_dynamic_code)
        ))
        self.assertFalse(analyzer._extract_unresolved_api_probe_specs(resolved_dynamic_code))
        typed_display_code = (
            "displayNode = slicer.vtkMRMLMarkupsDisplayNode.SafeDownCast("
            "curveNode.GetDisplayNode())\n"
            "if displayNode is not None:\n"
            "    displayNode.AddViewNodeID('vtkMRMLViewNode1')\n"
        )
        typed_display_specs = analyzer._extract_api_probe_specs(typed_display_code)
        self.assertTrue(any(
            spec.get("chain") == "displayNode.AddViewNodeID"
            and spec.get("receiver_expr") == "slicer.vtkMRMLMarkupsDisplayNode"
            and spec.get("proof_kind") == "class_type"
            for spec in typed_display_specs
        ))
        self.assertFalse(analyzer._extract_unresolved_api_probe_specs(typed_display_code))
        unresolved_dynamic_code = (
            "displayNode = get_runtime_display_node()\n"
            "displayNode.SetVisibility(True)\n"
        )
        unresolved_dynamic_specs = analyzer._extract_unresolved_api_probe_specs(unresolved_dynamic_code)
        self.assertTrue(unresolved_dynamic_specs)
        self.assertEqual(unresolved_dynamic_specs[0]["proof_kind"], "unproven_receiver")
        self.assertEqual(
            analyzer._classify_validation_issue(
                "templates/step.py.tpl: Unproven dynamic API receiver for 'displayNode.SetVisibility'"
            ),
            "UnprovenReceiver",
        )
        sanitized = analyzer._sanitize_templates({
            "templates/slice.py.tpl": (
                "mode = vtkMRMLSliceNode.SliceFOVMatchVolumesSpacingMatch2DView\n"
            )
        })
        self.assertIn(
            "slicer.vtkMRMLSliceNode.SliceFOVMatchVolumesSpacingMatch2DView",
            sanitized["templates/slice.py.tpl"],
        )

        routed_issues = analyzer._validation_issues_from_result(
            {
                "errors": [
                    "templates/step.py.tpl: Parameter 'initialSpace' for logic.consume() uses low-confidence typed_read_safe_fallback default '0.0'"
                ]
            },
            generators=[{"template_file": "templates/step.py.tpl"}],
            workflow_contract={"steps": []},
        )
        self.assertEqual(routed_issues[0]["issue_class"], "dataflow")
        self.assertEqual(routed_issues[0]["repair_route"], "rebuild_dataflow")

        # Stage 4 deterministically repairs schema-level contract drift while
        # leaving genuinely semantic source-selection errors for the LLM loop.
        stage4_context = {
            "steps": [{
                "step_number": 1,
                "operation_type": "user_choice",
                "description": "How many items?",
            }],
            "logic_methods": [],
            "extension_functions": [],
            "widgets": [],
            "ui_parameter_bindings": [],
            "parameter_roles": [{"role": "sourceBackedNode"}],
            "allowed_slicer_op_categories": [],
            "allowed_interaction_kinds": ["none"],
            "allowed_node_classes": ["vtkMRMLMarkupsPlaneNode"],
            "allowed_operation_intents": [],
            "allowed_node_role_kinds": ["choice_input", "interaction_output"],
        }
        normalized_stage4, normalization_notes = analyzer._normalize_stage4_semantic_result(
            {
                "steps": [{
                    "step_number": 1,
                    "operation_type": "extension_op",
                    "extension_method_hint": "inventedMethod",
                    "confidence": "high",
                    "interaction_kind": "none",
                    "operation_intents": [],
                    "choice": None,
                    "node_roles": [
                        {
                            "role_kind": "choice_input",
                            "node_class": "",
                            "parameter_name": "runtimeCount",
                        },
                        {
                            "role_kind": "interaction_output",
                            "node_class": "vtkMRMLMarkupsPlaneNode",
                            "parameter_name": "inventedNodeRole",
                        },
                        {
                            "role_kind": "interaction_output",
                            "node_class": "vtkMRMLMarkupsPlaneNode",
                            "parameter_name": "sourceBackedNode",
                        },
                    ],
                }],
                "repeat_blocks": [],
            },
            stage4_context,
        )
        normalized_step = normalized_stage4["steps"][0]
        self.assertEqual(normalized_step["operation_type"], "user_choice")
        self.assertEqual(normalized_step["choice"]["question"], "How many items?")
        self.assertEqual(normalized_step["choice"]["parameter_name"], "choice_step_1")
        self.assertEqual(normalized_step["node_roles"], [
            {
                "role_kind": "interaction_output",
                "node_class": "vtkMRMLMarkupsPlaneNode",
                "parameter_name": "",
            },
            {
                "role_kind": "interaction_output",
                "node_class": "vtkMRMLMarkupsPlaneNode",
                "parameter_name": "sourceBackedNode",
            },
        ])
        self.assertTrue(normalization_notes)
        stage4_errors = analyzer._validate_stage4_semantic_result(
            normalized_stage4, stage4_context
        )
        self.assertTrue(any("unknown method" in error for error in stage4_errors))
        self.assertFalse(any("node role" in error for error in stage4_errors))

        # Stage 4 uses LLM semantics for repeat intent while preserving the
        # authoritative one-operation-per-step cookbook contract.
        from SlicerAIAgentLib.CookbookParser import CookbookParser
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fp:
            fp.write(
                "1. [op=user_choice] How many items?\n"
                "2. [op=extension_op] Create one item.\n"
                "3. [op=user_interaction] Place one item, repeating the create "
                "and place actions N times.\n"
                "4. [op=user_interaction] Adjust the existing result.\n"
                "5. [op=extension_op] Recompute the result; repeat steps 4-5 "
                "until the result is satisfactory.\n"
                "6. [op=extension_op] Finalize.\n"
            )
            repeat_cookbook_path = fp.name
        try:
            repeat_cookbook = CookbookParser().parse(repeat_cookbook_path)
            semantic_steps = []
            for step in repeat_cookbook.steps:
                semantic_steps.append({
                    "step_number": step.step_number,
                    "operation_type": step.operation_type,
                    "semantic_intent": step.description,
                    "extension_method_hint": None,
                    "widget_name": None,
                    "ui_parameter_binding": None,
                    "slicer_op_category": None,
                    "slicer_api_keywords": [],
                    "interaction_kind": (
                        "markup_placement"
                        if step.operation_type == "user_interaction" else "none"
                    ),
                    "interaction_type": None,
                    "node_class": (
                        "vtkMRMLMarkupsFiducialNode"
                        if step.operation_type == "user_interaction" else None
                    ),
                    "placement_instructions": None,
                    "choice": (
                        {
                            "question": step.description,
                            "choices": [],
                            "parameter_name": "item_count",
                            "default_value": None,
                            "value_kind": "integer",
                        }
                        if step.operation_type == "user_choice" else None
                    ),
                    "is_optional": False,
                    "operation_intents": [],
                    "node_roles": [],
                    "confidence": "high",
                    "evidence_ids": [],
                })
            semantic_result = {
                "steps": semantic_steps,
                "repeat_blocks": [
                    {
                        "repeat_id": "create_items",
                        "body_steps": [2, 3],
                        "controller": {
                            "kind": "count",
                            "source_step": 1,
                            "prompt": "",
                            "exit_value": None,
                        },
                        "evidence_step_ids": [1, 3],
                        "confidence": "high",
                    },
                    {
                        "repeat_id": "refine",
                        "body_steps": [4, 5],
                        "controller": {
                            "kind": "until_choice",
                            "source_step": None,
                            "prompt": "Is the result satisfactory?",
                            "exit_value": True,
                        },
                        "evidence_step_ids": [5],
                        "confidence": "high",
                    },
                ],
            }
            analyzer._call_llm = lambda prompt: json.dumps(semantic_result)
            stage_map = analyzer._stage4_cookbook_decomposition(
                repeat_cookbook, {"methods": []}
            )
            repeat_blocks = stage_map["repeat_blocks"]
            self.assertEqual(len(repeat_blocks), 2)
            self.assertEqual(
                repeat_blocks[0]["body_steps"], ["cb_step_2", "cb_step_3"]
            )
            self.assertEqual(
                repeat_blocks[0]["controller"]["kind"], "count"
            )
            self.assertEqual(
                repeat_blocks[1]["controller"]["prompt"],
                "Is the result satisfactory?",
            )
            self.assertEqual(
                repeat_blocks[1]["body_steps"], ["cb_step_4", "cb_step_5"]
            )
            self.assertEqual(repeat_blocks[1]["controller"]["kind"], "until_choice")
            self.assertEqual(
                [step.operation_type for step in repeat_cookbook.steps],
                [
                    "user_choice", "extension_op", "user_interaction",
                    "user_interaction", "extension_op", "extension_op",
                ],
            )
        finally:
            os.unlink(repeat_cookbook_path)

        # API repair acceptance must depend on valid Python, not on imports.
        analyzer._call_llm = lambda prompt: "node.SetVisibility(True)"
        repaired = analyzer._revise_template_for_api(
            "templates/step.py.tpl",
            "node.ToggleVisibility()",
            [{"chain": "node.ToggleVisibility", "error": "missing"}],
        )
        self.assertEqual(repaired, "node.SetVisibility(True)")

        # Evidence-complete API proof inventory and generic type propagation.
        from SlicerAIAgentLib.extension_cli_analyzer.api_proof import (
            ApiProofValidator,
            RepairCoordinator,
            TemplateApiAnalyzer,
            TypeProvenanceGraph,
        )
        provenance_code = (
            "from typing import cast, List\n"
            "def make_node() -> vtkGenericNode:\n"
            "    return slicer.vtkGenericNode()\n"
            "first = make_node()\n"
            "alias = first\n"
            "typed = cast(vtkGenericNode, alias)\n"
            "items: List[vtkGenericNode] = [typed]\n"
            "for item in items:\n"
            "    child = item.GetChild()\n"
            "    child.SetValue(1)\n"
        )
        contracts = {
            "vtkGenericNode": {
                "methods": {
                    "GetChild": {
                        "exists": True,
                        "effect": "read_only",
                        "return_type": "vtkGenericChild",
                        "source": "synthetic_wrapper_metadata",
                    },
                },
            },
            "vtkGenericChild": {
                "methods": {
                    "SetValue": {
                        "exists": True,
                        "effect": "state_change",
                        "source": "synthetic_wrapper_metadata",
                    },
                    "MissingMethod": {
                        "exists": False,
                        "effect": "state_change",
                        "source": "synthetic_wrapper_metadata",
                    },
                },
            },
        }
        graph = TypeProvenanceGraph(provenance_code, return_contracts=contracts)
        self.assertEqual(graph.receiver_candidates("alias")[0]["type"], "vtkGenericNode")
        self.assertEqual(graph.receiver_candidates("typed")[0]["type"], "vtkGenericNode")
        self.assertEqual(graph.receiver_candidates("item")[0]["type"], "vtkGenericNode")
        self.assertEqual(graph.receiver_candidates("child")[0]["type"], "vtkGenericChild")

        inventory = TemplateApiAnalyzer().analyze(
            "templates/generic.py.tpl", provenance_code
        )
        self.assertTrue(inventory["complete"])
        call_ids = [call["call_id"] for call in inventory["calls"]]
        self.assertEqual(len(call_ids), len(set(call_ids)))
        self.assertEqual(len(call_ids), len([
            node for node in ast.walk(ast.parse(provenance_code))
            if isinstance(node, ast.Call)
        ]))

        proof_code = (
            "node = slicer.vtkGenericChild()\n"
            "node.SetValue(1)\n"
        )
        proof_inventory = TemplateApiAnalyzer().analyze(
            "templates/proven.py.tpl", proof_code
        )
        proof_graph = TypeProvenanceGraph(proof_code, return_contracts=contracts)
        proof = ApiProofValidator(type_contracts=contracts).validate_inventory(
            proof_inventory, proof_graph
        )
        self.assertFalse(proof["issues"], proof["issues"])

        invalid_code = (
            "node = slicer.vtkGenericChild()\n"
            "node.MissingMethod()\n"
        )
        invalid_proof = ApiProofValidator(type_contracts=contracts).validate_inventory(
            TemplateApiAnalyzer().analyze("templates/invalid.py.tpl", invalid_code),
            TypeProvenanceGraph(invalid_code, return_contracts=contracts),
        )
        self.assertTrue(any(
            issue["issue_type"] == "InvalidApiCall" and issue["blocking"]
            for issue in invalid_proof["issues"]
        ))

        unknown_write = "runtime_value.SetValue(1)\n"
        unknown_write_proof = ApiProofValidator().validate_inventory(
            TemplateApiAnalyzer().analyze("templates/write.py.tpl", unknown_write),
            TypeProvenanceGraph(unknown_write),
        )
        self.assertTrue(unknown_write_proof["issues"][0]["blocking"])
        unknown_read = "value = runtime_value.GetValue()\n"
        unknown_read_proof = ApiProofValidator().validate_inventory(
            TemplateApiAnalyzer().analyze("templates/read.py.tpl", unknown_read),
            TypeProvenanceGraph(unknown_read),
        )
        self.assertFalse(unknown_read_proof["issues"][0]["blocking"])
        self.assertFalse(
            TemplateApiAnalyzer().analyze("templates/broken.py.tpl", "node.SetValue(")["complete"]
        )

        coordinator = RepairCoordinator()
        self.assertTrue(coordinator.can_attempt("call", "source_backed_cast"))
        coordinator.record("call", "source_backed_cast", "failed")
        self.assertFalse(coordinator.can_attempt("call", "source_backed_cast"))

        analyzer._workflow_metadata = {
            "api_type_contracts": contracts,
            "parameter_bindings": {},
            "extension_callable_inventory": {
                "logic_methods": [],
                "module_functions": [],
            },
        }
        proof_report = analyzer._build_api_proof_report({
            "templates/proven.py.tpl": proof_code,
            "templates/read.py.tpl": unknown_read,
        })
        self.assertTrue(proof_report["valid"])
        self.assertEqual(
            proof_report["api_proof_coverage"]["warning_unproven_reads"], 1
        )
        warning_manifest = {}
        analyzer._finalize_package_validation_state(warning_manifest, {
            "valid": True,
            "api_proof_coverage": proof_report["api_proof_coverage"],
        })
        self.assertEqual(warning_manifest["status"], "validated_with_warnings")

        incomplete_report = analyzer._build_api_proof_report({
            "templates/broken.py.tpl": "node.SetValue("
        })
        self.assertFalse(incomplete_report["valid"])
        self.assertFalse(
            incomplete_report["api_proof_coverage"]["inventory_complete"]
        )

        structured_issues = analyzer._validation_issues_from_result({
            "errors": [],
            "validation_issues": unknown_write_proof["issues"],
        })
        self.assertEqual(structured_issues[0]["issue_type"], "UnprovenReceiver")
        self.assertEqual(
            structured_issues[0]["evidence_diagnosis"]["diagnosis"],
            "receiver_type_unproven",
        )

        live_alias_code = (
            "layoutManager = slicer.app.layoutManager()\n"
            "layoutManager.setLayout(1)\n"
        )
        live_alias_probe = {
            "probed": 2,
            "failures": [],
            "api_probe_coverage": {
                "resolved": [
                    {
                        "chain": "slicer.app.layoutManager",
                        "receiver_expr": "slicer.app",
                        "original_receiver_expr": "slicer.app",
                        "expanded_receiver_expr": "slicer.app",
                        "attr": "layoutManager",
                        "method_exists": True,
                        "live_receiver_type": "qSlicerApplication",
                    },
                    {
                        "chain": "slicer.app.layoutManager().setLayout",
                        "receiver_expr": "slicer.app.layoutManager()",
                        "original_receiver_expr": "layoutManager",
                        "expanded_receiver_expr": "slicer.app.layoutManager()",
                        "attr": "setLayout",
                        "method_exists": True,
                        "live_receiver_type": "qSlicerLayoutManager",
                    },
                ],
            },
        }
        analyzer._workflow_metadata = {
            "api_type_contracts": {},
            "parameter_bindings": {},
            "extension_callable_inventory": {},
        }
        live_alias_report = analyzer._build_api_proof_report(
            {"templates/live.py.tpl": live_alias_code},
            api_probe_result=live_alias_probe,
        )
        self.assertTrue(live_alias_report["valid"], live_alias_report["issues"])

        invalid_alias_report = analyzer._build_api_proof_report(
            {
                "templates/invalid_live.py.tpl": (
                    "node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLModelNode')\n"
                    "node.BadMethod()\n"
                ),
            },
            api_probe_result={
                "probed": 1,
                "failures": [{
                    "chain": "slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLModelNode').BadMethod",
                    "receiver_expr": (
                        "slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLModelNode')"
                    ),
                    "original_receiver_expr": "node",
                    "expanded_receiver_expr": (
                        "slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLModelNode')"
                    ),
                    "attr": "BadMethod",
                    "receiver_type": "vtkMRMLModelNode",
                }],
                "api_probe_coverage": {"resolved": []},
            },
        )
        self.assertFalse(invalid_alias_report["valid"])
        self.assertTrue(any(
            issue.get("issue_type") == "InvalidApiCall"
            and issue.get("method") == "BadMethod"
            for issue in invalid_alias_report["issues"]
        ))

        imported_logic_code = (
            "import GenericExtension\n"
            "logic = GenericExtension.GenericLogic()\n"
            "logic.run()\n"
            "parameterNode = logic.getParameterNode()\n"
            "parameterNode.SetParameter('role', 'value')\n"
        )
        imported_graph = TypeProvenanceGraph(
            imported_logic_code,
            logic_class_name="GenericLogic",
        )
        self.assertEqual(
            imported_graph.receiver_candidates("logic")[0]["type"],
            "GenericLogic",
        )
        self.assertEqual(
            imported_graph.receiver_candidates("parameterNode")[0]["type"],
            "vtkMRMLScriptedModuleNode",
        )

        analyzer._workflow_metadata = {
            "extension_callable_inventory": {
                "logic_methods": ["run"],
                "module_functions": [],
            },
            "logic_class_name": "GenericLogic",
            "parameter_bindings": {},
            "api_type_contracts": {
                "vtkMRMLScriptedModuleNode": {
                    "methods": {
                        "SetParameter": {
                            "exists": True,
                            "effect": "state_change",
                            "source": "wrapper_metadata",
                        },
                    },
                },
            },
        }
        imported_logic_report = analyzer._build_api_proof_report({
            "templates/logic.py.tpl": imported_logic_code,
        })
        self.assertTrue(imported_logic_report["valid"], imported_logic_report["issues"])

        view_adjustment_contract = analyzer._validate_template_contract(
            "templates/adjust_pre.py.tpl",
            "print('Adjust the existing object, then press Done.')\n",
            {
                "role": "pre",
                "generator": {
                    "pre_template_file": "templates/adjust_pre.py.tpl",
                    "interaction_descriptor": {
                        "interaction_kind": "view_adjustment",
                        "node_class": "",
                    },
                    "sub_operations": [],
                },
            },
            {},
        )
        self.assertFalse(
            any("stub" in error.lower() for error in view_adjustment_contract["errors"])
        )

        precondition_code = (
            "import slicer\n"
            "from GenericExtension import GenericLogic\n"
            "# precondition:begin\n"
            "slicer.util.selectModule('GenericExtension')\n"
            "# precondition:end\n"
            "logic = GenericLogic()\n"
            "logic.run()\n"
        )
        analyzer._workflow_metadata["extension_callable_inventory"]["logic_methods"] = ["run"]
        precondition_contract = analyzer._validate_template_contract(
            "templates/logic.py.tpl",
            precondition_code,
            {
                "role": "code",
                "generator": {
                    "template_file": "templates/logic.py.tpl",
                    "method_name": "run",
                    "operation_model": {"allow_module_switch": False},
                    "sub_operations": [{
                        "op_type": "extension_op",
                        "extension_method_hint": "run",
                        "description": "Run source-backed logic",
                    }],
                },
            },
            {},
        )
        self.assertFalse(any(
            "switches the active Slicer module" in error
            for error in precondition_contract["errors"]
        ))

        # Live API probe failures remain blocking through validation.
        validation = analyzer._stage9_validate(
            {"templates/step.py.tpl": "slicer.app.badApi()\n"},
            [{"template_file": "templates/step.py.tpl", "step_type": "automated"}],
            logic_analysis=None,
            api_probe_result={
                "unresolved_failures": [
                    {
                        "template": "templates/step.py.tpl",
                        "chain": "slicer.app.badApi",
                        "error": "API call does not exist",
                    }
                ]
            },
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("InvalidApiCall" in e for e in validation["errors"]))

        # Unsupported state-changing calls block without prescribing an
        # API-error-specific replacement.
        validation = analyzer._stage9_validate(
            {"templates/state.py.tpl": "receiver.SetUnsupportedState(True)\n"},
            [{
                "template_file": "templates/state.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Apply a state change",
                    "slicer_op_category": "generic_state_change",
                    "confidence": "high",
                }],
                "operation_model": {
                    "invokes_slicer_api": True,
                    "implementation_uses_slicer_api": True,
                },
            }],
            logic_analysis=None,
        )
        self.assertFalse(validation["valid"])
        unsupported_errors = [
            e for e in validation["errors"] if "SetUnsupportedState" in e
        ]
        self.assertTrue(unsupported_errors)
        self.assertTrue(any("UnprovenApiCall" in e for e in unsupported_errors))
        self.assertFalse(any("replacement" in e.lower() for e in unsupported_errors))

        # Method dependency metadata must include node-reference reads, not just
        # scalar GetParameter reads, so runtime guards can validate geometry.
        source = """
class Logic:
    def runGeometry(self):
        p = self.getParameterNode()
        p.GetNodeReference("modelNode")
        self.helper()
    def helper(self):
        p = self.getParameterNode()
        float(p.GetParameter("spacing"))
    def initializeCentroid(self):
        p = self.getParameterNode()
        p.SetParameter("centroidX", "0.0")
"""
        roles = analyzer._extract_parameter_roles_from_source(source)
        deps = analyzer._extract_parameter_method_dependencies(source, roles)
        effects = analyzer._extract_parameter_method_effects(source)
        self.assertIn("modelNode", deps["runGeometry"].get("node_roles", []))
        self.assertIn("spacing", deps["runGeometry"].get("parameter_roles", []))
        self.assertNotIn("initializeCentroid", deps)
        self.assertIn("centroidX", effects["initializeCentroid"]["writes"])

        # Node-reference requirements are based on actual use sites, including
        # parameter guards and references created by the method itself.
        conditional_source = """
class Logic:
    def run(self):
        parameterNode = self.getParameterNode()
        enabled = parameterNode.GetParameter("enableRepair") == "True"
        repairPlane = parameterNode.GetNodeReference("repairPlane")
        nestedPlane = parameterNode.GetNodeReference("nestedPlane")
        requiredModel = parameterNode.GetNodeReference("requiredModel")
        guardedModel = parameterNode.GetNodeReference("guardedModel")
        conditionallyCreated = parameterNode.GetNodeReference("conditionallyCreated")
        if enabled:
            repairPlane.GetID()
            if parameterNode.GetParameter("mode") == "advanced":
                nestedPlane.GetID()
            parameterNode.SetNodeReferenceID("conditionallyCreated", "node-id")
        if guardedModel:
            guardedModel.GetID()
        conditionallyCreated.GetID()
        requiredModel.GetID()
        self.createdHelper()
    def createdHelper(self):
        parameterNode = self.getParameterNode()
        parameterNode.SetNodeReferenceID("createdPlane", "node-id")
        createdPlane = parameterNode.GetNodeReference("createdPlane")
        createdPlane.GetID()
"""
        conditional_roles = analyzer._extract_parameter_roles_from_source(conditional_source)
        conditional_deps = analyzer._extract_parameter_method_dependencies(
            conditional_source, conditional_roles
        )
        requirements = conditional_deps["run"]["node_requirements"]
        self.assertEqual(requirements["requiredModel"]["requirement"], "required")
        self.assertEqual(requirements["conditionallyCreated"]["requirement"], "required")
        self.assertEqual(requirements["guardedModel"]["requirement"], "optional_unknown")
        self.assertEqual(requirements["createdPlane"]["requirement"], "produced_by_method")
        self.assertEqual(requirements["repairPlane"], {
            "requirement": "conditional",
            "conditions": [{
                "parameter": "enableRepair",
                "operator": "equals",
                "value": "True",
            }],
            "condition_groups": [[{
                "parameter": "enableRepair",
                "operator": "equals",
                "value": "True",
            }]],
        })
        self.assertEqual(requirements["nestedPlane"]["condition_groups"], [[
            {
                "parameter": "mode",
                "operator": "equals",
                "value": "advanced",
            },
            {
                "parameter": "enableRepair",
                "operator": "equals",
                "value": "True",
            },
        ]])

        # Runtime preconditions skip a false-gated missing reference and enforce
        # the same reference when its source-derived condition becomes true.
        from SlicerAIAgentLib.ExtensionCLILoader import _WorkflowContext, _build_runtime_prelude
        runtime_ctx = _WorkflowContext(
            ext_name="Generic",
            ext_dir="",
            tool_name="Generic",
            workflow_graph={},
            target_step={"step_id": "cb_step_1", "method_name": "conditionalOnly"},
            target_gen={"method_name": "conditionalOnly"},
            arguments={},
            user_action="start",
            done=set(),
            metadata={
                "extension_module_name": "GenericExtension",
                "logic_class_name": "GenericLogic",
                "parameter_bindings": {
                    "enableRepair": {
                        "role": "enableRepair",
                        "accesses": ["parameter_read"],
                        "value_types": ["bool"],
                    },
                    "repairPlane": {
                        "role": "repairPlane",
                        "accesses": ["node_reference_read"],
                        "node_class": "vtkMRMLMarkupsPlaneNode",
                    },
                },
                "parameter_method_dependencies": {
                    "conditionalOnly": {
                        "parameter_roles": ["enableRepair"],
                        "node_roles": ["repairPlane"],
                        "node_requirements": {
                            "repairPlane": requirements["repairPlane"],
                        },
                    },
                },
            },
        )

        class _FakeParameterNode:
            def __init__(self):
                self.enabled = "False"

            def GetParameter(self, role):
                return self.enabled if role == "enableRepair" else ""

            def GetNodeReference(self, role):
                return None

        class _FakeLogic:
            def __init__(self):
                self.parameterNode = _FakeParameterNode()

            def getParameterNode(self):
                return self.parameterNode

        runtime_namespace = {"_generic_logic": _FakeLogic()}
        exec(_build_runtime_prelude(runtime_ctx), runtime_namespace)
        runtime_namespace["_workflow_validate_method_preconditions"]()
        runtime_namespace["_generic_logic"].parameterNode.enabled = "True"
        with self.assertRaisesRegex(RuntimeError, "Missing conditional node reference"):
            runtime_namespace["_workflow_validate_method_preconditions"]()

        required_metadata = dict(runtime_ctx.metadata)
        required_metadata["parameter_bindings"] = {
            "requiredModel": {
                "role": "requiredModel",
                "accesses": ["node_reference_read"],
                "node_class": "vtkMRMLModelNode",
            },
        }
        required_metadata["parameter_method_dependencies"] = {
            "requiredOnly": {
                "node_roles": ["requiredModel"],
                "node_requirements": {
                    "requiredModel": {
                        "requirement": "required",
                        "conditions": [],
                        "condition_groups": [],
                    },
                },
            },
        }
        required_ctx = _WorkflowContext(
            "Generic", "", "Generic", {},
            {"step_id": "cb_step_2", "method_name": "requiredOnly"},
            {"method_name": "requiredOnly"}, {}, "start", set(), required_metadata,
        )
        required_namespace = {"_generic_logic": _FakeLogic()}
        exec(_build_runtime_prelude(required_ctx), required_namespace)
        with self.assertRaisesRegex(RuntimeError, "Missing required node reference"):
            required_namespace["_workflow_validate_method_preconditions"]()

        # Metadata generated before requirement classification is treated as
        # unknown instead of incorrectly requiring every transitive read.
        legacy_metadata = dict(required_metadata)
        legacy_metadata["parameter_method_dependencies"] = {
            "legacy": {"node_roles": ["requiredModel"]},
        }
        legacy_ctx = _WorkflowContext(
            "Generic", "", "Generic", {},
            {"step_id": "cb_step_3", "method_name": "legacy"},
            {"method_name": "legacy"}, {}, "start", set(), legacy_metadata,
        )
        legacy_namespace = {"_generic_logic": _FakeLogic()}
        exec(_build_runtime_prelude(legacy_ctx), legacy_namespace)
        legacy_namespace["_workflow_validate_method_preconditions"]()

        # Placeholder scanning ignores braces inside strings/f-strings.
        placeholders = analyzer._find_template_placeholders(
            "print(f'{value}')\n{vol_lookup}\nnode = {node_name: None}\n"
        )
        self.assertEqual(
            placeholders,
            [
                {"name": "node_name", "has_default": True},
                {"name": "vol_lookup", "has_default": False},
            ],
        )

        # Module/panel phrases in user-facing cookbooks are UI-location context,
        # not instructions to leave the SlicerAIAgent module.
        from SlicerAIAgentLib.CookbookParser import CookbookDef, CookbookStep
        cookbook = CookbookDef(
            extension_name="GenericExtension",
            source_file="cookbook.md",
            steps=[CookbookStep(
                step_number=1,
                title="Display",
                description=(
                    "In the Markups module's Display > Advanced panel, configure "
                    "View to show in both View 1 and Red."
                ),
            )],
        )
        stage_map = analyzer._build_stage_map_from_decomposition(
            {
                "steps": [{
                    "step_number": 1,
                    "sub_operations": [
                        {
                            "op_type": "slicer_op",
                            "description": "Switch to Markups module",
                            "slicer_op_category": "module_switching",
                            "slicer_api_keywords": ["selectModule", "Markups"],
                            "evidence_type": "slicer_core",
                            "confidence": "high",
                        },
                        {
                            "op_type": "slicer_op",
                            "description": "Configure display View to show in View 1 and Red",
                            "slicer_op_category": "markups_display",
                            "slicer_api_keywords": ["AddViewNodeID", "View"],
                            "evidence_type": "slicer_core",
                            "confidence": "high",
                        },
                    ],
                }]
            },
            cookbook,
            {"methods": []},
        )
        sub_ops = stage_map["stages"][0]["sub_operations"]
        self.assertEqual(len(sub_ops), 1)
        self.assertEqual(sub_ops[0]["slicer_op_category"], "markups_display")
        self.assertFalse(any(so.get("slicer_op_category") == "module_switching" for so in sub_ops))

        # Unbound open-ended user choices produce a validation warning.
        validation = analyzer._stage9_validate(
            {},
            [{
                "step_type": "user_choice",
                "param_signature": {"workflow_step": "cb_step_1"},
                "description": "Select the input model node",
                "choice_descriptor": {
                    "question": "Which model should be used?",
                    "choices": [],
                    "parameter_name": "inputModel",
                },
            }],
        )
        self.assertTrue(any("no source-derived parameter binding" in w for w in validation["warnings"]))

        # Closed-form choices can bind to non-node extension parameters.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
            fp.write(
                "class GenericLogic:\n"
                "    def updateParameterNodeFromGUI(self):\n"
                "        parameterNode.SetParameter('rightSideLegFibula', 'True')\n"
                "    def transform(self):\n"
                "        parameterNode.GetParameter('rightSideLegFibula')\n"
            )
            source_path = fp.name
        try:
            metadata = analyzer._build_workflow_metadata(
                {"entry_module": source_path, "logic_class": {"class_name": "GenericLogic"}},
                {},
                {
                    "steps": [{
                        "step_id": "cb_step_1",
                        "step_type": "user_choice",
                        "op_type": "user_choice",
                        "description": "If the source side is right, choose right.",
                        "choice_info": {
                            "question": "Is this the right side?",
                            "choices": [{"label": "Yes", "value": True}, {"label": "No", "value": False}],
                            "parameter_name": "right_side",
                        },
                        "sub_operations": [{"op_type": "user_choice"}],
                    }]
                },
            )
            self.assertEqual(
                metadata["choice_bindings"]["cb_step_1"]["parameter_name"],
                "rightSideLegFibula",
            )
        finally:
            os.unlink(source_path)

        # Scalar parameter-node defaults are inferred from UI widgets and
        # transitive helper-method reads before automated extension calls.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
            fp.write(
                "class GenericLogic:\n"
                "    def updateParameterNodeFromGUI(self):\n"
                "        parameterNode.SetParameter('initialSpace', str(self.ui.initialSpinBox.value))\n"
                "    def generate(self):\n"
                "        self.transform()\n"
                "    def transform(self):\n"
                "        initialSpace = float(parameterNode.GetParameter('initialSpace'))\n"
            )
            source_path = fp.name
        with tempfile.NamedTemporaryFile("w", suffix=".ui", delete=False) as fp:
            fp.write(
                "<ui><widget class=\"ctkDoubleSpinBox\" name=\"initialSpinBox\">"
                "<property name=\"value\"><double>1.5</double></property>"
                "</widget></ui>"
            )
            ui_path = fp.name
        try:
            metadata = analyzer._build_workflow_metadata(
                {
                    "entry_module": source_path,
                    "ui_files": [ui_path],
                    "logic_class": {"class_name": "GenericLogic"},
                },
                {},
                {"steps": []},
            )
            self.assertEqual(
                metadata["parameter_defaults"]["initialSpace"]["value"],
                "1.5",
            )
            self.assertEqual(
                metadata["parameter_method_dependencies"]["generate"]["parameter_roles"],
                ["initialSpace"],
            )
            analyzer._workflow_metadata = metadata
            validation = analyzer._stage9_validate(
                {"templates/cb_step_scalar.py.tpl": "logic.generate()\n"},
                [{
                    "template_file": "templates/cb_step_scalar.py.tpl",
                    "step_type": "automated",
                    "op_type": "extension_op",
                    "operation_model": {
                        "step_type": "automated",
                        "op_types": ["extension_op"],
                        "invokes_extension_method": True,
                    },
                    "sub_operations": [{
                        "op_type": "extension_op",
                        "description": "Generate result.",
                        "extension_method_hint": "generate",
                    }],
                }],
                logic_analysis={"methods": [{"name": "generate", "parameters": []}]},
            )
            self.assertTrue(validation["valid"], validation.get("errors"))
        finally:
            os.unlink(source_path)
            os.unlink(ui_path)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
            fp.write(
                "class GenericLogic:\n"
                "    def updateParameterNodeFromGUI(self):\n"
                "        parameterNode.SetParameter('implicitSpace', str(self.ui.implicitSpinBox.value))\n"
                "    def generate(self):\n"
                "        implicitSpace = float(parameterNode.GetParameter('implicitSpace'))\n"
            )
            source_path = fp.name
        with tempfile.NamedTemporaryFile("w", suffix=".ui", delete=False) as fp:
            fp.write(
                "<ui><widget class=\"ctkDoubleSpinBox\" name=\"implicitSpinBox\">"
                "<property name=\"decimals\"><number>1</number></property>"
                "</widget></ui>"
            )
            ui_path = fp.name
        try:
            metadata = analyzer._build_workflow_metadata(
                {
                    "entry_module": source_path,
                    "ui_files": [ui_path],
                    "logic_class": {"class_name": "GenericLogic"},
                },
                {},
                {"steps": []},
            )
            self.assertEqual(
                metadata["parameter_defaults"]["implicitSpace"]["source"],
                "ui_widget_implicit_default",
            )
            self.assertEqual(
                metadata["parameter_defaults"]["implicitSpace"]["value"],
                "0.0",
            )
        finally:
            os.unlink(source_path)
            os.unlink(ui_path)

        analyzer._workflow_metadata = {
            "parameter_bindings": {
                "internalCounter": {
                    "value_types": ["int"],
                    "node_class": "",
                },
            },
            "parameter_defaults": {},
            "parameter_method_dependencies": {
                "increment": {
                    "parameter_roles": ["internalCounter"],
                },
            },
            "method_parameter_effects": {
                "increment": {
                    "reads": ["internalCounter"],
                    "writes": ["internalCounter"],
                },
            },
            "choice_bindings": {},
            "interaction_bindings": {},
            "workflow_steps": [{"step_id": "cb_step_1"}],
        }
        contract_audit = analyzer._audit_workflow_contract({
            "steps": [{
                "step_id": "cb_step_1",
                "description": "Increment internal counter",
                "operations": [{"extension_method_hint": "increment"}],
            }]
        })
        self.assertTrue(contract_audit["valid"], contract_audit.get("errors"))
        scalar_contract = analyzer._validate_scalar_parameter_contract(
            "logic.increment()\n",
            step_id="cb_step_1",
        )
        self.assertFalse(scalar_contract["errors"])

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
            fp.write(
                "class GenericLogic:\n"
                "    def generate(self):\n"
                "        initialSpace = float(parameterNode.GetParameter('initialSpace'))\n"
            )
            source_path = fp.name
        try:
            metadata = analyzer._build_workflow_metadata(
                {"entry_module": source_path, "logic_class": {"class_name": "GenericLogic"}},
                {},
                {"steps": []},
            )
            metadata["parameter_defaults"].pop("initialSpace", None)
            analyzer._workflow_metadata = metadata
            validation = analyzer._stage9_validate(
                {"templates/cb_step_scalar_missing.py.tpl": "logic.generate()\n"},
                [{
                    "template_file": "templates/cb_step_scalar_missing.py.tpl",
                    "step_type": "automated",
                    "op_type": "extension_op",
                    "operation_model": {
                        "step_type": "automated",
                        "op_types": ["extension_op"],
                        "invokes_extension_method": True,
                    },
                    "sub_operations": [{
                        "op_type": "extension_op",
                        "description": "Generate result.",
                        "extension_method_hint": "generate",
                    }],
                }],
                logic_analysis={"methods": [{"name": "generate", "parameters": []}]},
            )
            self.assertFalse(validation["valid"])
            self.assertTrue(any("depends on scalar parameter 'initialSpace'" in e for e in validation["errors"]))
        finally:
            os.unlink(source_path)

        # Slicer API code cannot hide in an extension_op without an extension method.
        validation = analyzer._stage9_validate(
            {"templates/cb_step_1.py.tpl": "slicer.app.layoutManager().setLayout(1)"},
            [{
                "template_file": "templates/cb_step_1.py.tpl",
                "step_type": "automated",
                "op_type": "extension_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["extension_op"],
                    "invokes_extension_method": False,
                    "invokes_slicer_api": False,
                },
                "sub_operations": [{
                    "op_type": "extension_op",
                    "description": "Change a Slicer view",
                    "extension_method_hint": None,
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("extension_op without an extension method" in e for e in validation["errors"]))

        # Destructive operations require an explicit contract.
        validation = analyzer._stage9_validate(
            {"templates/cb_step_2.py.tpl": "displayNode.RemoveAllViewNodeIDs()"},
            [{
                "template_file": "templates/cb_step_2.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                },
                "sub_operations": [{"op_type": "slicer_op", "description": "Set display views"}],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("destructive operations" in e for e in validation["errors"]))

        # Display view-list resets are allowed only when followed by explicit scoped view IDs.
        # Slice-view targets also require class-specific 2D/slice visibility handling.
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2b.py.tpl": (
                    "displayNode.RemoveAllViewNodeIDs()\n"
                    "displayNode.AddViewNodeID('vtkMRMLViewNode1')\n"
                    "displayNode.AddViewNodeID('vtkMRMLSliceNodeRed')\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2b.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["view_display_scope"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Set markup display views",
                    "slicer_op_category": "markups_display",
                    "node_class": "vtkMRMLMarkupsDisplayNode",
                    "slicer_api_keywords": ["Display", "View", "Red"],
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("Markups display step targets a slice view" in e for e in validation["errors"]))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2c.py.tpl": (
                    "displayNode.SetVisibility(True)\n"
                    "displayNode.SetSliceProjection(True)\n"
                    "displayNode.RemoveAllViewNodeIDs()\n"
                    "displayNode.AddViewNodeID('vtkMRMLViewNode1')\n"
                    "displayNode.AddViewNodeID('vtkMRMLSliceNodeRed')\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2c.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["view_display_scope"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Set markup display views",
                    "slicer_op_category": "markups_display",
                    "node_class": "vtkMRMLMarkupsDisplayNode",
                    "slicer_api_keywords": ["Display", "View", "Red"],
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2d.py.tpl": (
                    "modelDisplayNode.SetVisibility(True)\n"
                    "modelDisplayNode.RemoveAllViewNodeIDs()\n"
                    "modelDisplayNode.AddViewNodeID('vtkMRMLSliceNodeRed')\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2d.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["view_display_scope"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Show model in Red slice view",
                    "slicer_op_category": "node_display",
                    "node_class": "vtkMRMLModelNode",
                    "slicer_api_keywords": ["Display", "Red"],
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("Model display step targets a slice view" in e for e in validation["errors"]))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2e.py.tpl": (
                    "modelDisplayNode.SetVisibility(True)\n"
                    "modelDisplayNode.SetVisibility2D(True)\n"
                    "modelDisplayNode.RemoveAllViewNodeIDs()\n"
                    "modelDisplayNode.AddViewNodeID('vtkMRMLSliceNodeRed')\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2e.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["view_display_scope"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Show model in Red slice view",
                    "slicer_op_category": "node_display",
                    "node_class": "vtkMRMLModelNode",
                    "slicer_api_keywords": ["Display", "Red"],
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        # Automated display-view operations must not switch the active Slicer module.
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2b_module.py.tpl": (
                    "slicer.util.selectModule('Markups')\n"
                    "displayNode.AddViewNodeID('vtkMRMLViewNode1')\n"
                    "displayNode.AddViewNodeID('vtkMRMLSliceNodeRed')\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2b_module.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "allow_module_switch": False,
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Configure Markups display View to show in View 1 and Red",
                    "slicer_op_category": "markups_display",
                    "slicer_api_keywords": ["Display", "View"],
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("switches the active Slicer module" in e for e in validation["errors"]))

        # Explicit module-switch steps may still use selectModule.
        validation = analyzer._stage9_validate(
            {"templates/cb_step_2b_explicit_module.py.tpl": "slicer.util.selectModule('Markups')\n"},
            [{
                "template_file": "templates/cb_step_2b_explicit_module.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "allow_module_switch": True,
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Switch to the Markups module",
                    "slicer_op_category": "module_switching",
                    "slicer_api_keywords": ["selectModule"],
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        # Runtime prelude is generated for defaults even when the user has not answered a choice.
        from SlicerAIAgentLib.ExtensionCLILoader import _WorkflowContext, _build_choice_prelude
        ctx = _WorkflowContext(
            ext_name="Generic",
            ext_dir="",
            tool_name="Generic",
            workflow_graph={},
            target_step={"step_id": "cb_step_1"},
            target_gen=None,
            arguments={},
            user_action="start",
            done=set(),
            metadata={
                "extension_module_name": "GenericExtension",
                "logic_class_name": "GenericLogic",
                "parameter_bindings": {
                    "initialSpace": {
                        "role": "initialSpace",
                        "accesses": ["parameter_read"],
                        "value_types": ["float"],
                    }
                },
                "parameter_defaults": {
                    "initialSpace": {
                        "value": "1.5",
                        "value_type": "float",
                        "source": "ui_widget_default",
                    }
                },
            },
        )
        prelude = _build_choice_prelude(ctx)
        self.assertIn("_workflow_defaults", prelude)
        self.assertIn("apply_workflow_metadata(parameterNode", prelude)

        # Fallback node resolution must not use a different Markups class than metadata expects.
        analyzer._workflow_metadata = {
            "parameter_bindings": {
                "fibulaLine": {
                    "role": "fibulaLine",
                    "node_class": "vtkMRMLMarkupsLineNode",
                }
            }
        }
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_node_class.py.tpl": (
                    "role = 'fibulaLine'\n"
                    "nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsCurveNode')\n"
                )
            },
            [{
                "template_file": "templates/cb_step_node_class.py.tpl",
                "step_type": "automated",
                "op_type": "extension_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["extension_op"],
                    "invokes_extension_method": True,
                },
                "sub_operations": [{
                    "op_type": "extension_op",
                    "description": "Resolve fibula line.",
                    "extension_method_hint": "centerLine",
                }],
            }],
            logic_analysis={"methods": [{"name": "centerLine", "parameters": []}]},
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("metadata expects vtkMRMLMarkupsLineNode" in e for e in validation["errors"]))

        # Slice intersection visibility must visibly refresh slice views when
        # implemented through direct slice display-node setters.
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2c.py.tpl": (
                    "sliceDisplayNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')\n"
                    "for sliceDisplayNode in sliceDisplayNodes:\n"
                    "    sliceDisplayNode.SetIntersectingSlicesVisibility(True)\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2c.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Toggle on slice intersection visibility.",
                    "slicer_op_category": "crosshair",
                    "slicer_api_keywords": ["crosshair"],
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("Slice intersection visibility step" in e for e in validation["errors"]))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2c_refresh.py.tpl": (
                    "sliceDisplayNodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')\n"
                    "for sliceDisplayNode in sliceDisplayNodes:\n"
                    "    sliceDisplayNode.SetIntersectingSlicesVisibility(True)\n"
                    "sliceNodes = slicer.util.getNodesByClass('vtkMRMLSliceNode')\n"
                    "for sliceNode in sliceNodes:\n"
                    "    sliceNode.Modified()\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2c_refresh.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Toggle on slice intersection visibility.",
                    "slicer_op_category": "crosshair",
                    "slicer_api_keywords": ["slice intersection visibility"],
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2c_app_logic.py.tpl": (
                    "appLogic = slicer.app.applicationLogic()\n"
                    "appLogic.SetIntersectingSlicesEnabled("
                    "appLogic.IntersectingSlicesVisibility, True)\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2c_app_logic.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Turn on slice intersection visibility.",
                    "slicer_op_category": "crosshair",
                    "slicer_api_keywords": ["slice intersection visibility"],
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2c_crosshair.py.tpl": (
                    "crosshairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')\n"
                    "crosshairNode.SetCrosshairMode(slicer.vtkMRMLCrosshairNode.ShowBasic)\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2c_crosshair.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                    "operation_intents": ["slice_intersection_visibility"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Turn on slice intersection visibility.",
                    "slicer_op_category": "crosshair",
                    "slicer_api_keywords": ["slice intersection visibility"],
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("changes only crosshair state" in e for e in validation["errors"]))

        # Final-state Slicer operations must not invert current state.
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2c_state.py.tpl": (
                    "appLogic = slicer.app.applicationLogic()\n"
                    "currentVisibility = appLogic.GetIntersectingSlicesEnabled(None)\n"
                    "appLogic.SetIntersectingSlicesEnabled(None, not currentVisibility)\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2c_state.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Toggle on slice intersection visibility.",
                    "slicer_op_category": "crosshair",
                    "slicer_api_keywords": ["visibility"],
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("Final-state operation requests" in e for e in validation["errors"]))

        # Per-template API evidence can validate new Slicer API footprints
        # without adding category-specific validator hints.
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_2d.py.tpl": (
                    "controller = slicer.app.applicationLogic()\n"
                    "controller.SetNewSlicerApiVariant(True)\n"
                )
            },
            [{
                "template_file": "templates/cb_step_2d.py.tpl",
                "step_type": "automated",
                "op_type": "slicer_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["slicer_op"],
                    "invokes_slicer_api": True,
                },
                "api_evidence": {
                    "accepted_footprints": ["SetNewSlicerApiVariant"],
                    "api_chains": ["controller.SetNewSlicerApiVariant"],
                },
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Use a newly discovered Slicer API variant.",
                    "slicer_op_category": "unknown_future_category",
                    "slicer_api_keywords": ["not_in_template"],
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        evidence = analyzer._build_template_api_evidence(
            (
                "nodes = slicer.util.getNodesByClass('vtkMRMLSliceDisplayNode')\n"
                "for node in nodes:\n"
                "    node.SetIntersectingSlicesVisibility(True)\n"
            ),
            {
                "sub_operations": [{
                    "op_type": "slicer_op",
                    "description": "Toggle slice intersections",
                    "slicer_op_category": "crosshair",
                    "slicer_api_keywords": ["crosshair"],
                }]
            },
        )
        self.assertIn("SetIntersectingSlicesVisibility", evidence["accepted_footprints"])
        self.assertIn("vtkMRMLSliceDisplayNode", evidence["accepted_footprints"])

        # Extension-owned interactions must not create a second Markups node.
        validation = analyzer._stage9_validate(
            {"templates/cb_step_3_pre.py.tpl": "node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', 'L')"},
            [{
                "pre_template_file": "templates/cb_step_3_pre.py.tpl",
                "step_type": "interactive",
                "interaction_descriptor": {
                    "node_class": "vtkMRMLMarkupsLineNode",
                    "interaction_owner": "previous_extension_method",
                    "placement_starter_method": "addLine",
                },
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("pre-template creates a new Markups node" in e for e in validation["errors"]))

        # Repeat groups cannot call the same placement starter in both start and interaction steps.
        contract = analyzer._validate_generator_contracts([
            {
                "step_type": "automated",
                "param_signature": {"workflow_step": "cb_step_4"},
                "interaction_descriptor": {"placement_starter_method": "addPlane"},
            },
            {
                "step_type": "mixed",
                "param_signature": {"workflow_step": "cb_step_5"},
                "repeat_group": {
                    "group_id": "repeat_cb_step_4_cb_step_5",
                    "start_step": "cb_step_4",
                    "interaction_step": "cb_step_5",
                },
                "interaction_descriptor": {
                    "node_class": "vtkMRMLMarkupsPlaneNode",
                    "interaction_owner": "extension_method",
                    "placement_starter_method": "addPlane",
                },
            },
        ])
        self.assertTrue(any("both call placement starter" in e for e in contract["errors"]))

        # Normalization removes duplicate repeat starter ownership before templates are built.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
            fp.write(
                "def setCustomLayout():\n"
                "    slicer.app.layoutManager().setLayout(123)\n"
            )
            module_source = fp.name
        try:
            analyzer._placement_starter_methods = {"addPlane": {"node_classes": ["vtkMRMLMarkupsPlaneNode"]}}
            workflow_graph = {
                "steps": [
                    {
                        "step_id": "cb_step_1",
                        "step_type": "user_choice",
                        "op_type": "user_choice",
                        "description": "How many planes are needed?",
                        "choice_info": {
                            "question": "How many planes?",
                            "parameter_name": "number_of_planes",
                            "choices": [],
                        },
                        "sub_operations": [{"op_type": "user_choice"}],
                    },
                    {
                        "step_id": "cb_step_2",
                        "step_type": "automated",
                        "op_type": "extension_op",
                        "description": "Click Add plane.",
                        "method_name": "addPlane",
                        "sub_operations": [{
                            "op_type": "extension_op",
                            "description": "Add a plane.",
                            "extension_method_hint": "addPlane",
                        }],
                    },
                    {
                        "step_id": "cb_step_3",
                        "step_type": "mixed",
                        "op_type": "mixed",
                        "description": "Repeat adding and placing a plane for each requested plane.",
                        "sub_operations": [
                            {
                                "op_type": "extension_op",
                                "description": "Add a plane.",
                                "extension_method_hint": "addPlane",
                            },
                            {
                                "op_type": "user_interaction",
                                "description": "Place the plane.",
                                "node_class": "vtkMRMLMarkupsPlaneNode",
                                "interaction_kind": "markup_placement",
                            },
                        ],
                    },
                    {
                        "step_id": "cb_step_4",
                        "step_type": "automated",
                        "op_type": "extension_op",
                        "description": "Restore the custom layout.",
                        "sub_operations": [{
                            "op_type": "extension_op",
                            "description": "Restore the custom layout.",
                            "extension_method_hint": None,
                        }],
                    },
                ]
            }
            repeat_group = {
                "group_id": "repeat_cb_step_1_cb_step_2_cb_step_3",
                "count_parameter": "number_of_planes",
                "count_step": "cb_step_1",
                "start_step": "cb_step_2",
                "interaction_step": "cb_step_3",
            }
            for step in workflow_graph["steps"][:3]:
                step["repeat_group"] = repeat_group
            metadata = {
                "repeat_groups": {repeat_group["group_id"]: repeat_group},
                "choice_bindings": {},
                "interaction_bindings": {},
                "parameter_bindings": {},
            }
            analyzer._normalize_workflow_contracts(
                workflow_graph,
                metadata,
                {"entry_module": module_source, "logic_class": {"class_name": "GenericLogic"}},
                {"methods": []},
            )
            interaction_step = workflow_graph["steps"][2]
            self.assertEqual(interaction_step["step_type"], "user_interaction")
            self.assertEqual(interaction_step["interaction_owner"], "previous_extension_method")
            self.assertFalse(any(
                so.get("extension_method_hint") == "addPlane"
                for so in interaction_step["sub_operations"]
            ))
            layout_step = workflow_graph["steps"][3]
            self.assertEqual(layout_step["extension_function_name"], "setCustomLayout")
            self.assertEqual(
                layout_step["sub_operations"][0]["extension_function_hint"],
                "setCustomLayout",
            )
        finally:
            os.unlink(module_source)

        # Placement starters remain bound across intervening Slicer configuration steps.
        analyzer._placement_starter_methods = {
            "addCurve": {
                "node_classes": ["vtkMRMLMarkupsCurveNode"],
                "starts_markup_placement": True,
                "placement_mode": "single",
            }
        }
        workflow_graph = {
            "steps": [
                {
                    "step_id": "cb_step_10",
                    "step_type": "automated",
                    "op_type": "extension_op",
                    "description": "Click Add curve.",
                    "method_name": "addCurve",
                    "sub_operations": [{
                        "op_type": "extension_op",
                        "description": "Click Add curve.",
                        "extension_method_hint": "addCurve",
                    }],
                },
                {
                    "step_id": "cb_step_11",
                    "step_type": "automated",
                    "op_type": "slicer_op",
                    "description": "Show the active curve in the desired views.",
                    "sub_operations": [{
                        "op_type": "slicer_op",
                        "description": "Show the active curve in the desired views.",
                    }],
                },
                {
                    "step_id": "cb_step_12",
                    "step_type": "interactive",
                    "op_type": "user_interaction",
                    "description": "Manually draw the curve.",
                    "node_class": "vtkMRMLMarkupsCurveNode",
                    "interaction_kind": "markup_placement",
                    "sub_operations": [{
                        "op_type": "user_interaction",
                        "description": "Manually draw the curve.",
                        "node_class": "vtkMRMLMarkupsCurveNode",
                        "interaction_kind": "markup_placement",
                    }],
                },
            ]
        }
        metadata = {
            "repeat_groups": {},
            "choice_bindings": {},
            "interaction_bindings": {},
            "parameter_bindings": {},
        }
        analyzer._normalize_workflow_contracts(
            workflow_graph,
            metadata,
            {"entry_module": "", "logic_class": {"class_name": "GenericLogic"}},
            {"methods": []},
        )
        interaction_step = workflow_graph["steps"][2]
        self.assertEqual(interaction_step["interaction_owner"], "previous_extension_method")
        self.assertEqual(interaction_step["placement_starter_method"], "addCurve")
        self.assertEqual(interaction_step["placement_starter_step_id"], "cb_step_10")
        self.assertEqual(
            metadata["interaction_policies"]["cb_step_12"]["placement_binding_reason"],
            "recent_same_node_class_placement_starter",
        )

        # Placement-starter classification records source-derived placement mode.
        placement_source = """
class GenericLogic:
    def addPlane(self):
        node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsPlaneNode")
        slicer.mrmlScene.AddNode(node)
        slicer.modules.markups.logic().SetActiveListID(node)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToSinglePlaceMode()

    def addPoint(self):
        node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.mrmlScene.AddNode(node)
        slicer.modules.markups.logic().SetActiveListID(node)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToPersistentPlaceMode()
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
            fp.write(placement_source)
            placement_file = fp.name
        try:
            starters = analyzer._classify_placement_starter_methods({
                "_logic_file": placement_file,
                "methods": [{"name": "addPlane"}, {"name": "addPoint"}],
            })
            self.assertEqual(starters["addPlane"]["placement_mode"], "single")
            self.assertEqual(starters["addPoint"]["placement_mode"], "persistent")
            self.assertTrue(starters["addPlane"]["starts_markup_placement"])
        finally:
            os.unlink(placement_file)

        # Extension-owned repeated interactions preserve the starter's placement mode.
        analyzer._placement_starter_methods = {
            "addPlane": {
                "node_classes": ["vtkMRMLMarkupsPlaneNode"],
                "starts_markup_placement": True,
                "placement_mode": "single",
            }
        }
        repeat_group = {
            "group_id": "repeat_count_add_place",
            "count_step": "cb_step_1",
            "start_step": "cb_step_2",
            "interaction_step": "cb_step_3",
        }
        repeated_step = {
            "step_id": "cb_step_3",
            "description": "Place a plane. Repeat for each requested plane.",
            "node_class": "vtkMRMLMarkupsPlaneNode",
            "interaction_type": "plane",
            "interaction_owner": "previous_extension_method",
            "placement_starter_method": "addPlane",
            "repeat_group": repeat_group,
            "sub_operations": [{
                "op_type": "user_interaction",
                "node_class": "vtkMRMLMarkupsPlaneNode",
                "placement_instructions": "Place a plane. Repeat for each requested plane.",
            }],
        }
        analyzer._normalize_repeat_interaction_instructions(repeated_step)
        pre_tpl = analyzer._generate_existing_placement_pre_template(
            "GenericExtension",
            repeated_step,
            "addPlane",
        )
        self.assertNotIn("SwitchToPersistentPlaceMode", pre_tpl)
        self.assertNotIn("SwitchToSinglePlaceMode", pre_tpl)
        self.assertNotIn("SetActiveListID", pre_tpl)
        self.assertIn("Place this plane, then click Done.", pre_tpl)

        validation = analyzer._stage9_validate(
            {"templates/cb_step_3_pre.py.tpl": pre_tpl},
            [{
                "pre_template_file": "templates/cb_step_3_pre.py.tpl",
                "step_type": "interactive",
                "param_signature": {"workflow_step": "cb_step_3"},
                "repeat_group": repeat_group,
                "interaction_descriptor": {
                    "node_class": "vtkMRMLMarkupsPlaneNode",
                    "interaction_owner": "previous_extension_method",
                    "placement_starter_method": "addPlane",
                    "repeat_group": repeat_group,
                },
            }],
        )
        self.assertTrue(validation["valid"], validation["errors"])

        bad_validation = analyzer._stage9_validate(
            {
                "templates/cb_step_3_pre.py.tpl": (
                    "interactionNode = slicer.mrmlScene.GetNodeByID('vtkMRMLInteractionNodeSingleton')\n"
                    "interactionNode.SwitchToPersistentPlaceMode()\n"
                    "print('Repeat for each requested plane')\n"
                )
            },
            [{
                "pre_template_file": "templates/cb_step_3_pre.py.tpl",
                "step_type": "interactive",
                "param_signature": {"workflow_step": "cb_step_3"},
                "repeat_group": repeat_group,
                "interaction_descriptor": {
                    "node_class": "vtkMRMLMarkupsPlaneNode",
                    "interaction_owner": "previous_extension_method",
                    "placement_starter_method": "addPlane",
                    "repeat_group": repeat_group,
                },
            }],
        )
        self.assertFalse(bad_validation["valid"])
        self.assertTrue(any("enters Markups placement mode" in e for e in bad_validation["errors"]))
        self.assertTrue(any("persistent placement mode" in e for e in bad_validation["errors"]))

        # Runtime-created repeated markup placement uses one item per Done.
        runtime_step = {
            "step_id": "cb_step_6",
            "description": "Place one point for each requested item.",
            "node_class": "vtkMRMLMarkupsFiducialNode",
            "interaction_type": "fiducial",
            "interaction_owner": "runtime_template",
            "repeat_group": {
                "group_id": "repeat_count_place",
                "count_step": "cb_step_4",
                "start_step": "cb_step_6",
                "interaction_step": "cb_step_6",
            },
        }
        runtime_tpl = analyzer._generate_pre_interaction_template(
            "GenericExtension",
            runtime_step,
            "GenericLogic",
            "GenericExtension",
        )
        self.assertIn("SwitchToSinglePlaceMode", runtime_tpl)
        self.assertNotIn("SwitchToPersistentPlaceMode", runtime_tpl)

        contract = analyzer._validate_generator_contracts([{
            "step_type": "interactive",
            "param_signature": {"workflow_step": "cb_step_6"},
            "repeat_group": runtime_step["repeat_group"],
            "interaction_descriptor": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "placement_instructions": "Repeat for each requested point.",
                "repeat_group": runtime_step["repeat_group"],
            },
        }])
        self.assertTrue(any("one item per Done" in e for e in contract["errors"]))

        # Generation-time UI guidance is distinct from cookbook text and repeat-aware.
        guidance_graph = {
            "steps": [
                {
                    "step_id": "cb_step_16",
                    "step_type": "user_choice",
                    "description": "Set how many cut planes to create.",
                    "choice_info": {
                        "question": "How many mandibular cutting planes would you like to create?",
                        "choices": [],
                        "parameter_name": "numberOfCutPlanes",
                        "default_value": "1",
                    },
                },
                {
                    "step_id": "cb_step_18",
                    "step_type": "interactive",
                    "description": "Place cut plane. Repeat for each requested plane.",
                    "node_class": "vtkMRMLMarkupsPlaneNode",
                    "interaction_type": "plane",
                    "repeat_group": {
                        "group_id": "repeat_cut_planes",
                        "count_step": "cb_step_16",
                        "start_step": "cb_step_17",
                        "interaction_step": "cb_step_18",
                    },
                    "sub_operations": [{
                        "op_type": "user_interaction",
                        "node_class": "vtkMRMLMarkupsPlaneNode",
                    }],
                },
            ],
            "step_count": 2,
        }
        analyzer._synthesize_workflow_ui_guidance(
            guidance_graph,
            {},
            {"logic_class": {"class_name": "GenericLogic"}},
            {"methods": []},
        )
        choice_guidance = guidance_graph["steps"][0]["ui_guidance"]
        self.assertEqual(choice_guidance["input_label"], "Number of cutting planes")
        repeat_guidance = guidance_graph["steps"][1]["ui_guidance"]
        self.assertIn("cutting plane", repeat_guidance["title"].lower())
        self.assertIn("click Done", repeat_guidance["instruction"])
        self.assertNotIn("Repeat for each", repeat_guidance["instruction"])

        # Invalid LLM guidance falls back deterministically.
        analyzer.llm_client = object()
        analyzer._call_llm = lambda prompt: "not json"
        fallback_graph = {
            "steps": [{
                "step_id": "cb_step_1",
                "step_type": "interactive",
                "description": "Draw the contour.",
                "node_class": "vtkMRMLMarkupsCurveNode",
                "sub_operations": [{"op_type": "user_interaction", "node_class": "vtkMRMLMarkupsCurveNode"}],
            }]
        }
        analyzer._synthesize_workflow_ui_guidance(
            fallback_graph,
            {},
            {"logic_class": {"class_name": "GenericLogic"}},
            {"methods": []},
        )
        self.assertEqual(fallback_graph["steps"][0]["ui_guidance"]["title"], "Draw curve")

        # Loader results carry generated guidance and repeat progress.
        from SlicerAIAgentLib.ExtensionCLILoader import (
            _WorkflowContext,
            _handle_interactive_start,
            _handle_user_choice_step,
            _workflow_repeat_state,
        )
        _workflow_repeat_state.clear()
        loader_step = {
            "step_id": "cb_step_18",
            "step_type": "interactive",
            "description": "Raw cookbook placement text.",
            "ui_guidance": repeat_guidance,
            "repeat_group": guidance_graph["steps"][1]["repeat_group"],
        }
        loader_gen = {
            "step_type": "interactive",
            "pre_template_file": "",
            "post_template_file": "",
            "interaction_descriptor": {
                "node_class": "vtkMRMLMarkupsPlaneNode",
                "placement_instructions": "Raw placement instructions.",
                "ui_guidance": repeat_guidance,
                "repeat_group": guidance_graph["steps"][1]["repeat_group"],
            },
            "repeat_group": guidance_graph["steps"][1]["repeat_group"],
        }
        _workflow_repeat_state["GenericExtension"] = {
            "repeat_cut_planes": {"target": 3, "completed": 1}
        }
        ctx = _WorkflowContext(
            "GenericExtension", tempfile.gettempdir(), "GenericExtension",
            {"steps": [loader_step]}, loader_step, loader_gen,
            {"workflow_step": "cb_step_18", "user_action": "start"},
            "start", set(), {},
        )
        loader_result = _handle_interactive_start(ctx)
        self.assertEqual(loader_result["ui_guidance"]["title"], repeat_guidance["title"])
        self.assertEqual(loader_result["repeat_progress"]["current"], 2)
        self.assertEqual(loader_result["repeat_progress"]["total"], 3)

        choice_step = guidance_graph["steps"][0]
        choice_gen = {
            "step_type": "user_choice",
            "choice_descriptor": {
                "question": choice_step["choice_info"]["question"],
                "choices": [],
                "parameter_name": "numberOfCutPlanes",
                "default_value": "1",
                "ui_guidance": choice_guidance,
            },
        }
        choice_ctx = _WorkflowContext(
            "GenericExtension", tempfile.gettempdir(), "GenericExtension",
            {"steps": [choice_step]}, choice_step, choice_gen,
            {"workflow_step": "cb_step_16", "user_action": "start"},
            "start", set(), {},
        )
        choice_result = _handle_user_choice_step(choice_ctx)
        self.assertEqual(choice_result["ui_guidance"]["input_label"], "Number of cutting planes")

        # Source-derived callable effects are available to Stage 4's LLM candidates.
        layout_source = """
def addBRPLayout():
    layoutNode = slicer.mrmlScene.GetSingletonNode('vtkMRMLLayoutNodeSingleton', 'vtkMRMLLayoutNode')
    layoutNode.AddLayoutDescription(slicer.BRPLayoutId, '<layout/>')

def setBRPLayout():
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.BRPLayoutId)
"""
        layout_tree = ast.parse(layout_source)
        layout_inventory = {}
        for node in layout_tree.body:
            if isinstance(node, ast.FunctionDef):
                layout_inventory[node.name] = {
                    "name": node.name,
                    "effects": analyzer._infer_callable_effects(
                        node,
                        ast.get_source_segment(layout_source, node) or "",
                    ),
                }
        self.assertEqual(layout_inventory["addBRPLayout"]["effects"], ["layout_register"])
        self.assertEqual(layout_inventory["setBRPLayout"]["effects"], ["layout_activate"])

        analyzer._workflow_metadata = {
            "extension_callable_inventory": {
                "module_functions": ["addBRPLayout", "setBRPLayout"],
                "module_function_effects": {
                    "addBRPLayout": ["layout_register"],
                    "setBRPLayout": ["layout_activate"],
                },
            }
        }
        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_layout_bad.py.tpl": (
                    "from BoneReconstructionPlanner import addBRPLayout\n"
                    "addBRPLayout()\n"
                )
            },
            [{
                "template_file": "templates/cb_step_layout_bad.py.tpl",
                "step_type": "automated",
                "op_type": "extension_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["extension_op"],
                    "invokes_extension_function": True,
                    "operation_intents": ["layout_activate"],
                },
                "sub_operations": [{
                    "op_type": "extension_op",
                    "description": "Change the layout to BoneReconstructionPlanner.",
                    "extension_function_hint": "addBRPLayout",
                }],
            }],
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any("Layout activation step" in e for e in validation["errors"]))

        validation = analyzer._stage9_validate(
            {
                "templates/cb_step_layout_good.py.tpl": (
                    "from BoneReconstructionPlanner import setBRPLayout\n"
                    "setBRPLayout()\n"
                )
            },
            [{
                "template_file": "templates/cb_step_layout_good.py.tpl",
                "step_type": "automated",
                "op_type": "extension_op",
                "operation_model": {
                    "step_type": "automated",
                    "op_types": ["extension_op"],
                    "invokes_extension_function": True,
                    "operation_intents": ["layout_activate"],
                },
                "sub_operations": [{
                    "op_type": "extension_op",
                    "description": "Change the layout to BoneReconstructionPlanner.",
                    "extension_function_hint": "setBRPLayout",
                }],
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))
        analyzer._workflow_metadata = {}

        # Template repairs synchronize extension-function evidence back to generator contracts.
        analyzer._workflow_metadata = {
            "extension_callable_inventory": {
                "module_functions": ["setBRPLayout"],
            }
        }
        workflow_graph = {
            "steps": [{
                "step_id": "cb_step_9",
                "step_type": "automated",
                "op_type": "extension_op",
                "description": "Restore custom layout.",
                "sub_operations": [{
                    "op_type": "extension_op",
                    "description": "Restore custom layout.",
                    "extension_method_hint": None,
                }],
            }]
        }
        generators = [{
            "template_file": "templates/cb_step_9.py.tpl",
            "step_type": "automated",
            "op_type": "extension_op",
            "param_signature": {"workflow_step": "cb_step_9"},
            "sub_operations": [{
                "op_type": "extension_op",
                "description": "Restore custom layout.",
                "extension_method_hint": None,
            }],
        }]
        analyzer._sync_template_contracts(
            {
                "templates/cb_step_9.py.tpl": (
                    "from GenericExtension import setBRPLayout\n"
                    "setBRPLayout()\n"
                )
            },
            generators,
            workflow_graph=workflow_graph,
        )
        self.assertEqual(generators[0]["extension_function_name"], "setBRPLayout")
        self.assertTrue(generators[0]["operation_model"]["invokes_extension_function"])
        self.assertEqual(
            workflow_graph["steps"][0]["sub_operations"][0]["extension_function_hint"],
            "setBRPLayout",
        )

        syntax_issues = analyzer._syntax_check_templates({
            "templates/good.py.tpl": "x = {value: 1}\n",
            "templates/bad.py.tpl": "if True print('bad')\n",
        })
        self.assertEqual(len(syntax_issues), 1)
        self.assertEqual(syntax_issues[0]["template"], "templates/bad.py.tpl")

        # Implementation-level Slicer scaffolding is allowed when explicitly modeled.
        validation = analyzer._stage9_validate(
            {"templates/cb_step_6_pre.py.tpl": "slicer.mrmlScene.GetNodeByID('vtkMRMLInteractionNodeSingleton')"},
            [{
                "pre_template_file": "templates/cb_step_6_pre.py.tpl",
                "step_type": "interactive",
                "operation_model": {
                    "step_type": "interactive",
                    "op_types": ["user_interaction"],
                    "invokes_slicer_api": False,
                    "implementation_uses_slicer_api": True,
                },
                "interaction_descriptor": {
                    "node_class": "vtkMRMLMarkupsPlaneNode",
                    "interaction_owner": "previous_extension_method",
                    "placement_starter_method": "addPlane",
                },
            }],
        )
        self.assertTrue(validation["valid"], validation.get("errors"))

        # Previous-step-owned repeat interactions can reference the starter without being treated as a second call.
        contract = analyzer._validate_generator_contracts([
            {
                "step_type": "automated",
                "param_signature": {"workflow_step": "cb_step_7"},
                "sub_operations": [{
                    "op_type": "extension_op",
                    "extension_method_hint": "addPlane",
                }],
            },
            {
                "step_type": "interactive",
                "param_signature": {"workflow_step": "cb_step_8"},
                "repeat_group": {
                    "group_id": "repeat_cb_step_7_cb_step_8",
                    "start_step": "cb_step_7",
                    "interaction_step": "cb_step_8",
                },
                "interaction_descriptor": {
                    "node_class": "vtkMRMLMarkupsPlaneNode",
                    "interaction_owner": "previous_extension_method",
                    "placement_starter_method": "addPlane",
                },
            },
        ])
        self.assertFalse(contract["errors"], contract["errors"])

        # Multi-line cookbook descriptions must not create bare indented Python text.
        multiline_step = dict(repeated_step)
        multiline_step["description"] = (
            "Place the fibular cut plane.\n"
            "This can be translated and rotated in slice views."
        )
        multiline_tpl = analyzer._generate_existing_placement_pre_template(
            "GenericExtension",
            multiline_step,
            "addPlane",
        )
        ast.parse(analyzer._fill_remaining_placeholders(multiline_tpl))
        self.assertIn("# This can be translated", multiline_tpl)

        # Sanitization repairs LLM-revised templates with split raw headers.
        sanitized = analyzer._sanitize_templates({
            "templates/cb_step_18_pre.py.tpl": (
                "# --- GenericExtension: Place the fibular cut plane.\n"
                "  This can be translated and rotated in slice views. (Setup) ---\n"
                "import slicer\n"
            )
        })
        ast.parse(sanitized["templates/cb_step_18_pre.py.tpl"])
        self.assertIn(
            "# This can be translated and rotated",
            sanitized["templates/cb_step_18_pre.py.tpl"],
        )

        # Placeholder filling for validation preserves braces inside Python strings/f-strings.
        filled = analyzer._fill_remaining_placeholders(
            "for ref_name in ['Mandible']:\n"
            "    print(f'Missing {ref_name}')\n"
            "threshold = {threshold: 1.0}\n"
        )
        self.assertIn("{ref_name}", filled)
        self.assertIn("threshold = 1.0", filled)
        ast.parse(filled)

        # Semantic validation recognizes tuple-unpacked loop variables.
        semantic = analyzer._semantic_validate(
            "required_refs = [('a', 'b', 1)]\n"
            "for ref_name, param_name, default_value in required_refs:\n"
            "    print(f'{ref_name}: {param_name}={default_value}')\n",
            {"methods": []},
        )
        self.assertFalse(semantic["errors"], semantic["errors"])

        # Adjusting existing objects is a view adjustment, not new markup placement.
        adjusted = {"op_type": "user_interaction"}
        analyzer._enrich_typed_user_interaction(
            adjusted,
            "Manually adjust the existing planes by dragging their handles.",
            {"interaction_candidates": [{
                "interaction_kind": "markup_placement",
                "interaction_type": "plane",
                "node_class": "vtkMRMLMarkupsPlaneNode",
            }]},
        )
        self.assertEqual(adjusted["interaction_kind"], "view_adjustment")
        self.assertEqual(adjusted["node_class"], "")

        self.delayDisplay("ExtensionCLIAnalyzer contract tests passed")
