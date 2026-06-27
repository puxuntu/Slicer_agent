from .common import *
import json


class WorkflowTestsMixin:
    def test_TurnRouterWorkflowRoutes(self):
        """Test validated LLM routing for active generated workflows."""
        from SlicerAIAgentLib.TurnRouter import (
            ROUTE_TRADITIONAL,
            ROUTE_WORKFLOW_CONFLICT,
            ROUTE_WORKFLOW_CONTROL,
            ROUTE_WORKFLOW_UNRESOLVED,
        )
        from SlicerAIAgentLib.WorkflowIntentResolver import WorkflowIntentResolver

        class _FakeClient:
            api_key = "test"

        resolver = WorkflowIntentResolver(_FakeClient())
        self.assertEqual(
            resolver.resolve("load a volume", {"active": False}).route_type,
            ROUTE_TRADITIONAL,
        )

        active = {
            "active": True,
            "current_step": "cb_step_18",
            "status": "waiting_for_user",
        }
        resolver._call_llm = lambda messages: json.dumps({
            "route_type": ROUTE_WORKFLOW_CONTROL,
            "action": "proceed",
            "step_id": "cb_step_18",
            "choice_value": None,
            "confidence": 0.94,
            "reason": "user completed the current action",
        })
        done_route = resolver.resolve("that placement is complete", active)
        self.assertEqual(done_route.route_type, ROUTE_WORKFLOW_CONTROL)
        self.assertEqual(done_route.action, "proceed")
        self.assertEqual(done_route.step_id, "cb_step_18")

        resolver._call_llm = lambda messages: json.dumps({
            "route_type": ROUTE_WORKFLOW_CONTROL,
            "action": "choice_made",
            "step_id": "cb_step_1",
            "choice_value": "right",
            "confidence": 0.91,
            "reason": "selected supplied choice",
        })
        choice_route = resolver.resolve("use the right side", {
            "active": True,
            "current_step": "cb_step_1",
            "status": "waiting_for_choice",
        }, {"choices": [{"label": "Left", "value": "left"}, {"label": "Right", "value": "right"}]})
        self.assertEqual(choice_route.route_type, ROUTE_WORKFLOW_CONTROL)
        self.assertEqual(choice_route.action, "choice_made")
        self.assertEqual(choice_route.choice_value, "right")

        resolver._call_llm = lambda messages: json.dumps({
            "route_type": ROUTE_WORKFLOW_CONFLICT,
            "action": None,
            "step_id": None,
            "choice_value": None,
            "confidence": 0.95,
            "reason": "separate task",
        })
        conflict = resolver.resolve("segment this CT first", active)
        self.assertEqual(conflict.route_type, ROUTE_WORKFLOW_CONFLICT)

        resolver._call_llm = lambda messages: json.dumps({
            "route_type": ROUTE_WORKFLOW_CONTROL,
            "action": "proceed",
            "step_id": "cb_step_18",
            "choice_value": None,
            "confidence": 0.40,
            "reason": "ambiguous",
        })
        unresolved = resolver.resolve("maybe", active)
        self.assertEqual(unresolved.route_type, ROUTE_WORKFLOW_UNRESOLVED)

    def test_WorkflowRuntimeStateTransitions(self):
        """Test deterministic workflow runtime state updates without Slicer execution."""
        from SlicerAIAgentLib.WorkflowRuntime import WorkflowRuntime, WorkflowSession

        runtime = WorkflowRuntime()
        runtime.session = WorkflowSession(
            extension_name="FakeExtension",
            tool_name="FakeExtension",
            workflow_id="fake_1",
            current_step="cb_step_1",
        )

        wait_result = runtime.handle_execution_result(
            {"type": "interactive", "step_id": "cb_step_1"},
            {"success": True},
        )
        self.assertEqual(wait_result["step_id"], "cb_step_1")
        self.assertEqual(runtime.session.status, "waiting_for_user")
        self.assertEqual(runtime.session.current_step, "cb_step_1")

        count = runtime.queue_traditional_prompt("show me the loaded nodes")
        self.assertEqual(count, 1)
        self.assertEqual(runtime.pop_queued_prompts(), ["show me the loaded nodes"])

        # Generic repeat blocks keep steps atomic while controlling transitions.
        graph = {
            "steps": [
                {"step_id": "cb_step_1", "step_type": "user_choice", "depends_on": []},
                {"step_id": "cb_step_2", "step_type": "extension_op", "depends_on": ["cb_step_1"]},
                {"step_id": "cb_step_3", "step_type": "user_interaction", "depends_on": ["cb_step_2"]},
                {"step_id": "cb_step_4", "step_type": "extension_op", "depends_on": ["cb_step_3"]},
                {"step_id": "cb_step_5", "step_type": "extension_op", "depends_on": ["cb_step_4"]},
                {"step_id": "cb_step_6", "step_type": "extension_op", "depends_on": ["cb_step_5"]},
            ],
            "repeat_blocks": [
                {
                    "repeat_id": "create_items",
                    "body_steps": ["cb_step_2", "cb_step_3"],
                    "entry_step": "cb_step_2",
                    "terminal_step": "cb_step_3",
                    "exit_step": "cb_step_4",
                    "controller": {"kind": "count", "source_step": "cb_step_1"},
                    "max_iterations": 5,
                },
                {
                    "repeat_id": "refine",
                    "body_steps": ["cb_step_4", "cb_step_5"],
                    "entry_step": "cb_step_4",
                    "terminal_step": "cb_step_5",
                    "exit_step": "cb_step_6",
                    "controller": {
                        "kind": "until_choice",
                        "prompt": "Is the result satisfactory?",
                        "exit_value": True,
                    },
                    "max_iterations": 4,
                },
            ],
        }
        import importlib
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        original_get_workflow_graph = wf_mod.get_workflow_graph
        wf_mod.get_workflow_graph = lambda extension_name: graph
        try:
            runtime = WorkflowRuntime()
            runtime.session = WorkflowSession(
                extension_name="FakeExtension",
                tool_name="FakeExtension",
                workflow_id="repeat_1",
                current_step="cb_step_1",
            )
            count_choice = runtime.handle_execution_result(
                {
                    "type": "choice_made",
                    "step_id": "cb_step_1",
                    "choice_value": "2",
                    "next_step": {"step_id": "cb_step_2"},
                },
                {"success": True},
            )
            self.assertEqual(count_choice["next_step"]["step_id"], "cb_step_2")

            runtime.handle_execution_result(
                {"type": "automated", "step_id": "cb_step_2"},
                {"success": True},
            )
            repeat_count = runtime.handle_execution_result(
                {"type": "interactive_done", "step_id": "cb_step_3"},
                {"success": True},
            )
            self.assertEqual(repeat_count["next_step"]["step_id"], "cb_step_2")
            self.assertEqual(repeat_count["repeat_progress"]["current"], 2)
            self.assertNotIn("cb_step_2", runtime.session.completed_steps)
            self.assertNotIn("cb_step_3", runtime.session.completed_steps)

            runtime.handle_execution_result(
                {"type": "automated", "step_id": "cb_step_2"},
                {"success": True},
            )
            count_done = runtime.handle_execution_result(
                {"type": "interactive_done", "step_id": "cb_step_3"},
                {"success": True},
            )
            self.assertEqual(count_done["next_step"]["step_id"], "cb_step_4")

            runtime.handle_execution_result(
                {"type": "automated", "step_id": "cb_step_4"},
                {"success": True},
            )
            decision = runtime.handle_execution_result(
                {"type": "automated", "step_id": "cb_step_5"},
                {"success": True},
            )
            self.assertEqual(decision["type"], "user_choice")
            self.assertEqual(decision["question"], "Is the result satisfactory?")
            self.assertEqual(runtime.session.status, "waiting_for_choice")

            repeat_decision = runtime.run_step(
                "cb_step_5", "choice_made", {"choice_value": False}
            )
            self.assertEqual(repeat_decision["next_step"]["step_id"], "cb_step_4")
            self.assertEqual(repeat_decision["repeat_progress"]["current"], 2)
            self.assertEqual(runtime.session.current_step, "cb_step_4")
            self.assertEqual(runtime.session.status, "running")

            runtime.handle_execution_result(
                {"type": "automated", "step_id": "cb_step_4"},
                {"success": True},
            )
            runtime.handle_execution_result(
                {"type": "automated", "step_id": "cb_step_5"},
                {"success": True},
            )
            accepted = runtime.run_step(
                "cb_step_5", "choice_made", {"choice_value": True}
            )
            self.assertEqual(accepted["next_step"]["step_id"], "cb_step_6")
            self.assertEqual(runtime.session.current_step, "cb_step_6")
        finally:
            wf_mod.get_workflow_graph = original_get_workflow_graph

    def test_WorkflowReplayCheckpoints(self):
        """Replay timeline: live checkpoint recording, rewind truncation, loop resume."""
        import importlib
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        WorkflowRuntime = wf_mod.WorkflowRuntime
        WorkflowSession = wf_mod.WorkflowSession
        from SlicerAIAgentLib.ExtensionCLILoader import reset_workflow_session

        graph = {
            "step_count": 4,
            "steps": [
                {"step_id": "s1", "step_type": "user_choice", "operation_type": "user_choice",
                 "description": "Pick count",
                 "choice_info": {"parameter_name": "numCuts", "choices": []}},
                {"step_id": "s2", "step_type": "extension_op", "operation_type": "extension_op",
                 "description": "Body A"},
                {"step_id": "s3", "step_type": "extension_op", "operation_type": "extension_op",
                 "description": "Body B"},
                {"step_id": "s4", "step_type": "extension_op", "operation_type": "extension_op",
                 "description": "After loop"},
            ],
            "repeat_blocks": [
                {"repeat_id": "loop", "body_steps": ["s2", "s3"], "entry_step": "s2",
                 "terminal_step": "s3", "exit_step": "s4",
                 "controller": {"kind": "count", "source_step": "s1"}, "max_iterations": 5},
            ],
        }
        original_get_workflow_graph = wf_mod.get_workflow_graph
        wf_mod.get_workflow_graph = lambda extension_name: graph

        def _new_runtime():
            runtime = WorkflowRuntime()
            runtime.session = WorkflowSession(
                extension_name="FakeReplayExt", tool_name="FakeReplayExt",
                workflow_id="replay_1", current_step="s1",
            )
            # Stub all scene ops so the test is pure state logic and never
            # touches the live Slicer scene (capture/restore/delete/commit).
            runtime._scene_node_ids = lambda: set()
            runtime._capture_sceneview = lambda step_id: None
            runtime._delete_sceneview = lambda nid: None
            runtime._delete_all_replay_sceneviews = lambda: None
            runtime._restore_to_view = lambda index: None
            runtime._commit_node_state = lambda index: None
            return runtime

        def _run_count_loop(runtime, target):
            """Drive s1(count)->[s2,s3]xN->s4, mirroring run_step's pending capture."""
            reset_workflow_session("FakeReplayExt")
            runtime._begin_pending_checkpoint("s1", {"user_action": "choice_made",
                                                     "choice_value": str(target)})
            runtime.handle_execution_result(
                {"type": "choice_made", "step_id": "s1", "choice_value": str(target),
                 "next_step": {"step_id": "s2"}},
                {"success": True},
            )
            for _ in range(target):
                runtime._begin_pending_checkpoint("s2", {"user_action": "start"})
                runtime.handle_execution_result(
                    {"type": "automated", "step_id": "s2"}, {"success": True})
                runtime._begin_pending_checkpoint("s3", {"user_action": "start"})
                runtime.handle_execution_result(
                    {"type": "automated", "step_id": "s3"}, {"success": True})
            runtime._begin_pending_checkpoint("s4", {"user_action": "start"})
            runtime.handle_execution_result(
                {"type": "automated", "step_id": "s4"}, {"success": True})

        try:
            # --- Recording: target=2 produces s1 + 2x(s2,s3) + s4 = 6 checkpoints ---
            runtime = _new_runtime()
            _run_count_loop(runtime, 2)
            cps = runtime.session.checkpoints
            self.assertEqual([cp.step_id for cp in cps],
                             ["s1", "s2", "s3", "s2", "s3", "s4"])
            self.assertEqual(cps[0].kind, "loop_count")
            self.assertTrue(cps[0].editable)
            self.assertEqual(cps[0].args, {"choice_value": "2"})
            self.assertEqual(cps[0].parameter_name, "numCuts")
            # Automated body steps are not editable but carry loop iteration info.
            self.assertFalse(cps[1].editable)
            self.assertEqual(cps[1].repeat.get("repeat_id"), "loop")
            self.assertEqual(cps[1].repeat.get("iteration"), 1)
            self.assertEqual(cps[3].repeat.get("iteration"), 2)
            self.assertEqual(cps[5].kind, "automated")
            # Timeline UI projection mirrors the checkpoints.
            ui = runtime.state_for_ui({"type": "automated", "step_id": "s4"})
            self.assertTrue(ui["can_replay"])
            self.assertEqual(len(ui["timeline"]), 6)
            self.assertEqual(ui["timeline"][0]["editable"], True)

            # --- Rewind into iteration 2 (checkpoint index 3 = s2, iteration 2) ---
            descriptor = runtime.rewind_to_checkpoint(3)
            self.assertEqual(descriptor["step_id"], "s2")
            self.assertEqual(descriptor["action"], "start")
            self.assertEqual(runtime.session.completed_steps, ["s1"])
            self.assertEqual(runtime.session.repeat_states["loop"]["iteration"], 2)
            self.assertEqual(runtime.session.repeat_states["loop"]["target"], 2)
            self.assertEqual(len(runtime.session.checkpoints), 3)
            self.assertEqual(runtime.session.active_checkpoint_index, 2)
            # Module-global mirror truncated to the same prefix.
            from SlicerAIAgentLib.ExtensionCLILoader import get_workflow_choices
            self.assertEqual(
                runtime.session.completed_instances,
                [{"step_id": "s1", "repeat": {}},
                 {"step_id": "s2", "repeat": {"repeat_id": "loop", "iteration": 1}},
                 {"step_id": "s3", "repeat": {"repeat_id": "loop", "iteration": 1}}],
            )

            # --- Rewind to the loop-count source with a MODIFIED value ---
            runtime2 = _new_runtime()
            _run_count_loop(runtime2, 2)
            descriptor2 = runtime2.rewind_to_checkpoint(0, {"choice_value": "4"})
            self.assertEqual(descriptor2["step_id"], "s1")
            self.assertEqual(descriptor2["action"], "choice_made")
            self.assertEqual(descriptor2["args"], {"choice_value": "4"})
            self.assertEqual(runtime2.session.completed_steps, [])
            self.assertEqual(runtime2.session.checkpoints, [])
            self.assertIsNone(runtime2.session.active_checkpoint_index)

            # --- Invalid index is rejected, not crashing ---
            self.assertIn("error", runtime2.rewind_to_checkpoint(9))
        finally:
            wf_mod.get_workflow_graph = original_get_workflow_graph
            reset_workflow_session("FakeReplayExt")

    def test_WorkflowReplayNavigation(self):
        """Back/Forward stepper: non-destructive scene+guidance navigation."""
        import importlib
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        WorkflowRuntime = wf_mod.WorkflowRuntime
        WorkflowSession = wf_mod.WorkflowSession
        from SlicerAIAgentLib.ExtensionCLILoader import reset_workflow_session

        graph = {
            "step_count": 3,
            "steps": [
                {"step_id": "s1", "step_type": "extension_op", "operation_type": "extension_op",
                 "description": "Step One"},
                {"step_id": "s2", "step_type": "extension_op", "operation_type": "extension_op",
                 "description": "Step Two"},
                {"step_id": "s3", "step_type": "extension_op", "operation_type": "extension_op",
                 "description": "Step Three"},
            ],
        }
        original_get_workflow_graph = wf_mod.get_workflow_graph
        wf_mod.get_workflow_graph = lambda extension_name: graph
        try:
            reset_workflow_session("FakeNavExt")
            runtime = WorkflowRuntime()
            runtime.session = WorkflowSession(
                extension_name="FakeNavExt", tool_name="FakeNavExt",
                workflow_id="nav_1", current_step="s1",
            )
            # Stub all scene ops so navigation is pure state logic.
            runtime._scene_node_ids = lambda: set()
            runtime._capture_sceneview = lambda step_id: None
            runtime._delete_sceneview = lambda nid: None
            runtime._delete_all_replay_sceneviews = lambda: None
            runtime._restore_to_view = lambda index: None
            runtime._commit_node_state = lambda index: None

            for sid in ("s1", "s2", "s3"):
                runtime._begin_pending_checkpoint(sid, {"user_action": "start"})
                runtime.handle_execution_result(
                    {"type": "automated", "step_id": sid}, {"success": True})

            self.assertEqual(len(runtime.session.checkpoints), 3)
            self.assertEqual(runtime.session.completed_steps, ["s1", "s2", "s3"])
            self.assertIsNone(runtime.session.preview_index)

            # Back from live -> last step; non-destructive (nothing deleted).
            st = runtime.navigate_back()
            self.assertEqual(runtime.session.preview_index, 2)
            self.assertEqual(st["description"], "Step Three")
            self.assertTrue(st["replay_previewing"])
            self.assertEqual(len(runtime.session.checkpoints), 3)         # unchanged
            self.assertEqual(runtime.session.completed_steps, ["s1", "s2", "s3"])

            st = runtime.navigate_back()
            self.assertEqual(runtime.session.preview_index, 1)
            self.assertEqual(st["description"], "Step Two")

            st = runtime.navigate_back()
            self.assertEqual(runtime.session.preview_index, 0)
            self.assertEqual(st["description"], "Step One")
            self.assertFalse(st["replay_can_back"])                       # at the first step

            # Forward returns step by step, then back to the live state.
            st = runtime.navigate_forward()
            self.assertEqual(runtime.session.preview_index, 1)
            self.assertEqual(st["description"], "Step Two")
            runtime.navigate_forward()                                    # -> index 2
            st = runtime.navigate_forward()                              # -> live
            self.assertIsNone(runtime.session.preview_index)
            self.assertFalse(st.get("replay_previewing"))
            # Still non-destructive after the full round trip.
            self.assertEqual(len(runtime.session.checkpoints), 3)
            self.assertEqual(runtime.session.completed_steps, ["s1", "s2", "s3"])

            # Still non-destructive after a full round trip.
            self.assertEqual(len(runtime.session.checkpoints), 3)
            self.assertEqual(runtime.session.completed_steps, ["s1", "s2", "s3"])
        finally:
            wf_mod.get_workflow_graph = original_get_workflow_graph
            reset_workflow_session("FakeNavExt")

    def test_WorkflowNodeSelectionBinding(self):
        """Node-selection steps surface node_class/keywords for the dropdown."""
        import importlib
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        WorkflowRuntime = wf_mod.WorkflowRuntime
        WorkflowSession = wf_mod.WorkflowSession
        WorkflowCheckpoint = wf_mod.WorkflowCheckpoint
        from SlicerAIAgentLib.ExtensionCLILoader import reset_workflow_session

        graph = {"step_count": 2, "steps": [
            {"step_id": "cb_step_2", "operation_type": "user_choice", "step_type": "user_choice",
             "description": "Choose the mandibular segmentation",
             "choice_info": {"parameter_name": "mandibularSegmentation", "choices": []}},
            {"step_id": "cb_step_16", "operation_type": "user_choice", "step_type": "user_choice",
             "description": "How many cut planes?",
             "choice_info": {"parameter_name": "numberOfCutPlanes", "choices": []}}]}
        meta = {"FakeNodeExt": {"workflow_metadata": {"parameter_bindings": {
            "mandibularSegmentation": {"node_class": "vtkMRMLSegmentationNode",
                                       "keywords": ["mandibular", "segmentation"]},
            "numberOfCutPlanes": {"node_class": "", "keywords": []}}}}}
        orig_graph = wf_mod.get_workflow_graph
        orig_ext = wf_mod.get_validated_extensions
        wf_mod.get_workflow_graph = lambda extension_name: graph
        wf_mod.get_validated_extensions = lambda: meta
        try:
            reset_workflow_session("FakeNodeExt")
            runtime = WorkflowRuntime()
            runtime.session = WorkflowSession(
                extension_name="FakeNodeExt", tool_name="FakeNodeExt",
                workflow_id="node_1", current_step="cb_step_2")

            b = runtime._node_binding_for_param("mandibularSegmentation")
            self.assertEqual(b.get("node_class"), "vtkMRMLSegmentationNode")
            self.assertIn("mandibular", b.get("keywords", []))

            # node-selection step → node_class carried; free-form param → empty
            st = runtime.state_for_ui({"type": "user_choice", "step_id": "cb_step_2",
                                       "parameter_name": "mandibularSegmentation", "choices": []})
            self.assertEqual(st["node_class"], "vtkMRMLSegmentationNode")
            self.assertTrue(st["needs_choice_input"])
            st2 = runtime.state_for_ui({"type": "user_choice", "step_id": "cb_step_16",
                                        "parameter_name": "numberOfCutPlanes", "choices": []})
            self.assertEqual(st2["node_class"], "")

            # replay preview carries node_class via the checkpoint's parameter_name
            cp = WorkflowCheckpoint(index=0, step_id="cb_step_2", kind="choice",
                                    action="choice_made", args={"choice_value": "MandibleSegmentation"},
                                    recorded_value="MandibleSegmentation",
                                    parameter_name="mandibularSegmentation", editable=True, choices=[])
            runtime.session.checkpoints = [cp]
            runtime.session.preview_index = 0
            pv = runtime._preview_ui_state()
            self.assertEqual(pv["node_class"], "vtkMRMLSegmentationNode")
            self.assertEqual(pv["default_value"], "MandibleSegmentation")
            self.assertTrue(pv["needs_choice_input"])
        finally:
            wf_mod.get_workflow_graph = orig_graph
            wf_mod.get_validated_extensions = orig_ext
            reset_workflow_session("FakeNodeExt")

    def test_WorkflowStepInstructions(self):
        """Clinical step instructions override the panel + the generation stage."""
        import importlib, json
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        WorkflowRuntime = wf_mod.WorkflowRuntime
        WorkflowSession = wf_mod.WorkflowSession
        from SlicerAIAgentLib.ExtensionCLILoader import reset_workflow_session

        graph = {"step_count": 1, "steps": [
            {"step_id": "cb_step_10", "operation_type": "user_interaction", "step_type": "interactive",
             "description": "Adjust slice intersection",
             "ui_guidance": {"title": "Adjust slice intersection", "instruction": "Drag the cross lines."}}]}
        ext = {"BRP": {"workflow_metadata": {}, "step_instructions": {"version": 1, "steps": {
            "cb_step_10": {"title": "Position the cutting plane",
                           "simple": "Drag the crosshair to set the cut.",
                           "detailed": "Defines the osteotomy plane.", "edited": False}}}}}
        og, oe = wf_mod.get_workflow_graph, wf_mod.get_validated_extensions
        wf_mod.get_workflow_graph = lambda e: graph
        wf_mod.get_validated_extensions = lambda: ext
        try:
            reset_workflow_session("BRP")
            r = WorkflowRuntime()
            r.session = WorkflowSession(extension_name="BRP", tool_name="BRP",
                                        workflow_id="1", current_step="cb_step_10")
            st = r.state_for_ui({"type": "interactive", "step_id": "cb_step_10",
                                 "interaction": {"placement_instructions": "Drag the cross lines."}})
            self.assertEqual(st["description"], "Position the cutting plane")
            self.assertEqual(st["instructions"], "Drag the crosshair to set the cut.")
            self.assertEqual(st["instructions_detailed"], "Defines the osteotomy plane.")
            # No saved instructions -> fall back to ui_guidance
            ext["BRP"]["step_instructions"] = {"version": 1, "steps": {}}
            st2 = r.state_for_ui({"type": "interactive", "step_id": "cb_step_10",
                                  "interaction": {"placement_instructions": "Drag the cross lines."}})
            self.assertEqual(st2["description"], "Adjust slice intersection")
            self.assertEqual(st2["instructions_detailed"], "")
        finally:
            wf_mod.get_workflow_graph = og
            wf_mod.get_validated_extensions = oe
            reset_workflow_session("BRP")

        # Generation stage: LLM result, edit-preservation, and fallback.
        from SlicerAIAgentLib.extension_cli_analyzer.step_instructions import AnalyzerStepInstructionsMixin

        class _FakeAnalyzer(AnalyzerStepInstructionsMixin):
            llm_client = object()
            def _build_guidance_context(self, step, md, g):
                return {"step_id": step.get("step_id", ""), "description": step.get("description", "")}
            def _clean_guidance_title(self, t):
                return t
            def _call_llm(self, prompt, call_class=None):
                return '{"steps":[{"step_id":"s1","title":"T1","simple":"S1","detailed":"D1"}]}'
            def _parse_json_response(self, raw):
                return json.loads(raw)

        wfg = {"steps": [
            {"step_id": "s1", "description": "do A", "ui_guidance": {"title": "A"}},
            {"step_id": "s2", "description": "do B"},
            {"step_id": "s3", "description": "do C", "ui_guidance": {"title": "C", "instruction": "click C"}}]}
        existing = {"version": 1, "steps": {"s2": {"title": "kept", "simple": "kept",
                                                   "detailed": "kept", "edited": True}}}
        res = _FakeAnalyzer().generate_step_instructions(
            wfg, {}, {"module_name": "X"}, {}, cookbook_def=None, existing=existing)
        steps = res["steps"]
        self.assertEqual(steps["s1"], {"title": "T1", "simple": "S1", "detailed": "D1", "edited": False})
        self.assertEqual(steps["s2"]["edited"], True)          # preserved
        self.assertEqual(steps["s2"]["title"], "kept")
        self.assertEqual(steps["s3"]["simple"], "click C")     # fallback from ui_guidance
        self.assertEqual(steps["s3"]["edited"], False)

    def test_WorkflowRuntimeUiStateMapping(self):
        """Test compact workflow UI state for progress and controls."""
        import importlib
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        WorkflowRuntime = wf_mod.WorkflowRuntime
        WorkflowSession = wf_mod.WorkflowSession

        graph = {
            "step_count": 4,
            "steps": [
                {"step_id": "cb_step_1", "step_type": "automated", "description": "Load data"},
                {
                    "step_id": "cb_step_2",
                    "step_type": "interactive",
                    "description": "Cookbook says click the old plane instructions.",
                    "is_optional": True,
                    "ui_guidance": {
                        "title": "Place cutting plane",
                        "instruction": "Position this cutting plane, then click Done.",
                        "done_label": "Plane placed",
                        "object_label": "cutting plane",
                    },
                },
                {
                    "step_id": "cb_step_3",
                    "step_type": "user_choice",
                    "description": "Choose count",
                    "choice_info": {
                        "choices": [
                            {"label": "One", "value": "1"},
                            {"label": "Two", "value": "2"},
                        ],
                    },
                },
                {
                    "step_id": "cb_step_16",
                    "step_type": "user_choice",
                    "description": "Choose cut plane count",
                    "choice_info": {
                        "question": "How many mandibular cut planes would you like to create?",
                        "choices": [],
                        "parameter_name": "numberOfCutPlanes",
                        "default_value": "1",
                    },
                    "ui_guidance": {
                        "title": "Choose cutting plane count",
                        "instruction": "Enter the number of cutting planes.",
                        "input_label": "Number of cutting planes",
                        "choice_label": "Number of cutting planes",
                    },
                },
            ],
        }
        original_get_workflow_graph = wf_mod.get_workflow_graph
        wf_mod.get_workflow_graph = lambda extension_name: graph
        try:
            runtime = WorkflowRuntime()
            runtime.session = WorkflowSession(
                extension_name="FakeExtension",
                tool_name="FakeExtension",
                workflow_id="fake_ui_1",
                current_step="cb_step_1",
            )

            automated = runtime.state_for_ui({"type": "automated", "step_id": "cb_step_1"})
            self.assertEqual(automated["total_steps"], 4)
            self.assertEqual(automated["current_index"], 1)
            self.assertFalse(automated["can_done"])

            runtime.session.status = "waiting_for_user"
            runtime.session.current_step = "cb_step_2"
            interactive = runtime.state_for_ui({
                "type": "interactive",
                "step_id": "cb_step_2",
                "interaction_instructions": "Hold Shift and move mouse in a view.",
                "repeat_progress": {"current": 2, "total": 3},
            })
            self.assertEqual(interactive["status"], "Waiting for your interaction")
            self.assertTrue(interactive["can_done"])
            self.assertTrue(interactive["can_skip"])
            self.assertEqual(interactive["description"], "Place cutting plane")
            self.assertIn("Position this cutting plane", interactive["instructions"])
            self.assertEqual(interactive["done_label"], "Plane placed")
            self.assertEqual(interactive["repeat_progress"]["current"], 2)

            runtime.session.status = "waiting_for_choice"
            runtime.session.current_step = "cb_step_3"
            choice = runtime.state_for_ui({
                "type": "user_choice",
                "step_id": "cb_step_3",
                "question": "How many planes?",
                "instruction": "Ask the user, then call FakeExtension with user_action='choice_made'.",
            })
            self.assertEqual(choice["status"], "Waiting for your choice")
            self.assertEqual(choice["description"], "How many planes?")
            self.assertEqual(choice["instructions"], "")
            self.assertFalse(choice["can_done"])
            self.assertEqual(choice["choices"][1]["value"], "2")

            compact_choice = runtime.state_for_ui({
                "type": "user_choice",
                "step_id": "cb_step_99",
                "question": "Use right side leg?",
                "instruction": (
                    "Ask the user: 'Use right side leg?'\n"
                    "Options:\n"
                    "  1. Yes\n"
                    "  2. No\n"
                    "Wait for the user's response."
                ),
            })
            self.assertEqual(compact_choice["choices"][0]["label"], "Yes")
            self.assertEqual(compact_choice["choices"][0]["value"], "true")
            self.assertEqual(compact_choice["choices"][1]["value"], "false")

            runtime.session.current_step = "cb_step_16"
            count_choice = runtime.state_for_ui({
                "type": "user_choice",
                "step_id": "cb_step_16",
                "question": "How many mandibular cut planes would you like to create?",
                "default_value": "1",
                "parameter_name": "numberOfCutPlanes",
            })
            self.assertEqual(count_choice["choices"], [])
            self.assertTrue(count_choice["needs_choice_input"])
            self.assertEqual(count_choice["default_value"], "1")
            self.assertEqual(count_choice["description"], "Choose cutting plane count")
            self.assertEqual(count_choice["input_label"], "Number of cutting planes")

            runtime.session.status = "completed"
            runtime.session.current_step = None
            runtime.session.completed_steps = ["cb_step_1", "cb_step_2", "cb_step_3"]
            completed = runtime.state_for_ui({"workflow_completed": True})
            self.assertEqual(completed["status"], "Completed")
            self.assertEqual(completed["completed_steps"], 3)
            self.assertFalse(completed["can_cancel"])
        finally:
            wf_mod.get_workflow_graph = original_get_workflow_graph

    def test_WorkflowWidgetClearsStaleMarkers(self):
        """Test stale workflow tool results are cleared after a workflow turn."""
        from SlicerAIAgent import SlicerAIAgentWidget

        class FakeLogic:
            pass

        interactive_step = {
            "tool": "BoneReconstructionPlanner",
            "step_id": "cb_step_10",
            "type": "interactive",
        }
        same_cached_step = dict(interactive_step)

        widget = object.__new__(SlicerAIAgentWidget)
        widget.logic = FakeLogic()
        widget.logic._lastInteractiveStep = interactive_step
        widget.logic._lastWorkflowStep = same_cached_step
        widget._currentWorkflowStepInfo = interactive_step
        widget._waitingForUser = True
        widget._autoAdvanceWorkflowStep = {"step_id": "cb_step_11"}
        widget._activeWorkflowId = "BoneReconstructionPlanner_1"
        widget._taskWorkflowPanelActive = True

        self.assertTrue(widget._sameWorkflowStepMarker(interactive_step, same_cached_step))

        widget._clearWorkflowResultMarkers()
        self.assertIsNone(widget.logic._lastInteractiveStep)
        self.assertIsNone(widget.logic._lastWorkflowStep)

        widget.logic._lastInteractiveStep = interactive_step
        widget.logic._lastWorkflowStep = same_cached_step
        widget._clearCompletedWorkflowState()
        self.assertIsNone(widget.logic._lastInteractiveStep)
        self.assertIsNone(widget.logic._lastWorkflowStep)
        self.assertIsNone(widget._currentWorkflowStepInfo)
        self.assertFalse(widget._waitingForUser)
        self.assertIsNone(widget._autoAdvanceWorkflowStep)
        self.assertIsNone(widget._activeWorkflowId)
        self.assertFalse(widget._taskWorkflowPanelActive)

    def test_SafeExecutor(self):
        """Test SafeExecutor functionality."""
        from SlicerAIAgentLib import SafeExecutor

        executor = SafeExecutor()

        # Test simple execution
        code = "result = 2 + 2"
        result = executor.execute(code)
        self.assertTrue(result["success"], f"Simple code should execute: {result.get('error')}")

        # Test output capture
        code = "print('Hello, World!')"
        result = executor.execute(code)
        self.assertTrue(result["success"])
        self.assertIn("Hello, World!", result["output"])

        # Test exception handling
        code = "raise ValueError('Test error')"
        result = executor.execute(code)
        self.assertFalse(result["success"])
        self.assertIn("Test error", result["error"])
        self.assertIsNotNone(result["traceback"])

        # Test globals
        executor.addGlobal("test_var", 42)
        code = "print(test_var)"
        result = executor.execute(code)
        self.assertTrue(result["success"])
        self.assertIn("42", result["output"])

        # Test history
        self.assertTrue(len(executor.getHistory()) > 0)
        executor.clearHistory()
        self.assertEqual(len(executor.getHistory()), 0)

        self.delayDisplay("SafeExecutor tests passed")

    def test_ProgressBarFilter(self):
        """Test SafeExecutor progress bar filtering."""
        from SlicerAIAgentLib.SafeExecutor import SafeExecutor

        # Test 1: Strip tqdm lines
        noisy = (
            "0%| | 0/25 [00:00<?, ?it/s] 4%|4 | 1/25 [00:39<15:44, 39.35s/it]\n"
            "100%|##########| 25/25 [13:27<00:00, 32.28s/it]\n"
        )
        self.assertEqual(SafeExecutor._filter_progress_bars(noisy), "")

        # Test 2: Strip checkpoint loading lines
        shards = (
            "Loading checkpoint shards: 0%| | 0/2 [00:00<?, ?it/s] "
            "Loading checkpoint shards: 100%|##########| 2/2 [00:35<00:00, 17.65s/it]\n"
        )
        self.assertEqual(SafeExecutor._filter_progress_bars(shards), "")

        # Test 3: Preserve real errors
        mixed = (
            "0%| | 0/25 [00:00<?, ?it/s]\n"
            "[VTK ERROR] Invalid segment\n"
            "Loading checkpoint shards: 50%|##### | 1/2 [00:22<00:22, 22.69s/it]\n"
            "RuntimeError: model not found\n"
        )
        filtered = SafeExecutor._filter_progress_bars(mixed)
        self.assertIn("[VTK ERROR] Invalid segment", filtered)
        self.assertIn("RuntimeError: model not found", filtered)
        self.assertNotIn("0%|", filtered)
        self.assertNotIn("Loading checkpoint shards", filtered)

        # Test 4: Empty input
        self.assertEqual(SafeExecutor._filter_progress_bars(""), "")
        self.assertEqual(SafeExecutor._filter_progress_bars(None), None)

        self.delayDisplay("Progress bar filter tests passed")

    def test_SkillPath(self):
        """Test skill path resolution and mode detection in logic."""
        from SlicerAIAgent import SlicerAIAgentLogic

        logic = SlicerAIAgentLogic()

        self.assertIsNotNone(logic.skill_path)
        self.assertTrue(os.path.exists(logic.skill_path) or logic.skill_mode == "unknown")
        self.assertIn(logic.skill_mode, ["full", "lightweight", "web", "unknown"])

        logic.cleanup()
        self.delayDisplay("Skill path tests passed")

    def test_ConversationStore(self):
        """Test ConversationStore functionality."""
        from SlicerAIAgentLib import ConversationStore

        store = ConversationStore()
        store.clearAll()

        # Test adding exchanges (with reasoning_content)
        store.addExchange("Hello", {"message": "Hi", "reasoning_content": "Thinking about greeting", "tokens": 10, "cost": 0.001})
        store.addExchange("How are you?", {"message": "I'm good", "tokens": 15, "cost": 0.0015})

        # Test retrieval
        current = store.getCurrentSession()
        self.assertEqual(len(current), 2)
        self.assertEqual(current[0].get("reasoning_content"), "Thinking about greeting")
        self.assertEqual(current[1].get("reasoning_content"), "")

        # Test stats
        stats = store.getStats()
        self.assertEqual(stats["current_session_exchanges"], 2)
        self.assertEqual(stats["current_session_tokens"], 25)

        # Test search
        results = store.search("Hello")
        self.assertEqual(len(results), 1)

        # Test export/import with temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            export_path = os.path.join(temp_dir, "test_conversation.json")
            store.exportSession(export_path)
            self.assertTrue(os.path.exists(export_path))

            store.clearAll()
            self.assertEqual(len(store.getAllConversations()), 0)

            store.importSession(export_path)
            self.assertTrue(len(store.getAllConversations()) > 0)
        finally:
            shutil.rmtree(temp_dir)

        # Test new session
        old_session = store.current_session_id
        store.newSession()
        self.assertNotEqual(store.current_session_id, old_session)

        self.delayDisplay("ConversationStore tests passed")

    def test_SlicerCodeTemplates(self):
        """Test SlicerCodeTemplates functionality."""
        from SlicerAIAgentLib import SlicerCodeTemplates

        # Test getting specific template
        template = SlicerCodeTemplates.getTemplate("load_volume")
        self.assertIsNotNone(template)
        self.assertIn("code", template)

        # Test getting all templates
        all_templates = SlicerCodeTemplates.getAllTemplates()
        self.assertTrue(len(all_templates) > 0)

        # Test finding by tag
        results = SlicerCodeTemplates.findByTag("volume")
        self.assertTrue(len(results) > 0)

        # Test finding by keyword
        results = SlicerCodeTemplates.findByKeyword("load")
        self.assertTrue(len(results) > 0)

        # Test formatting for prompt
        formatted = SlicerCodeTemplates.formatForPrompt(results[:2])
        self.assertIn("```python", formatted)

        self.delayDisplay("SlicerCodeTemplates tests passed")

    def test_GenerateSegmentationCode(self):
        """Test the GenerateSegmentationCode tool."""
        from SlicerAIAgentLib import SkillTools

        tools = SkillTools.SkillTools()

        # Test 1: Basic organ segmentation — passes through arbitrary prompts
        result = tools.execute("GenerateSegmentationCode", {"prompt": "segment the liver and spleen"})
        self.assertNotIn("error", result, f"Tool returned error: {result.get('error')}")
        self.assertEqual(result["tool"], "GenerateSegmentationCode")
        self.assertIn("liver", result["extracted_targets"])
        self.assertIn("spleen", result["extracted_targets"])
        self.assertIn("code", result)
        self.assertIn("VoxTell", result["code"])
        self.assertIn("runSegmentation", result["code"])
        self.assertIn("explanation", result)
        self.assertIn("requirements", result)

        # Test 2: Arbitrary anatomical terms (no keyword filtering)
        result = tools.execute("GenerateSegmentationCode", {"prompt": "find the hippocampus and amygdala"})
        self.assertIn("hippocampus", result["extracted_targets"])
        self.assertIn("amygdala", result["extracted_targets"])
        self.assertIn("runSegmentation", result["code"])

        # Test 3: Custom volume node name and output name
        result = tools.execute(
            "GenerateSegmentationCode",
            {
                "prompt": "segment the aorta",
                "volume_node_name": "CTChest",
                "output_segmentation_name": "AortaSeg",
            }
        )
        self.assertIn("CTChest", result["code"])
        self.assertIn("AortaSeg", result["code"])
        self.assertIn("aorta", result["extracted_targets"])

        # Test 4: No recognizable anatomical targets (fallback)
        result = tools.execute("GenerateSegmentationCode", {"prompt": "do something vague"})
        self.assertEqual(result["extracted_targets"], ["structure of interest"])
        self.assertIn("runSegmentation", result["code"])

        # Test 5: Verify GPU detection, widget-based logic access, offline mode, model check
        result = tools.execute("GenerateSegmentationCode", {"prompt": "segment kidneys"})
        self.assertIn("torch.cuda.is_available()", result["code"])
        self.assertIn("8_000_000_000", result["code"])
        self.assertIn("useGpu=use_gpu", result["code"])
        self.assertIn("widgetRepresentation().self()", result["code"])
        self.assertIn("_voxtell_widget.logic", result["code"])
        self.assertIn("isModelInstalled", result["code"])
        self.assertNotIn("slicer.modules.voxtell.logic()", result["code"])
        self.assertNotIn("import os", result["code"])

        # Test 6: Tool spec includes the new tool
        spec = tools.getToolsSpec()
        tool_names = [t["function"]["name"] for t in spec]
        self.assertIn("GenerateSegmentationCode", tool_names)

        self.delayDisplay("GenerateSegmentationCode tests passed")

    def test_NodeSelectionFallbackNodeClass(self):
        """Node-selection steps without an inferred binding still expose a node class.

        Regression: SlicerOrbitSurgerySim's node selectors use a
        qMRMLSubjectHierarchyComboBox / parameterNodeWrapper pattern the pipeline's
        binding inference does not recognize, so no binding (and thus no node
        class) reached the panel and it showed a free-text box instead of a node
        dropdown. The runtime now falls back to the node class the step graph
        already records on its node_roles / sub_operations.
        """
        from SlicerAIAgentLib.WorkflowRuntime import WorkflowRuntime

        fn = WorkflowRuntime._node_class_from_step_meta

        # Orbit-style: node choice with no binding; class on node_roles + sub_op.
        orbit_step = {
            "node_roles": [
                {"role_kind": "choice_input", "node_class": "vtkMRMLModelNode", "parameter_name": ""}
            ],
            "sub_operations": [{"value_kind": "node", "node_class": "vtkMRMLModelNode"}],
        }
        self.assertEqual(fn(orbit_step), "vtkMRMLModelNode")

        # Class only on the sub-operation (node_roles class blank).
        sub_only = {
            "node_roles": [{"role_kind": "choice_input", "node_class": ""}],
            "sub_operations": [{"value_kind": "node", "node_class": "vtkMRMLSegmentationNode"}],
        }
        self.assertEqual(fn(sub_only), "vtkMRMLSegmentationNode")

        # Boolean checkbox choice (e.g. BRP "right side leg"): no node class, so
        # the panel keeps its button/free-text rendering, not a node dropdown.
        bool_step = {
            "node_roles": [{"role_kind": "choice_input", "node_class": "", "parameter_name": "rightSideLegFibula"}],
            "sub_operations": [{"value_kind": "bool", "node_class": None}],
        }
        self.assertEqual(fn(bool_step), "")

        # Defensive: malformed input never raises.
        self.assertEqual(fn(None), "")
        self.assertEqual(fn({}), "")

        self.delayDisplay("Node-selection fallback node class resolves correctly")

    def test_IsNodeSelectionStep(self):
        """_is_node_selection_step flags node picks, excludes bool/enum/segments."""
        from SlicerAIAgentLib.WorkflowRuntime import WorkflowRuntime

        fn = WorkflowRuntime._is_node_selection_step

        # Node pick via the sub-op value_kind.
        self.assertTrue(fn({"sub_operations": [
            {"value_kind": "node", "node_class": "vtkMRMLScalarVolumeNode"}]}))
        # Node pick via a choice_input node-role alone (older artifacts).
        self.assertTrue(fn({"node_roles": [
            {"role_kind": "choice_input", "node_class": "vtkMRMLModelNode"}]}))
        # Boolean checkbox: no node class -> keeps its buttons, not a node pick.
        self.assertFalse(fn({
            "sub_operations": [{"value_kind": "bool", "node_class": None}],
            "node_roles": [{"role_kind": "choice_input", "node_class": ""}]}))
        # Segment-visibility selection is routed separately -> excluded even when
        # a choice_input node-role carries a segmentation class.
        self.assertFalse(fn({
            "sub_operations": [{"value_kind": "segment_visibility_selection",
                                "node_class": "vtkMRMLSegmentationNode"}],
            "node_roles": [{"role_kind": "choice_input",
                            "node_class": "vtkMRMLSegmentationNode"}]}))
        # Defensive: malformed input never raises.
        self.assertFalse(fn(None))
        self.assertFalse(fn({}))

        self.delayDisplay("_is_node_selection_step classifies node picks correctly")

    def test_NodePickDropsLiteralChoices(self):
        """A node-pick user_choice drops literal cookbook labels -> node tree.

        Regression for PelvicFracturePlanning cb_step_1 ("choose the Pelvic
        Volume"): the LLM tagged the step value_kind=node + node_class but ALSO
        emitted a literal choices=[{"label":"Pelvic Volume"}]. A non-empty
        choices array short-circuited needs_choice_input, so the panel rendered
        one button instead of the scene node tree. Node picks now drop the
        literal choices in both the live and replay paths; genuine boolean/loop
        decisions keep their buttons.
        """
        import importlib
        wf_mod = importlib.import_module("SlicerAIAgentLib.WorkflowRuntime")
        WorkflowRuntime = wf_mod.WorkflowRuntime
        WorkflowSession = wf_mod.WorkflowSession
        WorkflowCheckpoint = wf_mod.WorkflowCheckpoint
        from SlicerAIAgentLib.ExtensionCLILoader import reset_workflow_session

        node_choices = [{"label": "Pelvic Volume", "value": "inputVolume"}]
        graph = {"step_count": 2, "steps": [
            {"step_id": "cb_step_1", "operation_type": "user_choice", "step_type": "user_choice",
             "description": "In the \"Input CT Volume\" option, choose the Pelvic Volume.",
             "choice_info": {"parameter_name": "inputVolume", "choices": node_choices},
             "node_roles": [{"role_kind": "choice_input", "node_class": "vtkMRMLScalarVolumeNode",
                             "parameter_name": "inputVolume"}],
             "sub_operations": [{"value_kind": "node", "node_class": "vtkMRMLScalarVolumeNode"}]},
            {"step_id": "cb_step_8", "operation_type": "user_choice", "step_type": "user_choice",
             "description": "Manually adjust a fragment?",
             "choice_info": {"parameter_name": "manualAdjust",
                             "choices": [{"label": "Yes", "value": True}, {"label": "No", "value": False}]},
             "node_roles": [{"role_kind": "choice_input", "node_class": "", "parameter_name": "manualAdjust"}],
             "sub_operations": [{"value_kind": "bool", "node_class": None}]}]}
        # Empty parameter_bindings mirrors Pelvic (binding inference found nothing).
        meta = {"PelvicFracturePlanning": {"workflow_metadata": {"parameter_bindings": {}}}}
        orig_graph = wf_mod.get_workflow_graph
        orig_ext = wf_mod.get_validated_extensions
        wf_mod.get_workflow_graph = lambda extension_name: graph
        wf_mod.get_validated_extensions = lambda: meta
        try:
            reset_workflow_session("PelvicFracturePlanning")
            runtime = WorkflowRuntime()
            runtime.session = WorkflowSession(
                extension_name="PelvicFracturePlanning", tool_name="PelvicFracturePlanning",
                workflow_id="pelvic_1", current_step="cb_step_1")

            # Live path: node-pick step drops the literal choice -> node tree.
            st = runtime.state_for_ui({"type": "user_choice", "step_id": "cb_step_1",
                                       "parameter_name": "inputVolume", "choices": list(node_choices)})
            self.assertEqual(st["choices"], [])
            self.assertTrue(st["needs_choice_input"])
            self.assertEqual(st["node_class"], "vtkMRMLScalarVolumeNode")

            # Live path control: a boolean step keeps its Yes/No buttons.
            st_bool = runtime.state_for_ui({"type": "user_choice", "step_id": "cb_step_8",
                                            "parameter_name": "manualAdjust",
                                            "choices": [{"label": "Yes", "value": True},
                                                        {"label": "No", "value": False}]})
            self.assertEqual(len(st_bool["choices"]), 2)
            self.assertFalse(st_bool["needs_choice_input"])
            self.assertEqual(st_bool["node_class"], "")

            # Replay path: a recorded node-pick checkpoint also drops the literal
            # choice and recovers the node class from the step graph.
            cp = WorkflowCheckpoint(index=0, step_id="cb_step_1", kind="choice",
                                    action="choice_made", args={"choice_value": "PelvicCT"},
                                    recorded_value="PelvicCT", parameter_name="inputVolume",
                                    editable=True, choices=list(node_choices))
            runtime.session.checkpoints = [cp]
            runtime.session.preview_index = 0
            pv = runtime._preview_ui_state()
            self.assertEqual(pv["choices"], [])
            self.assertTrue(pv["needs_choice_input"])
            self.assertEqual(pv["node_class"], "vtkMRMLScalarVolumeNode")

            # Replay control: a loop decision on the same step keeps its Yes/No.
            cp_loop = WorkflowCheckpoint(index=0, step_id="cb_step_1", kind="loop_decision",
                                         action="repeat_decision", args={}, recorded_value="Yes",
                                         parameter_name="inputVolume", editable=True,
                                         choices=[{"label": "Yes", "value": "yes"},
                                                  {"label": "No", "value": "no"}])
            runtime.session.checkpoints = [cp_loop]
            runtime.session.preview_index = 0
            pv_loop = runtime._preview_ui_state()
            self.assertEqual(len(pv_loop["choices"]), 2)
        finally:
            wf_mod.get_workflow_graph = orig_graph
            wf_mod.get_validated_extensions = orig_ext
            reset_workflow_session("PelvicFracturePlanning")

        self.delayDisplay("Node-pick steps drop literal choices in live + replay")

    def test_UnboundNodeChoiceMaterialization(self):
        """Unbound node choices are materialized into the globals templates read.

        Companion to the UI fallback: when a node selector has no inferred binding
        (subject-hierarchy combo / parameterNodeWrapper), the recorded choice was
        never applied, so a downstream template reading the cached node id raised
        "Missing required input". The choice handler now emits code that resolves
        the node by name and sets both the namespace variable and the
        `_<ext_slug>_<param>_id` cross-stage global. Bound and non-node choices
        emit nothing (the prelude / button path handles those).
        """
        from SlicerAIAgentLib.extension_cli_loader.choice_helpers import (
            _build_node_choice_materialization_code,
        )

        class _Ctx:
            def __init__(self, ext, step, target_step, metadata, target_gen=None):
                self.ext_name = ext
                self.workflow_step = step
                self.target_step = target_step
                self.target_gen = target_gen or {}
                self.metadata = metadata or {}

        # Orbit-style unbound node choice -> materialize, valid Python, exact global.
        orbit_step = {
            "step_id": "cb_step_1",
            "node_roles": [{"role_kind": "choice_input", "node_class": "vtkMRMLModelNode", "parameter_name": ""}],
            "sub_operations": [{"value_kind": "node", "node_class": "vtkMRMLModelNode"}],
            "choice_info": {"parameter_name": "orbit_model_node"},
        }
        ctx = _Ctx("SlicerOrbitSurgerySim", "cb_step_1", orbit_step,
                   {"parameter_bindings": {}, "choice_bindings": {}})
        code = _build_node_choice_materialization_code(ctx, "orbit_model_node", "SkullModel")
        self.assertTrue(code)
        # Cross-stage global name must match the generator convention (and so the
        # name cb_step_9's template reads).
        self.assertIn("_slicerorbitsurgerysim_orbit_model_node_id = _chosen_node.GetID()", code)
        self.assertIn("orbit_model_node = _chosen_node", code)
        compile(code, "<gen>", "exec")  # raises if the emitted code is not valid Python

        # Bound node choice (BRP) -> applied by the prelude, so no materialization.
        bound_step = {
            "step_id": "cb_step_2",
            "node_roles": [{"role_kind": "choice_input", "node_class": "vtkMRMLSegmentationNode", "parameter_name": "mandibularSegmentation"}],
        }
        ctx2 = _Ctx("BoneReconstructionPlanner", "cb_step_2", bound_step,
                    {"parameter_bindings": {"mandibularSegmentation": {"node_class": "vtkMRMLSegmentationNode"}}})
        self.assertEqual(_build_node_choice_materialization_code(ctx2, "mandibularSegmentation", "MandibleSegmentation"), "")

        # Boolean choice -> no node class -> no materialization.
        bool_step = {"step_id": "cb_step_1", "sub_operations": [{"value_kind": "bool"}], "node_roles": []}
        ctx3 = _Ctx("BoneReconstructionPlanner", "cb_step_1", bool_step, {})
        self.assertEqual(_build_node_choice_materialization_code(ctx3, "rightSideLegFibula", True), "")

        self.delayDisplay("Unbound node choice materialization verified")

    def test_InstalledExtensionSearchRedirect(self):
        """Search/read of an installed extension resolves to the runtime copy, not the KB.

        Regression for the knowledge-base-vs-installed version skew (a method like
        rotation_p_stop present only in the newer KB copy but absent at runtime).
        _resolve_path redirects an installed extension's slicer-extensions/<Ext>/
        and ext:<Ext>/ paths to the installed source registered in extra_roots,
        with a basename fallback through the differing installed subtree.
        """
        import os
        import tempfile
        from SlicerAIAgentLib import SkillTools

        work = tempfile.mkdtemp(prefix="cli_resolve_")
        skill = os.path.join(work, "kb")
        os.makedirs(os.path.join(skill, "slicer-extensions", "OtherExt"))
        installed = os.path.join(work, "inst", "SlicerOrbitSurgerySim")
        deep = os.path.join(installed, "lib", "qt-scripted-modules")
        os.makedirs(deep)
        open(os.path.join(deep, "PlateRegistration.py"), "w").close()

        ex = SkillTools.SkillToolExecutor(skill)
        ex.extra_roots["SlicerOrbitSurgerySim"] = installed

        def _norm(p):
            return os.path.normpath(p)

        runtime_file = _norm(os.path.join(deep, "PlateRegistration.py"))
        # KB-style path of an installed extension -> redirected to runtime copy.
        self.assertEqual(
            _norm(ex._resolve_path(
                "slicer-extensions/SlicerOrbitSurgerySim/PlateRegistration/PlateRegistration.py"
            )),
            runtime_file,
        )
        # ext:<Ext>/ form too.
        self.assertEqual(
            _norm(ex._resolve_path("ext:SlicerOrbitSurgerySim/PlateRegistration.py")),
            runtime_file,
        )
        # Directory (grep root) -> installed root.
        self.assertEqual(
            _norm(ex._resolve_path("slicer-extensions/SlicerOrbitSurgerySim")),
            _norm(installed),
        )
        # A non-installed extension stays in the knowledge base.
        self.assertEqual(
            _norm(ex._resolve_path("slicer-extensions/OtherExt/Foo.py")),
            _norm(os.path.join(skill, "slicer-extensions", "OtherExt", "Foo.py")),
        )
        self.delayDisplay("Installed extension search redirects to the runtime copy")

    def test_WrapperExtensionSkipsClassicPrelude(self):
        """A parameterNodeWrapper extension skips the classic param-node prelude.

        Regression: the prelude's apply_workflow_metadata crashed on the wrapper
        with 'no attribute SetParameter'. For wrappers the data flows via the UI
        selectors (set by the choice steps) + the button/checkbox handlers, so the
        classic plumbing is both useless and harmful — it is skipped.
        """
        from SlicerAIAgentLib.extension_cli_loader.choice_helpers import _build_choice_prelude
        from SlicerAIAgentLib.extension_cli_loader.workflow_handlers import _WorkflowContext

        def _ctx(metadata):
            return _WorkflowContext(
                "X", "", "X", {}, {"step_id": "cb_step_5"}, {}, {}, "start", set(), metadata
            )

        classic = {
            "parameter_bindings": {"r": {"node_class": ""}},
            "parameter_defaults": {"r": {"value": "1"}},
            "extension_module_name": "X",
            "logic_class_name": "XLogic",
        }
        self.assertIn("apply_workflow_metadata", _build_choice_prelude(_ctx(classic)))

        wrapper = dict(classic)
        wrapper["parameter_node_wrapper"] = {"class_name": "XParameterNode", "fields": {}}
        self.assertEqual(_build_choice_prelude(_ctx(wrapper)), "")
        self.delayDisplay("Wrapper extension skips the classic param-node prelude")

    def test_ApplyWorkflowMetadataWrapperSafe(self):
        """apply_workflow_metadata no-ops on a wrapper, still applies on a classic node."""
        from SlicerAIAgentLib.workflow_state import apply_workflow_metadata

        class Wrapper:  # parameterNodeWrapper: no classic SetParameter
            orbitLm = None

        # Must not raise (defense-in-depth hasattr guard).
        apply_workflow_metadata(Wrapper(), {"r": "v"}, {"r": {"node_class": ""}}, {})

        class Classic:
            def __init__(self):
                self.params = {}

            def GetParameter(self, k):
                return self.params.get(k, "")

            def SetParameter(self, k, v):
                self.params[k] = v

        node = Classic()
        apply_workflow_metadata(node, {"r": "v"}, {"r": {"node_class": ""}}, {})
        self.assertEqual(node.params.get("r"), "v")
        self.delayDisplay("apply_workflow_metadata is wrapper-safe and classic-preserving")

    def test_Integration(self):
        """Integration test of multiple components."""
        from SlicerAIAgent import SlicerAIAgentLogic
        from SlicerAIAgentLib import (
            LLMClient,
            CodeValidator,
            SafeExecutor,
        )

        # Create components
        client = LLMClient()
        logic = SlicerAIAgentLogic()
        validator = CodeValidator()
        executor = SafeExecutor()

        # Test workflow: component initialization and basic operations
        prompt = "load a volume"
        self.assertIsNotNone(logic.skill_path)

        # Test code validation
        test_code = "volume = slicer.util.loadVolume('/path/to/volume.nrrd')"
        validation = validator.validate(test_code)
        self.assertTrue(validation["valid"])

        # Test execution
        result = executor.execute("x = 5 + 10")
        self.assertTrue(result["success"])

        self.delayDisplay("Integration tests passed")
