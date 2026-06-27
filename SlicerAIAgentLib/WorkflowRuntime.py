"""Deterministic runtime for validated generated extension CLI workflows."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .ExtensionCLILoader import (
    clear_workflow_step_completions,
    dispatch_extension_cli_tool,
    find_next_workflow_step,
    get_validated_extensions,
    get_workflow_choices,
    get_workflow_graph,
    mark_workflow_step_completed,
    reset_workflow_session,
    set_all_workflow_repeat_states,
    set_workflow_choices,
    set_workflow_repeat_state,
    truncate_workflow_completions,
)


WAIT_TYPES = {"interactive", "user_interaction", "user_choice"}
COMPLETE_TYPES = {
    "automated",
    "extension_op",
    "slicer_op",
    "interactive_done",
    "choice_made",
    "skipped",
}


@dataclass
class WorkflowCheckpoint:
    """One rewindable point in a workflow run, recorded live as it executes.

    Back/Forward navigation shows/hides workflow nodes by step using
    ``before_node_ids`` / ``created_node_ids`` (never deletes, so display state
    is preserved). The commit (rewind_to_checkpoint) deletes the downstream
    ``created_node_ids``, truncates session/global state to the captured prefix,
    then re-dispatches ``step_id`` with ``action``/``args``. ``sceneview_node_id``
    is unused (kept only so older saved snapshots are cleaned up).
    """

    index: int
    step_id: str
    kind: str                       # automated | choice | interaction | loop_count | loop_decision
    action: str                     # replay user_action: start | proceed | choice_made | skip
    args: Dict[str, Any] = field(default_factory=dict)
    recorded_value: Any = None      # captured choice / placement summary (shown + editable)
    parameter_name: str = ""        # choice parameter name, "" when none
    editable: bool = False          # True for choice / loop_count / loop_decision
    choices: List[Dict[str, Any]] = field(default_factory=list)    # enumerated choices, [] = free-form
    guidance_description: str = ""  # the step's description shown in the panel (Back/Forward)
    guidance_instructions: str = "" # the step's instruction shown in the panel (Back/Forward)
    repeat: Dict[str, Any] = field(default_factory=dict)            # {"repeat_id", "iteration"} or {}
    repeat_states_snapshot: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    completed_prefix: List[str] = field(default_factory=list)       # completed_steps BEFORE this step
    completed_instances_len: int = 0                                # len(completed_instances) BEFORE this step
    choices_prefix: Dict[str, Any] = field(default_factory=dict)    # _workflow_choices[ext] BEFORE this step
    before_node_ids: List[str] = field(default_factory=list)        # scene node IDs BEFORE this step (clean-slate target)
    created_node_ids: List[str] = field(default_factory=list)       # scene nodes this step produced
    layout_before: int = -1                                         # layoutManager.layout BEFORE this step (-1 = unknown)
    sceneview_node_id: Optional[str] = None                         # vtkMRMLSceneViewNode (scene before step)
    summary: str = ""
    timestamp: float = 0.0


@dataclass
class WorkflowSession:
    """State for one generated CLI workflow run."""

    extension_name: str
    tool_name: str
    workflow_id: str
    current_step: Optional[str] = None
    status: str = "running"
    completed_steps: List[str] = field(default_factory=list)
    queued_prompts: List[str] = field(default_factory=list)
    repeat_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    completed_instances: List[Dict[str, Any]] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    started_at: float = field(default_factory=time.time)
    checkpoints: List[WorkflowCheckpoint] = field(default_factory=list)
    active_checkpoint_index: Optional[int] = None
    baseline_node_ids: List[str] = field(default_factory=list)   # scene nodes before step 1 (loaded inputs)
    preview_index: Optional[int] = None          # None = live; i = previewing step i
    live_layout: int = -1                        # layoutManager.layout at the live state (captured on first Back)
    live_sceneview_id: Optional[str] = None      # full snapshot of the live scene (captured on first Back)

    @property
    def active(self) -> bool:
        return self.status in {"running", "waiting_for_user", "waiting_for_choice"}


class WorkflowRuntime:
    """Run generated CLI workflow steps without re-entering the LLM loop."""

    def __init__(self, log_dir: Optional[str] = None):
        self.session: Optional[WorkflowSession] = None
        self.log_dir = log_dir
        # Snapshot captured at run_step (before dispatch), finalized into a
        # WorkflowCheckpoint when the step completes. One step is in flight at
        # a time, so a single pending slot is sufficient.
        self._pending_checkpoint: Optional[Dict[str, Any]] = None

    def has_active_workflow(self) -> bool:
        return bool(self.session and self.session.active)

    def state_for_router(self) -> Dict[str, Any]:
        if not self.session:
            return {"active": False}
        return {
            "active": self.session.active,
            "extension_name": self.session.extension_name,
            "tool_name": self.session.tool_name,
            "workflow_id": self.session.workflow_id,
            "current_step": self.session.current_step,
            "status": self.session.status,
            "completed_steps": list(self.session.completed_steps),
            "repeat_states": copy.deepcopy(self.session.repeat_states),
        }

    def state_for_ui(self, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a compact, user-facing state for workflow progress UI."""
        if not self.session:
            return {"active": False}

        graph = get_workflow_graph(self.session.extension_name) or {}
        steps = graph.get("steps", []) if isinstance(graph, dict) else []
        step_ids = [step.get("step_id") for step in steps if step.get("step_id")]
        step_map = {step.get("step_id"): step for step in steps if step.get("step_id")}
        total_steps = graph.get("step_count") or len(steps)

        source = result if isinstance(result, dict) else self.session.last_result or {}
        next_step = source.get("next_step") or {}
        current_step = (
            source.get("step_id")
            or self.session.current_step
            or next_step.get("step_id")
        )
        current_meta = step_map.get(current_step, {})
        current_index = 0
        if current_step in step_ids:
            current_index = step_ids.index(current_step) + 1
        elif self.session.status == "completed" and total_steps:
            current_index = int(total_steps)

        completed_count = len(set(self.session.completed_steps))
        if self.session.status == "completed" and total_steps:
            completed_count = int(total_steps)

        result_type = source.get("type", "")
        status = self._ui_status_label(self.session.status, result_type)
        ui_guidance = self._ui_guidance_from_result(source, current_meta)
        guidance_title = str(ui_guidance.get("title", "") or "").strip()
        guidance_instruction = str(ui_guidance.get("instruction", "") or "").strip()
        is_repeat_decision = result_type == "user_choice" and bool(source.get("repeat_decision"))
        if is_repeat_decision:
            # A loop continue/exit decision: surface the decision's own question
            # and instruction, NOT the underlying step's guidance. Otherwise the
            # panel shows e.g. "Regenerate reconstruction / Repeat steps 27-30 if
            # needed" — making the Yes/No buttons read as "Yes = repeat" when Yes
            # actually accepts and exits the loop.
            description = (
                source.get("question")
                or guidance_title
                or current_meta.get("description")
                or ""
            )
            instructions = source.get("instruction") or guidance_instruction
        elif result_type == "user_choice":
            description = (
                guidance_title
                or source.get("question")
                or source.get("explanation")
                or current_meta.get("description")
                or ""
            )
            instructions = guidance_instruction or self._instructions_from_result(source)
        else:
            description = (
                guidance_title
                or source.get("explanation")
                or source.get("instruction")
                or current_meta.get("description")
                or ""
            )
            instructions = guidance_instruction or self._instructions_from_result(source)
        choices = self._choices_from_result(source, current_meta)
        # A node-pick user_choice must drive the scene node tree, not literal
        # cookbook-label buttons. Dropping stray literal choices flips
        # needs_choice_input True (below) so the panel reaches
        # _renderWorkflowNodeTree. Loop (repeat) decisions keep their synthetic
        # Yes/No buttons.
        if (result_type == "user_choice"
                and not is_repeat_decision
                and self._is_node_selection_step(current_meta)):
            choices = []
        is_optional = bool(source.get("is_optional") or current_meta.get("is_optional"))
        choice_info = current_meta.get("choice_info", {}) if isinstance(current_meta, dict) else {}
        default_value = source.get("default_value")
        if default_value is None:
            default_value = choice_info.get("default_value")
        parameter_name = source.get("parameter_name") or choice_info.get("parameter_name") or ""
        repeat_progress = source.get("repeat_progress") or {}
        # Node-selection steps bind a parameter to a node class; surface it (and
        # its keywords) so the panel can offer a dropdown of matching scene
        # nodes instead of a free-text box.
        binding = source.get("binding") or self._node_binding_for_param(parameter_name)
        node_class = source.get("node_class") or binding.get("node_class", "")
        if not node_class and result_type == "user_choice":
            # No binding was inferred (e.g. a subject-hierarchy selector or a
            # parameterNodeWrapper-bound field): fall back to the node class the
            # step graph records so the panel still offers a node dropdown.
            node_class = self._node_class_from_step_meta(current_meta)
        node_keywords = binding.get("keywords", []) or []
        # Clinically-informed instructions override the terse ui_guidance when
        # present: title -> description, detailed -> the primary instruction text
        # (shown by default), simple -> the terse "Show brief" body. The widget
        # picks which to show. Re-looked up live so manual edits show at once.
        instr = self._step_instructions_for(current_step)
        if instr.get("title"):
            description = instr["title"]
        if instr.get("simple"):
            instructions = instr["simple"]
        instructions_detailed = instr.get("detailed", "")
        active = self.session.active or self.session.status in {"completed", "cancelled"}

        return {
            "active": active,
            "extension_name": self.session.extension_name,
            "workflow_title": self._display_name(self.session.extension_name),
            "tool_name": self.session.tool_name,
            "workflow_id": self.session.workflow_id,
            "current_step": current_step,
            "current_index": current_index,
            "completed_steps": completed_count,
            "total_steps": int(total_steps or 0),
            "status": status,
            "raw_status": self.session.status,
            "result_type": result_type,
            "description": description,
            "instructions": instructions,
            "instructions_detailed": instructions_detailed,
            "choices": choices,
            "default_value": default_value,
            "parameter_name": parameter_name,
            "node_class": node_class,
            "node_keywords": node_keywords,
            "choice_label": ui_guidance.get("choice_label", ""),
            "input_label": ui_guidance.get("input_label", ""),
            "done_label": ui_guidance.get("done_label", "Done") or "Done",
            "object_label": ui_guidance.get("object_label", ""),
            "repeat_progress": repeat_progress,
            "needs_choice_input": result_type == "user_choice" and not choices,
            "can_done": self.session.status == "waiting_for_user",
            "can_skip": self.session.active and is_optional,
            "can_cancel": self.session.active,
            "timeline": self._timeline_for_ui(),
            "active_checkpoint_index": self.session.active_checkpoint_index,
            "can_replay": bool(self.session.checkpoints),
            "replay_can_back": bool(self.session.checkpoints),
            "replay_can_forward": self.session.preview_index is not None,
            "replay_can_action": self.session.preview_index is not None,
            "replay_previewing": self.session.preview_index is not None,
        }

    def start_from_result(self, result: Dict[str, Any]) -> Optional[WorkflowSession]:
        """Create a session from a generated CLI tool result if needed."""
        if not isinstance(result, dict) or not result.get("tool"):
            return self.session
        tool_name = result.get("tool")
        ext_name = self._extension_name_for_tool(tool_name) or tool_name
        if not ext_name:
            return self.session
        if self.session and self.session.extension_name == ext_name and self.session.active:
            if result.get("step_id"):
                self.session.current_step = result.get("step_id")
            self._apply_pre_execution_state(result)
            return self.session
        workflow_id = f"{ext_name}_{int(time.time() * 1000)}"
        reset_workflow_session(ext_name)
        # Drop any prior session's replay snapshots before starting fresh.
        self._delete_all_replay_sceneviews()
        self._pending_checkpoint = None
        self.session = WorkflowSession(
            extension_name=ext_name,
            tool_name=tool_name,
            workflow_id=workflow_id,
            current_step=result.get("step_id"),
        )
        self._apply_pre_execution_state(result)
        # The first step is dispatched via the LLM tool path (not run_step), so
        # capture its before-scene snapshot here so it is rewindable too. A
        # later run_step for the same step (a user_choice answer / interactive
        # Done) refreshes the args while keeping this before-scene.
        if result.get("step_id"):
            self._begin_pending_checkpoint(
                result.get("step_id"), {"user_action": "start"}
            )
        self._write_event("workflow_started", {"initial_result": self._compact_result(result)})
        return self.session

    def start_for_extension(self, extension_name: str, tool_name: Optional[str] = None) -> WorkflowSession:
        """Start a session explicitly for a validated generated extension."""
        ext_data = get_validated_extensions().get(extension_name)
        if not ext_data:
            raise ValueError(f"Validated extension CLI not found: {extension_name}")
        resolved_tool = tool_name or self._tool_name_for_extension(ext_data) or extension_name
        workflow_id = f"{extension_name}_{int(time.time() * 1000)}"
        reset_workflow_session(extension_name)
        self._delete_all_replay_sceneviews()
        self._pending_checkpoint = None
        graph = get_workflow_graph(extension_name)
        first_step = find_next_workflow_step(extension_name, set())
        self.session = WorkflowSession(
            extension_name=extension_name,
            tool_name=resolved_tool,
            workflow_id=workflow_id,
            current_step=(first_step or {}).get("step_id"),
        )
        self._write_event("workflow_started", {"workflow": graph.get("name", extension_name)})
        return self.session

    def run_step(
        self,
        step_id: Optional[str] = None,
        action: str = "start",
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch one generated CLI step and return the generated result."""
        if not self.session:
            return {"type": "error", "error": "No active generated CLI workflow"}

        target_step = step_id or self.session.current_step
        if not target_step and action != "cancel":
            return {"type": "error", "error": "No current workflow step"}

        call_args = dict(args or {})
        call_args.setdefault("workflow_step", target_step)
        call_args.setdefault("user_action", action)
        call_args.setdefault("_workflow_id", self.session.workflow_id)

        self.session.current_step = target_step
        self.session.status = "running"
        # Record a replay checkpoint snapshot of the scene/state BEFORE this
        # step runs. Idempotent across an interactive step's start->proceed
        # pair (keeps the before-placement scene). Finalized in _mark_completed
        # (or for a loop decision, just below).
        self._begin_pending_checkpoint(target_step, call_args)
        repeat_decision = self._handle_pending_repeat_decision(
            target_step, action, call_args
        )
        if repeat_decision is not None:
            self._finalize_loop_decision_checkpoint(target_step, call_args, repeat_decision)
            self.session.last_result = repeat_decision
            self._apply_pre_execution_state(repeat_decision)
            next_step = repeat_decision.get("next_step")
            if next_step:
                self.session.current_step = next_step.get("step_id")
                self.session.status = "running"
            else:
                self.session.current_step = None
                self.session.status = "completed"
                repeat_decision["workflow_completed"] = True
            self._write_event(
                "repeat_decision",
                {"args": call_args, "result": self._compact_result(repeat_decision)},
            )
            return repeat_decision

        result = dispatch_extension_cli_tool(self.session.tool_name, call_args)
        if result is None:
            result = {"type": "error", "error": f"Unknown workflow tool: {self.session.tool_name}"}
        elif isinstance(result, dict):
            result = copy.deepcopy(result)
            result.setdefault("tool", self.session.extension_name)
            result.setdefault("step_id", target_step)

        self.session.last_result = result
        self._apply_pre_execution_state(result)
        self._write_event("step_dispatched", {"args": call_args, "result": self._compact_result(result)})
        return result

    def handle_execution_result(
        self,
        step_result: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update workflow state after generated code execution completes."""
        if not self.session:
            return step_result
        result = dict(step_result or {})
        result_type = result.get("type")
        step_id = result.get("step_id") or self.session.current_step
        success = bool(execution_result.get("success")) and not execution_result.get("timed_out", False)

        if not success:
            self.session.status = "running"
            self._write_event("step_execution_failed", {
                "step_id": step_id,
                "error": execution_result.get("error"),
            })
            return result

        if result_type in {"interactive", "user_interaction"}:
            self.session.status = "waiting_for_user"
            self.session.current_step = step_id
            self._write_event("step_waiting_for_user", {"step_id": step_id})
            return result

        if result_type == "user_choice":
            self.session.status = "waiting_for_choice"
            self.session.current_step = step_id
            self._write_event("step_waiting_for_choice", {"step_id": step_id})
            return result

        if result.get("repeat_decision"):
            # A loop continue/exit decision result is already fully transitioned
            # by _handle_pending_repeat_decision in run_step (body cleared +
            # iteration advanced for "repeat", or exit target chosen). Re-marking
            # its terminal step complete here would undo _clear_repeat_body and
            # make the loop skip its own body on the next pass. Just advance to
            # the already-computed next_step.
            next_step = result.get("next_step")
            if next_step:
                self.session.current_step = next_step.get("step_id")
                self.session.status = "running"
            else:
                self.session.current_step = None
                self.session.status = "completed"
                result["workflow_completed"] = True
            self._write_event("step_execution_completed", {
                "step_id": step_id,
                "result_type": result_type,
                "next_step": next_step,
                "workflow_completed": result.get("workflow_completed", False),
            })
            return result

        if result_type in COMPLETE_TYPES and step_id:
            self._mark_completed(step_id)
            repeat_transition = self._repeat_transition_after_completion(
                step_id, result
            )
            if repeat_transition:
                result.update(repeat_transition)
                result_type = result.get("type")
                if result_type == "user_choice":
                    self.session.last_result = result
                    self.session.status = "waiting_for_choice"
                    self.session.current_step = step_id
                    return result
            elif not result.get("next_step"):
                result["next_step"] = self._next_step()

        next_step = result.get("next_step")
        if next_step:
            self.session.current_step = next_step.get("step_id")
            self.session.status = "running"
        elif result_type in COMPLETE_TYPES:
            self.session.current_step = None
            self.session.status = "completed"
            result["workflow_completed"] = True

        self._write_event("step_execution_completed", {
            "step_id": step_id,
            "result_type": result_type,
            "next_step": next_step,
            "workflow_completed": result.get("workflow_completed", False),
        })
        return result

    def queue_traditional_prompt(self, prompt: str) -> int:
        if not self.session:
            return 0
        self.session.queued_prompts.append(prompt)
        self._write_event("traditional_prompt_queued", {"prompt": prompt})
        return len(self.session.queued_prompts)

    def pop_queued_prompts(self) -> List[str]:
        if not self.session:
            return []
        queued = list(self.session.queued_prompts)
        self.session.queued_prompts.clear()
        return queued

    def get_prompt_fragment(self) -> str:
        if not self.session or not self.session.active:
            return ""
        completed = ", ".join(self.session.completed_steps) or "none"
        return (
            f"### Active Generated CLI Workflow: {self.session.extension_name}\n"
            f"- Workflow ID: {self.session.workflow_id}\n"
            f"- Status: {self.session.status}\n"
            f"- Current step: {self.session.current_step}\n"
            f"- Completed steps: {completed}\n"
        )

    def _apply_pre_execution_state(self, result: Dict[str, Any]) -> None:
        if not self.session or not isinstance(result, dict):
            return
        result_type = result.get("type")
        if result_type == "cancelled":
            self.session.status = "cancelled"
            self.session.current_step = None
        elif result_type == "user_choice":
            self.session.status = "waiting_for_choice"
            self.session.current_step = result.get("step_id")
        elif result_type in {"interactive", "user_interaction"}:
            self.session.status = "running"
            self.session.current_step = result.get("step_id")
        elif result_type == "skipped" and result.get("step_id"):
            self._mark_completed(result["step_id"])
            next_step = result.get("next_step") or self._next_step()
            if next_step:
                result["next_step"] = next_step
                self.session.current_step = next_step.get("step_id")
                self.session.status = "running"
            else:
                self.session.current_step = None
                self.session.status = "completed"

    def _mark_completed(self, step_id: str) -> None:
        if not self.session or not step_id:
            return
        if step_id not in self.session.completed_steps:
            self.session.completed_steps.append(step_id)
            self.session.completed_instances.append({
                "step_id": step_id,
                "repeat": self._repeat_instance_for_step(step_id),
            })
        mark_workflow_step_completed(self.session.extension_name, step_id)
        self._finalize_checkpoint(step_id)

    def _next_step(self) -> Optional[Dict[str, Any]]:
        if not self.session:
            return None
        return find_next_workflow_step(
            self.session.extension_name,
            set(self.session.completed_steps),
        )

    def _repeat_blocks(self) -> List[Dict[str, Any]]:
        if not self.session:
            return []
        graph = get_workflow_graph(self.session.extension_name) or {}
        return [
            block for block in graph.get("repeat_blocks", []) or []
            if isinstance(block, dict) and block.get("repeat_id")
        ]

    def _repeat_instance_for_step(self, step_id: str) -> Dict[str, Any]:
        if not self.session:
            return {}
        for block in self._repeat_blocks():
            if step_id in (block.get("body_steps") or []):
                state = self.session.repeat_states.get(block["repeat_id"], {})
                return {
                    "repeat_id": block["repeat_id"],
                    "iteration": int(state.get("iteration", 1) or 1),
                }
        return {}

    def _sync_repeat_state(self, repeat_id: str) -> None:
        if not self.session:
            return
        state = self.session.repeat_states.get(repeat_id, {})
        set_workflow_repeat_state(
            self.session.extension_name,
            repeat_id,
            state,
        )

    def _clear_repeat_body(self, block: Dict[str, Any]) -> None:
        if not self.session:
            return
        body_steps = list(block.get("body_steps") or [])
        body_set = set(body_steps)
        self.session.completed_steps = [
            step_id for step_id in self.session.completed_steps
            if step_id not in body_set
        ]
        clear_workflow_step_completions(self.session.extension_name, body_steps)

    @staticmethod
    def _normalize_control_value(value: Any) -> Any:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
            try:
                return int(lowered)
            except ValueError:
                return value.strip()
        return value

    def _repeat_transition_after_completion(
        self,
        step_id: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a control-flow transition after one atomic step succeeds."""
        if not self.session:
            return {}
        if result.get("repeat_decision"):
            return {}
        for block in self._repeat_blocks():
            repeat_id = block["repeat_id"]
            controller = block.get("controller", {}) or {}
            kind = controller.get("kind", "")
            state = self.session.repeat_states.setdefault(
                repeat_id, {"iteration": 1}
            )

            if kind == "count" and step_id == controller.get("source_step"):
                raw_target = self._normalize_control_value(result.get("choice_value"))
                try:
                    target = int(raw_target)
                except (TypeError, ValueError):
                    target = 0
                state.update({"iteration": 1, "target": max(0, target)})
                self._sync_repeat_state(repeat_id)
                if target <= 0:
                    for body_step in block.get("body_steps", []) or []:
                        self._mark_completed(body_step)
                    next_step = self._step_summary(block.get("exit_step"))
                    return {
                        "next_step": next_step,
                        "repeat_progress": {
                            "repeat_id": repeat_id,
                            "current": 0,
                            "completed": 0,
                            "total": 0,
                        },
                    }

            if step_id != block.get("terminal_step"):
                continue

            iteration = int(state.get("iteration", 1) or 1)
            max_iterations = int(block.get("max_iterations", 20) or 20)
            if kind == "count":
                target = int(state.get("target", 1) or 1)
                if iteration < min(target, max_iterations):
                    self._clear_repeat_body(block)
                    state["iteration"] = iteration + 1
                    self._sync_repeat_state(repeat_id)
                    return {
                        "next_step": self._step_summary(block.get("entry_step")),
                        "repeat_progress": {
                            "repeat_id": repeat_id,
                            "current": iteration + 1,
                            "completed": iteration,
                            "total": target,
                        },
                    }
                return {
                    "next_step": self._step_summary(block.get("exit_step")),
                    "repeat_limit_reached": target > max_iterations,
                    "repeat_progress": {
                        "repeat_id": repeat_id,
                        "current": iteration,
                        "completed": iteration,
                        "total": target,
                    },
                }

            if kind in {"until_choice", "while_choice"}:
                state["waiting_for_decision"] = True
                self._sync_repeat_state(repeat_id)
                prompt = controller.get("prompt") or "Continue the repeated workflow?"
                choices = [
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ]
                # Derive which answer exits vs repeats from the controller, so the
                # instruction is correct for both until_choice (exit on the exit
                # value) and while_choice. This makes the Yes/No meaning explicit
                # instead of leaving the user to guess.
                exit_value = self._normalize_control_value(controller.get("exit_value"))
                exit_label = next(
                    (c["label"] for c in choices
                     if self._normalize_control_value(c["value"]) == exit_value),
                    "Yes",
                )
                loop_label = next(
                    (c["label"] for c in choices
                     if self._normalize_control_value(c["value"]) != exit_value),
                    "No",
                )
                decision_instruction = (
                    f'Choose "{loop_label}" to repeat this section and refine the '
                    f'result; choose "{exit_label}" to accept it and continue.'
                )
                return {
                    "type": "user_choice",
                    "question": prompt,
                    "instruction": decision_instruction,
                    "choices": choices,
                    "parameter_name": f"repeat_decision:{repeat_id}",
                    "repeat_decision": repeat_id,
                    "repeat_progress": {
                        "repeat_id": repeat_id,
                        "current": iteration,
                        "completed": iteration,
                        "total": 0,
                    },
                    "next_step": None,
                }
        return {}

    def _handle_pending_repeat_decision(
        self,
        step_id: str,
        action: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.session or action != "choice_made":
            return None
        for block in self._repeat_blocks():
            repeat_id = block["repeat_id"]
            state = self.session.repeat_states.get(repeat_id, {})
            if not state.get("waiting_for_decision"):
                continue
            if step_id != block.get("terminal_step"):
                continue
            controller = block.get("controller", {}) or {}
            choice = self._normalize_control_value(args.get("choice_value"))
            exit_value = self._normalize_control_value(controller.get("exit_value"))
            should_exit = choice == exit_value
            iteration = int(state.get("iteration", 1) or 1)
            max_iterations = int(block.get("max_iterations", 20) or 20)
            state["waiting_for_decision"] = False
            limit_reached = not should_exit and iteration >= max_iterations
            if should_exit or limit_reached:
                next_step = self._step_summary(block.get("exit_step"))
            else:
                self._clear_repeat_body(block)
                state["iteration"] = iteration + 1
                next_step = self._step_summary(block.get("entry_step"))
            self._sync_repeat_state(repeat_id)
            return {
                "tool": self.session.tool_name,
                "type": "choice_made",
                "step_id": step_id,
                "choice_value": choice,
                "repeat_decision": repeat_id,
                "repeat_limit_reached": limit_reached,
                "next_step": next_step,
                "repeat_progress": {
                    "repeat_id": repeat_id,
                    "current": int(state.get("iteration", iteration) or iteration),
                    "completed": iteration,
                    "total": 0,
                },
            }
        return None

    def _step_summary(self, step_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self.session or not step_id:
            return None
        graph = get_workflow_graph(self.session.extension_name) or {}
        for step in graph.get("steps", []) or []:
            if step.get("step_id") == step_id:
                return {
                    "step_id": step_id,
                    "operation_type": (
                        step.get("operation_type")
                        or step.get("op_type")
                        or step.get("step_type", "extension_op")
                    ),
                    "step_type": step.get("step_type", "extension_op"),
                    "description": step.get("description", ""),
                    "is_optional": bool(step.get("is_optional", False)),
                    "ui_guidance": step.get("ui_guidance", {}),
                }
        return None

    @staticmethod
    def _instructions_from_result(result: Dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return ""
        if result.get("type") == "user_choice":
            return ""
        interaction = result.get("interaction") or {}
        return (
            interaction.get("placement_instructions")
            or result.get("interaction_instructions")
            or ""
        )

    @staticmethod
    def _ui_guidance_from_result(result: Dict[str, Any], step_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Return generated UI guidance from result or workflow metadata."""
        guidance = {}
        if isinstance(step_meta, dict) and isinstance(step_meta.get("ui_guidance"), dict):
            guidance.update(step_meta["ui_guidance"])
        if isinstance(result, dict) and isinstance(result.get("ui_guidance"), dict):
            guidance.update(result["ui_guidance"])
        if isinstance(result, dict):
            interaction = result.get("interaction") or {}
            if isinstance(interaction, dict) and isinstance(interaction.get("ui_guidance"), dict):
                guidance.update(interaction["ui_guidance"])
        return guidance

    @staticmethod
    def _choices_from_result(result: Dict[str, Any], step_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_choices = []
        if isinstance(result, dict):
            source_choices = result.get("choices") or []
        if not source_choices and isinstance(step_meta, dict):
            source_choices = (step_meta.get("choice_info") or {}).get("choices") or []
        if not source_choices and isinstance(result, dict):
            source_choices = WorkflowRuntime._choices_from_instruction(result.get("instruction", ""))
        choices = []
        for choice in source_choices:
            if not isinstance(choice, dict):
                continue
            label = str(choice.get("label", choice.get("value", ""))).strip()
            value = str(choice.get("value", label)).strip()
            if label or value:
                choices.append({"label": label or value, "value": value or label})
        return choices

    @staticmethod
    def _choices_from_instruction(instruction: str) -> List[Dict[str, Any]]:
        """Best-effort fallback for compact user_choice results."""
        choices = []
        for line in str(instruction or "").splitlines():
            match = re.match(r"\s*\d+[.)]\s+(.+?)\s*$", line)
            if not match:
                continue
            label = match.group(1).strip()
            if label:
                lowered = label.lower()
                if lowered == "yes":
                    value = "true"
                elif lowered == "no":
                    value = "false"
                else:
                    value = label
                choices.append({"label": label, "value": value})
        return choices

    @staticmethod
    def _ui_status_label(status: str, result_type: str = "") -> str:
        if status == "waiting_for_user":
            return "Waiting for your interaction"
        if status == "waiting_for_choice":
            return "Waiting for your choice"
        if status == "completed":
            return "Completed"
        if status == "cancelled":
            return "Cancelled"
        if result_type == "user_choice":
            return "Waiting for your choice"
        return "Running"

    @staticmethod
    def _display_name(name: str) -> str:
        text = str(name or "").replace("_", " ").replace("-", " ").strip()
        if not text:
            return "Workflow"
        return re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip()

    @staticmethod
    def _tool_name_for_extension(ext_data: Dict[str, Any]) -> Optional[str]:
        for schema in ext_data.get("schemas", []):
            func = schema.get("function", {})
            if func.get("name"):
                return func["name"]
        return None

    @classmethod
    def _extension_name_for_tool(cls, tool_name: str) -> Optional[str]:
        for ext_name, ext_data in get_validated_extensions().items():
            if cls._tool_name_for_extension(ext_data) == tool_name:
                return ext_name
        return None

    @staticmethod
    def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
        compact = {}
        for key in (
            "tool",
            "type",
            "step_id",
            "instruction",
            "explanation",
            "interaction_instructions",
            "question",
            "choices",
            "default_value",
            "ui_guidance",
            "repeat_progress",
            "next_step",
            "workflow_completed",
            "error",
        ):
            if key in result:
                compact[key] = result[key]
        if result.get("code"):
            compact["code_chars"] = len(result.get("code") or "")
        return compact

    # =====================================================================
    # Replay timeline — live checkpoint recording, rewind, scene snapshots
    # =====================================================================

    def _begin_pending_checkpoint(self, step_id: str, call_args: Dict[str, Any]) -> None:
        """Snapshot the scene + state BEFORE ``step_id`` dispatches.

        Idempotent across an interactive step's start->proceed pair: the second
        call keeps the before-placement scene captured at start and only
        refreshes the recorded args.
        """
        if not self.session or not step_id:
            return
        if self._pending_checkpoint and self._pending_checkpoint.get("step_id") == step_id:
            self._pending_checkpoint["call_args"] = dict(call_args or {})
            return
        # A different step's pending was never finalized (prior step failed or
        # was retried by self-correction); drop it first.
        self._discard_pending_checkpoint()
        self._pending_checkpoint = {
            "step_id": step_id,
            "call_args": dict(call_args or {}),
            "completed_prefix": list(self.session.completed_steps),
            "completed_instances_len": len(self.session.completed_instances),
            "repeat_states_snapshot": copy.deepcopy(self.session.repeat_states),
            "choices_prefix": get_workflow_choices(self.session.extension_name),
            "node_ids_before": self._scene_node_ids(),
            "layout_before": self._current_layout(),
            # Full before-step snapshot; restored by property-copy (never
            # delete) so ALL node state recovers without losing display nodes.
            "sceneview_node_id": self._capture_sceneview(step_id),
        }

    def _discard_pending_checkpoint(self) -> None:
        """Drop an un-finalized pending checkpoint and its orphan snapshot."""
        pending = self._pending_checkpoint
        self._pending_checkpoint = None
        if pending:
            self._delete_sceneview(pending.get("sceneview_node_id"))

    def _finalize_checkpoint(self, step_id: str) -> None:
        """Promote the pending snapshot for ``step_id`` into a WorkflowCheckpoint."""
        pending = self._pending_checkpoint
        if not pending or pending.get("step_id") != step_id:
            return
        self._pending_checkpoint = None
        self._record_checkpoint(pending)

    def _finalize_loop_decision_checkpoint(
        self,
        step_id: str,
        call_args: Dict[str, Any],
        repeat_decision: Dict[str, Any],
    ) -> None:
        """Record a checkpoint for a loop continue/exit decision (run_step path)."""
        if not self.session:
            return
        pending = self._pending_checkpoint
        if not pending or pending.get("step_id") != step_id:
            pending = {
                "step_id": step_id,
                "call_args": dict(call_args or {}),
                "completed_prefix": list(self.session.completed_steps),
                "completed_instances_len": len(self.session.completed_instances),
                "repeat_states_snapshot": copy.deepcopy(self.session.repeat_states),
                "choices_prefix": get_workflow_choices(self.session.extension_name),
                "node_ids_before": self._scene_node_ids(),
                "layout_before": self._current_layout(),
                "sceneview_node_id": self._capture_sceneview(step_id),
            }
        self._pending_checkpoint = None
        progress = repeat_decision.get("repeat_progress") or {}
        repeat = {
            "repeat_id": progress.get("repeat_id", "") or "",
            "iteration": progress.get("current") or progress.get("completed") or 0,
        }
        self._record_checkpoint(pending, force_kind="loop_decision", repeat=repeat)

    def _record_checkpoint(
        self,
        pending: Dict[str, Any],
        force_kind: Optional[str] = None,
        repeat: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.session:
            return
        step_id = pending["step_id"]
        meta = self._step_meta(step_id)
        op = (
            meta.get("operation_type")
            or meta.get("op_type")
            or meta.get("step_type")
            or "extension_op"
        )
        call_args = pending.get("call_args") or {}
        choice_value = call_args.get("choice_value")
        repeat_info = repeat if repeat is not None else self._repeat_instance_for_step(step_id)

        choices: List[Dict[str, Any]] = []
        if force_kind == "loop_decision":
            kind = "loop_decision"
            action = "choice_made"
            args = {"choice_value": choice_value}
            editable = True
            recorded_value = choice_value
            parameter_name = f"repeat_decision:{(repeat_info or {}).get('repeat_id', '')}"
            choices = [{"label": "Yes", "value": True}, {"label": "No", "value": False}]
        elif op == "user_choice" or choice_value is not None:
            kind = "loop_count" if self._is_loop_count_source(step_id) else "choice"
            action = "choice_made"
            args = {"choice_value": choice_value}
            editable = True
            recorded_value = choice_value
            choice_info = meta.get("choice_info") or {}
            parameter_name = choice_info.get("parameter_name", "") or ""
            choices = self._normalize_choices(choice_info.get("choices") or [])
        elif op == "user_interaction" or meta.get("step_type") in (
            "interactive", "mixed", "user_interaction",
        ):
            kind = "interaction"
            action = "start"
            args = {}
            editable = False
            recorded_value = self._interaction_summary(step_id)
            parameter_name = ""
        else:
            kind = "automated"
            action = "start"
            args = {}
            editable = False
            recorded_value = None
            parameter_name = ""

        before = pending.get("node_ids_before") or set()
        created = [nid for nid in self._scene_node_ids() if nid not in before]
        index = len(self.session.checkpoints)
        if index == 0 and not self.session.baseline_node_ids:
            # Scene before the first step = loaded inputs (e.g. the CT volume);
            # used by the UI to avoid warning about deleting them on rewind.
            self.session.baseline_node_ids = sorted(before)
        summary = self._checkpoint_summary(meta, kind, recorded_value, repeat_info)
        guidance = self._ui_guidance_from_result(self.session.last_result or {}, meta)
        guidance_description = (
            str(guidance.get("title", "") or "").strip()
            or str(meta.get("description", "") or "").strip()
        )
        guidance_instructions = (
            str(guidance.get("instruction", "") or "").strip()
            or self._instructions_from_result(self.session.last_result or {})
        )
        checkpoint = WorkflowCheckpoint(
            index=index,
            step_id=step_id,
            kind=kind,
            action=action,
            args=args,
            recorded_value=recorded_value,
            parameter_name=parameter_name,
            editable=editable,
            choices=choices,
            guidance_description=guidance_description,
            guidance_instructions=guidance_instructions,
            repeat=dict(repeat_info or {}),
            repeat_states_snapshot=pending.get("repeat_states_snapshot") or {},
            completed_prefix=list(pending.get("completed_prefix") or []),
            completed_instances_len=int(pending.get("completed_instances_len") or 0),
            choices_prefix=dict(pending.get("choices_prefix") or {}),
            before_node_ids=sorted(before),
            created_node_ids=created,
            layout_before=int(pending.get("layout_before", -1)),
            sceneview_node_id=pending.get("sceneview_node_id"),
            summary=summary,
            timestamp=round(time.time(), 3),
        )
        self.session.checkpoints.append(checkpoint)
        self.session.active_checkpoint_index = index
        self._write_event("checkpoint_recorded", {
            "index": index,
            "step_id": step_id,
            "kind": kind,
            "editable": editable,
            "repeat": repeat_info,
        })

    def _timeline_for_ui(self) -> List[Dict[str, Any]]:
        if not self.session:
            return []
        active = self.session.active_checkpoint_index
        items = []
        for cp in self.session.checkpoints:
            cp_repeat = cp.repeat or {}
            items.append({
                "index": cp.index,
                "step_id": cp.step_id,
                "kind": cp.kind,
                "summary": cp.summary,
                "repeat_id": cp_repeat.get("repeat_id", "") or "",
                "iteration": int(cp_repeat.get("iteration", 0) or 0),
                "is_current": cp.index == active,
                "recorded_value": cp.recorded_value,
                "parameter_name": cp.parameter_name,
                "editable": bool(cp.editable),
                "choices": list(cp.choices or []),
                "can_restore": bool(cp.sceneview_node_id),
            })
        return items

    # ---- Back/Forward navigation: full property recovery, never delete ----
    #
    # "Recover" means recover EVERYTHING at that step, not just node existence:
    # baseline-node display (e.g. the loaded segmentation's 3D surface/contours),
    # the layout, slice/view state, transforms, colors. We capture a full
    # vtkMRMLSceneViewNode per step but RESTORE it by COPYING the stored node
    # properties onto the matching live nodes (by ID) — never deleting or
    # re-creating a node, so display nodes are never lost. Nodes that did not
    # exist yet at that step are simply hidden. Slicer's own RestoreScene can't
    # be used here: removeNodes=False aborts when later nodes are present, and
    # removeNodes=True deletes+recreates (which is what drops display nodes).

    def navigate_back(self) -> Dict[str, Any]:
        """Step one step back: recover that step's full scene state. Non-destructive."""
        if not self.session or not self.session.checkpoints:
            return self._preview_ui_state()
        if self.session.preview_index is None:
            # Leaving the live state: snapshot it (scene + layout) so Forward can
            # return to it exactly.
            self.session.live_layout = self._current_layout()
            self.session.live_sceneview_id = self._capture_sceneview("live")
            self.session.preview_index = len(self.session.checkpoints) - 1
        elif self.session.preview_index > 0:
            self.session.preview_index -= 1
        else:
            return self._preview_ui_state()  # already at the first checkpoint
        self._restore_to_view(self.session.preview_index)
        self._write_event("replay_navigate", {
            "direction": "back", "preview_index": self.session.preview_index,
        })
        return self._preview_ui_state()

    def navigate_forward(self) -> Dict[str, Any]:
        """Step one step forward, or back to the live state. Non-destructive."""
        if not self.session or self.session.preview_index is None:
            return self._preview_ui_state()  # already live
        last = len(self.session.checkpoints) - 1
        if self.session.preview_index < last:
            self.session.preview_index += 1
            self._restore_to_view(self.session.preview_index)
        else:
            # Past the last checkpoint: recover the live state and exit preview.
            self._restore_to_view(None)
            self._delete_sceneview(self.session.live_sceneview_id)
            self.session.live_sceneview_id = None
            self.session.preview_index = None
        self._write_event("replay_navigate", {
            "direction": "forward", "preview_index": self.session.preview_index,
        })
        return self._preview_ui_state()

    def _restore_to_view(self, index: Optional[int]) -> None:
        """Recover the full scene state at step ``index`` (None = live)."""
        if not self.session:
            return
        if index is None:
            self._restore_scene_properties(self.session.live_sceneview_id)
            self._set_layout(self.session.live_layout)
            return
        cp = self.session.checkpoints[index]
        # 1. Copy every stored node's properties onto its live counterpart —
        #    recovers baseline-node display, transforms, slice/view state, etc.
        self._restore_scene_properties(cp.sceneview_node_id)
        # 2. Hide nodes that did not exist yet at this step (the snapshot can't
        #    set their visibility because it doesn't contain them).
        self._hide_nodes_after(index)
        # 3. Restore the layout ("overlay").
        self._set_layout(cp.layout_before)

    def _all_workflow_node_ids(self) -> set:
        """All MRML node IDs the workflow has created across all checkpoints."""
        ids = set()
        if self.session:
            for cp in self.session.checkpoints:
                ids.update(cp.created_node_ids or [])
        return ids

    def _restore_scene_properties(self, sceneview_id: Optional[str]) -> None:
        """Copy stored node properties onto matching live nodes (never delete/add).

        This recovers ALL node state (display visibility/color/view assignments,
        transforms, slice/view nodes, layout node) to the snapshot without
        removing or re-creating any node, so no display node is ever lost.
        """
        if not sceneview_id:
            return
        try:
            import slicer
        except Exception:
            return
        try:
            sv = slicer.mrmlScene.GetNodeByID(sceneview_id)
            if sv is None:
                return
            stored = sv.GetStoredScene()
            if stored is None:
                return
            for i in range(stored.GetNumberOfNodes()):
                snode = stored.GetNthNode(i)
                if snode is None or not snode.GetID():
                    continue
                if snode.IsA("vtkMRMLSceneViewNode"):
                    continue
                live = slicer.mrmlScene.GetNodeByID(snode.GetID())
                if live is None or live is snode:
                    continue
                try:
                    live.Copy(snode)
                except Exception:
                    pass
        except Exception:
            logger.debug("Scene property restore failed", exc_info=True)

    def _hide_nodes_after(self, index: int) -> None:
        """Hide workflow nodes created at/after step ``index`` (not in its snapshot)."""
        try:
            import slicer
        except Exception:
            return
        if not self.session:
            return
        keep = set(self.session.checkpoints[index].before_node_ids or [])
        try:
            for node_id in self._all_workflow_node_ids():
                if node_id in keep:
                    continue  # existed at this step; its state came from the snapshot
                node = slicer.mrmlScene.GetNodeByID(node_id)
                if node is not None:
                    self._set_node_visibility(node, False)
        except Exception:
            logger.debug("Hide-after toggle failed", exc_info=True)

    @staticmethod
    def _set_node_visibility(node, visible: bool) -> None:
        """Toggle a node's display visibility; create a display node if missing."""
        try:
            if not hasattr(node, "GetNumberOfDisplayNodes"):
                return
            count = node.GetNumberOfDisplayNodes()
            if count == 0 and visible:
                try:
                    node.CreateDefaultDisplayNodes()
                    count = node.GetNumberOfDisplayNodes()
                except Exception:
                    count = 0
            for i in range(count):
                dn = node.GetNthDisplayNode(i)
                if dn is not None:
                    dn.SetVisibility(1 if visible else 0)
        except Exception:
            pass

    @staticmethod
    def _current_layout() -> int:
        """Return the current Slicer layout id, or -1 if unavailable."""
        try:
            import slicer
            return int(slicer.app.layoutManager().layout)
        except Exception:
            return -1

    @staticmethod
    def _set_layout(layout_id: int) -> None:
        """Restore the Slicer layout (the conventional/extension view 'overlay')."""
        try:
            if layout_id is None or int(layout_id) < 0:
                return
            import slicer
            lm = slicer.app.layoutManager()
            if lm is not None and lm.layout != int(layout_id):
                lm.setLayout(int(layout_id))
        except Exception:
            logger.debug("Layout restore failed", exc_info=True)

    def _commit_node_state(self, index: int) -> None:
        """Recover the before-step state, then DELETE downstream nodes for a clean re-run."""
        try:
            import slicer
        except Exception:
            return
        if not self.session:
            return
        cp = self.session.checkpoints[index]
        # Recover all upstream/baseline node state to the before-step snapshot
        # (display, transforms, layout, slice/view) without deleting anything.
        self._restore_scene_properties(cp.sceneview_node_id)
        keep = set(cp.before_node_ids or [])
        try:
            removed = 0
            for node_id in self._all_workflow_node_ids():
                if node_id in keep:
                    continue
                node = slicer.mrmlScene.GetNodeByID(node_id)
                if node is not None:
                    try:
                        slicer.mrmlScene.RemoveNode(node)    # downstream: delete for re-run
                        removed += 1
                    except Exception:
                        pass
            if removed:
                logger.info("[Replay] Removed %d downstream node(s) for re-run", removed)
        except Exception:
            logger.debug("Commit node state failed", exc_info=True)
        self._set_layout(cp.layout_before)

    def _preview_ui_state(self) -> Dict[str, Any]:
        """UI-state dict for the current navigation position.

        When live (preview_index is None) this is the normal state_for_ui. When
        previewing checkpoint i it returns that step's guidance + progress AND
        the choice buttons / parameter input that the step originally showed, so
        the user can see (and re-pick) the recorded choice; clicking one re-runs
        from that step. Done/Skip stay hidden so only Back/Forward/Action and the
        choices drive.
        """
        if not self.session:
            return {"active": False}
        if self.session.preview_index is None:
            return self.state_for_ui()
        idx = self.session.preview_index
        cp = self.session.checkpoints[idx]
        graph = get_workflow_graph(self.session.extension_name) or {}
        steps = graph.get("steps", []) if isinstance(graph, dict) else []
        step_ids = [s.get("step_id") for s in steps if s.get("step_id")]
        total_steps = graph.get("step_count") or len(steps)
        current_index = (step_ids.index(cp.step_id) + 1) if cp.step_id in step_ids else 0
        # Re-surface the step's interactive elements from the checkpoint so they
        # reappear while scrubbing. Free-form params (no enumerated choices) show
        # a prefilled input; clicking/submitting re-runs from here with that value.
        meta = self._step_meta(cp.step_id)
        guidance = (meta.get("ui_guidance") or {}) if isinstance(meta, dict) else {}
        choices = list(cp.choices or [])
        # Mirror the live path: a node-pick step drives the scene node tree,
        # never the recorded literal buttons. Loop decisions (Yes/No) are left
        # untouched.
        if cp.kind != "loop_decision" and self._is_node_selection_step(meta):
            choices = []
        needs_input = bool(cp.editable) and not choices
        binding = self._node_binding_for_param(cp.parameter_name)
        node_class = binding.get("node_class", "")
        if not node_class and needs_input:
            # Mirror the live path: surface the step graph's node class when the
            # pipeline inferred no binding, so replayed node steps keep a dropdown.
            node_class = self._node_class_from_step_meta(meta)
        # Clinically-informed instructions override the recorded guidance (live
        # lookup so manual edits show in replay too).
        instr = self._step_instructions_for(cp.step_id)
        return {
            "active": True,
            "extension_name": self.session.extension_name,
            "workflow_title": self._display_name(self.session.extension_name),
            "current_step": cp.step_id,
            "current_index": current_index,
            "completed_steps": idx,
            "total_steps": int(total_steps or 0),
            "status": f"Reviewing step {idx + 1}",
            "description": instr.get("title") or cp.guidance_description or cp.summary,
            "instructions": instr.get("simple") or cp.guidance_instructions,
            "instructions_detailed": instr.get("detailed", ""),
            "choices": choices,
            "needs_choice_input": needs_input,
            "default_value": cp.recorded_value,
            "parameter_name": cp.parameter_name,
            "node_class": node_class,
            "node_keywords": binding.get("keywords", []) or [],
            "choice_label": guidance.get("choice_label", ""),
            "input_label": guidance.get("input_label", ""),
            "repeat_progress": cp.repeat or {},
            "can_done": False,
            "can_skip": False,
            "can_cancel": True,
            "replay_can_back": idx > 0,
            "replay_can_forward": True,
            "replay_can_action": True,
            "replay_previewing": True,
            "preview_index": idx,
        }

    @staticmethod
    def _normalize_choices(source_choices) -> List[Dict[str, Any]]:
        choices = []
        for choice in source_choices or []:
            if not isinstance(choice, dict):
                continue
            label = str(choice.get("label", choice.get("value", ""))).strip()
            value = choice.get("value", label)
            if label or (value not in (None, "")):
                choices.append({"label": label or str(value), "value": value})
        return choices

    def rewind_to_checkpoint(
        self,
        index: int,
        modified_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Restore scene + truncate all state to BEFORE checkpoint ``index``.

        Returns a {step_id, action, args} descriptor for the widget to feed to
        _runWorkflowStepDirect; this method does not execute step code itself.
        """
        if not self.session:
            return {"error": "No active workflow session"}
        checkpoints = self.session.checkpoints
        if not isinstance(index, int) or index < 0 or index >= len(checkpoints):
            return {"error": f"Invalid checkpoint index: {index}"}
        cp = checkpoints[index]
        ext = self.session.extension_name

        # 1. Recover the before-step state (property-copy, no delete) then delete
        #    the downstream nodes so the re-run recreates them on a clean scene.
        self._commit_node_state(index)

        # 2. Truncate the in-memory session to the checkpoint prefix.
        self.session.completed_steps = list(cp.completed_prefix)
        self.session.completed_instances = (
            self.session.completed_instances[: cp.completed_instances_len]
        )
        self.session.repeat_states = copy.deepcopy(cp.repeat_states_snapshot)
        self.session.current_step = cp.step_id
        self.session.status = "running"
        self.session.last_result = None
        self._discard_pending_checkpoint()

        # 3. Truncate the module-global mirrors to the same prefix.
        truncate_workflow_completions(ext, set(cp.completed_prefix))
        set_workflow_choices(ext, dict(cp.choices_prefix))
        set_all_workflow_repeat_states(ext, cp.repeat_states_snapshot)

        # 4. Drop interaction-node memory for nodes the restore removed.
        try:
            from .workflow_state import prune_missing_interaction_nodes
            prune_missing_interaction_nodes(ext)
        except Exception:
            logger.debug("prune_missing_interaction_nodes failed", exc_info=True)

        # 5. Truncate the checkpoint list to the prefix and delete the discarded
        #    snapshots (and the live snapshot); re-running records new ones.
        for stale in checkpoints[index:]:
            self._delete_sceneview(stale.sceneview_node_id)
        self._delete_sceneview(self.session.live_sceneview_id)
        self.session.live_sceneview_id = None
        self.session.checkpoints = checkpoints[:index]
        self.session.active_checkpoint_index = index - 1 if index > 0 else None
        self.session.preview_index = None

        args = dict(modified_args) if modified_args else dict(cp.args)
        self._write_event("rewind_to_checkpoint", {
            "index": index,
            "step_id": cp.step_id,
            "action": cp.action,
            "modified": modified_args is not None,
        })
        return {"step_id": cp.step_id, "action": cp.action, "args": args}

    def clear_checkpoints(self) -> None:
        """Reset the timeline (cancel/complete). Recovers the live state if mid-navigation."""
        if self.session:
            # If the user was mid-navigation, restore the live scene first.
            if self.session.preview_index is not None and self.session.live_sceneview_id:
                self._restore_to_view(None)
            for cp in self.session.checkpoints:
                self._delete_sceneview(getattr(cp, "sceneview_node_id", None))
            self._delete_sceneview(self.session.live_sceneview_id)
            self.session.live_sceneview_id = None
            self.session.checkpoints = []
            self.session.active_checkpoint_index = None
            self.session.preview_index = None
        self._pending_checkpoint = None
        self._delete_all_replay_sceneviews()

    def _step_meta(self, step_id: str) -> Dict[str, Any]:
        if not self.session or not step_id:
            return {}
        graph = get_workflow_graph(self.session.extension_name) or {}
        for step in graph.get("steps", []) or []:
            if step.get("step_id") == step_id:
                return step
        return {}

    @staticmethod
    def _node_class_from_step_meta(meta: Dict[str, Any]) -> str:
        """Node class for a node-selection choice taken from the step graph itself.

        The pipeline's UI->parameter binding inference only recognizes the
        classic ``qMRMLNodeComboBox`` + ``parameterNode.SetNodeReferenceID``
        pattern. Selectors using a ``qMRMLSubjectHierarchyComboBox`` or the
        ``parameterNodeWrapper`` / ``connectGui`` declarative style yield no
        binding, so ``parameter_bindings`` / ``choice_binding`` carry no node
        class and the panel would fall back to a free-text box. The LLM
        decomposition still tags the choice's node class on the step's
        ``node_roles`` / ``sub_operations``; surface that as a fallback so any
        node-selection step still offers a dropdown of matching scene nodes.
        Returns "" for non-node choices (e.g. a boolean checkbox).
        """
        if not isinstance(meta, dict):
            return ""
        roles = meta.get("node_roles") or []
        for role in roles:
            if isinstance(role, dict) and role.get("role_kind") == "choice_input":
                nc = str(role.get("node_class") or "").strip()
                if nc:
                    return nc
        for so in meta.get("sub_operations") or []:
            if isinstance(so, dict) and str(so.get("value_kind") or "").strip() == "node":
                nc = str(so.get("node_class") or "").strip()
                if nc:
                    return nc
        for role in roles:
            if isinstance(role, dict):
                nc = str(role.get("node_class") or "").strip()
                if nc:
                    return nc
        return ""

    @staticmethod
    def _is_node_selection_step(meta: Dict[str, Any]) -> bool:
        """True when a user_choice step's selection is an MRML node pick.

        A node pick must drive the scene node tree, never literal cookbook-label
        buttons. The authoritative signal is a user_choice sub-op with
        ``value_kind == "node"``; a ``choice_input`` node-role carrying a
        ``node_class`` is accepted as a fallback for older artifacts. Segment-
        visibility steps (``value_kind == "segment_visibility_selection"``) are
        routed separately and excluded. Returns False for boolean / enum /
        numeric choices (no node_class) so those keep their buttons.
        """
        if not isinstance(meta, dict):
            return False
        sub_ops = [s for s in (meta.get("sub_operations") or []) if isinstance(s, dict)]
        kinds = {str(s.get("value_kind") or "").strip() for s in sub_ops}
        if "segment_visibility_selection" in kinds:
            return False
        if "node" in kinds:
            return True
        for role in meta.get("node_roles") or []:
            if (isinstance(role, dict)
                    and role.get("role_kind") == "choice_input"
                    and str(role.get("node_class") or "").strip()):
                return True
        return False

    def _node_binding_for_param(self, parameter_name: str) -> Dict[str, Any]:
        """Return the parameter binding ({node_class, keywords, ...}) from metadata.

        Node-selection steps bind a parameter (e.g. mandibularSegmentation) to a
        node class; that mapping lives in workflow_metadata.parameter_bindings.
        Used to offer a dropdown of matching scene nodes. Fail-soft to {}.
        """
        if not self.session or not parameter_name:
            return {}
        try:
            ext_data = get_validated_extensions().get(self.session.extension_name) or {}
            metadata = ext_data.get("workflow_metadata", {}) or {}
            bindings = metadata.get("parameter_bindings", {}) or {}
            return dict(bindings.get(parameter_name, {}) or {})
        except Exception:
            return {}

    def _step_instructions_for(self, step_id: str) -> Dict[str, Any]:
        """Return the clinically-informed {title, simple, detailed} for a step, or {}.

        Read live from the loaded step_instructions.json so manual edits take
        effect on the next render (live and replay) without restarting.
        """
        if not self.session or not step_id:
            return {}
        try:
            ext_data = get_validated_extensions().get(self.session.extension_name) or {}
            steps = (ext_data.get("step_instructions", {}) or {}).get("steps", {}) or {}
            return dict(steps.get(step_id, {}) or {})
        except Exception:
            return {}

    def _is_loop_count_source(self, step_id: str) -> bool:
        for block in self._repeat_blocks():
            controller = block.get("controller", {}) or {}
            if controller.get("kind") == "count" and controller.get("source_step") == step_id:
                return True
        return False

    def _interaction_summary(self, step_id: str) -> str:
        try:
            from .workflow_state import latest_interaction_node_for_step
            node = latest_interaction_node_for_step(step_id)
            if node is not None:
                try:
                    return f"Placed {node.GetName()}"
                except Exception:
                    return "Placed markup"
        except Exception:
            pass
        return "Interaction"

    @staticmethod
    def _checkpoint_summary(
        meta: Dict[str, Any],
        kind: str,
        value: Any,
        repeat: Optional[Dict[str, Any]],
    ) -> str:
        desc = str(meta.get("description") or meta.get("step_id") or "Step").strip()
        if kind in ("choice", "loop_count"):
            label = (meta.get("choice_info") or {}).get("parameter_name") or desc
            return f"{label}: {value}"
        if kind == "loop_decision":
            return f"Loop decision: {value}"
        return desc

    @staticmethod
    def _scene_node_ids() -> set:
        ids = set()
        try:
            import slicer
            for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                node = slicer.mrmlScene.GetNthNode(i)
                if node is None or not node.GetID():
                    continue
                if node.IsA("vtkMRMLSceneViewNode"):
                    continue
                ids.add(node.GetID())
        except Exception:
            pass
        return ids

    def _prune_to_node_ids(self, keep_ids) -> None:
        """Delete storable, non-singleton nodes not in ``keep_ids`` after a restore.

        Belt-and-braces for RestoreScene: it should remove nodes created after a
        snapshot, but interactively-created markup/display nodes can survive,
        which breaks generated templates that assume a clean node set on re-run
        (e.g. a step that does GetFirstNodeByClass and expects exactly one
        curve). This removes those leftovers while preserving infrastructure
        (singletons such as views/slices/selection, replay snapshots) and the
        baseline inputs (which are in keep_ids). Node IDs are preserved across
        StoreScene/RestoreScene, so a restored node keeps its original ID and
        stays in keep_ids; only genuinely newer nodes are dropped.
        """
        try:
            import slicer
        except Exception:
            return
        keep = set(keep_ids or [])
        try:
            doomed = []
            for i in range(slicer.mrmlScene.GetNumberOfNodes()):
                node = slicer.mrmlScene.GetNthNode(i)
                if node is None or not node.GetID():
                    continue
                if node.GetID() in keep:
                    continue
                if node.IsA("vtkMRMLSceneViewNode"):
                    continue
                if node.GetSingletonTag():
                    continue
                try:
                    if not node.GetSaveWithScene():
                        continue
                except Exception:
                    continue
                doomed.append(node)
            for node in doomed:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
            if doomed:
                logger.info(
                    "[Replay] Pruned %d stale node(s) after scene restore", len(doomed)
                )
        except Exception:
            logger.debug("Replay node prune failed", exc_info=True)

    def _capture_sceneview(self, step_id: str) -> Optional[str]:
        try:
            import slicer
        except Exception:
            return None
        try:
            ext = self.session.extension_name if self.session else "wf"
            name = f"_wfReplay_{ext}_{step_id}_{int(time.time() * 1000)}"
            sv = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSceneViewNode", name)
            if sv is None:
                return None
            sv.SetHideFromEditors(True)
            sv.SetAttribute("SlicerAIAgentReplay", "1")
            # Keep replay snapshots out of saved scenes / node selectors; they
            # are transient, in-memory, current-session only.
            try:
                sv.SetSaveWithScene(False)
            except Exception:
                pass
            sv.StoreScene()
            return sv.GetID()
        except Exception:
            logger.debug("Replay scene snapshot failed", exc_info=True)
            return None

    def _restore_sceneview(self, sceneview_node_id: Optional[str]) -> bool:
        if not sceneview_node_id:
            return False
        try:
            import slicer
        except Exception:
            return False
        try:
            sv = slicer.mrmlScene.GetNodeByID(sceneview_node_id)
            if sv is None:
                return False
            sv.RestoreScene()
            return True
        except Exception:
            logger.debug("Replay scene restore failed", exc_info=True)
            return False

    def _delete_sceneview(self, sceneview_node_id: Optional[str]) -> None:
        if not sceneview_node_id:
            return
        try:
            import slicer
            sv = slicer.mrmlScene.GetNodeByID(sceneview_node_id)
            if sv is not None:
                slicer.mrmlScene.RemoveNode(sv)
        except Exception:
            logger.debug("Replay scene snapshot delete failed", exc_info=True)

    @staticmethod
    def _delete_all_replay_sceneviews() -> None:
        try:
            import slicer
        except Exception:
            return
        try:
            collection = slicer.mrmlScene.GetNodesByClass("vtkMRMLSceneViewNode")
            if collection is None:
                return
            collection.UnRegister(None)
            doomed = []
            for i in range(collection.GetNumberOfItems()):
                node = collection.GetItemAsObject(i)
                if node is not None and node.GetAttribute("SlicerAIAgentReplay") == "1":
                    doomed.append(node)
            for node in doomed:
                slicer.mrmlScene.RemoveNode(node)
        except Exception:
            logger.debug("Replay scene snapshot bulk delete failed", exc_info=True)

    def _write_event(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.log_dir or not self.session:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, "workflow_runtime.jsonl")
            entry = {
                "time": round(time.time(), 3),
                "event": event,
                "workflow_id": self.session.workflow_id,
                "extension_name": self.session.extension_name,
                "current_step": self.session.current_step,
                "status": self.session.status,
                "payload": payload,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
