from __future__ import annotations

from .cache import *


def reset_workflow_state(extension_name: Optional[str] = None) -> None:
    """Clear accumulated workflow state so a workflow can be re-run cleanly.

    Args:
        extension_name: If provided, only clear state for that extension.
            If None, clear all workflow state.
    """
    if extension_name is not None:
        _workflow_completed_steps.pop(extension_name, None)
        _workflow_choices.pop(extension_name, None)
        _workflow_repeat_state.pop(extension_name, None)
    else:
        _workflow_completed_steps.clear()
        _workflow_choices.clear()
        _workflow_repeat_state.clear()
    try:
        from SlicerAIAgentLib.workflow_state import clear_workflow_state
        clear_workflow_state(extension_name)
    except Exception:
        logger.debug("Failed to clear generated workflow runtime state", exc_info=True)


def reset_workflow_session(extension_name: Optional[str] = None) -> None:
    """Public alias used by deterministic generated-CLI workflow runtime."""
    reset_workflow_state(extension_name)


def get_workflow_graph(extension_name: str) -> Dict:
    """Return the parsed workflow graph for a validated generated extension."""
    _ensure_cache()
    ext_data = _cli_cache.get(extension_name)
    if not ext_data:
        return {}
    workflow_path = os.path.join(ext_data["dir"], "workflow.json")
    if not os.path.isfile(workflow_path):
        return {}
    with open(workflow_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_next_workflow_step(
    extension_name: str,
    completed_steps: Optional[set] = None,
) -> Optional[Dict]:
    """Return the next workflow step whose dependencies are satisfied."""
    graph = get_workflow_graph(extension_name)
    if not graph:
        return None
    completed = (
        set(completed_steps)
        if completed_steps is not None
        else set(_workflow_completed_steps.get(extension_name, set()))
    )
    return _find_next_step_local(graph, completed)


def mark_workflow_step_completed(extension_name: str, step_id: str) -> None:
    """Record a generated CLI workflow step as completed."""
    if not extension_name or not step_id:
        return
    _workflow_completed_steps.setdefault(extension_name, set()).add(step_id)


def clear_workflow_step_completions(
    extension_name: str,
    step_ids: List[str],
) -> None:
    """Clear atomic step completion markers before a repeat iteration."""
    completed = _workflow_completed_steps.setdefault(extension_name, set())
    completed.difference_update(step_ids or [])


def set_workflow_repeat_state(
    extension_name: str,
    repeat_id: str,
    state: Dict[str, Any],
) -> None:
    """Publish generic runtime repeat state for generated template context."""
    if not extension_name or not repeat_id:
        return
    _workflow_repeat_state.setdefault(extension_name, {})[repeat_id] = dict(state or {})


def _find_next_step_local(
    workflow_graph: Dict, completed: set
) -> Optional[Dict]:
    """Find the next workflow step whose dependencies are all completed."""
    for step in workflow_graph.get("steps", []):
        sid = step.get("step_id", "")
        if sid in completed:
            continue
        deps = step.get("depends_on", [])
        if all(d in completed for d in deps):
            is_optional = bool(step.get("is_optional", False))
            return {
                "step_id": sid,
                "operation_type": step.get("operation_type") or step.get("op_type") or step.get("step_type", "extension_op"),
                "step_type": step.get("step_type", step.get("operation_type", "extension_op")),
                "description": step.get("description", ""),
                "is_optional": is_optional,
                "ui_guidance": step.get("ui_guidance", {}),
            }
    return None


def dispatch_workflow_step(
    ext_name: str,
    ext_data: Dict,
    tool_name: str,
    arguments: Dict,
) -> Dict:
    """
    Dispatch a tool call for an interactive workflow extension.

    Reads the workflow.json, matches the requested step, and delegates to
    the appropriate per-type handler.

    Returns:
        Dict with type-specific fields:
        - type: runtime result status such as "automated", "interactive",
          "user_choice", or "error"
        - For user_interaction: pre_code, interaction descriptor, instructions
        - For extension_op/slicer_op: code, instruction
    """
    ext_dir = ext_data["dir"]
    generators = ext_data["generators"]
    metadata = ext_data.get("workflow_metadata", {}) or {}

    # Load workflow graph
    workflow_path = os.path.join(ext_dir, "workflow.json")
    if not os.path.isfile(workflow_path):
        return {"error": f"workflow.json not found for {ext_name}"}

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_graph = json.load(f)

    # Get requested step and action
    workflow_step = arguments.get("workflow_step", "")
    user_action = arguments.get("user_action", "start")

    if user_action == "cancel":
        return {
            "tool": tool_name,
            "type": "cancelled",
            "message": "Workflow cancelled.",
        }

    # Find the matching step
    target_step = None
    for step in workflow_graph.get("steps", []):
        if step["step_id"] == workflow_step:
            target_step = step
            break

    if not target_step:
        available = [s["step_id"] for s in workflow_graph.get("steps", [])]
        return {
            "error": f"Unknown workflow step '{workflow_step}'. Available: {available}",
        }

    operation_type = (
        target_step.get("operation_type")
        or target_step.get("op_type")
        or target_step.get("step_type", "extension_op")
    )
    legacy_step_type = {
        "extension_op": "automated",
        "slicer_op": "automated",
        "user_interaction": "interactive",
        "user_choice": "user_choice",
    }.get(operation_type, operation_type)

    # Track step completion for the local next-step resolver.
    # When start/proceed is called, the current step's depends_on are all done.
    # Add them to the completed set so subsequent skip calls can compute
    # the correct next step.
    done = _workflow_completed_steps.setdefault(ext_name, set())
    for dep in target_step.get("depends_on", []):
        done.add(dep)

    # Find the matching generator entry
    target_gen = None
    for gen in generators:
        gen_step = gen.get("param_signature", {}).get("workflow_step", "")
        if gen_step == workflow_step:
            target_gen = gen
            break

    # Handle skip uniformly for all step types
    if user_action == "skip":
        return _handle_skip(
            ext_name, tool_name, workflow_step, workflow_graph, done,
        )

    # Delegate to per-type handler
    ctx = _WorkflowContext(
        ext_name=ext_name,
        ext_dir=ext_dir,
        tool_name=tool_name,
        workflow_graph=workflow_graph,
        target_step=target_step,
        target_gen=target_gen,
        arguments=arguments,
        user_action=user_action,
        done=done,
        metadata=metadata,
    )

    handlers = {
        "extension_op": _handle_automated_step,
        "slicer_op": _handle_automated_step,
        "user_interaction": _handle_interactive_step,
        "user_choice": _handle_user_choice_step,
        # Legacy generated packages are mapped only when no canonical
        # operation_type/op_type exists.
        "automated": _handle_automated_step,
        "interactive": _handle_interactive_step,
    }
    handler = handlers.get(operation_type) or handlers.get(legacy_step_type)
    if not handler:
        return {"error": f"Unknown operation type: {operation_type}"}
    result = handler(ctx)

    # Centralized tagging: any template-derived result with executable code
    # is marked as generator-pipeline origin so the runtime fast path and the
    # self-correction classifier can route on it. ``_prelude_globals`` carries
    # the workflow-metadata dicts the prelude expects in the executor
    # namespace; the executor registers each key via addGlobal before exec.
    if isinstance(result, dict) and result.get("code"):
        result.setdefault("origin", "generated_template")
        try:
            prelude_globals = _build_prelude_globals(ctx)
        except Exception:
            logger.debug(
                "Failed to build prelude globals for step %s",
                workflow_step,
                exc_info=True,
            )
            prelude_globals = {}
        if prelude_globals:
            result["_prelude_globals"] = prelude_globals
    return result


# =====================================================================
# Shared helpers used by the per-type handlers
# =====================================================================
