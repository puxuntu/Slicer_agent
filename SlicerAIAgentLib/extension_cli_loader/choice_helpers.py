from __future__ import annotations

from .cache import *
from .templates import _workflow_choices, _workflow_repeat_state
from .workflow_state import _find_next_step_local


def _build_format_kwargs(arguments: Dict) -> Dict[str, str]:
    """Convert tool arguments to template format kwargs (repr-wrapped)."""
    format_kwargs = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            format_kwargs[key] = repr(value)
        elif value is None:
            format_kwargs[key] = "None"
        else:
            format_kwargs[key] = repr(value)
    return format_kwargs


def _build_next_step_instruction(tool_name: str, next_step: Optional[Dict], prefix: str = "") -> str:
    """Build the standard 'next step' instruction string."""
    if not next_step:
        return f"{prefix}All steps complete. Workflow is done." if prefix else "All steps complete. Workflow is done."
    is_opt = next_step.get("is_optional", False)
    if is_opt:
        return (
            f"{prefix}"
            f"Next step '{next_step['step_id']}' is optional: "
            f"{next_step['description']} "
            f"Ask the user if they want to proceed. "
            f"If yes, call {tool_name} with workflow_step='{next_step['step_id']}' user_action='start'. "
            f"If no, call with workflow_step='{next_step['step_id']}' user_action='skip'."
        )
    return (
        f"{prefix}"
        f"Proceed by calling {tool_name} with "
        f"workflow_step='{next_step['step_id']}' user_action='start'."
    )


def _semantic_tokens(text: str) -> set:
    """Tokenize names/questions for generic confidence matching."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
    stop = {
        "the", "and", "for", "with", "node", "select", "choose", "current",
        "which", "what", "option", "step", "user", "choice", "number",
        "many", "want", "volume", "segmentation", "model", "markup",
    }
    tokens = {w for w in words if w not in stop and len(w) > 2}
    tokens.update(w[:6] for w in list(tokens) if len(w) >= 6)
    return tokens


def _choice_is_closed_form(choices: List[Dict]) -> bool:
    """Return True for non-scene choices that should be asked explicitly."""
    if not choices:
        return False
    labels = {str(c.get("label", "")).strip().lower() for c in choices}
    values = {str(c.get("value", "")).strip().lower() for c in choices}
    labels_and_values = labels | values
    normalized = {
        re.sub(r"[^a-z0-9]+", " ", item).strip()
        for item in labels_and_values
        if item
    }
    compact = {item.replace(" ", "") for item in normalized}
    boolean_options = {"yes", "no", "true", "false"}
    side_options = {
        "left", "right", "left leg", "right leg",
        "left side", "right side", "left fibula", "right fibula",
    }
    compact_side_options = {item.replace(" ", "") for item in side_options}
    return (
        normalized <= boolean_options
        or compact <= boolean_options
        or normalized <= side_options
        or compact <= compact_side_options
    )


def _choice_is_count_question(choice_desc: Dict, step: Dict) -> bool:
    """Return True for numeric/count choices that should not auto-select nodes."""
    text = " ".join([
        str(choice_desc.get("parameter_name", "")),
        str(choice_desc.get("question", "")),
        str(step.get("description", "")),
    ]).lower()
    return any(token in text for token in ("how many", "number of", "count", "num", "amount"))


def _ui_guidance_for_context(ctx: _WorkflowContext, descriptor: Optional[Dict] = None) -> Dict:
    """Return the best available generated UI guidance for this step."""
    descriptor = descriptor or {}
    guidance = {}
    for source in (
        ctx.target_step.get("ui_guidance", {}),
        (ctx.target_gen or {}).get("ui_guidance", {}),
        descriptor.get("ui_guidance", {}),
    ):
        if isinstance(source, dict):
            guidance.update({k: v for k, v in source.items() if v not in (None, "")})
    return guidance


def _repeat_progress_for_context(ctx: _WorkflowContext) -> Dict:
    """Return repeat progress for one-item-at-a-time interaction UI."""
    repeat_group = (
        (ctx.target_gen or {}).get("repeat_block")
        or ctx.target_step.get("repeat_block")
        or (ctx.target_gen or {}).get("repeat_group")
        or (ctx.target_gen or {}).get("interaction_descriptor", {}).get("repeat_group")
        or (ctx.target_gen or {}).get("choice_descriptor", {}).get("repeat_group")
        or ctx.target_step.get("repeat_group")
    )
    if not repeat_group:
        return {}
    group_id = (
        repeat_group.get("repeat_id")
        or repeat_group.get("group_id")
        or ctx.workflow_step
    )
    state = _workflow_repeat_state.get(ctx.ext_name, {}).get(group_id, {})
    target = int(state.get("target", 0) or 0)
    iteration = int(state.get("iteration", 0) or 0)
    completed = int(state.get("completed", max(0, iteration - 1)) or 0)
    if target <= 0:
        return {
            "group_id": group_id,
            "current": iteration,
            "completed": completed,
            "total": 0,
        }
    current = min(iteration or completed + 1, target)
    return {
        "group_id": group_id,
        "current": current,
        "completed": completed,
        "total": target,
    }


def _record_choice_and_advance(
    ctx: _WorkflowContext,
    param_name: str,
    choice_value: Any,
    auto_selected: bool = False,
) -> Dict:
    """Store a choice, initialize repeat state when needed, and advance."""
    choice_desc = (ctx.target_gen or {}).get("choice_descriptor", {}) or {}
    choices = choice_desc.get("choices") or (ctx.target_step.get("choice_info") or {}).get("choices") or []
    choice_value = _normalize_choice_value(choice_value, choices)
    ext_choices = _workflow_choices.setdefault(ctx.ext_name, {})
    if param_name:
        ext_choices[param_name] = choice_value

    repeat_group = (
        (ctx.target_gen or {}).get("repeat_group")
        or (ctx.target_gen or {}).get("choice_descriptor", {}).get("repeat_group")
        or ctx.target_step.get("repeat_group")
    )
    if repeat_group:
        try:
            target = int(choice_value)
        except (TypeError, ValueError):
            target = 0
        if target > 0:
            group_id = repeat_group.get("group_id") or ctx.workflow_step
            _workflow_repeat_state.setdefault(ctx.ext_name, {})[group_id] = {
                "target": target,
                "completed": 0,
            }

    ctx.done.add(ctx.workflow_step)
    next_step = _find_next_step_local(ctx.workflow_graph, ctx.done)
    result = {
        "tool": ctx.tool_name,
        "type": "choice_made",
        "step_id": ctx.workflow_step,
        "parameter_name": param_name,
        "choice_value": choice_value,
        "message": f"Selected '{choice_value}' for '{param_name}'.",
    }
    if auto_selected:
        result["auto_selected"] = True
        result["message"] = (
            f"Automatically selected '{choice_value}' for '{param_name}' "
            "from the loaded scene."
        )
    choice_code = _build_choice_parameter_update_code(ctx, param_name, choice_value)
    if choice_code:
        result["type"] = "choice_made"
        result["code"] = choice_code
        result["instruction"] = (
            "STOP. Do NOT make any more tool calls. "
            "Your NEXT response must be an ```agent_plan JSON block followed by a ```python block "
            "containing the 'code' field above VERBATIM. "
            "Do NOT call any more tools until this code has been executed."
        )
        if next_step:
            result["next_step"] = next_step
            result["after_execution_instruction"] = _build_next_step_instruction(ctx.tool_name, next_step)
        return result
    result["instruction"] = _build_next_step_instruction(ctx.tool_name, next_step)
    if next_step:
        result["next_step"] = next_step
    return result


def _normalize_choice_value(choice_value: Any, choices: List[Dict]) -> Any:
    """Map labels like 'Yes'/'No' or strings to the declared choice value."""
    if isinstance(choice_value, str):
        raw = choice_value.strip()
        lowered = raw.lower()
    else:
        raw = choice_value
        lowered = str(choice_value).strip().lower()
    for choice in choices or []:
        label = str(choice.get("label", "")).strip().lower()
        value = choice.get("value")
        value_text = str(value).strip().lower()
        if lowered in {label, value_text}:
            return value
    if lowered in ("true", "yes", "right", "right leg", "right side"):
        return True
    if lowered in ("false", "no", "left", "left leg", "left side"):
        return False
    return raw


def _python_bool_text(value: Any) -> str:
    normalized = _normalize_choice_value(value, [])
    if isinstance(normalized, bool):
        return "True" if normalized else "False"
    lowered = str(normalized).strip().lower()
    return "True" if lowered in ("true", "yes", "right", "right leg", "right side", "1") else "False"


def _build_choice_parameter_update_code(
    ctx: _WorkflowContext,
    param_name: str,
    choice_value: Any,
) -> str:
    """Return executable code for user_choice steps bound to parameter updates."""
    sub_ops = (
        (ctx.target_gen or {}).get("sub_operations")
        or ctx.target_step.get("sub_operations")
        or []
    )
    choice_desc = (ctx.target_gen or {}).get("choice_descriptor", {}) or {}
    choices = choice_desc.get("choices") or (ctx.target_step.get("choice_info") or {}).get("choices") or []
    normalized_value = _normalize_choice_value(choice_value, choices)
    matched = None
    for so in sub_ops:
        if (
            so.get("operation_intent") == "extension_parameter_update"
            and so.get("parameter_name") == param_name
        ):
            matched = so
            break
    if not matched:
        return ""

    module_name = ctx.metadata.get("extension_module_name") or ctx.ext_name
    logic_class_name = ctx.metadata.get("logic_class_name") or f"{ctx.ext_name}Logic"
    logic_var = f"_{ctx.ext_name.lower()}_logic"
    value_property = matched.get("value_property", "")
    if value_property == "checked":
        value_text = _python_bool_text(normalized_value)
    else:
        value_text = str(normalized_value)

    return "\n".join([
        f"# --- {ctx.ext_name}: {ctx.workflow_step} choice ---",
        "import slicer",
        f"from {module_name} import {logic_class_name}",
        "",
        "try:",
        f"    logic = {logic_var}",
        "except NameError:",
        f"    logic = {logic_class_name}()",
        f"    {logic_var} = logic",
        "",
        "parameterNode = logic.getParameterNode()",
        f"parameterNode.SetParameter({param_name!r}, {value_text!r})",
        "try:",
        "    parameterNode.Modified()",
        "except Exception:",
        "    pass",
        f"{logic_var} = logic",
        f"print(\"[{ctx.ext_name}] Step '{ctx.workflow_step}' choice applied.\")",
        "",
    ])


def _try_auto_select_choice(ctx: _WorkflowContext, choice_desc: Dict) -> Optional[Dict]:
    """Auto-select a scene node for high-confidence node-binding choices."""
    if slicer is None or ctx.user_action != "start":
        return None

    choices = choice_desc.get("choices", [])
    if _choice_is_closed_form(choices) or _choice_is_count_question(choice_desc, ctx.target_step):
        return None

    binding = (
        choice_desc.get("binding")
        or ctx.target_step.get("choice_binding")
        or ctx.metadata.get("choice_bindings", {}).get(ctx.workflow_step, {})
    )
    node_class = binding.get("node_class", "") if binding else ""
    if not node_class:
        return None

    param_name = (
        binding.get("choice_parameter_name")
        or choice_desc.get("parameter_name")
        or binding.get("parameter_name", "")
    )
    binding_param = binding.get("parameter_name", param_name)
    keywords = set(binding.get("keywords", []))
    question_tokens = _semantic_tokens(
        " ".join([
            param_name,
            choice_desc.get("question", ""),
            ctx.target_step.get("description", ""),
        ])
    )
    target_tokens = keywords | question_tokens

    nodes = slicer.mrmlScene.GetNodesByClass(node_class)
    candidates = []
    for i in range(nodes.GetNumberOfItems()):
        node = nodes.GetItemAsObject(i)
        if node is None:
            continue
        name = node.GetName() or ""
        node_tokens = _semantic_tokens(name)
        score = len(target_tokens & node_tokens)
        if str(name).lower() == str(param_name).lower():
            score += 5
        if keywords and keywords <= node_tokens:
            score += 2
        candidates.append((score, name, node))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    best = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else -1

    # High confidence means either there is only one class-compatible node, or
    # the best semantic name match is clearly separated from the next candidate.
    if not (len(candidates) == 1 or (best[0] >= 1 and best[0] >= second_score + 1)):
        return None

    # Store both the user-facing cookbook parameter and the actual parameter-node
    # role when they differ, so template prelude can bind the Slicer parameter node.
    ext_choices = _workflow_choices.setdefault(ctx.ext_name, {})
    if binding_param and binding_param != param_name:
        ext_choices[binding_param] = best[1]
    return _record_choice_and_advance(ctx, param_name, best[1], auto_selected=True)


def _build_choice_prelude(ctx: _WorkflowContext) -> str:
    """Emit a tiny shim that imports and invokes the metadata-application helper.

    The choices/bindings/defaults dicts are passed as Python objects via the
    executor namespace (see _build_prelude_globals); the prelude only needs
    to import the helper and call it. This collapses ~60 lines of inlined
    Python source per dispatch into ~10, and eliminates the json.dumps-vs-Python
    literal class of bugs (booleans like ``false`` are not Python literals).
    """
    choices = _workflow_choices.get(ctx.ext_name, {})
    bindings = ctx.metadata.get("parameter_bindings", {}) or {}
    defaults = ctx.metadata.get("parameter_defaults", {}) or {}
    module_name = ctx.metadata.get("extension_module_name", "")
    logic_class_name = ctx.metadata.get("logic_class_name", "")
    if (not choices and not defaults) or not bindings or not module_name or not logic_class_name:
        return ""

    logic_var = f"_{ctx.ext_name.lower()}_logic"
    return (
        "# [Workflow metadata] Apply source-derived defaults and stored user choices\n"
        "import slicer\n"
        f"from {module_name} import {logic_class_name}\n"
        "from SlicerAIAgentLib.workflow_state import apply_workflow_metadata\n"
        "try:\n"
        f"    logic = {logic_var}\n"
        "except NameError:\n"
        f"    logic = {logic_class_name}()\n"
        "parameterNode = logic.getParameterNode()\n"
        "apply_workflow_metadata(parameterNode, _workflow_choices, _workflow_bindings, _workflow_defaults)\n"
        f"{logic_var} = logic\n\n"
    )


def _repeat_index_for_context(ctx: _WorkflowContext) -> int:
    repeat_group = (
        (ctx.target_gen or {}).get("repeat_block")
        or ctx.target_step.get("repeat_block")
        or (ctx.target_gen or {}).get("repeat_group")
        or (ctx.target_gen or {}).get("interaction_descriptor", {}).get("repeat_group")
        or (ctx.target_gen or {}).get("choice_descriptor", {}).get("repeat_group")
        or ctx.target_step.get("repeat_group")
        or {}
    )
    group_id = repeat_group.get("repeat_id") or repeat_group.get("group_id", "")
    if not group_id:
        return 0
    state = _workflow_repeat_state.get(ctx.ext_name, {}).get(group_id, {})
    if state.get("iteration"):
        return int(state["iteration"])
    return int(state.get("completed", 0)) + 1


def _method_name_for_context(ctx: _WorkflowContext) -> str:
    method_name = (ctx.target_gen or {}).get("method_name") or ctx.target_step.get("method_name") or ""
    if method_name:
        return method_name
    for so in (ctx.target_gen or {}).get("sub_operations", []) or ctx.target_step.get("sub_operations", []) or []:
        if so.get("op_type") == "extension_op" and so.get("extension_method_hint"):
            return so.get("extension_method_hint", "")
    return ""


def _build_runtime_prelude(ctx: _WorkflowContext) -> str:
    """Build hidden runtime context and an import shim for precondition checks.

    The validator function itself lives in SlicerAIAgentLib.workflow_state —
    the prelude just imports it and resolves the per-step ``_workflow_logic``
    instance. ``_workflow_required_bindings`` and ``_workflow_method_name``
    are passed as Python objects via the _prelude_globals side channel; the
    injected precondition call before each ``logic.<method>()`` site reads
    them from the executor namespace.
    """
    workflow_id = str(ctx.arguments.get("_workflow_id") or ctx.arguments.get("workflow_id") or "")
    repeat_index = _repeat_index_for_context(ctx)
    step_id = ctx.workflow_step
    method_name = _method_name_for_context(ctx)
    metadata = ctx.metadata or {}
    module_name = metadata.get("extension_module_name", "")
    logic_class_name = metadata.get("logic_class_name", "")

    lines = [
        "# [Workflow runtime] Hidden generated-CLI workflow context",
        f"_workflow_runtime_extension = {ctx.ext_name!r}",
        f"_workflow_runtime_id = {workflow_id!r}",
        f"_workflow_runtime_step = {step_id!r}",
        f"_workflow_runtime_repeat_index = {repeat_index}",
        "from SlicerAIAgentLib.workflow_state import remember_interaction_node, resolve_interaction_node",
        "",
    ]
    if not (method_name and module_name and logic_class_name):
        return "\n".join(lines) + "\n"

    lines.extend([
        "# [Workflow preconditions] Validate source-derived node references before extension logic calls",
        "from SlicerAIAgentLib.workflow_state import validate_method_preconditions",
        f"try:\n    _workflow_logic = _{ctx.ext_name.lower()}_logic\nexcept NameError:\n    from {module_name} import {logic_class_name}\n    _workflow_logic = {logic_class_name}()",
        f"_{ctx.ext_name.lower()}_logic = _workflow_logic",
        "",
    ])
    return "\n".join(lines) + "\n"


def _has_method_preconditions(ctx: _WorkflowContext) -> bool:
    """True iff this step will emit method-precondition metadata.

    Cheap structural check — only depends on whether the step has a method
    name AND parameter bindings referenced by that method's dependencies.
    Used by _inject_method_precondition_call to avoid rebuilding the entire
    prelude just to test membership.
    """
    method_name = _method_name_for_context(ctx)
    if not method_name:
        return False
    metadata = ctx.metadata or {}
    bindings = metadata.get("parameter_bindings", {}) or {}
    method_deps = (metadata.get("parameter_method_dependencies", {}) or {}).get(method_name, {}) or {}
    role_names = list(method_deps.get("parameter_roles") or []) + list(method_deps.get("node_roles") or [])
    return any(isinstance(bindings.get(role), dict) for role in role_names)


def _inject_method_precondition_call(ctx: _WorkflowContext, code: str) -> str:
    method_name = _method_name_for_context(ctx)
    if not method_name or not _has_method_preconditions(ctx):
        return code
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*)logic\.{re.escape(method_name)}\s*\(",
        re.MULTILINE,
    )
    match = pattern.search(code)
    if not match:
        return code
    insertion = (
        f"{match.group('indent')}"
        "validate_method_preconditions(_workflow_logic, _workflow_required_bindings)\n"
    )
    return code[:match.start()] + insertion + code[match.start():]


def _build_prelude_globals(ctx: _WorkflowContext) -> Dict[str, Any]:
    """Return the Python-object side channel for the prelude.

    The dispatch result attaches this dict under ``_prelude_globals``; the
    executor registers each key as a ``__main__`` global before exec, so the
    prelude can reference choices/bindings/defaults directly without
    serializing them to source text. Always emits all-or-nothing per group
    (choice metadata vs precondition metadata) so the prelude never sees a
    missing global it would otherwise reference.
    """
    globals_dict: Dict[str, Any] = {}
    choices = _workflow_choices.get(ctx.ext_name, {})
    bindings = ctx.metadata.get("parameter_bindings", {}) or {}
    defaults = ctx.metadata.get("parameter_defaults", {}) or {}
    module_name = ctx.metadata.get("extension_module_name", "")
    logic_class_name = ctx.metadata.get("logic_class_name", "")
    if (choices or defaults) and bindings and module_name and logic_class_name:
        globals_dict["_workflow_choices"] = dict(choices)
        globals_dict["_workflow_bindings"] = dict(bindings)
        globals_dict["_workflow_defaults"] = dict(defaults)

    method_name = _method_name_for_context(ctx)
    if method_name:
        method_deps = (ctx.metadata.get("parameter_method_dependencies", {}) or {}).get(method_name, {}) or {}
        role_names = list(method_deps.get("parameter_roles") or []) + list(method_deps.get("node_roles") or [])
        node_requirements = method_deps.get("node_requirements") or {}
        required_bindings = {
            role: {
                **bindings.get(role, {}),
                "_workflow_requirement": node_requirements.get(
                    role, {
                        "requirement": "optional_unknown",
                        "conditions": [],
                        "condition_groups": [],
                    }
                ),
            }
            for role in role_names
            if isinstance(bindings.get(role), dict)
        }
        if required_bindings and module_name and logic_class_name:
            globals_dict["_workflow_required_bindings"] = required_bindings
            globals_dict["_workflow_method_name"] = method_name
    return globals_dict


def _prepend_choice_prelude(ctx: _WorkflowContext, code: Optional[str]) -> Optional[str]:
    if not code:
        return code
    runtime_prelude = _build_runtime_prelude(ctx)
    prelude = _build_choice_prelude(ctx)
    code = _inject_method_precondition_call(ctx, code)
    return (prelude or "") + runtime_prelude + code


def build_assembled_code_for_validation(
    ext_name: str,
    metadata: Dict[str, Any],
    generator: Dict[str, Any],
    template_code: str,
) -> str:
    """Assemble prelude + filled template for offline dry-validation.

    Mirrors what ``dispatch_workflow_step`` produces at runtime, minus the
    actual choice values (which are runtime state). Used by Stage 9 to
    validate the *assembled* artifact (template + prelude) at generation
    time, so generator-side prelude bugs surface here instead of at every
    runtime step.

    Args:
        ext_name: Extension name (e.g., "BoneReconstructionPlanner").
        metadata: workflow_metadata.json contents (parameter_bindings, etc.).
        generator: Generator entry from code_generators.json.
        template_code: Template file content with all format-kwargs already
            filled by the caller.

    Returns:
        Assembled code string (prelude + template) ready for ``ast.parse``
        and ``CodeValidator.validate``.
    """
    workflow_step = (
        (generator.get("param_signature") or {}).get("workflow_step")
        or generator.get("workflow_step")
        or "validation_sample_step"
    )
    target_step = {
        "step_id": workflow_step,
        "method_name": generator.get("method_name", ""),
        "sub_operations": generator.get("sub_operations", []),
        "repeat_block": generator.get("repeat_block"),
        "repeat_group": generator.get("repeat_group"),
        "interaction_descriptor": generator.get("interaction_descriptor", {}),
        "choice_descriptor": generator.get("choice_descriptor", {}),
    }
    ctx = _WorkflowContext(
        ext_name=ext_name,
        ext_dir="",
        tool_name=f"{ext_name}_validation",
        workflow_graph={"steps": []},
        target_step=target_step,
        target_gen=generator,
        arguments={"_workflow_id": "validation"},
        user_action="start",
        done=set(),
        metadata=metadata or {},
    )
    assembled = _prepend_choice_prelude(ctx, template_code)
    return assembled if assembled is not None else template_code


def _store_generated_interaction_node_code(
    extension_name: str,
    step_id: str,
    node_expr: str = "node",
) -> str:
    node_var = f"_{extension_name.lower()}_{step_id}_id"
    return "\n".join([
        f"{node_var} = {node_expr}.GetID()",
        (
            "remember_interaction_node("
            f"_workflow_runtime_extension, _workflow_runtime_id, {step_id!r}, "
            f"{node_var}, _workflow_runtime_repeat_index)"
        ),
    ])


def _handle_repeat_after_interaction(ctx: _WorkflowContext) -> Optional[Dict]:
    """Return a repeat-next response when an interaction should loop."""
    if (ctx.target_gen or {}).get("repeat_block") or ctx.target_step.get("repeat_block"):
        # Generic repeat blocks advance only after post-code execution succeeds.
        return None
    repeat_group = (
        (ctx.target_gen or {}).get("repeat_group")
        or (ctx.target_gen or {}).get("interaction_descriptor", {}).get("repeat_group")
        or ctx.target_step.get("repeat_group")
    )
    if not repeat_group:
        return None
    if repeat_group.get("interaction_step") != ctx.workflow_step:
        return None

    group_id = repeat_group.get("group_id") or ctx.workflow_step
    state = _workflow_repeat_state.setdefault(ctx.ext_name, {}).setdefault(
        group_id, {"target": 0, "completed": 0}
    )
    state["completed"] = int(state.get("completed", 0)) + 1
    target = int(state.get("target", 0))
    if target <= 0 or state["completed"] >= target:
        return None

    start_step_id = repeat_group.get("start_step")
    if start_step_id:
        ctx.done.discard(start_step_id)
    ctx.done.discard(ctx.workflow_step)
    next_step = {
        "step_id": start_step_id or ctx.workflow_step,
        "step_type": "automated",
        "description": (
            f"Repeat placement {state['completed'] + 1} of {target}"
        ),
        "is_optional": False,
    }
    return {
        "tool": ctx.tool_name,
        "type": "repeat_next",
        "step_id": ctx.workflow_step,
        "repeat_group": repeat_group,
        "repeat_progress": {
            "group_id": group_id,
            "completed": state["completed"],
            "current": state["completed"] + 1,
            "total": target,
        },
        "ui_guidance": _ui_guidance_for_context(ctx, (ctx.target_gen or {}).get("interaction_descriptor", {})),
        "message": (
            f"Repeat item {state['completed']} of {target} completed. "
            f"Continue with item {state['completed'] + 1}."
        ),
        "next_step": next_step,
        "instruction": _build_next_step_instruction(ctx.tool_name, next_step),
    }
