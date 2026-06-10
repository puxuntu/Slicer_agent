"""Runtime state helpers for generated extension CLI workflows.

Generated workflow templates execute as independent snippets.  This module
keeps the small amount of cross-snippet state that cannot safely live in a
Python global embedded in one generated code block.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

_interaction_nodes: Dict[Tuple[str, str, str, int], str] = {}


def _key(
    extension_name: str,
    workflow_id: str,
    step_id: str,
    repeat_index: Optional[int] = None,
) -> Tuple[str, str, str, int]:
    return (
        str(extension_name or ""),
        str(workflow_id or ""),
        str(step_id or ""),
        int(repeat_index or 0),
    )


def clear_workflow_state(extension_name: Optional[str] = None) -> None:
    """Clear generated workflow runtime state."""
    if not extension_name:
        _interaction_nodes.clear()
        return
    ext = str(extension_name)
    for key in list(_interaction_nodes):
        if key[0] == ext:
            _interaction_nodes.pop(key, None)


def remember_interaction_node(
    extension_name: str,
    workflow_id: str,
    step_id: str,
    node_id: str,
    repeat_index: Optional[int] = None,
) -> str:
    """Remember the MRML node ID produced for a workflow interaction step."""
    if node_id:
        _interaction_nodes[_key(extension_name, workflow_id, step_id, repeat_index)] = str(node_id)
    return str(node_id or "")


def get_interaction_node_id(
    extension_name: str,
    workflow_id: str,
    step_id: str,
    repeat_index: Optional[int] = None,
) -> str:
    """Return a remembered interaction node ID, or an empty string."""
    return _interaction_nodes.get(_key(extension_name, workflow_id, step_id, repeat_index), "")


def resolve_interaction_node(
    extension_name: str,
    workflow_id: str,
    step_id: str,
    node_class: str = "",
    repeat_index: Optional[int] = None,
):
    """Resolve a remembered MRML node and validate its class if requested."""
    node_id = get_interaction_node_id(extension_name, workflow_id, step_id, repeat_index)
    if not node_id:
        return None
    try:
        import slicer
        node = slicer.mrmlScene.GetNodeByID(node_id)
    except Exception:
        return None
    if node is None:
        return None
    if node_class and hasattr(node, "IsA") and not node.IsA(node_class):
        return None
    return node


# =====================================================================
# Workflow metadata + precondition helpers
#
# These functions back the (small) prelude emitted in front of every
# generated workflow step template. They used to be inlined as Python
# source text in every dispatch result; moving them here lets the
# prelude pass choices/bindings/defaults as Python objects via the
# executor namespace instead of round-tripping them through source
# (which was both bloated and bug-prone — see json.dumps vs Python
# literals).
# =====================================================================

def _workflow_tokens(text):
    """Split a name/label into a lowercased token set, expanding camelCase."""
    import re
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return set(re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower()))


def _workflow_find_node(value, node_class, keywords):
    """Find a scene node matching ``value`` restricted to ``node_class``.

    Tries exact-name lookup first, then falls back to token-overlap scoring
    using ``keywords`` from the parameter binding. Returns None if nothing
    matches with positive score.
    """
    if not node_class:
        return None
    try:
        import slicer
    except ImportError:
        return None
    value = str(value or "")
    try:
        node = slicer.util.getNode(value)
        if node and node.IsA(node_class):
            return node
    except Exception:
        pass
    target_tokens = _workflow_tokens(value) | set(keywords or [])
    nodes = slicer.mrmlScene.GetNodesByClass(node_class)
    best_node = None
    best_score = -1
    for _i in range(nodes.GetNumberOfItems()):
        candidate = nodes.GetItemAsObject(_i)
        if candidate is None:
            continue
        score = len(target_tokens & _workflow_tokens(candidate.GetName()))
        if candidate.GetName() == value:
            score += 10
        if score > best_score:
            best_score = score
            best_node = candidate
    return best_node if best_score > 0 else None


def apply_workflow_metadata(parameterNode, choices, bindings, defaults):
    """Apply source-derived defaults and stored user choices to a parameter node.

    Mirrors the two application loops that used to be inlined into every
    generated prelude. Idempotent: only sets defaults when the parameter is
    empty, and only sets choices when an explicit value is stored.
    """
    if parameterNode is None:
        return
    if not bindings:
        return
    # Defaults first — only fill empty slots.
    for _role, _default in (defaults or {}).items():
        _binding = bindings.get(_role, {})
        if _binding.get("node_class", ""):
            continue
        try:
            _current = parameterNode.GetParameter(_role)
        except Exception:
            _current = ""
        if _current in (None, ""):
            parameterNode.SetParameter(_role, str(_default.get("value", "")))
    # Then stored user choices (override defaults).
    for _role, _binding in (bindings or {}).items():
        _value = (choices or {}).get(_role)
        if _value is None:
            continue
        _node_class = _binding.get("node_class", "")
        if _node_class:
            _node = _workflow_find_node(_value, _node_class, _binding.get("keywords", []))
            if _node is not None:
                parameterNode.SetNodeReferenceID(_role, _node.GetID())
        else:
            parameterNode.SetParameter(
                _role,
                "True" if _value is True
                else "False" if _value is False
                else str(_value),
            )


def _workflow_validate_polydata(_node, _role):
    if not hasattr(_node, "GetPolyData"):
        return
    _poly = _node.GetPolyData()
    if _poly is None:
        raise RuntimeError("Model node has no polydata: %s" % _role)
    try:
        if _poly.GetNumberOfPoints() <= 0:
            raise RuntimeError("Model node has empty polydata: %s" % _role)
    except AttributeError:
        pass


def _workflow_validate_markups(_node, _role, _min_points):
    if _min_points <= 0 or not hasattr(_node, "GetNumberOfControlPoints"):
        return
    _count = _node.GetNumberOfControlPoints()
    if _count < _min_points:
        raise RuntimeError(
            "Markup node %s needs at least %d control points, got %d"
            % (_role, _min_points, _count)
        )


def _workflow_condition_is_active(_parameter_node, _condition):
    _parameter = _condition.get("parameter", "")
    if not _parameter:
        return False
    _actual = _parameter_node.GetParameter(_parameter)
    _expected = str(_condition.get("value", ""))
    if _condition.get("operator") == "not_equals":
        return str(_actual) != _expected
    return str(_actual) == _expected


def _workflow_node_is_required(_parameter_node, _binding):
    _requirement = _binding.get("_workflow_requirement") or {"requirement": "optional_unknown"}
    _kind = _requirement.get("requirement", "optional_unknown")
    if _kind == "required":
        return True
    if _kind == "conditional":
        _groups = _requirement.get("condition_groups") or []
        if _groups:
            return any(
                all(_workflow_condition_is_active(_parameter_node, _c) for _c in _group)
                for _group in _groups
            )
        return any(
            _workflow_condition_is_active(_parameter_node, _c)
            for _c in (_requirement.get("conditions") or [])
        )
    return False


def validate_method_preconditions(logic, required_bindings):
    """Validate node references for an extension logic method call.

    Walks ``required_bindings`` (parameter binding dicts annotated with
    ``_workflow_requirement``) and raises ``RuntimeError`` if a required
    node reference is missing, has the wrong type, or is empty. Called
    from the prelude immediately before the corresponding ``logic.<method>()``
    invocation. No-op when ``required_bindings`` is empty.
    """
    if not logic or not required_bindings:
        return
    parameter_node = logic.getParameterNode()
    for _role, _binding in required_bindings.items():
        _node_class = _binding.get("node_class", "")
        if not _node_class:
            continue
        _accesses = set(_binding.get("accesses") or [])
        if "node_reference_read" not in _accesses:
            continue
        _node = parameter_node.GetNodeReference(_role)
        if _node is None:
            if _workflow_node_is_required(parameter_node, _binding):
                _requirement = _binding.get("_workflow_requirement") or {}
                _conditions = _requirement.get("conditions") or []
                if _conditions:
                    raise RuntimeError(
                        "[GeneratedWorkflowPrecondition] Missing conditional node reference: %s; active condition: %s"
                        % (_role, _conditions)
                    )
                raise RuntimeError(
                    "[GeneratedWorkflowPrecondition] Missing required node reference: %s"
                    % _role
                )
            continue
        if hasattr(_node, "IsA") and not _node.IsA(_node_class):
            raise RuntimeError(
                "Node reference %s has wrong type: expected %s" % (_role, _node_class)
            )
        if _node_class == "vtkMRMLModelNode":
            _workflow_validate_polydata(_node, _role)
        if "Markups" in _node_class:
            _workflow_validate_markups(
                _node, _role, int(_binding.get("min_control_points", 0) or 0)
            )
