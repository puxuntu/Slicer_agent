"""
ExtensionCLILoader - Auto-discovery and dynamic loading of extension CLI tools.

Scans Resources/extension_CLI/*/ for generated extension tool definitions,
loads their schemas, code templates, and prompt fragments at runtime,
and dispatches tool calls from the LLM to the appropriate code generator.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

try:
    import slicer
except ImportError:
    slicer = None

logger = logging.getLogger(__name__)

# Module-level cache: {extension_name: {"manifest": ..., "schemas": ..., "generators": ..., ...}}
_cli_cache: Dict[str, Dict] = {}
_cache_valid = False


def get_cli_base_dir() -> str:
    """Return the path to Resources/extension_CLI/."""
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(module_dir, "Resources", "extension_CLI")


def _ensure_cache():
    """Populate the CLI cache by scanning the extension_CLI directory."""
    global _cache_valid, _cli_cache
    if _cache_valid:
        return

    base_dir = get_cli_base_dir()
    if not os.path.isdir(base_dir):
        logger.info("No extension_CLI directory found at %s", base_dir)
        _cache_valid = True
        return

    for entry in os.listdir(base_dir):
        ext_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(ext_dir):
            continue

        manifest_path = os.path.join(ext_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            status = manifest.get("status", "unknown")
            if status != "validated":
                logger.info(
                    "Skipping %s (status=%s, not validated)", entry, status
                )
                continue

            schemas_path = os.path.join(ext_dir, "tool_schemas.json")
            generators_path = os.path.join(ext_dir, "code_generators.json")
            prompt_path = os.path.join(ext_dir, "prompt_fragment.md")

            schemas = []
            if os.path.isfile(schemas_path):
                with open(schemas_path, "r", encoding="utf-8") as f:
                    schemas = json.load(f)

            generators = []
            if os.path.isfile(generators_path):
                with open(generators_path, "r", encoding="utf-8") as f:
                    generators = json.load(f)

            prompt_fragment = ""
            if os.path.isfile(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_fragment = f.read()

            _cli_cache[entry] = {
                "manifest": manifest,
                "schemas": schemas,
                "generators": generators,
                "prompt_fragment": prompt_fragment,
                "dir": ext_dir,
            }

            tool_names = [s.get("function", {}).get("name", "?") for s in schemas]
            logger.info(
                "Loaded extension CLI: %s (tools=%s)", entry, tool_names
            )

        except Exception as e:
            logger.warning("Failed to load extension CLI %s: %s", entry, e)

    _cache_valid = True


def invalidate_cache():
    """Force a cache refresh on the next access."""
    global _cache_valid
    _cache_valid = False
    _cli_cache.clear()


def get_validated_extensions() -> Dict[str, Dict]:
    """Return the full cache of validated extension CLI data."""
    _ensure_cache()
    return dict(_cli_cache)


def get_dynamic_extension_tools() -> List[Dict]:
    """Return tool schemas from all validated extension CLIs."""
    _ensure_cache()
    tools = []
    for ext_data in _cli_cache.values():
        tools.extend(ext_data["schemas"])
    return tools


def get_extension_prompt_fragments() -> str:
    """Return concatenated prompt fragments from all validated extension CLIs."""
    _ensure_cache()
    fragments = []
    for ext_name, ext_data in _cli_cache.items():
        fragment = ext_data.get("prompt_fragment", "").strip()
        if fragment:
            fragments.append(fragment)
    return "\n\n".join(fragments)


def dispatch_extension_cli_tool(tool_name: str, arguments: Dict) -> Optional[Dict]:
    """
    Dispatch a tool call to the appropriate extension CLI code generator.

    Returns a result dict with 'tool', 'code', 'instruction', 'explanation' keys,
    or None if the tool_name is not found in any extension CLI.
    """
    _ensure_cache()

    for ext_name, ext_data in _cli_cache.items():
        for schema in ext_data["schemas"]:
            func_info = schema.get("function", {})
            if func_info.get("name") != tool_name:
                continue

            # Found the matching tool — generate code from template
            return _generate_from_template(
                ext_name, ext_data, tool_name, arguments
            )

    return None

def _fill_template(template_str: str, kwargs: Dict[str, str]) -> str:
    """Fill a template supporting both {name} and {name: default_value} syntax.

    - {name}: replaced with kwargs[name], KeyError if missing
    - {name: default}: replaced with kwargs[name] if present, else default
    - {{ and }} are preserved as literal braces
    - Brace pairs inside Python string/f-string literals are left untouched
    """
    import re

    # Protect escaped braces
    sentinel_l = "\x00LBRACE\x00"
    sentinel_r = "\x00RBRACE\x00"
    buf = template_str.replace("{{", sentinel_l).replace("}}", sentinel_r)

    def _replace(match):
        name = match.group(1)
        # Detect {name: default} pattern: group(2) is the ": default" wrapper
        try:
            default = match.group(4)
            has_default = default is not None
        except IndexError:
            has_default = False
            default = None

        if name in kwargs:
            return kwargs[name]
        if has_default:
            return default
        raise KeyError(name)

    # Mask out Python string literals so regex does not touch braces inside them.
    # Handles: '...', "...", f"...", r"...", b"...", and triple-quoted variants.
    _STRING_PLACEHOLDER = "\x00STR\x00"
    string_ranges = []
    for m in re.finditer(
        r'(?:[fFrRbBuU]{0,2})("""|\'\'\'|"|\')(.*?)\1',
        buf, re.DOTALL
    ):
        string_ranges.append((m.start(), m.end()))

    def _in_string(pos):
        return any(s <= pos < e for s, e in string_ranges)

    def _replace_if_not_in_string(match):
        if _in_string(match.start()):
            return match.group(0)
        return _replace(match)

    # Match {name: default} — skip those inside string literals
    buf = re.sub(r'\{(\w+)((: )(.*?))\}', _replace_if_not_in_string, buf)
    # Match remaining simple {name} placeholders, but skip those inside strings
    buf = re.sub(r'\{(\w+)\}', _replace_if_not_in_string, buf)

    # Restore escaped braces
    buf = buf.replace(sentinel_l, "{").replace(sentinel_r, "}")
    return buf


def _generate_from_template(
    ext_name: str,
    ext_data: Dict,
    tool_name: str,
    arguments: Dict,
) -> Dict:
    """
    Fill a code template with the provided arguments and return the result dict.
    For interactive workflows, delegates to dispatch_workflow_step.
    """
    ext_dir = ext_data["dir"]
    generators = ext_data["generators"]
    manifest = ext_data["manifest"]

    # Route interactive workflows to the workflow dispatch
    if manifest.get("workflow_type") == "interactive":
        return dispatch_workflow_step(ext_name, ext_data, tool_name, arguments)

    # Find the matching generator entry
    stage = arguments.get("stage", "")
    best_match = None
    for gen in generators:
        if gen.get("tool_name") != tool_name:
            continue
        # If stage is specified, match it; otherwise use the first generator
        gen_stage = gen.get("param_signature", {}).get("stage", "")
        if stage and gen_stage == stage:
            best_match = gen
            break
        elif not best_match:
            best_match = gen

    if not best_match:
        return {
            "error": f"No code generator found for tool={tool_name} stage={stage}"
        }

    # Read the template file
    template_rel = best_match.get("template_file", "")
    template_path = os.path.join(ext_dir, template_rel)
    if not os.path.isfile(template_path):
        return {
            "error": f"Template file not found: {template_rel}"
        }

    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    # Build format kwargs from arguments
    format_kwargs = {}
    for key, value in arguments.items():
        # Convert Python values to safe Python source literals for template embedding
        if isinstance(value, str):
            format_kwargs[key] = repr(value)
        elif isinstance(value, bool):
            format_kwargs[key] = repr(value)
        elif isinstance(value, (list, dict, int, float)):
            format_kwargs[key] = repr(value)
        elif value is None:
            format_kwargs[key] = "None"
        else:
            format_kwargs[key] = repr(value)

    # Normalize argument names: add camelCase aliases for snake_case keys.
    # Schema params use snake_case (text_prompts) but template placeholders
    # use Python param names (textPrompts). Both must be available.
    aliases = {}
    for key in list(format_kwargs.keys()):
        if "_" in key:
            parts = key.split("_")
            camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
            if camel not in format_kwargs:
                aliases[camel] = format_kwargs[key]
    format_kwargs.update(aliases)

    # Add default values for common placeholders
    format_kwargs.setdefault("volume_node_name", "''")
    vn_raw = arguments.get("volume_node_name", "")
    if not vn_raw:
        format_kwargs["vol_lookup"] = (
            "inputVolume = slicer.mrmlScene.GetFirstNodeByClass"
            "('vtkMRMLScalarVolumeNode')"
        )
    else:
        format_kwargs["vol_lookup"] = (
            f'inputVolume = slicer.util.getNode("{vn_raw}")'
        )

    # Pre-scan template and pre-fill ALL placeholder values into format_kwargs.
    # This ensures _fill_template only does simple substitution — no default-handling
    # logic needed, which avoids regex edge cases with complex default expressions.
    import re as _re
    _scan_sentinel_l = "\x00SCANLBRACE\x00"
    _scan_sentinel_r = "\x00SCANRBRACE\x00"
    _scan_buf = template_str.replace("{{", _scan_sentinel_l).replace("}}", _scan_sentinel_r)
    for _m in _re.finditer(r'\{(\w+)((: )(.*?))?\}', _scan_buf):
        _pname = _m.group(1)
        if _pname in format_kwargs:
            continue  # Already have a value (from LLM args or camelCase alias)
        _has_default = _m.group(2) is not None
        if _has_default and _m.group(4) is not None:
            # Use the inline default from the template (e.g. logic.defaultModelPath())
            format_kwargs[_pname] = _m.group(4)
        else:
            # No default — use heuristic fallback based on parameter name
            _plower = _pname.lower()
            if "path" in _plower or "dir" in _plower:
                format_kwargs[_pname] = "''"
            elif "name" in _plower:
                format_kwargs[_pname] = "''"
            elif any(k in _plower for k in ("gpu", "use", "show", "enable", "verbose", "debug")):
                format_kwargs[_pname] = "True"
            elif any(k in _plower for k in ("callback", "handler")):
                format_kwargs[_pname] = "None"
            else:
                format_kwargs[_pname] = "None"

    # Fill the template using custom engine that supports {name} and {name: default}
    try:
        code = _fill_template(template_str, format_kwargs)
    except KeyError as e:
        return {"error": f"Template placeholder not filled: {e}"}

    return {
        "tool": tool_name,
        "stage": stage,
        "code": code,
        "instruction": (
            "OUTPUT THE 'code' FIELD ABOVE VERBATIM INSIDE A ```python BLOCK "
            "AS YOUR NEXT RESPONSE. Do not modify the code. "
            "Do not write analysis or explanation before the code block."
        ),
        "explanation": best_match.get("description", ""),
        "requirements": best_match.get("requirements", []),
    }


# Track completed/skipped steps per workflow so the skip handler can compute
# the next step without needing a reference to the WorkflowOrchestrator.
_workflow_completed_steps: Dict[str, set] = {}


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
            is_optional = (
                step.get("step_type") == "branch"
                or step.get("is_optional", False)
            )
            return {
                "step_id": sid,
                "step_type": step.get("step_type", "automated"),
                "description": step.get("description", ""),
                "is_optional": is_optional,
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

    Reads the workflow.json, matches the requested step, and returns
    either pre-interaction code (for interactive steps), full code (for
    automated steps), or branch info (for branch steps).

    Returns:
        Dict with type-specific fields:
        - type: "interactive" | "automated" | "branch" | "error"
        - For interactive: pre_code, interaction descriptor, instructions
        - For automated: code, instruction
        - For branch: condition, branches
    """
    ext_dir = ext_data["dir"]
    generators = ext_data["generators"]

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

    step_type = target_step.get("step_type", "automated")

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

    if user_action == "skip":
        # Mark the step as completed so _find_next_step_local skips it
        done = _workflow_completed_steps.setdefault(ext_name, set())
        done.add(workflow_step)
        # Find the next step whose dependencies are all satisfied
        next_step = _find_next_step_local(workflow_graph, done)
        result = {
            "tool": tool_name,
            "type": "skipped",
            "step_id": workflow_step,
            "message": f"Step '{workflow_step}' skipped.",
        }
        if next_step:
            result["next_step"] = next_step
            is_opt = next_step.get("is_optional", False)
            if is_opt:
                result["instruction"] = (
                    f"Next step '{next_step['step_id']}' is optional: "
                    f"{next_step['description']} "
                    f"Ask the user if they want to proceed. "
                    f"If yes, call {tool_name} with workflow_step='{next_step['step_id']}' user_action='start'. "
                    f"If no, call with workflow_step='{next_step['step_id']}' user_action='skip'."
                )
            else:
                result["instruction"] = (
                    f"Proceed by calling {tool_name} with "
                    f"workflow_step='{next_step['step_id']}' user_action='start'."
                )
        else:
            result["instruction"] = "All remaining optional steps have been handled. Workflow is complete."
        return result

    if step_type == "automated":
        # Fill and return the code template
        if not target_gen:
            return {"error": f"No generator for automated step '{workflow_step}'"}

        template_rel = target_gen.get("template_file", "")
        template_path = os.path.join(ext_dir, template_rel)
        if not os.path.isfile(template_path):
            return {"error": f"Template not found: {template_rel}"}

        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        # Build format kwargs from arguments
        format_kwargs = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                format_kwargs[key] = repr(value)
            elif value is None:
                format_kwargs[key] = "None"
            else:
                format_kwargs[key] = repr(value)

        try:
            code = _fill_template(template_str, format_kwargs)
        except KeyError as e:
            return {"error": f"Template placeholder not filled: {e}"}

        return {
            "tool": tool_name,
            "type": "automated",
            "code": code,
            "instruction": (
                "STOP. Do NOT make any more tool calls. "
                "Your NEXT response must be an ```agent_plan JSON block followed by a ```python block "
                "containing the 'code' field above VERBATIM. "
                "Do NOT call any more tools until this code has been executed."
            ),
            "explanation": target_step.get("description", ""),
            "step_id": workflow_step,
        }

    elif step_type == "interactive":
        # Return pre-interaction code for the LLM to output
        if not target_gen:
            return {"error": f"No generator for interactive step '{workflow_step}'"}

        pre_template_rel = target_gen.get("pre_template_file", "")
        if pre_template_rel:
            pre_template_path = os.path.join(ext_dir, pre_template_rel)
            if os.path.isfile(pre_template_path):
                with open(pre_template_path, "r", encoding="utf-8") as f:
                    pre_template = f.read()
            else:
                pre_template = None
        else:
            pre_template = None

        if pre_template:
            format_kwargs = {}
            for key, value in arguments.items():
                if isinstance(value, str):
                    format_kwargs[key] = repr(value)
                elif value is None:
                    format_kwargs[key] = "None"
                else:
                    format_kwargs[key] = repr(value)
            try:
                pre_code = _fill_template(pre_template, format_kwargs)
            except KeyError as e:
                return {"error": f"Pre-template placeholder not filled: {e}"}
        else:
            pre_code = None

        interaction_desc = target_gen.get("interaction_descriptor", {})

        return {
            "tool": tool_name,
            "type": "interactive",
            "pre_code": pre_code,
            "code": pre_code,
            "instruction": (
                "STOP. Do NOT make any more tool calls. "
                "Your NEXT response must be an ```agent_plan JSON block followed by a ```python block "
                "containing the 'code' field above VERBATIM. "
                "After the code executes, tell the user to perform the interaction "
                "described in 'interaction_instructions' and click Done when finished. "
                "Do NOT call any more tools until the user clicks Done."
            ),
            "interaction_instructions": interaction_desc.get("placement_instructions", ""),
            "interaction_type": interaction_desc.get("interaction_type", ""),
            "step_id": workflow_step,
            "explanation": target_step.get("description", ""),
        }

    elif step_type == "branch":
        condition = target_step.get("condition", "Optional step")
        return {
            "tool": tool_name,
            "type": "branch",
            "step_id": workflow_step,
            "condition": condition,
            "branches": target_step.get("branches", {}),
            "explanation": target_step.get("description", ""),
            "instruction": (
                f"Ask the user: '{condition}' "
                f"If yes, call {tool_name} with workflow_step='{workflow_step}' "
                f"user_action='start'. "
                f"If no, call with workflow_step='{workflow_step}' "
                f"user_action='skip'."
            ),
        }

    return {"error": f"Unknown step type: {step_type}"}


def get_all_cli_manifests() -> List[Dict]:
    """Return all extension CLI manifests (for UI display)."""
    _ensure_cache()
    results = []
    for ext_name, ext_data in _cli_cache.items():
        manifest = ext_data["manifest"].copy()
        manifest["cli_dir"] = ext_data["dir"]
        results.append(manifest)

    # Also scan for non-validated CLIs
    base_dir = get_cli_base_dir()
    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            if entry in _cli_cache:
                continue
            ext_dir = os.path.join(base_dir, entry)
            manifest_path = os.path.join(ext_dir, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    m["cli_dir"] = ext_dir
                    results.append(m)
                except Exception:
                    pass
    return results


def delete_cli(extension_name: str) -> bool:
    """Delete an extension CLI directory."""
    import shutil

    base_dir = get_cli_base_dir()
    ext_dir = os.path.join(base_dir, extension_name)
    if os.path.isdir(ext_dir):
        shutil.rmtree(ext_dir)
        invalidate_cache()
        logger.info("Deleted extension CLI: %s", extension_name)
        return True
    return False


def discover_installed_extensions() -> List[Dict]:
    """
    Discover Slicer extensions from the Extension Manager, additional module
    paths, and loaded scripted modules.

    Returns a list of dicts with:
    - name: extension display name
    - module_name: Python module name
    - install_path: file system path to the extension root
    - source_path: file system path usable for source analysis
    - has_python: whether the extension has Python modules
    - cli_status: 'validated', 'draft', 'failed', or None
    """
    results = []
    seen = set()

    def _add(name, install_path, source_path, source_type="extension_manager"):
        if name in seen:
            return
        seen.add(name)

        has_python = False
        if os.path.isdir(source_path):
            for root, _dirs, files in os.walk(source_path):
                if any(f.endswith(".py") for f in files):
                    has_python = True
                break

        cli_status = None
        cli_dir = os.path.join(get_cli_base_dir(), name)
        manifest_path = os.path.join(cli_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                cli_status = m.get("status")
            except Exception:
                pass

        results.append({
            "name": name,
            "module_name": name,
            "install_path": install_path,
            "source_path": source_path,
            "has_python": has_python,
            "cli_status": cli_status,
            "source_type": source_type,
        })

    # 1. Extensions installed via Extension Manager
    try:
        em = slicer.app.extensionsManagerModel()
        for ext_name in em.installedExtensions:
            install_path = em.extensionInstallPath(ext_name)
            _add(ext_name, install_path, install_path)
    except Exception as e:
        logger.warning("Failed to scan Extension Manager: %s", e)

    # 2. Additional module paths (loaded via Edit > Application Settings > Modules)
    try:
        settings = slicer.app.revisionUserSettings()
        raw_value = settings.value("Modules/AdditionalPaths")
        if raw_value is None:
            additionalPaths = []
        elif isinstance(raw_value, str):
            additionalPaths = [raw_value]
        elif isinstance(raw_value, (list, tuple)):
            additionalPaths = list(raw_value)
        else:
            additionalPaths = []

        # Convert relative paths to absolute
        absolute_paths = []
        for p in additionalPaths:
            abs_p = slicer.app.toSlicerHomeAbsolutePath(p)
            absolute_paths.append(abs_p)

        for mod_path in absolute_paths:
            if not os.path.isdir(mod_path):
                continue
            # Each additional path may point to a module directory directly,
            # or to a parent containing multiple module directories.
            if any(f.endswith(".py") for f in os.listdir(mod_path)):
                ext_name = os.path.basename(mod_path)
                _add(ext_name, mod_path, mod_path, source_type="additional_paths")
            else:
                # Scan subdirectories for extension-like folders
                for entry in os.listdir(mod_path):
                    subdir = os.path.join(mod_path, entry)
                    if not os.path.isdir(subdir):
                        continue
                    for f in os.listdir(subdir):
                        if f.endswith(".py"):
                            _add(entry, subdir, subdir, source_type="additional_paths")
                            break
    except Exception as e:
        logger.warning("Failed to scan additional module paths: %s", e)

    # 3. Loaded scripted modules (fallback — catches everything that's loaded)
    try:
        loaded_modules = slicer.util.moduleNames()
        for module_name in loaded_modules:
            if module_name in seen:
                continue
            try:
                mod_path = slicer.util.modulePath(module_name)
                if not mod_path or not os.path.isfile(mod_path):
                    continue
                module_dir = os.path.dirname(mod_path)
                parent_dir = os.path.dirname(module_dir)
                _add(module_name, parent_dir, module_dir, source_type="loaded_modules")
            except Exception:
                continue
    except Exception as e:
        logger.warning("Failed to scan loaded modules: %s", e)

    return results


def save_cli_package(
    extension_name: str,
    manifest: Dict,
    tool_schemas: List[Dict],
    code_generators: List[Dict],
    templates: Dict[str, str],
    prompt_fragment: str,
    generation_log_entry: Optional[Dict] = None,
) -> str:
    """
    Save a complete extension CLI package to disk.

    Args:
        extension_name: Name for the CLI directory
        manifest: Manifest dict (will be written as manifest.json)
        tool_schemas: List of OpenAI tool schema dicts
        code_generators: List of generator metadata dicts
        templates: Dict mapping template_file names to template content strings
        prompt_fragment: Markdown prompt fragment
        generation_log_entry: Optional entry to append to generation_log.json

    Returns:
        Path to the created CLI directory
    """
    base_dir = get_cli_base_dir()
    os.makedirs(base_dir, exist_ok=True)

    ext_dir = os.path.join(base_dir, extension_name)
    os.makedirs(ext_dir, exist_ok=True)
    os.makedirs(os.path.join(ext_dir, "templates"), exist_ok=True)

    # Write manifest
    with open(os.path.join(ext_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write tool schemas
    with open(os.path.join(ext_dir, "tool_schemas.json"), "w", encoding="utf-8") as f:
        json.dump(tool_schemas, f, indent=2, ensure_ascii=False)

    # Write code generators
    with open(os.path.join(ext_dir, "code_generators.json"), "w", encoding="utf-8") as f:
        json.dump(code_generators, f, indent=2, ensure_ascii=False)

    # Write templates
    for tpl_name, tpl_content in templates.items():
        if tpl_name == "workflow.json":
            # Workflow graph goes at CLI root, not in templates/
            tpl_path = os.path.join(ext_dir, tpl_name)
        else:
            # tpl_name may already include "templates/" prefix — strip it
            clean_name = tpl_name.removeprefix("templates/")
            tpl_path = os.path.join(ext_dir, "templates", clean_name)
        os.makedirs(os.path.dirname(tpl_path), exist_ok=True)
        with open(tpl_path, "w", encoding="utf-8") as f:
            f.write(tpl_content)

    # Write prompt fragment
    with open(os.path.join(ext_dir, "prompt_fragment.md"), "w", encoding="utf-8") as f:
        f.write(prompt_fragment)

    # Append generation log entry
    if generation_log_entry:
        log_path = os.path.join(ext_dir, "generation_log.json")
        log_entries = []
        if os.path.isfile(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_entries = json.load(f)
            except Exception:
                pass
        log_entries.append(generation_log_entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)

    # Invalidate cache so the new CLI is picked up
    invalidate_cache()

    logger.info("Saved extension CLI package: %s", ext_dir)
    return ext_dir
