"""
WorkflowOrchestrator - Runtime state machine for guided interactive extension workflows.

Coordinates between:
- The LLM agent (decides which step to execute)
- The user (performs interactive 3D operations)
- Slicer (runs code, creates nodes, enters placement modes)
- The InteractionManager (low-level Slicer interaction)
- The SafeExecutor (runs automated step code)
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import slicer
import qt

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Describes a single step in an interactive workflow."""
    step_id: str
    phase: str
    step_type: str              # "automated" | "interactive" | "branch"
    description: str

    # Interactive step fields
    interaction_type: Optional[str] = None     # "curve", "plane", "line", "fiducial", "point_list"
    node_class: Optional[str] = None
    placement_mode: Optional[str] = None       # "indefinite" | "single"
    placement_instructions: Optional[str] = None
    validation_rules: Optional[List[str]] = None
    min_control_points: int = 0
    reactive_chains: Optional[List[Dict]] = None
    display_properties: Optional[Dict] = None

    # Automated step fields
    method_name: Optional[str] = None

    # Branch step fields
    condition: Optional[str] = None
    branches: Optional[Dict[str, str]] = None

    # Dependency fields
    depends_on: List[str] = field(default_factory=list)
    produces_nodes: List[str] = field(default_factory=list)

    # Template references
    pre_template: Optional[str] = None    # Template file for pre-interaction setup
    post_template: Optional[str] = None   # Template file for post-interaction processing
    code_template: Optional[str] = None   # Template file for fully automated steps

    @classmethod
    def from_dict(cls, d: Dict) -> "WorkflowStep":
        return cls(
            step_id=d["step_id"],
            phase=d.get("phase", d["step_id"]),
            step_type=d.get("step_type", "automated"),
            description=d.get("description", ""),
            interaction_type=d.get("interaction_type"),
            node_class=d.get("node_class"),
            placement_mode=d.get("placement_mode", "indefinite"),
            placement_instructions=d.get("placement_instructions"),
            validation_rules=d.get("validation_rules", []),
            min_control_points=d.get("min_control_points", 0),
            reactive_chains=d.get("reactive_chains", []),
            display_properties=d.get("display_properties"),
            method_name=d.get("method_name"),
            condition=d.get("condition"),
            branches=d.get("branches"),
            depends_on=d.get("depends_on", []),
            produces_nodes=d.get("produces_nodes", []),
            pre_template=d.get("pre_template"),
            post_template=d.get("post_template"),
            code_template=d.get("code_template"),
        )


@dataclass
class WorkflowState:
    """Runtime state of an active workflow instance."""
    workflow_id: str
    extension_name: str
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    node_registry: Dict[str, str] = field(default_factory=dict)  # var_name -> node_id
    step_results: Dict[str, Dict] = field(default_factory=dict)  # step_id -> result
    status: str = "running"  # "running", "waiting_for_user", "completed", "error", "cancelled"
    error_message: Optional[str] = None


class WorkflowOrchestrator:
    """
    Manages the execution of interactive extension workflows.

    Usage:
        orchestrator = WorkflowOrchestrator(executor, interaction_manager)
        state = orchestrator.start_workflow("BoneReconstructionPlanner", workflow_graph)
        result = orchestrator.execute_step(state.workflow_id, "data_loading", {})
        # ... for interactive steps, user interacts, then:
        result = orchestrator.complete_interaction(state.workflow_id)
    """

    def __init__(self, executor=None, interaction_manager=None):
        """
        Args:
            executor: SafeExecutor instance for running code.
            interaction_manager: InteractionManager instance for 3D interactions.
        """
        self._executor = executor
        self._interaction_manager = interaction_manager
        self._active_workflows: Dict[str, WorkflowState] = {}
        self._workflow_graphs: Dict[str, Dict] = {}  # extension_name -> workflow graph
        self._workflow_steps: Dict[str, Dict[str, WorkflowStep]] = {}  # ext -> {step_id: step}

    @property
    def interaction_manager(self):
        return self._interaction_manager

    @interaction_manager.setter
    def interaction_manager(self, value):
        self._interaction_manager = value

    @property
    def executor(self):
        return self._executor

    @executor.setter
    def executor(self, value):
        self._executor = value

    def load_workflow_graph(self, extension_name: str, workflow_graph: Dict) -> None:
        """
        Load and cache a workflow graph for an extension.

        Args:
            extension_name: Extension identifier.
            workflow_graph: Parsed workflow.json dict with "steps" and "phases".
        """
        steps = {}
        for step_dict in workflow_graph.get("steps", []):
            step = WorkflowStep.from_dict(step_dict)
            steps[step.step_id] = step
        self._workflow_graphs[extension_name] = workflow_graph
        self._workflow_steps[extension_name] = steps
        logger.info(
            f"[WorkflowOrchestrator] Loaded workflow for {extension_name}: "
            f"{len(steps)} steps"
        )

    def start_workflow(self, extension_name: str, workflow_graph: Optional[Dict] = None) -> WorkflowState:
        """
        Initialize a new workflow instance.

        Args:
            extension_name: Extension to run.
            workflow_graph: Optional workflow dict (if not pre-loaded).

        Returns:
            WorkflowState for the new instance.
        """
        if workflow_graph:
            self.load_workflow_graph(extension_name, workflow_graph)

        if extension_name not in self._workflow_steps:
            raise ValueError(
                f"No workflow graph loaded for '{extension_name}'. "
                f"Call load_workflow_graph() first."
            )

        workflow_id = f"{extension_name}_{uuid.uuid4().hex[:8]}"
        state = WorkflowState(
            workflow_id=workflow_id,
            extension_name=extension_name,
        )
        self._active_workflows[workflow_id] = state

        # Find the first step (no dependencies)
        steps = self._workflow_steps[extension_name]
        for step_id, step in steps.items():
            if not step.depends_on:
                state.current_step = step_id
                break

        logger.info(
            f"[WorkflowOrchestrator] Started workflow {workflow_id} "
            f"for {extension_name}, first step: {state.current_step}"
        )
        return state

    def execute_step(
        self,
        workflow_id: str,
        step_id: str,
        arguments: Optional[Dict] = None,
        template_filler: Optional[Callable] = None,
    ) -> Dict:
        """
        Execute a single workflow step.

        For automated steps: runs code and returns result.
        For interactive steps: runs pre-interaction code and returns
        interaction instructions (does NOT complete the step).

        Args:
            workflow_id: Active workflow instance ID.
            step_id: Which step to execute.
            arguments: Arguments from the LLM tool call.
            template_filler: Callable(template_path, args) -> code_str.
                Used to fill template placeholders.

        Returns:
            Dict with keys depending on step type:
            - automated: {"type": "automated", "code": ..., "result": ...}
            - interactive: {"type": "interactive", "pre_code": ...,
                "interaction": ..., "step_info": ...}
            - branch: {"type": "branch", "condition": ..., "branches": ...}
        """
        state = self._get_state(workflow_id)
        if not state:
            return {"type": "error", "error": f"Workflow {workflow_id} not found"}

        steps = self._workflow_steps.get(state.extension_name, {})
        step = steps.get(step_id)
        if not step:
            return {"type": "error", "error": f"Step '{step_id}' not found"}

        args = arguments or {}

        if step.step_type == "automated":
            return self._execute_automated_step(state, step, args, template_filler)
        elif step.step_type == "interactive":
            return self._execute_interactive_step(state, step, args, template_filler)
        elif step.step_type == "branch":
            return self._handle_branch_step(state, step, args)
        else:
            return {"type": "error", "error": f"Unknown step_type: {step.step_type}"}

    def complete_interaction(
        self,
        workflow_id: str,
        user_message: Optional[str] = None,
        template_filler: Optional[Callable] = None,
    ) -> Dict:
        """
        Called when the user signals completion of an interactive step.
        Runs post-interaction code and advances the workflow.

        Args:
            workflow_id: Active workflow instance.
            user_message: Optional message from the user (e.g., why they skipped).
            template_filler: Callable for filling post-interaction templates.

        Returns:
            Dict with post-execution results and the next step info.
        """
        state = self._get_state(workflow_id)
        if not state:
            return {"type": "error", "error": f"Workflow {workflow_id} not found"}
        if state.status != "waiting_for_user":
            return {"type": "error", "error": "Not in waiting state"}

        steps = self._workflow_steps.get(state.extension_name, {})
        step = steps.get(state.current_step)
        if not step:
            return {"type": "error", "error": f"Current step '{state.current_step}' not found"}

        # Exit placement mode
        if self._interaction_manager:
            self._interaction_manager.exit_placement_mode()

        # Validate user input
        node_id = state.node_registry.get(step.step_id)
        if node_id and self._interaction_manager:
            valid, error_msg = self._interaction_manager.validate_node(
                node_id, min_points=step.min_control_points
            )
            if not valid:
                return {
                    "type": "error",
                    "error": error_msg,
                    "step_id": step.step_id,
                    "retry": True,
                }

        # Run post-interaction code
        post_code = None
        if template_filler and step.post_template:
            post_code = template_filler(step.post_template, {
                **state.node_registry,
                "user_message": user_message or "",
            })

        result = {"type": "interaction_complete", "step_id": step.step_id}
        if post_code and self._executor:
            exec_result = self._executor.execute(post_code)
            result["post_result"] = exec_result
            if not exec_result.get("success"):
                result["type"] = "error"
                result["error"] = exec_result.get("error", "Post-interaction code failed")
                return result
            # Capture any node IDs from post-interaction code
            self._capture_node_ids(state, step, exec_result)

        # Mark step complete and advance
        state.completed_steps.append(step.step_id)
        state.step_results[step.step_id] = {"status": "completed"}
        state.status = "running"

        # Find next step
        next_step = self._find_next_step(state)
        if next_step:
            state.current_step = next_step.step_id
            result["next_step"] = {
                "step_id": next_step.step_id,
                "step_type": next_step.step_type,
                "description": next_step.description,
                "interaction_type": next_step.interaction_type,
            }
        else:
            state.current_step = None
            state.status = "completed"
            result["workflow_completed"] = True

        logger.info(
            f"[WorkflowOrchestrator] Completed step '{step.step_id}', "
            f"next: {state.current_step or 'DONE'}"
        )
        return result

    def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a workflow and clean up all resources."""
        state = self._get_state(workflow_id)
        if not state:
            return

        state.status = "cancelled"

        # Exit placement mode
        if self._interaction_manager:
            self._interaction_manager.exit_placement_mode()
            self._interaction_manager.cleanup()

        # Clean up created nodes
        if self._interaction_manager:
            self._interaction_manager.cleanup_all_created_nodes()

        del self._active_workflows[workflow_id]
        logger.info(f"[WorkflowOrchestrator] Cancelled workflow {workflow_id}")

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Return serializable workflow state."""
        state = self._get_state(workflow_id)
        if not state:
            return None
        return {
            "workflow_id": state.workflow_id,
            "extension_name": state.extension_name,
            "current_step": state.current_step,
            "completed_steps": state.completed_steps,
            "status": state.status,
            "node_registry": state.node_registry,
        }

    def get_active_workflow_prompt_fragment(self) -> str:
        """
        Generate a prompt fragment describing active workflow state
        for injection into the LLM system prompt.
        """
        active = [
            (wid, s) for wid, s in self._active_workflows.items()
            if s.status in ("running", "waiting_for_user")
        ]
        if not active:
            return ""

        lines = []
        for workflow_id, state in active:
            steps = self._workflow_steps.get(state.extension_name, {})
            lines.append(f"### Active Workflow: {state.extension_name}")
            lines.append(f"- Workflow ID: {workflow_id}")
            lines.append(f"- Status: {state.status}")
            lines.append(f"- Current step: {state.current_step}")
            lines.append(f"- Completed steps: {', '.join(state.completed_steps) or 'none'}")

            if state.current_step and state.current_step in steps:
                step = steps[state.current_step]
                lines.append(f"- Step description: {step.description}")
                if step.step_type == "interactive":
                    lines.append(f"- Interaction type: {step.interaction_type}")
                    if step.placement_instructions:
                        lines.append(f"- Instructions: {step.placement_instructions}")

            # Show remaining steps
            remaining = [
                sid for sid in steps
                if sid not in state.completed_steps and sid != state.current_step
            ]
            if remaining:
                lines.append(f"- Remaining steps: {', '.join(remaining)}")

            lines.append("")

        return "\n".join(lines)

    def get_active_workflow_id(self) -> Optional[str]:
        """Return the ID of the first active (non-completed) workflow, or None."""
        for wid, state in self._active_workflows.items():
            if state.status in ("running", "waiting_for_user"):
                return wid
        return None

    def is_workflow_active(self) -> bool:
        """Check if any workflow is currently running or waiting."""
        return any(
            s.status in ("running", "waiting_for_user")
            for s in self._active_workflows.values()
        )

    def get_step(self, extension_name: str, step_id: str) -> Optional[WorkflowStep]:
        """Look up a step definition."""
        return self._workflow_steps.get(extension_name, {}).get(step_id)

    def get_ordered_steps(self, extension_name: str) -> List[WorkflowStep]:
        """Return steps in dependency order."""
        steps = self._workflow_steps.get(extension_name, {})
        if not steps:
            return []

        ordered = []
        visited = set()
        visiting = set()

        def visit(sid):
            if sid in visited:
                return
            if sid in visiting:
                return  # cycle guard
            visiting.add(sid)
            step = steps.get(sid)
            if step:
                for dep in step.depends_on:
                    visit(dep)
                ordered.append(step)
            visiting.discard(sid)
            visited.add(sid)

        for sid in steps:
            visit(sid)

        return ordered

    # --- Private methods ---

    def _get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        return self._active_workflows.get(workflow_id)

    def _execute_automated_step(
        self,
        state: WorkflowState,
        step: WorkflowStep,
        args: Dict,
        template_filler: Optional[Callable],
    ) -> Dict:
        """Execute a fully automated step."""
        code = None
        if template_filler and step.code_template:
            fill_args = {**state.node_registry, **args}
            code = template_filler(step.code_template, fill_args)

        if not code:
            return {
                "type": "error",
                "error": f"No code template for automated step '{step.step_id}'",
            }

        result = {"type": "automated", "code": code, "step_id": step.step_id}

        if self._executor:
            exec_result = self._executor.execute(code)
            result["exec_result"] = exec_result
            if not exec_result.get("success"):
                result["type"] = "error"
                result["error"] = exec_result.get("error", "Execution failed")
                return result

            self._capture_node_ids(state, step, exec_result)

        # Mark complete and find next
        state.completed_steps.append(step.step_id)
        state.step_results[step.step_id] = {"status": "completed"}

        next_step = self._find_next_step(state)
        if next_step:
            state.current_step = next_step.step_id
            result["next_step"] = {
                "step_id": next_step.step_id,
                "step_type": next_step.step_type,
                "description": next_step.description,
                "interaction_type": next_step.interaction_type,
            }
        else:
            state.current_step = None
            state.status = "completed"
            result["workflow_completed"] = True

        return result

    def _execute_interactive_step(
        self,
        state: WorkflowState,
        step: WorkflowStep,
        args: Dict,
        template_filler: Optional[Callable],
    ) -> Dict:
        """
        Execute the pre-interaction portion of an interactive step.
        Returns instructions and pre-code; does NOT complete the step.
        """
        pre_code = None
        if template_filler and step.pre_template:
            fill_args = {**state.node_registry, **args}
            pre_code = template_filler(step.pre_template, fill_args)

        result = {
            "type": "interactive",
            "step_id": step.step_id,
            "interaction": {
                "interaction_type": step.interaction_type,
                "node_class": step.node_class,
                "placement_instructions": step.placement_instructions,
                "validation_rules": step.validation_rules,
            },
            "step_info": {
                "step_id": step.step_id,
                "description": step.description,
            },
        }

        if pre_code:
            result["pre_code"] = pre_code
        else:
            # Generate minimal node creation code if no template
            result["pre_code"] = self._generate_minimal_placement_code(step)

        state.status = "waiting_for_user"
        logger.info(
            f"[WorkflowOrchestrator] Entering wait state for step '{step.step_id}' "
            f"(interaction: {step.interaction_type})"
        )
        return result

    def _handle_branch_step(
        self,
        state: WorkflowState,
        step: WorkflowStep,
        args: Dict,
    ) -> Dict:
        """Handle a branch decision point."""
        user_choice = args.get("choice") or args.get("user_action")

        return {
            "type": "branch",
            "step_id": step.step_id,
            "condition": step.condition,
            "branches": step.branches,
            "user_choice": user_choice,
            "description": step.description,
        }

    def _resolve_branch(
        self,
        workflow_id: str,
        step_id: str,
        choice: str,
    ) -> Dict:
        """
        Resolve a branch step and set the next step accordingly.

        Args:
            workflow_id: Active workflow ID.
            step_id: The branch step ID.
            choice: The chosen branch key.

        Returns:
            Dict with next step info.
        """
        state = self._get_state(workflow_id)
        if not state:
            return {"type": "error", "error": f"Workflow {workflow_id} not found"}

        steps = self._workflow_steps.get(state.extension_name, {})
        step = steps.get(step_id)
        if not step or step.step_type != "branch":
            return {"type": "error", "error": f"Step '{step_id}' is not a branch step"}

        next_step_id = step.branches.get(choice) if step.branches else None
        if not next_step_id:
            # Skip this branch
            state.completed_steps.append(step_id)
            next_step = self._find_next_step(state)
            if next_step:
                state.current_step = next_step.step_id
            else:
                state.current_step = None
                state.status = "completed"
            return {"type": "branch_resolved", "skipped": True}

        state.completed_steps.append(step_id)
        state.current_step = next_step_id
        next_step = steps.get(next_step_id)

        return {
            "type": "branch_resolved",
            "choice": choice,
            "next_step": {
                "step_id": next_step_id,
                "step_type": next_step.step_type if next_step else None,
                "description": next_step.description if next_step else "",
            },
        }

    def _find_next_step(self, state: WorkflowState) -> Optional[WorkflowStep]:
        """Find the next step whose dependencies are all satisfied."""
        steps = self._workflow_steps.get(state.extension_name, {})
        for step_id, step in steps.items():
            if step_id in state.completed_steps:
                continue
            if step_id == state.current_step:
                continue
            if all(dep in state.completed_steps for dep in step.depends_on):
                return step
        return None

    def _capture_node_ids(self, state: WorkflowState, step: WorkflowStep, exec_result: Dict) -> None:
        """
        Try to capture MRML node IDs created during execution.

        Looks for variables in the __main__ namespace matching the
        produces_nodes list.
        """
        import sys
        main_globals = sys.modules.get("__main__")
        if not main_globals:
            return

        for node_var in step.produces_nodes:
            obj = main_globals.__dict__.get(node_var)
            if obj and hasattr(obj, "GetID"):
                state.node_registry[node_var] = obj.GetID()
                logger.info(f"[WorkflowOrchestrator] Registered node '{node_var}' -> {obj.GetID()}")

    def _generate_minimal_placement_code(self, step: WorkflowStep) -> str:
        """Generate minimal code for creating a markup node and entering placement mode."""
        node_name = step.step_id.replace("_", " ").title()
        return (
            f"# --- {step.description} (Setup) ---\n"
            f"import slicer\n\n"
            f"node = slicer.mrmlScene.AddNewNodeByClass('{step.node_class}', '{node_name}')\n"
            f"displayNode = node.GetDisplayNode()\n"
            f"if displayNode is None:\n"
            f"    displayNode = node.CreateDefaultDisplayNode()\n"
            f"displayNode.SetVisibility(True)\n\n"
            f"print('[Workflow] {step.placement_instructions or step.description}')\n\n"
            f"selectionNode = slicer.app.applicationLogic().GetSelectionNode()\n"
            f"selectionNode.SetActivePlaceNodeID(node.GetID())\n"
            f"slicer.modules.markups.logic().StartPlaceMode("
            f"slicer.qSlicerMarkupsPlaceWidget.IndefinitePlace)\n"
        )
