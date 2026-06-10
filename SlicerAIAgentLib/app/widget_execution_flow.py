from .common import *


class WidgetExecutionFlowMixin:
    def _saveGeneratedCodeToFile(self, code, suffix=""):
        """Save the generated code to a local text file for user reference.

        Args:
            code: The generated Python code string.
            suffix: Optional suffix for the filename (e.g. '_correction_1').
        """
        try:
            turn_number = getattr(self, '_currentTurn', 1)
            latestPath = os.path.join(self._getCurrentLogDir(), f'{turn_number}{suffix}_code.txt')
            with open(latestPath, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            logger.warning(f"Failed to save generated code to file: {e}")

    def _saveAgentPlanToFile(self, plan, suffix=""):
        """Persist the parsed agent_plan JSON as a standalone debug artifact."""
        if not isinstance(plan, dict):
            return
        try:
            turn_number = getattr(self, '_currentTurn', 1)
            path = os.path.join(self._getCurrentLogDir(), f'{turn_number}{suffix}_agent_plan.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save agent plan to file: {e}")

    def _createRunLogDir(self, turn_number):
        """Create logs/YYYYmmdd_HHMMSS_turnN for all artifacts from one run."""
        from datetime import datetime
        moduleDir = SLICER_AI_AGENT_ROOT
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(moduleDir, "logs", f"{stamp}_turn{turn_number}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _getCurrentLogDir(self):
        """Return current run log directory, creating a fallback if needed."""
        if not self._currentLogDir:
            self._currentLogDir = self._createRunLogDir(getattr(self, "_currentTurn", 1))
            if self.logic and self.logic.llmClient:
                self.logic.llmClient.setDebugOutputDir(self._currentLogDir)
        os.makedirs(self._currentLogDir, exist_ok=True)
        return self._currentLogDir

    def _recordRoleEvent(self, role, event, details=None):
        """Record a structured event for the role-composed agent trace."""
        import time
        entry = {
            "time": round(time.time(), 3),
            "role": role,
            "event": event,
            "details": details or {},
        }
        self._roleTrace.append(entry)
        if self._timing is not None:
            self._timing["role_trace"] = list(self._roleTrace)

    def _saveRoleTraceToFile(self, suffix=""):
        """Persist the current role trace for academic/debug inspection."""
        try:
            turn_number = getattr(self, '_currentTurn', 1)
            path = os.path.join(self._getCurrentLogDir(), f'{turn_number}{suffix}_role_trace.json')
            payload = {
                "turn": turn_number,
                "prompt": getattr(self, "_lastUserPrompt", ""),
                "events": self._roleTrace,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save role trace: {e}")

    def _displayAgentPlanSummary(self, plan):
        """Show a compact plan summary in chat without adding a new UI surface."""
        if not plan:
            self.appendToChat("System", "No valid agent_plan block was returned. Auto-execution will be blocked.")
            return
        try:
            summary = plan.get("task_summary", "No task summary")
            overall_confidence = plan.get("overall_confidence") or self._deriveOverallPlanConfidence(plan)
            steps = plan.get("steps", [])
            step_lines = []
            for i, step in enumerate(steps[:5], start=1):
                if isinstance(step, dict):
                    action = step.get("action", "Unnamed step")
                    confidence = step.get("confidence", "unknown")
                    step_lines.append(f"{i}. {action} [{confidence}]")
            extra = "" if len(steps) <= 5 else f"\n... {len(steps) - 5} more step(s)"
            self.appendToChat(
                "System",
                "Verified plan received.\n"
                f"Task: {summary}\n"
                f"Overall confidence: {overall_confidence}\n"
                + ("\n".join(step_lines) if step_lines else "No steps listed")
                + extra
            )
        except Exception as e:
            logger.warning(f"Failed to display agent plan: {e}")

    def _deriveOverallPlanConfidence(self, plan):
        """Estimate whole-plan confidence for display when the plan lacks the field."""
        if not isinstance(plan, dict):
            return "unknown"
        steps = plan.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return "unknown"
        confidence_values = [
            str(step.get("confidence", "unknown")).lower()
            for step in steps
            if isinstance(step, dict)
        ]
        if not confidence_values:
            return "unknown"
        if any(value in ("low", "needs_lookup", "unknown") for value in confidence_values):
            return "low"
        if any(value == "medium" for value in confidence_values):
            return "medium"
        return "high"

    def _validateAgentPlan(self, plan):
        """Validate the structured plan enough to prevent blind execution."""
        errors = []
        warnings = []
        if not isinstance(plan, dict):
            return {
                "valid": False,
                "requires_confirmation": False,
                "errors": ["Missing or invalid agent_plan JSON block"],
                "warnings": [],
            }

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append("agent_plan.steps must be a non-empty list")

        risk = str(plan.get("risk_level", "unknown")).lower()
        plan_requires_confirmation = bool(plan.get("requires_confirmation")) or risk == "high"
        unresolved = plan.get("unverified_assumptions", [])
        if unresolved is None:
            unresolved = []
        if not isinstance(unresolved, list):
            warnings.append("agent_plan.unverified_assumptions should be a list")
            unresolved = [str(unresolved)]

        high_conf_steps = 0
        high_conf_without_evidence = 0
        needs_lookup = []
        if isinstance(steps, list):
            for idx, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    errors.append(f"Step {idx} must be an object")
                    continue
                if not step.get("action"):
                    errors.append(f"Step {idx} is missing action")
                confidence = str(step.get("confidence", "unknown")).lower()
                if confidence == "needs_lookup":
                    needs_lookup.append(step.get("action", f"step {idx}"))
                if confidence == "high":
                    high_conf_steps += 1
                    evidence = step.get("evidence")
                    if not evidence:
                        high_conf_without_evidence += 1
                # Validate expected_scene_change
                expected = step.get("expected_scene_change")
                if not isinstance(expected, dict) or not expected.get("type"):
                    errors.append(f"Step {idx} is missing required 'expected_scene_change'. Use 'not_checked' if this step has no scene effect.")
                else:
                    valid_types = {"node_count_delta", "node_exists", "node_modified",
                                   "node_has_display", "node_name_matches", "layout_changed",
                                   "selection_changed", "module_entered", "property_true", "not_checked"}
                    if str(expected.get("type")).lower() not in valid_types:
                        warnings.append(f"Step {idx} has unsupported expected_scene_change type '{expected.get('type')}'")

        if needs_lookup:
            errors.append("Plan still contains needs_lookup step(s): " + ", ".join(needs_lookup[:3]))
        if high_conf_steps and high_conf_without_evidence == high_conf_steps and high_conf_steps >= 2:
            warnings.append("Most high-confidence API steps have no explicit evidence")

        return {
            "valid": len(errors) == 0,
            "requires_confirmation": plan_requires_confirmation,
            "errors": errors,
            "warnings": warnings,
        }

    def _askExecutionConfirmation(self, reason, resume_callback):
        """Ask the user before executing high-risk or destructive code."""
        message_box = qt.QMessageBox(slicer.util.mainWindow())
        message_box.setWindowTitle("Confirm Agent Execution")
        message_box.setText("The generated plan or code requires confirmation before execution.")
        message_box.setInformativeText(reason)
        message_box.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        message_box.setDefaultButton(qt.QMessageBox.No)
        result = message_box.exec_()
        if result == qt.QMessageBox.Yes:
            self._recordRoleEvent("SafetyCritic", "execution_confirmed_by_user", {
                "reason": reason,
            })
            resume_callback()
        else:
            self._setReadyStatus()
            self.sendButton.setEnabled(True)
            self._recordRoleEvent("SafetyCritic", "execution_cancelled_by_user", {
                "reason": reason,
            })
            self._saveRoleTraceToFile()
            self.appendToChat("System", "Execution cancelled by user.")

    def _autoExecuteCode(self, attempt=1, max_attempts=5):
        """Auto-execute generated code with pre-validation and self-correction on failure."""
        if not hasattr(self, 'currentCode') or not self.currentCode:
            return

        self._setAgentStatus("Safety Critic", "Validating plan...")
        plan_validation = self._validateAgentPlan(getattr(self, "currentAgentPlan", None))
        self._recordRoleEvent("SafetyCritic", "plan_validation_completed", {
            "valid": plan_validation.get("valid"),
            "requires_confirmation": plan_validation.get("requires_confirmation"),
            "errors": plan_validation.get("errors", []),
            "warnings": plan_validation.get("warnings", []),
        })
        if not plan_validation["valid"]:
            error_msg = "; ".join(plan_validation["errors"])
            if attempt >= max_attempts:
                self._setReadyStatus()
                self.sendButton.setEnabled(True)
                self._recordRoleEvent("Repairer", "max_attempts_reached", {
                    "attempts": max_attempts,
                    "final_error": error_msg,
                    "stage": "plan_validation",
                })
                self._saveRoleTraceToFile()
                self.appendToChat(
                    "Error",
                    f"Plan validation failed after {max_attempts} attempts.\nFinal error: {error_msg}"
                )
                return
            self.appendToChat(
                "System",
                f"Plan validation failed (attempt {attempt}/{max_attempts}).\n"
                f"Error: {error_msg}\n"
                "Auto-correcting..."
            )
            if self.logic:
                self.logic.addExecutionFeedback(
                    f"Agent plan validation failed (attempt {attempt}/{max_attempts}):\n"
                    f"Error: {error_msg}\n"
                    "The response must include a valid ```agent_plan JSON block before the Python code."
                )
            self._selfCorrectCode(error_msg, attempt, max_attempts)
            return

        # Pre-validation: check for syntax errors and common issues
        import time
        if self._timing:
            self._timing['validation_start'] = time.time()

        validation = None
        if self.logic and hasattr(self.logic, 'codeValidator'):
            self._setAgentStatus("Safety Critic", "Validating code...")
            validation = self.logic.codeValidator.validate(self.currentCode)
            self._recordRoleEvent("SafetyCritic", "code_validation_completed", {
                "valid": validation.get("valid"),
                "requires_confirmation": validation.get("requires_confirmation"),
                "destructive_ops": validation.get("destructive_ops", []),
                "warnings": validation.get("warnings", []),
                "reason": validation.get("reason"),
            })
            if self._timing:
                self._timing['validation_end'] = time.time()
            if not validation['valid']:
                # Syntax error detected before execution
                error_msg = validation['reason']
                if attempt >= max_attempts:
                    self._setReadyStatus()
                    self.sendButton.setEnabled(True)
                    self._recordRoleEvent("Repairer", "max_attempts_reached", {
                        "attempts": max_attempts,
                        "final_error": error_msg,
                        "stage": "code_validation",
                    })
                    self._saveRoleTraceToFile()
                    self.appendToChat(
                        "Error",
                        f"Pre-validation failed after {max_attempts} attempts.\nFinal error: {error_msg}"
                    )
                    return
                self.appendToChat("System",
                    f"Pre-validation failed (attempt {attempt}/{max_attempts}).\n"
                    f"Error: {error_msg}\n"
                    f"Auto-correcting...")
                if self.logic:
                    self.logic.addExecutionFeedback(
                        f"Code pre-validation failed (attempt {attempt}/{max_attempts}):\n"
                        f"Error: {error_msg}\n"
                        "The code had syntax errors or violated safety rules before execution."
                    )
                self._selfCorrectCode(error_msg, attempt, max_attempts)
                return
        else:
            if self._timing:
                self._timing['validation_end'] = time.time()

        requires_confirmation = plan_validation.get("requires_confirmation", False)
        confirmation_reasons = []
        if plan_validation.get("warnings"):
            for warning in plan_validation["warnings"]:
                logger.warning(f"Agent plan warning: {warning}")
        if requires_confirmation:
            confirmation_reasons.append(
                f"Plan risk level: {getattr(self, 'currentAgentPlan', {}).get('risk_level', 'unknown')}"
            )
        if validation and validation.get("requires_confirmation"):
            destructive_ops = ", ".join(validation.get("destructive_ops", []))
            confirmation_reasons.append(f"Potentially destructive operation(s): {destructive_ops}")
        if confirmation_reasons:
            reason = "\n".join(confirmation_reasons)
            logger.info(f"Auto-approving execution with warnings: {reason}")
            self.appendToChat("Warning", f"Executing with potentially destructive operations: {reason}")
            self._recordRoleEvent("SafetyCritic", "confirmation_required", {
                "reason": reason,
            })

        self._autoExecuteCodeConfirmed(attempt, max_attempts)

    def _autoExecuteCodeConfirmed(self, attempt=1, max_attempts=5):
        """Run already-validated code after optional confirmation has passed."""
        import time

        self._setAgentStatus("Executor", f"Executing (attempt {attempt}/{max_attempts})...")
        slicer.app.processEvents()

        if self._timing and 'execution_start' not in self._timing:
            self._timing['execution_start'] = time.time()
        if self._timing:
            self._timing['execution_async_call'] = time.time()

        self._recordRoleEvent("Executor", "execution_scheduled", {
            "attempt": attempt,
            "code_chars": len(self.currentCode or ""),
        })

        # Capture scene state BEFORE execution for semantic verification
        before_snapshot = None
        if hasattr(self, 'buildSceneSnapshot'):
            try:
                before_snapshot = self.buildSceneSnapshot()
            except Exception as e:
                logger.warning(f"Failed to build pre-execution scene snapshot: {e}")

        def onExecutionComplete(result):
            if self._timing:
                self._timing['execution_callback_start'] = time.time()
            feedback_lines = []
            output_has_errors = False
            step_info = getattr(self, '_currentWorkflowStepInfo', None)
            runtime_managed = bool(self._workflowRuntime and self._workflowRuntime.session and step_info)
            self._recordRoleEvent("Executor", "execution_completed", {
                "success": result.get("success"),
                "timed_out": result.get("timed_out", False),
                "execution_time": result.get("execution_time", 0),
                "error": result.get("error"),
            })
            if result.get("timed_out", False):
                self._setReadyStatus()
                output = result.get('output', 'No output')
                exec_time = result.get('execution_time', 30)
                msg = f"Code execution timed out after {exec_time:.1f}s."
                if output:
                    msg += f"\nOutput: {output}"
                self.appendToChat("Warning", msg)
                if runtime_managed:
                    self._updateWorkflowPanel({
                        "active": True,
                        "workflow_title": self._currentWorkflowUiState.get("workflow_title", "Workflow"),
                        "status": "Failed",
                        "description": msg,
                        "total_steps": self._currentWorkflowUiState.get("total_steps", 0),
                        "completed_steps": self._currentWorkflowUiState.get("completed_steps", 0),
                        "current_index": self._currentWorkflowUiState.get("current_index", 0),
                        "can_cancel": True,
                    })
                feedback_lines.append(f"Status: timed_out\nExecution time: {exec_time:.1f}s\nOutput: {output}")
            elif result["success"]:
                output = result.get('output', 'No output')
                execution_time = result.get('execution_time', 0)
                msg = f"Code executed successfully in {execution_time:.2f}s."
                if output:
                    msg += f"\nOutput: {output}"
                if not runtime_managed:
                    self.appendToChat("System", msg)
                feedback_lines.append(f"Status: success\nExecution time: {execution_time:.2f}s\nOutput: {output}")
                # Detect actual errors (excluding VTK warnings which are often benign)
                lower_output = output.lower()
                # Strip VTK output lines — they frequently contain words like "error"
                # in benign messages (e.g., "Decimation completed without errors")
                non_vtk_output = "\n".join(
                    line for line in output.split("\n")
                    if not line.strip().startswith("[VTK")
                ).lower()
                if any(k in non_vtk_output for k in ('traceback', 'exception', 'failed', 'error')):
                    output_has_errors = True
                    feedback_lines.append("Warning: execution output contains error indicators even though no uncaught exception was raised.")
                self._lastExecutionResult = dict(result)
                self._lastOutputHasErrors = output_has_errors
            else:
                # Execution failed
                error_msg = result.get('error', 'Unknown error')
                if "[GeneratedWorkflowPrecondition]" in error_msg:
                    result["failure_category"] = "generated_workflow_precondition"
                    self._recordRoleEvent("SafetyCritic", "generated_precondition_failed", {
                        "error": error_msg[:1000],
                    })
                execution_time = result.get('execution_time', 0)
                msg = f"Execution failed (attempt {attempt}/{max_attempts}).\nError: {error_msg[:200]}"
                self.appendToChat("System", msg)
                feedback_lines.append(f"Status: failed\nExecution time: {execution_time:.2f}s\nError: {error_msg[:500]}")
                self._lastExecutionResult = dict(result)
                self._lastOutputHasErrors = True

            # Semantic scene verification: compare before/after snapshots against agent_plan expectations
            if result.get("success") and not result.get("timed_out", False) and before_snapshot:
                try:
                    if hasattr(self, 'buildSceneSnapshot') and hasattr(self, 'verifySceneAgainstPlan'):
                        after_snapshot = self.buildSceneSnapshot()
                        plan = getattr(self, 'currentAgentPlan', None)
                        if plan:
                            verification = self.verifySceneAgainstPlan(before_snapshot, after_snapshot, plan)
                            if not verification.get("valid", True):
                                output_has_errors = True
                                verr = "; ".join(verification.get("errors", []))
                                feedback_lines.append(f"Scene verification failed: {verr}")
                                self.appendToChat("System", f"Scene verification failed: {verr}")
                                self._recordRoleEvent("SafetyCritic", "scene_verification_failed", {
                                    "errors": verification.get("errors", []),
                                    "warnings": verification.get("warnings", []),
                                    "diagnostics": verification.get("diagnostics", []),
                                })
                            elif verification.get("warnings"):
                                self._recordRoleEvent("SafetyCritic", "scene_verification_warnings", {
                                    "warnings": verification.get("warnings", []),
                                })
                except Exception as e:
                    logger.warning(f"Scene verification raised exception: {e}")

            # Add execution feedback to conversation history
            if self.logic:
                feedback_text = "Code execution result:\n" + "\n".join(feedback_lines) + "\nThe MRML scene has been updated. Refer to the CURRENT SLICER SCENE in the next system prompt for the complete raw MRML."
                self.logic.addExecutionFeedback(feedback_text)

            # Record execution timing
            if self._timing:
                self._timing['execution_end'] = time.time()
                self._timing['execution_result'] = 'success' if result.get('success') else 'failed'
                # Record executor internal timing
                if 'executor_scheduled' in result:
                    self._timing['executor_scheduled'] = result['executor_scheduled']
                if 'executor_actual_start' in result:
                    self._timing['executor_actual_start'] = result['executor_actual_start']
                self._writeTimingReport()

            if result.get("success") and runtime_managed:
                updated_step = self._workflowRuntime.handle_execution_result(step_info, result)
                self._currentWorkflowStepInfo = updated_step
                self._applyWorkflowDisplayProperties(updated_step)
                self._updateWorkflowPanel(updated_step)
                if updated_step.get("type") in ("interactive", "mixed"):
                    self._recordRoleEvent("Workflow", "entering_wait", {
                        "step_id": updated_step.get("step_id"),
                    })
                    self._streamQueue.put(('workflow_wait', updated_step))
                    return
                if updated_step.get("type") == "user_choice":
                    self._displayWorkflowChoice(updated_step)
                    self._setReadyStatus()
                    self.sendButton.setEnabled(True)
                    return
                next_step = updated_step.get("next_step")
                if next_step:
                    completed_step_id = updated_step.get("step_id", "")
                    self._updateWorkflowPanel(self._workflowRuntime.state_for_ui(updated_step))
                    self._autoAdvanceWorkflowStep = next_step
                elif updated_step.get("workflow_completed"):
                    self._updateWorkflowPanel(updated_step)
                    self.appendToChat("System", "Generated CLI workflow complete.")
                    self._clearCompletedWorkflowState()
                    self._flushQueuedWorkflowPrompts()

            # Interactive workflow detection: if an interactive step just executed,
            # transition to waiting_for_user and enter wait mode.
            if (result.get("success")
                and self._workflowOrchestrator
                and self._activeWorkflowId
                and self._workflowOrchestrator.is_workflow_active()
                and not runtime_managed):
                wf_state = self._workflowOrchestrator._get_state(self._activeWorkflowId)
                if wf_state and wf_state.status == "running":
                    step_info = getattr(self, '_currentWorkflowStepInfo', None)
                    if step_info and step_info.get("type") in ("interactive", "mixed"):
                        # Apply display properties before entering wait state
                        self._applyWorkflowDisplayProperties(step_info)
                        wf_state.status = "waiting_for_user"
                        self._recordRoleEvent("Workflow", "entering_wait", {
                            "step_id": wf_state.current_step,
                        })
                        self._streamQueue.put(('workflow_wait', step_info))
                        return
                    # user_choice steps: no UI state change — the LLM handles the
                    # question/answer via normal chat turns. Just keep chat enabled.
                    if step_info and step_info.get("type") == "user_choice":
                        wf_state.status = "waiting_for_choice"
                        self._recordRoleEvent("Workflow", "waiting_for_choice", {
                            "step_id": wf_state.current_step,
                        })
                        self._displayWorkflowChoice(step_info)
                    # Automated step completed — auto-advance to next step
                    if step_info and step_info.get("type") == "repeat_next":
                        next_step = step_info.get("next_step")
                        if next_step:
                            self._updateWorkflowPanel(step_info)
                            self._autoAdvanceWorkflowStep = next_step
                    elif step_info and step_info.get("type") in (
                        "automated", "skipped", "choice_made",
                        "interactive_done", "mixed_done",
                    ):
                        # Apply display properties for automated steps
                        self._applyWorkflowDisplayProperties(step_info)
                        completed_step_id = step_info.get("step_id", "")
                        # Mark current step complete in both tracking systems
                        from SlicerAIAgentLib.ExtensionCLILoader import _find_next_step_local, _workflow_completed_steps
                        done = _workflow_completed_steps.setdefault(step_info.get("tool", ""), set())
                        done.add(completed_step_id)
                        # Also mark in WorkflowOrchestrator's state so
                        # complete_interaction()/_find_next_step stays in sync
                        if self._activeWorkflowId:
                            wf_st = self._workflowOrchestrator._get_state(self._activeWorkflowId)
                            if wf_st and completed_step_id not in wf_st.completed_steps:
                                wf_st.completed_steps.append(completed_step_id)
                            # Advance current_step so prompt fragment doesn't advertise the skipped step
                            if wf_st and step_info.get("type") == "skipped":
                                wf_st.current_step = None
                        # Load workflow graph for next-step lookup
                        try:
                            import os, json
                            from SlicerAIAgentLib.ExtensionCLILoader import get_cli_base_dir
                            ext_name = step_info.get("tool", "")
                            wf_path = os.path.join(get_cli_base_dir(), ext_name, "workflow.json")
                            if os.path.isfile(wf_path):
                                with open(wf_path) as wf_f:
                                    wf_graph = json.load(wf_f)
                                next_step = _find_next_step_local(wf_graph, done)
                                if next_step:
                                    self._updateWorkflowPanel(step_info)
                                    # Trigger a new turn with the next step
                                    self._autoAdvanceWorkflowStep = next_step
                                elif step_info.get("type") == "skipped":
                                    # No more steps after the skipped one — end the workflow
                                    if wf_st:
                                        wf_st.status = "completed"
                                    self._updateWorkflowPanel({
                                        "active": True,
                                        "workflow_title": step_info.get("tool", "Workflow"),
                                        "status": "Completed",
                                        "description": "Workflow complete.",
                                        "can_done": False,
                                        "can_skip": False,
                                        "can_cancel": False,
                                    })
                                    self._clearCompletedWorkflowState()
                        except Exception as e:
                            logger.warning(f"Workflow auto-advance failed: {e}")

            # Self-correction for failures or suspicious outputs (but not timeouts)
            # Skip self-correction during active workflow steps that succeeded —
            # VTK noise should not derail the workflow sequence.
            workflow_active = (
                (
                    self._workflowRuntime
                    and self._workflowRuntime.has_active_workflow()
                )
                or (
                    self._workflowOrchestrator
                    and self._activeWorkflowId
                    and self._workflowOrchestrator.is_workflow_active()
                )
            )
            if not result.get("timed_out", False) and (not result["success"] or (output_has_errors and not workflow_active)):
                if attempt < max_attempts:
                    self.appendToChat("System", "Auto-correcting...")
                    error_for_correction = result.get('error', '')
                    if not error_for_correction and output_has_errors:
                        # success=True but output contains clear failure keywords
                        error_for_correction = "\n".join(feedback_lines) or output

                    # Revision D: triage by failure type. If the failing code
                    # came from a generated CLI template AND the error is
                    # syntax/name-level (i.e. a generator-side bug), skip the
                    # LLM repair loop — it can't see the generator, so it
                    # would burn ~100s looking for a fix that has to come
                    # from regenerating the CLI. Fall through to repair on
                    # runtime/scene errors or any ambiguity.
                    step_info_for_triage = getattr(self, '_currentWorkflowStepInfo', None) or {}
                    if self._shouldSkipSelfCorrectionForGeneratedTemplate(
                        step_info_for_triage, error_for_correction,
                    ):
                        self._handleGeneratedTemplateGeneratorBug(
                            step_info_for_triage, error_for_correction,
                        )
                        return

                    self._recordRoleEvent("Repairer", "correction_requested", {
                        "attempt": attempt + 1,
                        "reason": error_for_correction[:1000] if error_for_correction else "",
                    })
                    self._selfCorrectCode(error_for_correction, attempt, max_attempts)
                else:
                    final_error = result.get('error', 'Unknown error') if not result["success"] else "Output contains errors"
                    if getattr(self, "_taskWorkflowPanelActive", False):
                        self._updateWorkflowPanel({
                            "active": True,
                            "mode": "task",
                            "workflow_title": "Task",
                            "status": "Failed",
                            "description": str(final_error),
                            "instructions": "Generated code and execution details are available in Debug.",
                            "total_steps": 0,
                            "can_done": False,
                            "can_skip": False,
                            "can_cancel": False,
                        })
                        self._taskWorkflowPanelActive = False
                    self._setReadyStatus()
                    self.appendToChat("Error",
                        f"Execution failed after {max_attempts} attempts.\n"
                        f"Final error: {final_error}")
                    self._recordRoleEvent("Repairer", "max_attempts_reached", {
                        "attempts": max_attempts,
                        "final_error": final_error,
                    })
                    self._saveRoleTraceToFile()
            else:
                self._recordRoleEvent("Executor", "task_completed", {
                    "execution_success": result.get("success"),
                })
                self._saveRoleTraceToFile()

                # Auto-advance workflow: if an automated step just completed and
                # there's a next step queued, trigger it automatically.
                next_step = getattr(self, '_autoAdvanceWorkflowStep', None)
                if next_step and result.get("success"):
                    self._autoAdvanceWorkflowStep = None
                    tool_name = next_step.get("step_id", "")
                    # Use a timer to schedule the next step as a new user-like turn
                    step_id = next_step["step_id"]
                    # Schedule the auto-advance on the Qt main thread
                    qt.QTimer.singleShot(100, lambda: self._autoAdvanceNextStep(next_step))
                else:
                    self._setReadyStatus()

        # Apply per-step prelude globals (workflow metadata as Python objects)
        # so the new compact prelude can reference _workflow_choices etc.
        # without round-tripping them through source text.
        self._applyPreludeGlobals()

        # Execute asynchronously
        self.logic.executeCodeAsync(self.currentCode, onExecutionComplete)

    def _applyPreludeGlobals(self):
        """Inject the current workflow step's _prelude_globals into __main__.

        The dispatch result attaches ``_prelude_globals`` (a dict of Python
        objects: choices/bindings/defaults/required_bindings) when the new
        compact prelude would otherwise reference them as undefined names.
        This registers each key on the executor's __main__ namespace before
        exec, then schedules cleanup so stale metadata from step N doesn't
        leak into step N+1.
        """
        try:
            executor = getattr(self.logic, 'executor', None)
            if executor is None or not hasattr(executor, 'addGlobal'):
                return
        except Exception:
            return

        step_info = getattr(self, '_currentWorkflowStepInfo', None) or {}
        prelude_globals = step_info.get('_prelude_globals') if isinstance(step_info, dict) else None
        if not isinstance(prelude_globals, dict):
            prelude_globals = {}

        # Cleanup-on-entry: remove keys injected by the previous step so they
        # don't leak across steps. Tracked in self._lastInjectedPreludeKeys.
        previous_keys = getattr(self, '_lastInjectedPreludeKeys', None) or []
        for key in previous_keys:
            try:
                executor.removeGlobal(key)
            except Exception:
                pass

        if not prelude_globals:
            self._lastInjectedPreludeKeys = []
            return

        for key, value in prelude_globals.items():
            try:
                executor.addGlobal(key, value)
            except Exception:
                pass
        self._lastInjectedPreludeKeys = list(prelude_globals.keys())

    # ================================================================
    # Revision D: Self-correction triage for generated CLI templates
    # ================================================================

    # Substrings that signal a runtime/scene error even when the code parses
    # cleanly. Their presence biases the classifier away from "syntax_or_name".
    _GENERATED_TEMPLATE_RUNTIME_ERROR_MARKERS = (
        "[GeneratedWorkflowPrecondition]",
        "Missing required node reference",
        "Missing conditional node reference",
        "RuntimeError",
        "ValueError",
        "AttributeError",
        "TypeError",
        "vtkMRML",
        "slicer.mrmlScene",
        "GetParameterNode",
        "Node reference",
    )

    def _classifyExecutionError(self, error_msg: str, code: str) -> str:
        """Classify an execution error for self-correction triage.

        Returns one of:
            ``"syntax_or_name"`` — code failed to parse, or execution raised
                a Python-level ``NameError``/``SyntaxError``. Suggests a
                generator-side bug for generated-template code.
            ``"runtime_or_scene"`` — code parsed fine but raised a runtime/
                scene error (missing node, bad VTK call, etc.). The existing
                LLM repair path can handle these.
            ``"other"`` — ambiguous; caller should fall through to existing
                repair to avoid blocking on uncertain classification.
        """
        if not error_msg:
            return "other"
        lowered = error_msg.lower()
        # Strong syntax/name signals — note that NameError's actual message
        # text is ``name 'X' is not defined`` (no exception type name), so we
        # also match that pattern directly. This catches the JSON-vs-Python
        # boolean bug class (``name 'false' is not defined``).
        import re as _re
        name_error_pattern = _re.compile(r"name\s+['\\\"][^'\"]+['\\\"]\s+is\s+not\s+defined")
        syntax_signals = (
            "syntaxerror" in lowered
            or "nameerror" in lowered
            or "indentationerror" in lowered
            or name_error_pattern.search(lowered) is not None
        )
        if syntax_signals:
            # But runtime errors sometimes mention these names too — confirm
            # the runtime markers aren't present before classifying as syntax.
            for marker in self._GENERATED_TEMPLATE_RUNTIME_ERROR_MARKERS:
                if marker.lower() in lowered:
                    return "runtime_or_scene"
            return "syntax_or_name"

        # ast.parse as a secondary signal: code that won't parse at all is
        # definitely syntax-level. (Generator bugs that produce unparseable
        # output land here.)
        if code:
            try:
                import ast as _ast
                _ast.parse(code)
            except (SyntaxError, IndentationError, ValueError):
                return "syntax_or_name"

        # Explicit runtime/scene markers
        for marker in self._GENERATED_TEMPLATE_RUNTIME_ERROR_MARKERS:
            if marker.lower() in lowered:
                return "runtime_or_scene"

        return "other"

    def _shouldSkipSelfCorrectionForGeneratedTemplate(
        self,
        step_info: dict,
        error_msg: str,
    ) -> bool:
        """Return True iff the failing code came from a generated CLI
        template AND the error is a generator-side syntax/name bug.

        Conservative: returns False (proceed with existing repair) on any
        ambiguity, on non-generated-template code, or on runtime/scene errors.
        """
        if not isinstance(step_info, dict):
            return False
        if step_info.get("origin") != "generated_template":
            return False
        code = self.currentCode or ""
        classification = self._classifyExecutionError(error_msg or "", code)
        return classification == "syntax_or_name"

    def _handleGeneratedTemplateGeneratorBug(
        self,
        step_info: dict,
        error_msg: str,
    ):
        """Surface a structured 'regenerate the CLI' error and cancel the workflow.

        Called when self-correction is skipped for a generated-template syntax
        error. The LLM repair path can't see the generator, so this would have
        been wasted effort. Instead we tell the user to regenerate the CLI.
        """
        step_id = (step_info or {}).get("step_id", "<unknown>")
        ext_name = (step_info or {}).get("tool", "<unknown>")
        # tool name is usually ``<ExtensionName>`` — strip a trailing suffix
        # if present.
        message = (
            f"CLI generator produced invalid code for step '{step_id}'. "
            f"This is a bug in the generation pipeline, not a runtime scene "
            f"issue. Original error: {error_msg or 'unknown'}. "
            f"Please regenerate the {ext_name} CLI."
        )
        self.appendToChat("Error", message)
        self._setReadyStatus()
        try:
            self.sendButton.setEnabled(True)
        except Exception:
            pass
        self._recordRoleEvent("Repairer", "generator_bug_detected", {
            "step_id": step_id,
            "extension": ext_name,
            "error": (error_msg or "")[:1000],
            "skipped_self_correction": True,
        })
        try:
            self._saveRoleTraceToFile()
        except Exception:
            pass
