from .common import *


class WidgetCorrectionMixin:
    def _selfCorrectCode(self, error_msg, attempt, max_attempts):
        """Generate corrected code with isolated context (no conversation history bloat)."""
        if not self.currentCode:
            return

        # Revision D (belt-and-suspenders): the caller in widget_execution_flow
        # is supposed to triage generated-template syntax errors before getting
        # here. As a defensive check, refuse to enter the LLM repair loop if
        # the current step is a generated template whose error classifies as
        # syntax/name-level — those indicate a generator bug the LLM can't fix.
        try:
            step_info = getattr(self, '_currentWorkflowStepInfo', None) or {}
            if (
                isinstance(step_info, dict)
                and step_info.get("origin") == "generated_template"
                and hasattr(self, "_classifyExecutionError")
                and self._classifyExecutionError(error_msg or "", self.currentCode or "")
                == "syntax_or_name"
            ):
                if hasattr(self, "_handleGeneratedTemplateGeneratorBug"):
                    self._handleGeneratedTemplateGeneratorBug(step_info, error_msg)
                    return
        except Exception:
            pass

        import time
        if self._timing:
            corrections = self._timing.setdefault('corrections', [])
            corrections.append({'attempt': attempt + 1, 'start': time.time()})

        error_detail = error_msg if error_msg else "Unknown error"
        self._recordRoleEvent("Repairer", "correction_started", {
            "attempt": attempt + 1,
            "error": error_detail[:1000],
        })
        self.appendToChat("You", f"[Auto-correction attempt {attempt+1}]")
        self._setAgentStatus("Repairer", "Correcting...")
        self._startThinkingTimer()
        if self.logic and self.logic.llmClient:
            self.logic.llmClient.debug_suffix = "_correction"

        # Build system prompt on the main thread BEFORE starting the background thread.
        # _buildSceneContext() calls slicer.mrmlScene.GetSceneXMLString() which is a Qt
        # method and must run on the main thread.
        system_content = "You are an expert 3D Slicer Python coding assistant."
        if self.logic and self.logic.llmClient and hasattr(self.logic.llmClient, '_buildSystemPrompt'):
            try:
                context = {"scene": self.logic._buildSceneContext()} if self.logic else None
                system_content = self.logic.llmClient._buildSystemPrompt(context)
            except Exception:
                if hasattr(self.logic.llmClient, '_loadSystemPromptTemplate'):
                    system_content = self.logic.llmClient._loadSystemPromptTemplate()
        elif self.logic and self.logic.llmClient and hasattr(self.logic.llmClient, '_loadSystemPromptTemplate'):
            system_content = self.logic.llmClient._loadSystemPromptTemplate()

        # Capture state needed by the background thread
        _logic = self.logic
        _currentCode = self.currentCode
        _lastUserPrompt = getattr(self, '_lastUserPrompt', '')
        _currentTurn = getattr(self, '_currentTurn', 1)
        _system_content = system_content
        _workflow_state = {}
        _workflow_repair_active = False
        try:
            if self._workflowRuntime and self._workflowRuntime.has_active_workflow():
                _workflow_state = self._workflowRuntime.state_for_router()
                _workflow_repair_active = True
        except Exception:
            _workflow_state = {}
            _workflow_repair_active = False

        def _filtered_repair_tools():
            tools = list(getattr(_logic, "skillTools", []) or [])
            if not _workflow_repair_active:
                return tools
            blocked_names = set()
            try:
                from SlicerAIAgentLib.ExtensionCLILoader import get_validated_extensions
                for ext_data in get_validated_extensions().values():
                    for schema in ext_data.get("schemas", []) or []:
                        name = (schema.get("function") or {}).get("name")
                        if name:
                            blocked_names.add(name)
            except Exception:
                pass
            if not blocked_names:
                return tools
            filtered = []
            for tool in tools:
                name = ""
                if isinstance(tool, dict):
                    name = (tool.get("function") or {}).get("name", "")
                if name and name in blocked_names:
                    continue
                filtered.append(tool)
            return filtered

        def _run_correction():
            """Run chatWithToolsIsolated in a background thread so the main Qt event loop
            stays alive and _processStreamQueue can consume progress deltas in real time."""
            try:
                isolated_messages = [{'role': 'system', 'content': _system_content}]

                if _lastUserPrompt:
                    isolated_messages.append({'role': 'user', 'content': _lastUserPrompt})

                # Inject prior tool trajectory so LLM doesn't fix blind
                prior_tool_messages = []
                if _logic and _logic.llmClient:
                    for msg in _logic.llmClient.conversation_history:
                        if msg.get('role') in ('assistant', 'tool'):
                            prior_tool_messages.append(msg)
                if prior_tool_messages:
                    isolated_messages.extend(prior_tool_messages)
                    isolated_messages.append({
                        'role': 'system',
                        'content': 'The messages above show the tool searches and file reads from the original attempt. Use them for reference.'
                    })

                isolated_messages.append({
                    'role': 'assistant',
                    'content': (
                        f"Previous agent_plan:\n"
                        f"```json\n{json.dumps(getattr(self, 'currentAgentPlan', None), ensure_ascii=False, indent=2)}\n```\n\n"
                        f"Previous Python code:\n"
                        f"```python\n{_currentCode}\n```"
                    )
                })

                user_content = (
                    f"CRITICAL: The previous Python code execution failed with this error:\n"
                    f"{error_detail}\n\n"
                    + (
                        "This is a generated workflow precondition failure. Do not create dummy MRML nodes "
                        "only to satisfy the precondition. Preserve required-reference checks, but remove or "
                        "bypass a check when the source method proves the reference is conditional, optional, "
                        "or created by the method itself.\n\n"
                        if "[GeneratedWorkflowPrecondition]" in error_detail else ""
                    )
                    + (
                        "This failure happened inside an active generated extension CLI workflow.\n"
                        f"Workflow state:\n{json.dumps(_workflow_state, ensure_ascii=False, indent=2)}\n\n"
                        "Repair ONLY the current failed Python code block. Do not restart the workflow, "
                        "do not call earlier or later workflow steps, and do not ask the user to choose "
                        "workflow options again. If the current step cannot be repaired safely, output "
                        "Python code that raises a clear RuntimeError for the current step.\n\n"
                        if _workflow_repair_active else ""
                    )
                    +
                    "You have Grep, ReadFile, and VectorSearch tools available. "
                    "If the error is caused by an incorrect API signature, missing parameter, or wrong module path, "
                    "use the tools to verify the correct usage before fixing. "
                    "Do NOT search unnecessarily — if you are confident in the fix, apply it directly.\n\n"
                    "Your task is to fix the error and output the COMPLETE corrected Python code in ONE ```python block."
                    " Also output a corrected ```agent_plan JSON block before the Python block."
                    " Do not emit tool-call markup, DSML, XML, or file-read instructions in the final response."
                )

                isolated_messages.append({
                    'role': 'user',
                    'content': user_content
                })

                # Save isolated prompt to debug file before sending
                try:
                    suffix = f"_correction_{attempt}"
                    first_debug = os.path.join(self._getCurrentLogDir(), f'{_currentTurn}{suffix}_first_prompt_debug.txt')
                    with open(first_debug, 'w', encoding='utf-8') as f:
                        for i, msg in enumerate(isolated_messages):
                            f.write(f"{'='*60}\n")
                            f.write(f"MESSAGE {i+1} | role: {msg.get('role', 'unknown')}\n")
                            f.write(f"{'='*60}\n")
                            f.write(f"{msg.get('content', '')}\n\n")
                except Exception:
                    pass

                # Use tool-calling isolated chat so LLM can re-search if needed
                def _on_correction_progress(progress):
                    self._streamQueue.put(('delta', dict(progress)))

                def _on_correction_status(status_text):
                    self._streamQueue.put(('status', status_text))

                def _on_correction_reasoning(reasoning_text, round_num):
                    self._appendThinkingToFile(reasoning_text, turn=getattr(self, '_currentTurn', 1))
                    self._streamQueue.put(('thinking_done', None))

                def _on_correction_reasoning_delta(chunk):
                    self._streamQueue.put(('thinking_delta', chunk))

                response = _logic.llmClient.chatWithToolsIsolated(
                    messages=isolated_messages,
                    tools=_filtered_repair_tools(),
                    tool_executor=_logic._executeTool,
                    max_tool_rounds=5,
                    on_progress=_on_correction_progress,
                    on_status=_on_correction_status,
                    on_reasoning=_on_correction_reasoning,
                    on_reasoning_delta=_on_correction_reasoning_delta,
                )

                # Dispatch result handling via _streamQueue so it runs on the main thread
                self._streamQueue.put(('correction_complete', {
                    'response': response,
                    'attempt': attempt,
                    'max_attempts': max_attempts,
                    'error_detail': error_detail,
                    'original_prompt': _lastUserPrompt,
                }))
            except Exception as e:
                self._streamQueue.put(('correction_error', {
                    'error_msg': str(e),
                }))

        thread = threading.Thread(target=_run_correction, daemon=True)
        thread.start()

    def _handleCorrectionResult(self, response, attempt, max_attempts, error_detail, original_prompt):
        """Handle successful self-correction response on the main thread."""
        # Save correction timing report and token usage
        if self._timing:
            corrections = self._timing.get('corrections', [])
            if corrections:
                if response.get('timing_report'):
                    corrections[-1]['timing_report'] = response['timing_report']
                if response.get('tokens'):
                    corrections[-1]['tokens'] = response['tokens']
                if response.get('cost') is not None:
                    corrections[-1]['cost'] = response['cost']

        # Accumulate per-turn token/cost from self-correction
        if response.get('tokens'):
            self._currentTurnTokens += response['tokens']
            self._currentTurnCost += response.get('cost', 0)
            self._updateTokenLabel()

        if self.logic and self.logic.llmClient:
            self.logic.llmClient.debug_suffix = ""

        if response.get("code"):
            self.currentCode = response["code"]
            self.currentAgentPlan = response.get("agent_plan")
            self._recordRoleEvent("Repairer", "correction_received", {
                "attempt": attempt + 1,
                "has_plan": bool(self.currentAgentPlan),
                "code_chars": len(self.currentCode or ""),
            })
            self.codeDisplay.setPlainText(response["code"])
            self._displayAgentPlanSummary(self.currentAgentPlan)
            self._saveAgentPlanToFile(self.currentAgentPlan, suffix=f"_correction_{attempt}")
            self._saveGeneratedCodeToFile(response["code"], suffix=f"_correction_{attempt}")
            self._stopThinkingTimer("Corrected")

            # Update conversation_history: replace wrong code with corrected code
            if self.logic and self.logic.llmClient:
                history = self.logic.llmClient.conversation_history
                # Find last assistant message containing a code block and replace it
                for i in range(len(history) - 1, -1, -1):
                    msg = history[i]
                    if msg.get('role') == 'assistant' and '```' in msg.get('content', ''):
                        history[i] = {
                            'role': 'assistant',
                            'content': response.get('message', f"```python\n{response['code']}\n```")
                        }
                        if response.get('reasoning_content'):
                            history[i]['reasoning_content'] = response['reasoning_content']
                        break

                # Append correction marker
                history.append({
                    'role': 'system',
                    'content': (
                        f"CORRECTION: The previous code failed with: {error_detail}. "
                        f"After correction attempt {attempt + 1}, the working version is above. "
                        f"The original search results remain valid."
                    )
                })

                # Append compressed correction-phase tool results
                correction_messages = response.get('intermediate_messages', [])
                if correction_messages:
                    compressed = self.logic.llmClient._compressToolResultsForHistory(
                        correction_messages, user_prompt=original_prompt
                    )
                    history.extend(compressed)

            self._autoExecuteCode(attempt + 1, max_attempts)
        else:
            raw_msg = response.get('message', '')[:300]
            self.appendToChat("System", f"Correction response contained no code block. Raw response preview:\n{raw_msg}")
            self._recordRoleEvent("Repairer", "correction_missing_code", {
                "attempt": attempt + 1,
                "raw_preview": raw_msg,
            })
            workflow_repair_active = False
            try:
                workflow_repair_active = bool(
                    self._workflowRuntime and self._workflowRuntime.has_active_workflow()
                )
            except Exception:
                workflow_repair_active = False
            if workflow_repair_active:
                self._stopThinkingTimer("Failed")
                self._setReadyStatus()
                return
            if attempt + 1 < max_attempts:
                retry_error = (
                    f"{error_detail}\n\n"
                    "The previous correction response contained no executable ```python block. "
                    f"Raw preview: {raw_msg}"
                )
                self._selfCorrectCode(retry_error, attempt + 1, max_attempts)
            else:
                self._stopThinkingTimer("Failed")
                self._setReadyStatus()

    def _appendThinkingToFile(self, reasoning_content: str, turn: int = 1):
        """Append LLM thinking/reasoning content to a local text file."""
        from datetime import datetime
        try:
            filepath = os.path.join(self._getCurrentLogDir(), f'{turn}_thinking_history.txt')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"Turn {turn} | {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{reasoning_content}\n\n")
        except Exception as e:
            logger.warning(f"Failed to write thinking history: {e}")

    def _handleCorrectionError(self, error_msg):
        """Handle self-correction error on the main thread."""
        self._stopThinkingTimer("Error")
        if self.logic and self.logic.llmClient:
            self.logic.llmClient.debug_suffix = ""
        self.appendToChat("Error", f"Self-correction failed: {error_msg}")
        self._setReadyStatus()
