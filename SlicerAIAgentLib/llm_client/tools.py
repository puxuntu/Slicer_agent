from .common import *


class LLMClientToolsMixin:
    def _runToolLoop(
        self,
        messages: List[Dict[str, Any]],
        url: str,
        tools: List[Dict],
        tool_executor: Callable[[str, Dict], Dict],
        max_tool_rounds: int,
        on_progress: Optional[Callable[[Dict], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        reasoning_effort: str = "high",
        on_reasoning: Optional[Callable[[str, int], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Core tool-calling loop. Operates on messages in-place.
        Does NOT touch conversation_history or turn_number.
        Returns dict with content, reasoning_content, data, timing_report,
        tool_calls_history, intermediate_messages, and has_code flag.

        Args:
            on_reasoning: Optional callback(reasoning_text, round_number) fired
                          after each API response with thinking content.
                          Used for streaming thinking to disk in real-time.
        """
        tool_calls_history = []
        intermediate_messages = []
        all_reasoning_parts = []  # accumulate reasoning across all tool rounds
        timing_report = {
            'api_calls': 0,
            'tool_rounds': 0,
            'total_api_time': 0.0,
            'total_tool_time': 0.0,
            'total_other_time': 0.0,
            'total_tokens': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'rounds': [],
        }

        pending_workflow_wait = False
        for round_num in range(max_tool_rounds):
            logger.info(f"Tool calling round {round_num + 1}")
            round_start = time.time()

            logger.debug(f"Payload messages count: {len(messages)}")

            if on_status:
                on_status("Thinking...")

            try:
                api_start = time.time()

                payload = self._buildPayload(messages, stream=False, tools=tools, reasoning_effort=reasoning_effort)
                request = self._buildRequest(url, payload)

                # On first round, run a quick TCP probe to catch network issues early
                if round_num == 0:
                    self._probeConnection(url)

                data = self._fetchWithDiagnostics(request)
                api_time = time.time() - api_start
                if self._isClaude():
                    data = self._normalizeClaudeResponse(data)

                assistant_message = data['choices'][0]['message']
                content = self._coerceText(assistant_message.get('content', ''))
                reasoning_content = self._coerceText(assistant_message.get('reasoning_content', ''))
                if reasoning_content:
                    all_reasoning_parts.append(f"[Round {round_num + 1}]\n{reasoning_content}")
                    # Stream thinking to callback for real-time file writing
                    if on_reasoning:
                        try:
                            on_reasoning(reasoning_content, round_num + 1)
                        except Exception:
                            pass

                tool_calls = assistant_message.get('tool_calls')

                if not tool_calls:
                    usage = data.get('usage', {})
                    round_tokens = usage.get('total_tokens', 0)
                    round_prompt = usage.get('prompt_tokens', 0)
                    round_completion = usage.get('completion_tokens', 0)
                    timing_report['total_tokens'] += round_tokens
                    timing_report['total_prompt_tokens'] += round_prompt
                    timing_report['total_completion_tokens'] += round_completion
                    timing_report['api_calls'] += 1
                    timing_report['total_api_time'] += api_time
                    other_time = max(0, time.time() - round_start - api_time)
                    timing_report['total_other_time'] += other_time

                    code = self._extractCode(content)

                    timing_report['rounds'].append({
                        'round': round_num + 1,
                        'phase': 'generate' if code else 'text',
                        'api_time': round(api_time, 3),
                        'tool_time': 0.0,
                        'other_time': round(other_time, 3),
                        'round_time': round(time.time() - round_start, 3),
                        'tools': [],
                        'tokens': round_tokens,
                        'thinking': True,
                    })

                    if code:
                        if on_status:
                            on_status("Generating...")
                        accumulated_reasoning = '\n\n'.join(all_reasoning_parts) if all_reasoning_parts else reasoning_content
                        return {
                            'content': content,
                            'reasoning_content': accumulated_reasoning,
                            'data': data,
                            'timing_report': timing_report,
                            'tool_calls_history': tool_calls_history,
                            'intermediate_messages': intermediate_messages,
                            'has_code': True,
                        }

                    # If a workflow step is waiting for user input, return the
                    # assistant's text instead of nudging the LLM to keep going.
                    if pending_workflow_wait:
                        messages.append({
                            'role': 'assistant',
                            'content': content,
                        })
                        if reasoning_content:
                            messages[-1]['reasoning_content'] = reasoning_content
                        intermediate_messages.append(messages[-1])
                        accumulated_reasoning = '\n\n'.join(all_reasoning_parts) if all_reasoning_parts else reasoning_content
                        return {
                            'content': content,
                            'reasoning_content': accumulated_reasoning,
                            'data': data,
                            'timing_report': timing_report,
                            'tool_calls_history': tool_calls_history,
                            'intermediate_messages': intermediate_messages,
                            'has_code': False,
                            'workflow_wait': True,
                        }

                    messages.append({
                        'role': 'assistant',
                        'content': content,
                    })
                    if reasoning_content:
                        messages[-1]['reasoning_content'] = reasoning_content
                    intermediate_messages.append(messages[-1])

                    if on_progress:
                        on_progress({
                            'reasoning_content': f'[Text] Round {round_num + 1}: LLM output text without tools or code. Continuing...\n',
                            'content': '',
                            'round': round_num + 1,
                            'phase': 'text',
                        })

                    # If the round produced no tool_calls and no code, nudge the LLM
                    # to take action instead of just narrating what it plans to do
                    if not code:
                        messages.append({
                            'role': 'user',
                            'content': (
                                'Your previous response did not call any tools and did not produce code. '
                                'You MUST now either:\n'
                                '1. Call a tool (Grep, ReadFile, or VectorSearch) to gather information, OR\n'
                                '2. Output the final ```python code block directly.\n'
                                'Do not just describe what you plan to do — actually do it.'
                            ),
                        })
                    continue

                # Execute tool calls in parallel
                tool_results = []
                tool_names = []
                tool_start = time.time()

                def _execute_single(tool_call):
                    tool_id = tool_call.get('id')
                    function = tool_call.get('function', {})
                    tool_name = function.get('name')
                    tool_args_str = function.get('arguments', '{}')
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    try:
                        result = tool_executor(tool_name, tool_args)
                        # Strip the _prelude_globals side channel before LLM
                        # serialization. The widget-side executor consumes it
                        # via _lastWorkflowStep (see _executeTool in logic_core);
                        # the LLM only needs the small ``code`` string, not the
                        # multi-KB metadata dicts the prelude references.
                        llm_result = result
                        if isinstance(result, dict) and "_prelude_globals" in result:
                            llm_result = {
                                k: v for k, v in result.items()
                                if k != "_prelude_globals"
                            }
                        return {
                            "tool_result": {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": json.dumps(llm_result, ensure_ascii=False, default=str),
                            },
                            "history_entry": {
                                "tool": tool_name,
                                "args": tool_args,
                                "result": llm_result,
                            },
                            "name": tool_name,
                        }
                    except Exception as e:
                        return {
                            "tool_result": {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                            },
                            "history_entry": None,
                            "name": f"{tool_name}(error)",
                        }

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    outputs = list(executor.map(_execute_single, tool_calls))

                for out in outputs:
                    tool_result = out["tool_result"]
                    # ReadFile and Grep now handle their own output shaping at the tool layer.
                    # No post-compression needed here.
                    tool_results.append(tool_result)
                    if out["history_entry"] is not None:
                        tool_calls_history.append(out["history_entry"])
                    tool_names.append(out["name"])

                # Detect workflow steps that require waiting for user input
                for out in outputs:
                    try:
                        result_data = json.loads(out["tool_result"]["content"])
                        if isinstance(result_data, dict) and result_data.get("type") == "user_choice":
                            pending_workflow_wait = True
                    except Exception:
                        pass

                tool_time = time.time() - tool_start
                usage = data.get('usage', {})
                round_tokens = usage.get('total_tokens', 0)
                round_prompt = usage.get('prompt_tokens', 0)
                round_completion = usage.get('completion_tokens', 0)
                timing_report['total_tokens'] += round_tokens
                timing_report['total_prompt_tokens'] += round_prompt
                timing_report['total_completion_tokens'] += round_completion
                timing_report['api_calls'] += 1
                timing_report['tool_rounds'] += 1
                timing_report['total_api_time'] += api_time
                timing_report['total_tool_time'] += tool_time
                other_time = max(0, time.time() - round_start - api_time - tool_time)
                timing_report['total_other_time'] += other_time

                has_search = any(n in ('Grep', 'SearchSymbol') for n in tool_names)
                has_readfile = any(n == 'ReadFile' for n in tool_names)
                has_vector = any(n == 'VectorSearch' for n in tool_names)
                has_segmentation = any(n == 'GenerateSegmentationCode' for n in tool_names)
                if has_search and has_readfile:
                    phase_label = "Search+Read"
                elif has_readfile:
                    phase_label = "Read"
                elif has_search or has_vector:
                    phase_label = "Search"
                elif has_segmentation:
                    phase_label = "Segmentation"
                else:
                    phase_label = "Tools"

                if on_status:
                    if has_readfile and (has_search or has_vector):
                        on_status("Searching & reading...")
                    elif has_readfile:
                        on_status("Reading...")
                    elif has_search or has_vector:
                        on_status("Searching...")
                    elif has_segmentation:
                        on_status("Generating segmentation code...")
                    else:
                        on_status("Running tools...")

                timing_report['rounds'].append({
                    'round': round_num + 1,
                    'phase': phase_label.lower(),
                    'api_time': round(api_time, 3),
                    'tool_time': round(tool_time, 3),
                    'other_time': round(other_time, 3),
                    'round_time': round(time.time() - round_start, 3),
                    'tools': tool_names,
                    'tokens': round_tokens,
                    'thinking': True,
                })

                assistant_msg = {
                    "role": "assistant",
                    "content": content if content else "",
                    "tool_calls": tool_calls,
                }
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)
                intermediate_messages.append(assistant_msg)
                messages.extend(tool_results)
                intermediate_messages.extend(tool_results)

                if on_progress:
                    progress_lines = [f"[{phase_label}] Round {round_num + 1}:"]
                    for tc in tool_calls_history[-len(tool_results):]:
                        tool_name = tc['tool']
                        args = tc['args']
                        if tool_name == 'Grep':
                            pattern = args.get('pattern', 'N/A')
                            path = args.get('path', 'N/A')
                            progress_lines.append(f'  Grep: "{pattern}" → {path}')
                        elif tool_name == 'ReadFile':
                            path = args.get('path', 'N/A')
                            progress_lines.append(f'  ReadFile: {path}')
                        elif tool_name == 'SearchSymbol':
                            pattern = args.get('pattern', 'N/A')
                            path = args.get('path', 'N/A')
                            sym_type = args.get('type', 'all')
                            progress_lines.append(f'  SearchSymbol({sym_type}): "{pattern}" → {path}')
                        elif tool_name == 'VectorSearch':
                            query = args.get('query', 'N/A')
                            progress_lines.append(f'  VectorSearch: "{query}"')
                        else:
                            progress_lines.append(f'  {tool_name}: {args}')
                    progress_msg = '\n'.join(progress_lines) + '\n'
                    on_progress({'reasoning_content': progress_msg, 'content': '', 'round': round_num + 1, 'phase': phase_label.lower()})

                logger.info(f"Round {round_num + 1} complete. Added {len(tool_results)} tool results. Proceeding to next round.")

            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8', errors='ignore')
                logger.error(f"HTTP Error {e.code} in chatWithTools round {round_num + 1}: {error_body}")
                try:
                    debug_msgs = json.dumps(messages, indent=2, default=str, ensure_ascii=False)[:3000]
                    logger.debug(f"Messages sent: {debug_msgs}")
                except:
                    pass
                raise RuntimeError(f"API Error {e.code}: {error_body}")
            except Exception as e:
                logger.error(f"Error in chatWithTools round {round_num + 1}: {e}")
                raise

        logger.warning(f"Max tool rounds ({max_tool_rounds}) reached — forcing final code generation")

        if on_status:
            on_status("Generating...")

        # Force one final call without tools to generate the code
        messages.append({
            'role': 'user',
            'content': (
                "You have reached the maximum number of search rounds. "
                "Stop searching and generate the final answer now. "
                "You MUST output exactly two fenced blocks in this order:\n"
                "1. ```agent_plan with valid JSON using the required schema.\n"
                "2. ```python with the complete executable code.\n"
                "Do not output Python without an agent_plan. "
                "Do not emit tool calls, DSML, XML, or search instructions."
            ),
        })
        intermediate_messages.append(messages[-1])

        final_start = time.time()
        payload = self._buildPayload(messages, stream=False, tools=None, reasoning_effort=reasoning_effort)
        request = self._buildRequest(url, payload)
        with self._openRequest(request) as response:
            data = json.loads(response.read().decode('utf-8'))
        api_time = time.time() - final_start
        if self._isClaude():
            data = self._normalizeClaudeResponse(data)

        assistant_message = data['choices'][0]['message']
        content = self._coerceText(assistant_message.get('content', ''))
        reasoning_content = self._coerceText(assistant_message.get('reasoning_content', ''))
        if reasoning_content:
            all_reasoning_parts.append(f"[Final Generation]\n{reasoning_content}")

        usage = data.get('usage', {})
        round_tokens = usage.get('total_tokens', 0)
        timing_report['total_tokens'] += round_tokens
        timing_report['total_prompt_tokens'] += usage.get('prompt_tokens', 0)
        timing_report['total_completion_tokens'] += usage.get('completion_tokens', 0)
        timing_report['api_calls'] += 1
        timing_report['total_api_time'] += api_time
        timing_report['rounds'].append({
            'round': max_tool_rounds + 1,
            'phase': 'generate',
            'api_time': round(api_time, 3),
            'tool_time': 0.0,
            'other_time': 0.0,
            'round_time': round(time.time() - final_start, 3),
            'tools': [],
            'tokens': round_tokens,
            'thinking': True,
        })

        code = self._extractCode(content)
        accumulated_reasoning = '\n\n'.join(all_reasoning_parts) if all_reasoning_parts else reasoning_content
        return {
            'content': content,
            'reasoning_content': accumulated_reasoning,
            'data': data,
            'timing_report': timing_report,
            'tool_calls_history': tool_calls_history,
            'intermediate_messages': intermediate_messages,
            'has_code': bool(code),
        }

    def chatWithTools(
        self,
        prompt: str,
        tools: List[Dict],
        tool_executor: Callable[[str, Dict], Dict],
        context: Optional[Dict] = None,
        max_tool_rounds: int = 10,
        on_progress: Optional[Callable[[Dict], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str, int], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request with tool calling support.

        The LLM has access to ALL tools (Grep, ReadFile, VectorSearch) from the start and autonomously
        decides when to search, when to read, and when to generate code. The loop terminates
        when the LLM outputs a ```python code block.

        Args:
            prompt: User's input prompt
            tools: List of tool definitions for the AI
            tool_executor: Function that executes tool calls (name, args) -> result
            context: Optional skill-based context
            max_tool_rounds: Maximum number of tool call rounds
            on_progress: Callback for progress updates (reasoning_content, content, round_info)
            on_reasoning_delta: Per-chunk callback for streaming thinking text

        Returns:
            Dictionary with final response, code, tokens, cost, and tool call history
        """
        if not self.api_key:
            raise RuntimeError("API key not configured")

        messages = self._buildMessages(prompt, context)
        url = self._getChatUrl()

        result = self._runToolLoop(
            messages=messages,
            url=url,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            on_progress=on_progress,
            on_status=on_status,
            reasoning_effort="high",
            on_reasoning=on_reasoning,
            on_reasoning_delta=on_reasoning_delta,
        )

        content = result['content']
        reasoning_content = result['reasoning_content']
        data = result['data']
        timing_report = result['timing_report']
        tool_calls_history = result['tool_calls_history']
        intermediate_messages = result['intermediate_messages']
        if result['has_code'] or result.get('workflow_wait'):
            # Final response with code or workflow wait - DEBUG: write messages to file
            try:
                debug_path = self._debugPath(f'{self.turn_number}_last_prompt_debug{self.debug_suffix}.txt')
                with open(debug_path, 'w', encoding='utf-8') as f:
                    total_user_msgs = sum(1 for m in messages if m.get('role') == 'user')
                    users_seen = 0
                    for i, msg in enumerate(messages):
                        if msg.get('role') == 'user':
                            users_seen += 1
                            turn_label = self.turn_number - total_user_msgs + users_seen
                            f.write(f"\n{'-'*40}\n")
                            f.write(f"--- Turn {turn_label} ---\n")
                            f.write(f"{'-'*40}\n")
                        f.write(f"{'='*60}\n")
                        f.write(f"MESSAGE {i+1} | role: {msg.get('role', 'unknown')}\n")
                        f.write(f"{'='*60}\n")
                        if 'tool_calls' in msg:
                            f.write("[tool_calls present]\n")
                        f.write(f"{msg.get('content', '')}\n\n")
            except Exception:
                pass

            # Persist full turn including tool calling trajectory (compressed for history)
            self.conversation_history.append({'role': 'user', 'content': prompt})
            if intermediate_messages:
                compressed_messages = self._compressToolResultsForHistory(intermediate_messages, user_prompt=prompt)
                self.conversation_history.extend(compressed_messages)
            assistant_entry = {'role': 'assistant', 'content': content}
            if reasoning_content:
                assistant_entry['reasoning_content'] = reasoning_content
            self.conversation_history.append(assistant_entry)
            self.turn_number += 1

        intermediate_messages = result['intermediate_messages']

        # Use accumulated usage across all tool rounds so response['tokens']
        # and response['cost'] reflect the total for this turn, not just the
        # last API call.
        accumulated_usage = {
            'prompt_tokens': timing_report['total_prompt_tokens'],
            'completion_tokens': timing_report['total_completion_tokens'],
            'total_tokens': timing_report['total_tokens'],
        }
        response = self._buildResponse(
            content,
            reasoning_content,
            accumulated_usage,
            data,
        )
        response['tool_calls_history'] = tool_calls_history
        response['timing_report'] = timing_report
        response['intermediate_messages'] = intermediate_messages
        response['workflow_wait'] = result.get('workflow_wait')
        return response

    def decomposeQuery(self, prompt: str) -> List[str]:
        """
        Decompose a complex user query into sub-task queries for multi-retrieval.

        Short or simple queries are returned as-is without API call.
        Complex multi-step queries are broken into 2-5 independent sub-queries
        via a lightweight LLM call.

        Returns:
            List of query strings. Always returns at least [prompt].
        """
        # Heuristic: only truly short and simple queries skip decomposition
        word_count = len(prompt.split())
        comma_count = prompt.count(',')
        and_count = prompt.lower().count(' and ')
        if word_count < 12 and comma_count == 0 and and_count <= 1:
            return [prompt]

        system_msg = (
            "You are a task decomposition assistant for 3D Slicer medical image analysis software.\n"
            "Analyze the user's request and break it into independent sub-tasks.\n"
            "Each sub-task should be a concise natural-language query suitable for code/API retrieval.\n\n"
            "Rules:\n"
            "- If the request is simple (1-2 clear steps), return it as a single task unchanged.\n"
            "- For complex multi-step requests, break it into independent sub-tasks. Group related operations together (e.g., loading data + initial display is one task); split only when the next step needs a different API domain or concept.\n"
            "- Each sub-task must be self-contained and mention the specific Slicer operation.\n"
            "- Output ONLY a JSON array of strings. No markdown, no explanation.\n\n"
            'Output format example: ["load a DICOM volume", '
            '"apply threshold segmentation to create a segment", '
            '"export segmentation as STL model"]'
        )

        # Save/restore conversation history to avoid polluting user dialog
        saved_history = list(self.conversation_history)
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"User request: {prompt}"},
            ]
            payload = self._buildPayload(messages, stream=False, thinking=False)
            url = self._getChatUrl()
            request = self._buildRequest(url, payload)

            with self._openRequest(request) as response:
                data = json.loads(response.read().decode('utf-8'))

            if self._isClaude():
                data = self._normalizeClaudeResponse(data)

            content = data['choices'][0]['message'].get('content', '').strip()

            # Extract JSON array from response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                try:
                    sub_queries = json.loads(match.group())
                    if isinstance(sub_queries, list) and len(sub_queries) > 0:
                        validated = [str(q).strip() for q in sub_queries if str(q).strip()]
                        if validated:
                            return validated
                except json.JSONDecodeError:
                    pass

            # Fallback: return original if parsing fails
            return [prompt]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [prompt]
        finally:
            self.conversation_history = saved_history

    def chatWithToolsIsolated(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict],
        tool_executor: Callable[[str, Dict], Dict],
        max_tool_rounds: int = 10,
        on_progress: Optional[Callable[[Dict], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str, int], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run tool-calling loop with fully isolated context.
        Does NOT read from or write to conversation_history.
        Does NOT increment turn_number.
        Accepts a pre-built messages list (e.g. for self-correction).

        Args:
            messages: Pre-built message list (including system prompt, user request, etc.)
            tools: List of tool definitions
            tool_executor: Function that executes tool calls
            max_tool_rounds: Maximum rounds (default 50)
            on_progress: Optional progress callback

        Returns:
            Dictionary with response, code, tokens, cost, timing_report, tool_calls_history,
            intermediate_messages
        """
        if not self.api_key:
            raise RuntimeError("API key not configured")

        url = self._getChatUrl()

        result = self._runToolLoop(
            messages=messages,
            url=url,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            on_progress=on_progress,
            reasoning_effort="high",
            on_reasoning=on_reasoning,
            on_reasoning_delta=on_reasoning_delta,
        )

        content = result['content']
        reasoning_content = result['reasoning_content']
        data = result['data']
        timing_report = result['timing_report']
        tool_calls_history = result['tool_calls_history']

        # Use accumulated usage across all tool rounds (same as chatWithTools).
        accumulated_usage = {
            'prompt_tokens': timing_report['total_prompt_tokens'],
            'completion_tokens': timing_report['total_completion_tokens'],
            'total_tokens': timing_report['total_tokens'],
        }
        response = self._buildResponse(
            content,
            reasoning_content,
            accumulated_usage,
            data,
        )
        response['tool_calls_history'] = tool_calls_history
        response['timing_report'] = timing_report
        return response
