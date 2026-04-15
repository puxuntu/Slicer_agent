"""
LLMClient - HTTP client for LLM API communication.

Supports streaming responses, conversation history, token tracking, and tool calling.
System prompt is loaded from external markdown file.

Compatible with OpenAI-compatible APIs including Kimi, OpenAI, and others.
"""

import json
import logging
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with LLM APIs (OpenAI-compatible).

    Features:
    - Streaming response handling
    - Multi-turn conversation support
    - Token usage tracking
    - Retry logic with exponential backoff
    - Compatible with Kimi, OpenAI, and other OpenAI-compatible APIs
    """

    # API Configuration
    # Compatible with OpenAI API format
    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
    DEFAULT_MODEL = "kimi-k2.5"
    DEFAULT_TIMEOUT = None  # No client-side timeout for API requests
    MAX_RETRIES = 5  # Retry up to 5 times for transient errors
    LEGACY_MODEL_ALIASES = {
        "kimi-latest": DEFAULT_MODEL,
        "kimi-k2-16k": DEFAULT_MODEL,
        "kimi-2.5": DEFAULT_MODEL,
    }

    # Pricing per 1K tokens (approximate, update as needed)
    # See: https://platform.moonshot.cn/docs/pricing
    MODEL_PRICING = {
        "kimi-k2.5": {"input": 0.002, "output": 0.006},
        "kimi-k2-thinking": {"input": 0.002, "output": 0.006},
        "kimi-k2-turbo-preview": {"input": 0.001, "output": 0.003},
        "kimi-k2-0905-preview": {"input": 0.002, "output": 0.006},
        "moonshot-v1-8k": {"input": 0.001, "output": 0.002},
        "moonshot-v1-32k": {"input": 0.002, "output": 0.004},
        "moonshot-v1-128k": {"input": 0.006, "output": 0.012},
    }

    # Path to system prompt file (relative to this file)
    SYSTEM_PROMPT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Resources', 'Prompts', 'system_prompt.md'
    )

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: API key (optional, can be set later)
            model: Model name to use (default: kimi-k2.5)
        """
        self.api_key = api_key
        self.model = self._normalizeModelName(model or self.DEFAULT_MODEL)
        self.base_url = self.DEFAULT_BASE_URL
        self.timeout = self.DEFAULT_TIMEOUT
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.turn_number = 1
        self._system_prompt_template = self._loadSystemPromptTemplate()

    def _normalizeModelName(self, model: Optional[str]) -> str:
        """Map legacy or empty model names to supported current model names."""
        normalized_model = (model or "").strip()
        if not normalized_model:
            return self.DEFAULT_MODEL
        return self.LEGACY_MODEL_ALIASES.get(normalized_model, normalized_model)

    def setApiKey(self, api_key: str):
        """Set or update the API key."""
        self.api_key = api_key

    def setModel(self, model: str):
        """Set the model to use."""
        self.model = self._normalizeModelName(model)

    def setBaseUrl(self, base_url: str):
        """Set a custom base URL (for enterprise deployments)."""
        self.base_url = base_url.rstrip('/')

    def clearHistory(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.turn_number = 1
        logger.info("Conversation history cleared")

    def getHistory(self) -> List[Dict[str, Any]]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def setHistory(self, history: List[Dict[str, Any]]):
        """Set conversation history."""
        self.conversation_history = history

    def _buildHeaders(self) -> Dict[str, str]:
        """Build HTTP headers for API requests."""
        if not self.api_key:
            raise RuntimeError("API key not configured. Please set your API key in Settings.")

        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _supportsThinking(self) -> bool:
        """Return True if the current model should receive the thinking parameter."""
        return self.model.startswith("kimi-k2")

    def _buildMessages(self, prompt: str, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Build the messages array for the API request.

        Args:
            prompt: User's input prompt
            context: Optional skill-based context

        Returns:
            List of message dictionaries
        """
        messages: List[Dict[str, Any]] = []

        if not self.conversation_history:
            system_content = self._buildSystemPrompt(context)
        else:
            # Lightweight system prompt for follow-up turns
            system_content = (
                "You are an expert 3D Slicer Python coding assistant. "
                "Critical rules still apply: exactly one ```python code block, no forbidden modules (os, subprocess, sys, open, eval, exec, getattr, etc.). "
                "Continue the conversation based on previous context and the current scene below.\n"
            )
            if context and context.get('scene'):
                scene = context['scene']
                system_content += "\n## CURRENT SLICER SCENE\n"
                system_content += "Raw unprocessed MRML scene context (let the AI analyze):\n```\n"
                try:
                    system_content += json.dumps(scene, ensure_ascii=False, indent=2)
                except Exception:
                    system_content += str(scene)
                system_content += "\n```\n"

        messages.append({"role": "system", "content": system_content})

        history_to_include = self.conversation_history[-50:]  # Keep last 50 messages for context
        messages.extend(history_to_include)

        messages.append({"role": "user", "content": prompt})

        # DEBUG: Write the first-turn prompt to a local file for inspection
        try:
            debug_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                f'{self.turn_number}_first_prompt_debug.txt'
            )
            with open(debug_path, 'w', encoding='utf-8') as f:
                for i, msg in enumerate(messages):
                    f.write(f"{'='*60}\n")
                    f.write(f"MESSAGE {i+1} | role: {msg.get('role', 'unknown')}\n")
                    f.write(f"{'='*60}\n")
                    if 'tool_calls' in msg:
                        f.write("[tool_calls present]\n")
                    f.write(f"{msg.get('content', '')}\n\n")
        except Exception:
            pass  # Silently ignore debug write failures

        return messages

    def _buildPayload(self, messages: List[Dict[str, Any]], stream: bool = False, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Build the API payload for chat completion requests."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if stream:
            payload["stream"] = True
        if self._supportsThinking():
            payload["thinking"] = {"type": "enabled"}
        if tools:
            payload["tools"] = tools
        return payload

    def _loadSystemPromptTemplate(self) -> str:
        """
        Load the system prompt template from external markdown file.
        
        Returns:
            System prompt template string
        """
        try:
            if os.path.exists(self.SYSTEM_PROMPT_PATH):
                with open(self.SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Loaded system prompt from {self.SYSTEM_PROMPT_PATH}")
                return content
            else:
                logger.warning(f"System prompt file not found at {self.SYSTEM_PROMPT_PATH}, using fallback")
                return self._getFallbackSystemPrompt()
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            return self._getFallbackSystemPrompt()
    
    def _getFallbackSystemPrompt(self) -> str:
        """
        Get a fallback system prompt if the file cannot be loaded.
        Matches the structure of Resources/Prompts/system_prompt.md.
        """
        return """## Slicer Programming Reference

For help writing 3D Slicer code, use the slicer skill located at:
Resources/Skills/slicer-skill-full

All searches should target paths under Resources/Skills/slicer-skill-full/.

## YOUR ROLE

You are an expert 3D Slicer Python coding assistant. Your job is to convert the user's natural language request into safe, executable Python code for 3D Slicer.

## WORKFLOW

1. Search when needed. If you are not 100% certain about the exact API name or usage, use the available tools (Grep, ReadFile, Glob) to search the Slicer skill knowledge base.
2. Stop searching once you know enough. Do not perform repeated, unnecessary searches for the same topic.
3. Write the final code immediately. Once you have confirmed the correct API, respond with the final Python code. Do not request more tools after you have enough information.

## RESPONSE FORMAT

Your response must contain exactly one ```python code block with the executable Slicer code.

You may optionally include 1-2 sentences of explanation before the code block. Do not write long essays.

## CRITICAL RULES - NEVER VIOLATE

### 1. Exactly One Code Block
- ONLY ONE ```python code block in the entire response.
- The code block must contain executable Slicer Python code only.
- NEVER put shell commands, subprocess calls, or grep commands inside the code block.
- NEVER put multiple code blocks.

### 2. Forbidden Modules & Functions
These CANNOT be used in the final code. Code using them will be rejected:
- System/OS: os, subprocess, sys, socket, ctypes, mmap, signal, pty, resource
- Execution: eval, exec, compile, execfile, __import__
- Networking: urllib, urllib2, http, ftplib, telnetlib
- Serialization: pickle, cPickle, shelve, marshal, imp
- File I/O: open(), file(), input(), raw_input()
- Reflection: getattr, setattr, delattr, globals, locals, vars, dir

### 3. Search with Tools, Not Code
- If you need to find API information, MUST use tools (Grep, ReadFile, Glob).
- NEVER write Python code to search the skill (no subprocess, no file open, no os.walk).
- Search results should guide your code generation.

### 4. Common Slicer Pitfalls
- After modifying volume arrays with arrayFromVolume(), always call arrayFromVolumeModified().
- Volume arrays are in KJI order (slice, row, column), not IJK.

## EXAMPLE GOOD RESPONSE

I searched the skill and found that SampleData.SampleDataLogic().downloadMRHead() downloads the MRHead sample volume. I'll use this to load the volume.

```python
import SampleData
volumeNode = SampleData.SampleDataLogic().downloadMRHead()
slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)
print(f"Loaded volume: {volumeNode.GetName()}")
```

## EXAMPLE BAD RESPONSE

Let me search for the API by running a shell command:

```python
import subprocess
result = subprocess.run(['grep', '-r', 'loadVolume', ...])
print(result.stdout)
```

This is wrong because it uses subprocess instead of the provided tools.
"""

    def _buildSystemPrompt(self, context: Optional[Dict] = None) -> str:
        """
        Build the comprehensive system prompt with Slicer expertise.
        Loads base prompt from external file and appends dynamic context.

        Args:
            context: Skill-based context with API hints, scene information, and tool availability

        Returns:
            System prompt string
        """
        # Start with the template from file
        base_prompt = self._system_prompt_template

        # Inject dynamic platform information
        import platform
        base_prompt += f"\n\n## PLATFORM INFORMATION\n"
        base_prompt += f"Current Platform: {platform.system()}\n"
        base_prompt += "The search tools (Grep, Glob, ReadFile) handle platform differences automatically.\n"
        base_prompt += "You only need to specify the relative path within the skill directory.\n"

        # Add dynamic scene context
        if context and context.get('scene'):
            scene = context['scene']
            base_prompt += "\n\n## CURRENT SLICER SCENE\n"
            base_prompt += "Raw unprocessed MRML scene context (let the AI analyze):\n```\n"
            try:
                base_prompt += json.dumps(scene, ensure_ascii=False, indent=2)
            except Exception:
                base_prompt += str(scene)
            base_prompt += "\n```\n"

        return base_prompt

    def _openRequest(self, request: urllib.request.Request):
        """Open an HTTP request with optional timeout support."""
        if self.timeout is None:
            return urllib.request.urlopen(request)
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _buildRequest(self, url: str, payload: Optional[Dict[str, Any]] = None, method: str = 'POST') -> urllib.request.Request:
        """Create an HTTP request for the LLM API."""
        data = None
        if payload is not None:
            data = json.dumps(payload).encode('utf-8')
        return urllib.request.Request(
            url,
            data=data,
            headers=self._buildHeaders(),
            method=method,
        )

    def _timeoutErrorMessage(self) -> str:
        """Build a user-facing timeout error message."""
        if self.timeout is None:
            return "Request timed out. Please check your network connection and try again."
        return f"Request timed out after {self.timeout} seconds. Please check your network connection and try again."

    def _coerceText(self, value: Any) -> str:
        """Convert streamed delta values into plain text safely."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    if 'text' in item:
                        parts.append(str(item.get('text', '')))
                    elif 'content' in item:
                        parts.append(str(item.get('content', '')))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return ''.join(parts)
        return str(value)

    def _iterSseDataLines(self, response) -> List[str]:
        """Yield merged SSE data payloads from a streaming HTTP response."""
        event_lines: List[str] = []
        for raw_line in response:
            line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
            if not line:
                if event_lines:
                    yield '\n'.join(event_lines)
                    event_lines = []
                continue
            if line.startswith('data:'):
                event_lines.append(line[5:].strip())
        if event_lines:
            yield '\n'.join(event_lines)

    def _parseStreamChunk(self, data_line: str) -> Dict[str, Any]:
        """Parse one SSE data payload into content and reasoning deltas."""
        if data_line == '[DONE]':
            return {
                'done': True,
                'content': '',
                'reasoning_content': '',
                'finish_reason': 'stop',
                'usage': {},
                'raw_chunk': None,
            }

        payload = json.loads(data_line)
        choice = (payload.get('choices') or [{}])[0]
        delta = choice.get('delta') or {}
        message = choice.get('message') or {}
        content = self._coerceText(delta.get('content')) or self._coerceText(message.get('content'))
        reasoning_content = self._coerceText(delta.get('reasoning_content')) or self._coerceText(message.get('reasoning_content'))
        finish_reason = choice.get('finish_reason')

        return {
            'done': False,
            'content': content,
            'reasoning_content': reasoning_content,
            'finish_reason': finish_reason,
            'usage': payload.get('usage', {}),
            'raw_chunk': payload,
        }

    def _appendConversation(self, prompt: str, assistant_message: str, reasoning_content: str = ''):
        """Store user and assistant messages in local conversation history."""
        self.conversation_history.append({'role': 'user', 'content': prompt})
        assistant_entry: Dict[str, Any] = {'role': 'assistant', 'content': assistant_message}
        if reasoning_content:
            assistant_entry['reasoning_content'] = reasoning_content
        self.conversation_history.append(assistant_entry)

    def _buildResponse(self, message: str, reasoning_content: str, usage: Dict[str, Any], raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Build the normalized response dictionary returned to callers."""
        tokens_used = usage.get('total_tokens', 0)
        self.total_tokens_used += tokens_used

        cost = self._calculateCost(usage)
        self.total_cost += cost

        code = self._extractCode(message)

        return {
            'message': message,
            'reasoning_content': reasoning_content,
            'code': code,
            'tokens': tokens_used,
            'cost': cost,
            'raw_response': raw_response,
        }

    def chat(self, prompt: str, context: Optional[Dict] = None, stream: bool = False) -> Dict[str, Any]:
        """
        Send a chat request to the LLM API.

        Args:
            prompt: User's input prompt
            context: Optional skill-based context
            stream: Whether to use streaming transport

        Returns:
            Dictionary with keys:
                - message: The text response
                - reasoning_content: The model's reasoning text if available
                - code: Extracted Python code (if any)
                - tokens: Total tokens used
                - cost: Estimated cost in USD
        """
        if stream:
            return self.chatStream(prompt, context=context)

        if not self.api_key:
            raise RuntimeError("API key not configured")

        messages = self._buildMessages(prompt, context)
        payload = self._buildPayload(messages, stream=False)
        url = f"{self.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                request = self._buildRequest(url, payload)
                with self._openRequest(request) as response:
                    data = json.loads(response.read().decode('utf-8'))

                assistant_payload = data['choices'][0]['message']
                assistant_message = self._coerceText(assistant_payload.get('content'))
                reasoning_content = self._coerceText(assistant_payload.get('reasoning_content'))
                self._appendConversation(prompt, assistant_message, reasoning_content)
                self.turn_number += 1
                return self._buildResponse(
                    assistant_message,
                    reasoning_content,
                    data.get('usage', {}),
                    data,
                )

            except urllib.error.HTTPError as e:
                last_error = e
                error_body = e.read().decode('utf-8')
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e.code} - {error_body}")

                if e.code == 401:
                    raise RuntimeError("Invalid API key. Please check your API key.")
                if e.code == 404:
                    error_data = json.loads(error_body) if error_body else {}
                    error_msg = error_data.get('error', {}).get('message', 'Model not found')
                    raise RuntimeError(
                        f"Model error: {error_msg}\n\n"
                        f"Current model: '{self.model}'\n"
                        f"Try using 'kimi-k2.5' or check available models at:\n"
                        f"https://platform.moonshot.cn/docs/models"
                    )
                if e.code == 429:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                if e.code >= 500:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"API request failed: {e.code} - {error_body}")

            except urllib.error.URLError as e:
                last_error = e
                if isinstance(e.reason, socket.timeout):
                    raise RuntimeError(self._timeoutErrorMessage())
                logger.warning(f"URL error on attempt {attempt + 1}: {e}")
                import time
                time.sleep(1)
                continue

            except socket.timeout:
                raise RuntimeError(self._timeoutErrorMessage())

            except Exception as e:
                last_error = e
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                import time
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.MAX_RETRIES} attempts. Last error: {last_error}")

    def chatStream(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        on_delta: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Send a streaming chat request to the LLM API and assemble the full result.

        Args:
            prompt: User's input prompt
            context: Optional skill-based context
            on_delta: Optional callback receiving incremental deltas with
                `content` and `reasoning_content` keys.

        Returns:
            Dictionary with the assembled assistant message, reasoning text,
            extracted code, token count, cost, and raw response summary.
        """
        if not self.api_key:
            raise RuntimeError("API key not configured")

        messages = self._buildMessages(prompt, context)
        payload = self._buildPayload(messages, stream=True)
        url = f"{self.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                request = self._buildRequest(url, payload)
                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                usage: Dict[str, Any] = {}

                with self._openRequest(request) as response:
                    for data_line in self._iterSseDataLines(response):
                        chunk = self._parseStreamChunk(data_line)
                        if chunk['done']:
                            break

                        if chunk['usage']:
                            usage = chunk['usage']

                        if chunk['reasoning_content']:
                            reasoning_parts.append(chunk['reasoning_content'])
                        if chunk['content']:
                            content_parts.append(chunk['content'])

                        if on_delta and (chunk['content'] or chunk['reasoning_content']):
                            on_delta({
                                'content': chunk['content'],
                                'reasoning_content': chunk['reasoning_content'],
                                'finish_reason': chunk['finish_reason'],
                                'raw_chunk': chunk['raw_chunk'],
                            })

                assistant_message = ''.join(content_parts)
                reasoning_content = ''.join(reasoning_parts)
                self._appendConversation(prompt, assistant_message, reasoning_content)
                self.turn_number += 1

                raw_response = {
                    'choices': [
                        {
                            'message': {
                                'content': assistant_message,
                                'reasoning_content': reasoning_content,
                            }
                        }
                    ],
                    'usage': usage,
                }
                return self._buildResponse(assistant_message, reasoning_content, usage, raw_response)

            except urllib.error.HTTPError as e:
                last_error = e
                error_body = e.read().decode('utf-8')
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e.code} - {error_body}")

                if e.code == 401:
                    raise RuntimeError("Invalid API key. Please check your API key.")
                if e.code == 404:
                    error_data = json.loads(error_body) if error_body else {}
                    error_msg = error_data.get('error', {}).get('message', 'Model not found')
                    raise RuntimeError(
                        f"Model error: {error_msg}\n\n"
                        f"Current model: '{self.model}'\n"
                        f"Try using 'kimi-k2.5' or check available models at:\n"
                        f"https://platform.moonshot.cn/docs/models"
                    )
                if e.code == 429:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                if e.code >= 500:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"API request failed: {e.code} - {error_body}")

            except urllib.error.URLError as e:
                last_error = e
                if isinstance(e.reason, socket.timeout):
                    raise RuntimeError(self._timeoutErrorMessage())
                logger.warning(f"URL error on attempt {attempt + 1}: {e}")
                import time
                time.sleep(1)
                continue

            except socket.timeout:
                raise RuntimeError(self._timeoutErrorMessage())

            except Exception as e:
                last_error = e
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                import time
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.MAX_RETRIES} attempts. Last error: {last_error}")

    def chatWithTools(
        self,
        prompt: str,
        tools: List[Dict],
        tool_executor: Callable[[str, Dict], Dict],
        context: Optional[Dict] = None,
        max_tool_rounds: int = 20,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request with tool calling support.
        
        This method handles the multi-turn conversation for tool use:
        1. Send prompt with tools registered
        2. If AI requests tool calls, execute them
        3. Return tool results to AI
        4. Get final response with generated code
        
        Args:
            prompt: User's input prompt
            tools: List of tool definitions for the AI
            tool_executor: Function that executes tool calls (name, args) -> result
            context: Optional skill-based context
            max_tool_rounds: Maximum number of tool call rounds
            on_progress: Callback for progress updates (reasoning_content, content, round_info)
            
        Returns:
            Dictionary with final response, code, tokens, cost, and tool call history
        """
        if not self.api_key:
            raise RuntimeError("API key not configured")
        
        messages = self._buildMessages(prompt, context)
        url = f"{self.base_url}/chat/completions"
        tool_calls_history = []
        intermediate_messages = []  # Persist tool calling trajectory to conversation history
        
        for round_num in range(max_tool_rounds):
            logger.info(f"Tool calling round {round_num + 1}")
            payload = self._buildPayload(messages, stream=False, tools=tools)
            
            # Log payload for debugging (truncate for readability)
            logger.debug(f"Payload messages count: {len(messages)}")
            
            try:
                request = self._buildRequest(url, payload)
                with self._openRequest(request) as response:
                    data = json.loads(response.read().decode('utf-8'))
                
                assistant_message = data['choices'][0]['message']
                content = self._coerceText(assistant_message.get('content', ''))
                reasoning_content = self._coerceText(assistant_message.get('reasoning_content', ''))
                
                # Check if there are tool calls
                tool_calls = assistant_message.get('tool_calls')
                
                if not tool_calls:
                    # No tool calls, we have the final response
                    # DEBUG: Write the complete messages (including any tool results) to a local file
                    try:
                        debug_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            f'{self.turn_number}_last_prompt_debug.txt'
                        )
                        with open(debug_path, 'w', encoding='utf-8') as f:
                            for i, msg in enumerate(messages):
                                f.write(f"{'='*60}\n")
                                f.write(f"MESSAGE {i+1} | role: {msg.get('role', 'unknown')}\n")
                                f.write(f"{'='*60}\n")
                                if 'tool_calls' in msg:
                                    f.write("[tool_calls present]\n")
                                f.write(f"{msg.get('content', '')}\n\n")
                    except Exception:
                        pass

                    # Persist full turn including tool calling trajectory
                    self.conversation_history.append({'role': 'user', 'content': prompt})
                    if intermediate_messages:
                        self.conversation_history.extend(intermediate_messages)
                    assistant_entry = {'role': 'assistant', 'content': content}
                    if reasoning_content:
                        assistant_entry['reasoning_content'] = reasoning_content
                    self.conversation_history.append(assistant_entry)
                    self.turn_number += 1

                    response = self._buildResponse(
                        content,
                        reasoning_content,
                        data.get('usage', {}),
                        data,
                    )
                    response['tool_calls_history'] = tool_calls_history
                    return response
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    tool_id = tool_call.get('id')
                    function = tool_call.get('function', {})
                    tool_name = function.get('name')
                    tool_args_str = function.get('arguments', '{}')
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Execute the tool
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    try:
                        result = tool_executor(tool_name, tool_args)
                        # Tool message format: role, tool_call_id, content
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        })
                        tool_calls_history.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": result,
                        })
                    except Exception as e:
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                        })
                
                # Add assistant message with tool_calls to conversation
                # Must include content (can be empty string), tool_calls, and reasoning_content for k2 models
                assistant_msg = {
                    "role": "assistant",
                    "content": content if content else "",
                    "tool_calls": tool_calls,
                }
                # Add reasoning_content if present (required for thinking-enabled models)
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)
                intermediate_messages.append(assistant_msg)
                # Add tool results
                messages.extend(tool_results)
                intermediate_messages.extend(tool_results)
                # Add reminder for AI to provide final answer (not another tool call)
                reminder = "Tool results provided above. Now provide your final answer with the Python code. DO NOT request more tools."
                # Remove any previous identical reminders to keep the prompt clean
                messages = [m for m in messages if not (m.get("role") == "system" and m.get("content") == reminder)]
                reminder_msg = {
                    "role": "system",
                    "content": reminder,
                }
                messages.append(reminder_msg)
                intermediate_messages.append(reminder_msg)
                
                # Report progress with detailed tool info
                if on_progress:
                    progress_lines = [f"[Search] Round {round_num + 1}:"]
                    for tc in tool_calls_history[-len(tool_results):]:
                        tool_name = tc['tool']
                        args = tc['args']
                        # Build detailed description based on tool type
                        if tool_name == 'Grep':
                            pattern = args.get('pattern', 'N/A')
                            path = args.get('path', 'N/A')
                            progress_lines.append(f"  Grep: \"{pattern}\" → {path}")
                        elif tool_name == 'ReadFile':
                            path = args.get('path', 'N/A')
                            progress_lines.append(f"  ReadFile: {path}")
                        elif tool_name == 'Glob':
                            pattern = args.get('pattern', 'N/A')
                            path = args.get('path', 'N/A')
                            progress_lines.append(f"  Glob: {pattern} in {path}")
                        else:
                            progress_lines.append(f"  {tool_name}: {args}")
                    progress_msg = '\n'.join(progress_lines) + '\n'
                    on_progress({'reasoning_content': progress_msg, 'content': '', 'round': round_num + 1})
                
                logger.info(f"Round {round_num + 1} complete. Added {len(tool_results)} tool results. Proceeding to next round.")
                
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8', errors='ignore')
                logger.error(f"HTTP Error {e.code} in chatWithTools round {round_num + 1}: {error_body}")
                # Log messages for debugging
                try:
                    debug_msgs = json.dumps(messages, indent=2, default=str, ensure_ascii=False)[:3000]
                    logger.debug(f"Messages sent: {debug_msgs}")
                except:
                    pass
                raise RuntimeError(f"API Error {e.code}: {error_body}")
            except Exception as e:
                logger.error(f"Error in chatWithTools round {round_num + 1}: {e}")
                raise
        
        # Max rounds reached, return last response
        logger.warning(f"Max tool rounds ({max_tool_rounds}) reached")
        response = self._buildResponse(
            content,
            reasoning_content,
            data.get('usage', {}),
            data,
        )
        response['tool_calls_history'] = tool_calls_history
        return response

    def _extractCode(self, message: str) -> Optional[str]:
        """
        Extract Python code from the assistant's message.

        Args:
            message: The full response message

        Returns:
            Extracted code string or None if no code found
        """
        code_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(code_pattern, message, re.DOTALL)
        if matches:
            return matches[0]  # Enforce exactly one code block

        code_pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(code_pattern, message, re.DOTALL)
        if matches:
            return matches[0]  # Enforce exactly one code block

        return None

    def _calculateCost(self, usage: Dict[str, Any]) -> float:
        """
        Calculate the estimated cost of the API call.

        Args:
            usage: Token usage dictionary from API response

        Returns:
            Estimated cost in USD
        """
        pricing = self.MODEL_PRICING.get(self.model) or self.MODEL_PRICING[self.DEFAULT_MODEL]
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        return (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']

    def getStats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_tokens': self.total_tokens_used,
            'total_cost': self.total_cost,
            'conversation_length': len(self.conversation_history),
        }

    def testConnection(self) -> Dict[str, Any]:
        """
        Test API connection with a simple request.

        Returns:
            Dictionary with success status and available models (if returned)
        """
        if not self.api_key:
            return {'success': False, 'error': 'API key not configured'}

        try:
            url = f"{self.base_url}/models"
            request = self._buildRequest(url, payload=None, method='GET')
            with self._openRequest(request) as response:
                data = json.loads(response.read().decode('utf-8'))
                return {
                    'success': True,
                    'models': [m.get('id') for m in data.get('data', [])],
                    'message': 'Connection successful',
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            return {
                'success': False,
                'error': f"HTTP {e.code}: {error_body}",
                'message': 'Connection failed',
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Connection failed',
            }

    def validateModel(self, model_name: str = None) -> bool:
        """
        Test if a specific model name is valid.

        Args:
            model_name: Model name to test (default: current model)

        Returns:
            True if model is accessible
        """
        model = self._normalizeModelName(model_name or self.model)
        if not self.api_key:
            return False

        try:
            payload = {
                'model': model,
                'messages': [{'role': 'user', 'content': 'Hi'}],
                'max_tokens': 10,
            }
            request = self._buildRequest(f"{self.base_url}/chat/completions", payload)
            with self._openRequest(request):
                return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def cleanup(self):
        """Cleanup resources."""
        self.conversation_history = []
        logger.info("LLMClient cleaned up")
