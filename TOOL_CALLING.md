# Tool Calling Feature

## Overview

SlicerKimiAgent now supports **Tool Calling** (Function Calling) with the Kimi API. This allows the AI to:

1. Search the slicer-skill repository using built-in tools
2. Get accurate API information before generating code
3. Generate more reliable Slicer Python code based on actual skill content

## How It Works

### User Flow

1. User enters a prompt (e.g., "Load an example volume")
2. AI automatically searches the skill using tools:
   - `Grep`: Search for function names or keywords
   - `ReadFile`: Read documentation files
   - `Glob`: Find files by pattern
3. AI gets search results and generates code based on actual API
4. Code is validated and presented to user

### Architecture

```
User Prompt → Kimi API → AI generates tool_calls → Execute tools → AI generates final response
```

## Components

### 1. SkillTools.py
Cross-platform tool executor for searching the skill:

- **Windows**: Uses PowerShell `Select-String` for fast regex search
- **Linux/macOS**: Uses system `grep` command
- **Fallback**: Pure Python regex implementation

Tools available:
- `Grep(pattern, path)`: Search for regex pattern in files
- `ReadFile(path)`: Read file content
- `Glob(pattern, path)`: Find files matching pattern

### 2. KimiClient.py
Extended with `chatWithTools()` method for multi-turn tool calling conversations.

### 3. System Prompt
Updated to instruct AI to always search the skill before writing code.

## Configuration

Tool calling is enabled by default. To disable (fallback to regular chat):

```python
response = self.logic.generateResponseStream(prompt, context, _onDelta, use_tools=False)
```

## Testing

Test tool calling with these prompts:

1. **Volume Loading**: "How do I load an example MR volume?"
2. **Segmentation**: "Show me how to create a segmentation"
3. **Volume Rendering**: "How do I set up volume rendering?"

Expected behavior:
- AI should call `Grep` or `ReadFile` tools
- AI should reference skill search results in [Thinking] section
- Generated code should use correct API names from skill

## Error Handling

If tool calling fails:
1. Logs warning message
2. Automatically falls back to regular streaming chat
3. User sees normal response (just without tool-based search)
