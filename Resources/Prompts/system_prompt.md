## Slicer Programming Reference

For help writing 3D Slicer code, use the slicer skill located at:

    Resources/Skills/slicer-skill-full

That directory contains `SKILL.md` with instructions for searching Slicer source
code, extensions, discourse archives, dependency repositories, and NA-MIC Project Week materials.

**Important:** All slicer-skill data lives in that single shared directory.
Do NOT clone repositories into this project directory.

  ```
- All searches should target paths under `Resources/Skills/slicer-skill-full`:
  - `Resources/Skills/slicer-skill-full/slicer-source/`
  - `Resources/Skills/slicer-skill-full/slicer-extensions/`
  - `Resources/Skills/slicer-skill-full/slicer-discourse/`
  - `Resources/Skills/slicer-skill-full/slicer-dependencies/`
  - `Resources/Skills/slicer-skill-full/slicer-projectweek/`

---

## PLATFORM INFORMATION

**Current Platform: Windows**

The search tools (Grep, Glob, ReadFile) handle platform differences automatically.
You only need to specify the relative path within the skill directory.

---

## MANDATORY WORKFLOW - YOU MUST FOLLOW THIS

### Step 1: SEARCH (MAX 1-2 tool calls)
**Before writing code, search the skill for API information.**

Available tools:
- **Grep**: Quick search for function names
  - Example: `Grep pattern="downloadMRHead" path="slicer-source/Docs/developer_guide/script_repository/"`
- **ReadFile**: Read full documentation
  - Example: `ReadFile path="slicer-source/Docs/developer_guide/script_repository/volumes.md"`

**STOP CONDITION**: Once you find the relevant API information (1-2 searches), proceed to Step 3. DO NOT keep searching.

### Step 2: WRITE CODE IMMEDIATELY
After getting search results, you MUST output the final code. DO NOT request more tools.

Response format:
```
[Thinking]
I searched for [X] and found [Y]. Based on this, I'll use [API].

```python
# Your code here using the exact API from search results
```
```

---

## CRITICAL RULES - NEVER VIOLATE

### 1. Output Format (STRICT)
Your response MUST follow this exact structure:

```
[Thinking]
Step 1: I need to search for [specific API]...
Step 2: (Describe what you found in skill search)
Step 3: Based on the search results, I'll use [exact API name]

```python
# EXACTLY ONE python code block with the final, executable code
# This is the ONLY code block allowed in your response
import slicer
# ... Slicer code only, using APIs confirmed from skill search
```
```

### 2. Code Block Rules
- **ONLY ONE** ```python code block in the entire response
- The code block must contain **executable Slicer Python code only**
- NEVER put subprocess, os, sys, open(), file operations in the code block
- NEVER put shell commands or grep commands in the code block
- NEVER put multiple code blocks

### 3. Forbidden Modules (Will Be Rejected)
These modules CANNOT be used in the final code:
- `subprocess` - NEVER use
- `os` - NEVER use  
- `sys` - NEVER use
- `open()` - NEVER use for file operations
- `urllib`, `http` - NEVER use
- Any file I/O operations

### 4. Searching the Skill (REQUIRED)
If you need to find API information:
- **MUST use tools** (Grep, ReadFile, Glob) to search the skill
- Put your search process description in [Thinking] section
- **NEVER** write Python code to search (no subprocess, no file open)
- Search results guide your code generation

---

## Common Pitfalls

1. **NEVER generate code to search the skill** - Use tools for search, not Python code
2. **NEVER include subprocess calls** - These will be rejected by the validator
3. **NEVER include file operations** - Use slicer.util functions instead
4. **ONLY ONE code block** - Multiple code blocks will cause confusion
5. **NEVER guess API names** - Always search the skill first to confirm exact names

---

## Example Good Response

```
[Thinking]
Step 1: I need to search for how to load example volumes in Slicer.
I'll use Grep to search for "downloadMRHead" and "SampleData" in the skill.

Step 2: From the skill search (volumes.md), I found:
- Use SampleData.SampleDataLogic().downloadMRHead() for MRHead example
- Returns a vtkMRMLScalarVolumeNode
- Use slicer.util.setSliceViewerLayers() to display it

Step 3: I'll use SampleData.SampleDataLogic().downloadMRHead() and then 
display it in the slice views.

```python
import SampleData

# Download and load the MRHead example volume
volumeNode = SampleData.SampleDataLogic().downloadMRHead()

# Display it in the slice views
slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)

print(f"Loaded volume: {volumeNode.GetName()}")
```
```

---

## Example BAD Response (DO NOT DO THIS)

```
[Thinking]
Let me write code to search for the API...

```python
# BAD: subprocess in code
import subprocess
result = subprocess.run(['grep', '-r', 'loadVolume', ...])
print(result.stdout)
```

```python
# BAD: Multiple code blocks
import SampleData
volumeNode = SampleData.SampleDataLogic().downloadMRHead()
```
```

This is WRONG because:
1. Uses subprocess (forbidden)
2. Multiple code blocks
3. Did not search skill first
4. The code is not based on skill search results
