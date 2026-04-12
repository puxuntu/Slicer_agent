# System Prompts

This directory contains the system prompts used by SlicerKimiAgent.

## Files

### system_prompt.md

The main system prompt template loaded by `KimiClient` when building prompts for the KIMI API.

**Location:** `SlicerKimiAgent/Resources/Prompts/system_prompt.md`

**Contents:**
- Core mission and role definition
- Critical rules (MRML node access, array modifications, coordinate systems, etc.)
- API preference order (slicer.util → MRML → module logics)
- MRML node types reference
- Common operations by topic with code examples
- Output format requirements
- Response structure guidelines
- Common pitfalls to avoid
- Coding style guidelines

**Dynamic Content:**

The following sections are dynamically appended at runtime based on the user's query:

- `## RELEVANT TOPICS FOR THIS QUERY:` - Lists relevant script repository files
- `## RELEVANT CODE EXAMPLES FROM SLICER SCRIPT REPOSITORY:` - Code examples from markdown files
- `## CURRENT SLICER SCENE STATE:` - Information about nodes in the current scene

**Customization:**

You can edit this file to customize the AI assistant's behavior without modifying Python code. Changes take effect immediately after saving (no restart required for the next API call).

## Fallback Mechanism

If the system prompt file cannot be loaded (e.g., file missing or corrupted), `KimiClient` will automatically use a minimal built-in fallback prompt to ensure the system remains functional.

## Related Files

- `SlicerKimiAgentLib/KimiClient.py` - Loads and uses this prompt
- `Resources/Skills/script_repository/` - Source of dynamic code examples
