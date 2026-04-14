"""
SlicerAIAgentLib - Core library for the SlicerAIAgent extension.

This package contains the core components for AI-powered assistance in 3D Slicer:
- LLMClient: HTTP client for LLM API communication (OpenAI-compatible)
- SkillTools: Tool calling support for searching the skill
- CodeValidator: Validates Python code for safety before execution
- SafeExecutor: Sandboxed execution environment for generated code
- ConversationStore: Manages conversation history
"""

__all__ = [
    'LLMClient',
    'SkillTools',
    'CodeValidator',
    'SafeExecutor',
    'ConversationStore',
    'SlicerCodeTemplates',
]

# Import main classes for convenient access
from .LLMClient import LLMClient
from .SkillTools import SkillToolExecutor, get_skill_tools
from .CodeValidator import CodeValidator
from .SafeExecutor import SafeExecutor
from .ConversationStore import ConversationStore
from .SlicerCodeTemplates import SlicerCodeTemplates
