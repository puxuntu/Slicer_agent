"""
SlicerKimiAgentLib - Core library for the SlicerKimiAgent extension.

This package contains the core components for AI-powered assistance in 3D Slicer:
- KimiClient: HTTP client for KIMI API communication
- SkillContextManager: Manages skill-based context for LLM prompts
- SkillTools: Tool calling support for searching the skill
- CodeValidator: Validates Python code for safety before execution
- SafeExecutor: Sandboxed execution environment for generated code
- ConversationStore: Manages conversation history
"""

__all__ = [
    'KimiClient',
    'SkillContextManager', 
    'SkillTools',
    'CodeValidator',
    'SafeExecutor',
    'ConversationStore',
    'SlicerCodeTemplates',
]

# Import main classes for convenient access
from .KimiClient import KimiClient
from .SkillContextManager import SkillContextManager
from .SkillTools import SkillToolExecutor, get_skill_tools
from .CodeValidator import CodeValidator
from .SafeExecutor import SafeExecutor
from .ConversationStore import ConversationStore
from .SlicerCodeTemplates import SlicerCodeTemplates
