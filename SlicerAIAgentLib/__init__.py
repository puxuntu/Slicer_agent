"""
SlicerAIAgentLib - Core library for the SlicerAIAgent extension.

This package contains the core components for AI-powered assistance in 3D Slicer:
- LLMClient: HTTP client for LLM API communication (OpenAI-compatible)
- SkillTools: Tool calling support for searching the skill
- SceneTools: Structured scene introspection (summary + node properties)
- SkillIndexer: Dense vector retrieval index (FAISS) for the knowledge base
- CodeValidator: Validates Python code for safety before execution
- SafeExecutor: Sandboxed execution environment for generated code
- ConversationStore: Manages conversation history
- InteractionManager: Low-level Slicer 3D interaction management
- WorkflowOrchestrator: Runtime state machine for guided interactive workflows
"""

__all__ = [
    'LLMClient',
    'SkillTools',
    'SkillIndexer',
    'CodeValidator',
    'SafeExecutor',
    'ConversationStore',
    'SlicerCodeTemplates',
    'SceneTools',
    'ExtensionCLILoader',
    'ExtensionCLIAnalyzer',
    'InteractionManager',
    'WorkflowOrchestrator',
]

# Import main classes for convenient access
from .LLMClient import LLMClient
from .SkillTools import SkillToolExecutor, get_skill_tools
from .SkillIndexer import (
    Chunker,
    VectorIndex,
    VectorRetriever,
    IndexBuilder,
    CodeChunk,
    RetrievedChunk,
)
from .CodeValidator import CodeValidator
from .SafeExecutor import SafeExecutor
from .ConversationStore import ConversationStore
from .SlicerCodeTemplates import SlicerCodeTemplates
from .SceneTools import buildSceneSummary, getNodeProperties, get_scene_tools
from .ExtensionCLILoader import get_dynamic_extension_tools, dispatch_extension_cli_tool, get_validated_extensions
from .InteractionManager import InteractionManager
from .WorkflowOrchestrator import WorkflowOrchestrator
