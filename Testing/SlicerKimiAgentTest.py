"""
Unit tests for SlicerKimiAgent extension.

Run these tests from Slicer's Python console:
    import unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('SlicerKimiAgentTest')
    unittest.TextTestRunner(verbosity=2).run(suite)
"""

import os
import sys
import unittest
import tempfile
import shutil

import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SlicerKimiAgentTest(ScriptedLoadableModuleTest):
    """
    Comprehensive test suite for SlicerKimiAgent.
    """

    def setUp(self):
        """Setup for each test - clear the scene."""
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        """Cleanup after each test."""
        pass

    def runTest(self):
        """Run all tests."""
        self.setUp()
        self.test_ModuleImport()
        self.tearDown()
        
        self.setUp()
        self.test_KimiClient()
        self.tearDown()
        
        self.setUp()
        self.test_CodeValidator()
        self.tearDown()
        
        self.setUp()
        self.test_SafeExecutor()
        self.tearDown()
        
        self.setUp()
        self.test_SkillContextManager()
        self.tearDown()
        
        self.setUp()
        self.test_ConversationStore()
        self.tearDown()
        
        self.setUp()
        self.test_SlicerCodeTemplates()
        self.tearDown()
        
        self.setUp()
        self.test_Integration()
        self.tearDown()

    def test_ModuleImport(self):
        """Test that all module components can be imported."""
        try:
            from SlicerKimiAgentLib import (
                KimiClient,
                SkillContextManager,
                CodeValidator,
                SafeExecutor,
                ConversationStore,
                SlicerCodeTemplates,
            )
            self.delayDisplay("All module components imported successfully")
        except Exception as e:
            self.delayDisplay(f"Module import failed: {e}")
            raise

    def test_KimiClient(self):
        """Test KimiClient functionality."""
        from SlicerKimiAgentLib import KimiClient

        client = KimiClient()

        # Test defaults
        self.assertEqual(client.model, "kimi-k2.5")
        self.assertIsNone(client.timeout)

        # Test API key management
        client.setApiKey("test_key")
        self.assertEqual(client.api_key, "test_key")

        # Test model selection and legacy normalization
        client.setModel("moonshot-v1-8k")
        self.assertEqual(client.model, "moonshot-v1-8k")
        client.setModel("kimi-latest")
        self.assertEqual(client.model, "kimi-k2.5")

        # Test conversation history
        client.clearHistory()
        self.assertEqual(len(client.conversation_history), 0)

        # Test history management
        client.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there", "reasoning_content": "Thinking..."},
        ]
        self.assertEqual(len(client.getHistory()), 2)

        client.clearHistory()
        self.assertEqual(len(client.getHistory()), 0)

        # Test code extraction
        message_with_code = """
Here's some code:
```python
x = 1 + 1
print(x)
```
More text.
"""
        code = client._extractCode(message_with_code)
        self.assertIsNotNone(code)
        self.assertIn("x = 1 + 1", code)

        # Test SSE parsing helpers
        done_chunk = client._parseStreamChunk("[DONE]")
        self.assertTrue(done_chunk["done"])

        reasoning_chunk = client._parseStreamChunk('{\"choices\":[{\"delta\":{\"reasoning_content\":\"step 1\"}}]}')
        self.assertEqual(reasoning_chunk["reasoning_content"], "step 1")
        self.assertEqual(reasoning_chunk["content"], "")

        content_chunk = client._parseStreamChunk('{\"choices\":[{\"delta\":{\"content\":\"final answer\"}},\"usage\":{\"total_tokens\":12}}]')
        self.assertEqual(content_chunk["content"], "final answer")

        # Test response assembly
        response = client._buildResponse("answer", "reasoning", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, {})
        self.assertEqual(response["message"], "answer")
        self.assertEqual(response["reasoning_content"], "reasoning")
        self.assertEqual(response["tokens"], 15)

        # Test stats
        stats = client.getStats()
        self.assertIn("total_tokens", stats)
        self.assertIn("total_cost", stats)

        # Test system prompt building with context
        context = {
            "file_descriptions": {"volumes.md": "Volume operations"},
            "examples": [{"source": "volumes.md", "code": "volume = slicer.util.loadVolume('test.nrrd')"}],
            "scene": {"node_counts": {"Volume": 1}, "sample_node_names": ["MRHead"]}
        }
        prompt = client._buildSystemPrompt(context)
        self.assertIn("Volume operations", prompt)
        self.assertIn("test.nrrd", prompt)

        self.delayDisplay("KimiClient tests passed")

    def test_CodeValidator(self):
        """Test CodeValidator functionality."""
        from SlicerKimiAgentLib import CodeValidator
        
        validator = CodeValidator()
        
        # Test safe code
        safe_code = "volume = slicer.util.loadVolume('test.nrrd')"
        result = validator.validate(safe_code)
        self.assertTrue(result["valid"], f"Safe code should pass: {result.get('reason')}")
        
        # Test empty code
        result = validator.validate("")
        self.assertFalse(result["valid"])
        self.assertIn("Empty", result["reason"])
        
        # Test syntax error
        bad_syntax = "def broken("
        result = validator.validate(bad_syntax)
        self.assertFalse(result["valid"])
        self.assertIn("Syntax", result["reason"])
        
        # Test blocked import
        blocked_import = "import os"
        result = validator.validate(blocked_import)
        self.assertFalse(result["valid"])
        self.assertIn("Blocked", result["reason"])
        
        # Test blocked function
        blocked_func = "eval('1+1')"
        result = validator.validate(blocked_func)
        self.assertFalse(result["valid"])
        self.assertIn("Blocked", result["reason"])
        
        # Test destructive operation detection
        destructive_code = "slicer.mrmlScene.RemoveNode(node)"
        result = validator.validate(destructive_code)
        self.assertTrue(result["requires_confirmation"])
        self.assertTrue(len(result["destructive_ops"]) > 0)
        
        self.delayDisplay("CodeValidator tests passed")

    def test_SafeExecutor(self):
        """Test SafeExecutor functionality."""
        from SlicerKimiAgentLib import SafeExecutor
        
        executor = SafeExecutor()
        
        # Test simple execution
        code = "result = 2 + 2"
        result = executor.execute(code)
        self.assertTrue(result["success"], f"Simple code should execute: {result.get('error')}")
        
        # Test output capture
        code = "print('Hello, World!')"
        result = executor.execute(code)
        self.assertTrue(result["success"])
        self.assertIn("Hello, World!", result["output"])
        
        # Test exception handling
        code = "raise ValueError('Test error')"
        result = executor.execute(code)
        self.assertFalse(result["success"])
        self.assertIn("Test error", result["error"])
        self.assertIsNotNone(result["traceback"])
        
        # Test globals
        executor.addGlobal("test_var", 42)
        code = "print(test_var)"
        result = executor.execute(code)
        self.assertTrue(result["success"])
        self.assertIn("42", result["output"])
        
        # Test history
        self.assertTrue(len(executor.getHistory()) > 0)
        executor.clearHistory()
        self.assertEqual(len(executor.getHistory()), 0)
        
        self.delayDisplay("SafeExecutor tests passed")

    def test_SkillContextManager(self):
        """Test SkillContextManager functionality with real Slicer skill."""
        from SlicerKimiAgentLib import SkillContextManager
        
        manager = SkillContextManager()
        
        # Test context building for volume operations
        context = manager.buildContext("load a volume and render it")
        self.assertIsNotNone(context)
        self.assertIn("volumes.md", context.get("topics", []))
        
        # Test context building for segmentation
        context = manager.buildContext("create a segmentation and threshold")
        self.assertIsNotNone(context)
        self.assertIn("segmentations.md", context.get("topics", []))
        
        # Test context building for markups
        context = manager.buildContext("add fiducial points")
        self.assertIsNotNone(context)
        self.assertIn("markups.md", context.get("topics", []))
        
        # Test context formatting
        context = manager.buildContext("load volume")
        formatted = manager.formatContextForPrompt(context)
        self.assertIn("RELEVANT TOPICS", formatted)
        
        # Test API help retrieval (now returns None for unknown APIs)
        help_info = manager.getApiHelp("slicer.util.loadVolume")
        # May be None if not in bundled index
        
        # Test cache refresh (should not raise)
        manager.refreshCache()
        
        self.delayDisplay("SkillContextManager tests passed")

    def test_ConversationStore(self):
        """Test ConversationStore functionality."""
        from SlicerKimiAgentLib import ConversationStore
        
        store = ConversationStore()
        store.clearAll()
        
        # Test adding exchanges (with reasoning_content)
        store.addExchange("Hello", {"message": "Hi", "reasoning_content": "Thinking about greeting", "tokens": 10, "cost": 0.001})
        store.addExchange("How are you?", {"message": "I'm good", "tokens": 15, "cost": 0.0015})

        # Test retrieval
        current = store.getCurrentSession()
        self.assertEqual(len(current), 2)
        self.assertEqual(current[0].get("reasoning_content"), "Thinking about greeting")
        self.assertEqual(current[1].get("reasoning_content"), "")
        
        # Test stats
        stats = store.getStats()
        self.assertEqual(stats["current_session_exchanges"], 2)
        self.assertEqual(stats["current_session_tokens"], 25)
        
        # Test search
        results = store.search("Hello")
        self.assertEqual(len(results), 1)
        
        # Test export/import with temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            export_path = os.path.join(temp_dir, "test_conversation.json")
            store.exportSession(export_path)
            self.assertTrue(os.path.exists(export_path))
            
            store.clearAll()
            self.assertEqual(len(store.getAllConversations()), 0)
            
            store.importSession(export_path)
            self.assertTrue(len(store.getAllConversations()) > 0)
        finally:
            shutil.rmtree(temp_dir)
        
        # Test new session
        old_session = store.current_session_id
        store.newSession()
        self.assertNotEqual(store.current_session_id, old_session)
        
        self.delayDisplay("ConversationStore tests passed")

    def test_SlicerCodeTemplates(self):
        """Test SlicerCodeTemplates functionality."""
        from SlicerKimiAgentLib import SlicerCodeTemplates
        
        # Test getting specific template
        template = SlicerCodeTemplates.getTemplate("load_volume")
        self.assertIsNotNone(template)
        self.assertIn("code", template)
        
        # Test getting all templates
        all_templates = SlicerCodeTemplates.getAllTemplates()
        self.assertTrue(len(all_templates) > 0)
        
        # Test finding by tag
        results = SlicerCodeTemplates.findByTag("volume")
        self.assertTrue(len(results) > 0)
        
        # Test finding by keyword
        results = SlicerCodeTemplates.findByKeyword("load")
        self.assertTrue(len(results) > 0)
        
        # Test formatting for prompt
        formatted = SlicerCodeTemplates.formatForPrompt(results[:2])
        self.assertIn("```python", formatted)
        
        self.delayDisplay("SlicerCodeTemplates tests passed")

    def test_Integration(self):
        """Integration test of multiple components."""
        from SlicerKimiAgentLib import (
            KimiClient,
            SkillContextManager,
            CodeValidator,
            SafeExecutor,
        )
        
        # Create components
        client = KimiClient()
        skill_manager = SkillContextManager()
        validator = CodeValidator()
        executor = SafeExecutor()
        
        # Test workflow: Build context -> Validate -> Execute (mock)
        prompt = "load a volume"
        context = skill_manager.buildContext(prompt)
        self.assertIsNotNone(context)
        
        # Test code validation
        test_code = "volume = slicer.util.loadVolume('/path/to/volume.nrrd')"
        validation = validator.validate(test_code)
        self.assertTrue(validation["valid"])
        
        # Test execution
        result = executor.execute("x = 5 + 10")
        self.assertTrue(result["success"])
        
        self.delayDisplay("Integration tests passed")


class SlicerKimiAgentLogicTest(unittest.TestCase):
    """Tests for the SlicerKimiAgentLogic class."""

    def setUp(self):
        """Setup for each test."""
        from SlicerKimiAgent import SlicerKimiAgentLogic
        self.logic = SlicerKimiAgentLogic()

    def tearDown(self):
        """Cleanup after each test."""
        if self.logic:
            self.logic.cleanup()

    def test_api_key_management(self):
        """Test API key setting."""
        self.logic.setApiKey("test_key")
        self.assertTrue(self.logic.hasApiKey())
        
        self.logic.setApiKey("")
        self.assertFalse(self.logic.hasApiKey())

    def test_model_setting(self):
        """Test model setting."""
        self.logic.setModel("moonshot-v1-32k")
        self.assertEqual(self.logic.kimiClient.model, "moonshot-v1-32k")

        self.logic.setModel("kimi-latest")
        self.assertEqual(self.logic.kimiClient.model, "kimi-k2.5")

    def test_conversation_clear(self):
        """Test conversation clearing."""
        self.logic.clearConversation()
        # Should not raise


# For running outside Slicer's test framework
if __name__ == "__main__":
    unittest.main()
