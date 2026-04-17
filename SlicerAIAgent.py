import json
import os
import queue
import threading
import unittest
import logging
from typing import List
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#------------------------------------------------------------------
# Module Class
#------------------------------------------------------------------
class SlicerAIAgent(ScriptedLoadableModule):
    """AI-powered assistant for 3D Slicer using LLM APIs."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Slicer AI Agent"
        self.parent.categories = ["AI"]
        self.parent.dependencies = []
        self.parent.contributors = ["Puxun (Agent Developer)"]
        self.parent.helpText = """
        An AI-powered assistant that helps you control 3D Slicer using natural language.

        Features:
        - Natural language to Python code generation
        - Scene manipulation and analysis
        - Guided workflows for common tasks
        - Integration with Slicer's skill knowledge base

        Usage:
        1. Enter your API key in Settings
        2. Type your request in the chat box
        3. Review and execute the generated code
        """
        self.parent.acknowledgementText = """
        This extension uses LLM APIs for code generation.
        Thanks to the 3D Slicer community for the comprehensive skill knowledge base.
        """
        moduleDir = os.path.dirname(__file__)
        iconPath = os.path.join(moduleDir, 'Resources', 'Icons', 'SlicerAIAgent.png')
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)

#------------------------------------------------------------------
# Widget Class
#------------------------------------------------------------------
class SlicerAIAgentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Main UI widget for SlicerAIAgent."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._chatEntriesHtml = []
        # Streaming state
        self._streamReasoning = ""
        self._streamContent = ""
        self._streaming = False
        # Thread-safe queue for streaming events (filled by worker, drained on main thread)
        self._streamQueue = queue.Queue()
        self._streamPollTimer = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiFilePath = os.path.join(os.path.dirname(__file__), 'Resources', 'UI', 'SlicerAIAgent.ui')
        if os.path.exists(uiFilePath):
            self.ui = slicer.util.loadUI(uiFilePath)
            self.layout.addWidget(self.ui)
            self._connectUIWidgets()
            self.setupConnections()
        else:
            self.setupUIProgrammatically()

        self.logic = SlicerAIAgentLogic()
        self.loadSettings()

        self._streamPollTimer = qt.QTimer()
        self._streamPollTimer.setInterval(50)
        self._streamPollTimer.timeout.connect(self._drainStreamQueue)
        self._streamPollTimer.start()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        logger.info("SlicerAIAgent widget setup complete")

    def _connectUIWidgets(self):
        self.providerSelector = self.ui.findChild(qt.QComboBox, "providerSelector")
        self.modelSelector = self.ui.findChild(qt.QComboBox, "modelSelector")
        self.baseUrlInput = self.ui.findChild(qt.QLineEdit, "baseUrlInput")
        self.apiKeyInput = self.ui.findChild(qt.QLineEdit, "apiKeyInput")
        self.saveSettingsButton = self.ui.findChild(qt.QPushButton, "saveSettingsButton")
        self.testConnectionButton = self.ui.findChild(qt.QPushButton, "testConnectionButton")
        self.chatHistory = self.ui.findChild(qt.QTextEdit, "chatHistory")
        self.codeDisplay = self.ui.findChild(qt.QTextEdit, "codeDisplay")
        # Note: executeButton and copyButton removed - code is auto-executed
        self.clearChatButton = self.ui.findChild(qt.QPushButton, "clearChatButton")
        self.promptInput = self.ui.findChild(qt.QTextEdit, "promptInput")
        self.sendButton = self.ui.findChild(qt.QPushButton, "sendButton")
        self.statusLabel = self.ui.findChild(qt.QLabel, "statusLabel")
        self.tokenLabel = self.ui.findChild(qt.QLabel, "tokenLabel")

    def setupUIProgrammatically(self):
        self.ui = ctk.ctkCollapsibleButton()
        self.ui.text = "Slicer AI Agent"
        self.layout.addWidget(self.ui)

        mainLayout = qt.QVBoxLayout(self.ui)

        settingsGroup = ctk.ctkCollapsibleGroupBox()
        settingsGroup.title = "Settings"
        settingsGroup.collapsed = True
        mainLayout.addWidget(settingsGroup)

        settingsLayout = qt.QFormLayout(settingsGroup)

        # Row 1: Provider + Model
        providerModelLayout = qt.QHBoxLayout()
        self.providerSelector = qt.QComboBox()
        self.providerSelector.addItems(["Kimi", "Claude"])
        self.providerSelector.setToolTip("Select AI provider")
        providerModelLayout.addWidget(self.providerSelector)

        self.modelSelector = qt.QComboBox()
        self.modelSelector.setEditable(True)
        self.modelSelector.setToolTip("Select or type a model name")
        providerModelLayout.addWidget(self.modelSelector)
        settingsLayout.addRow("Provider / Model:", providerModelLayout)

        # Row 2: Base URL
        self.baseUrlInput = qt.QLineEdit()
        self.baseUrlInput.setPlaceholderText("API base URL")
        settingsLayout.addRow("Base URL:", self.baseUrlInput)

        # Row 3: API Key + Test button
        apiKeyLayout = qt.QHBoxLayout()
        self.apiKeyInput = qt.QLineEdit()
        self.apiKeyInput.setEchoMode(qt.QLineEdit.Password)
        self.apiKeyInput.setPlaceholderText("Enter your API key")
        apiKeyLayout.addWidget(self.apiKeyInput)

        self.testConnectionButton = qt.QPushButton("Test")
        self.testConnectionButton.setToolTip("Test API connection")
        apiKeyLayout.addWidget(self.testConnectionButton)
        settingsLayout.addRow("API Key:", apiKeyLayout)

        # Row 4: Save Settings
        self.saveSettingsButton = qt.QPushButton("Save Settings")
        settingsLayout.addRow(self.saveSettingsButton)

        chatLabel = qt.QLabel("Conversation:")
        mainLayout.addWidget(chatLabel)

        self.chatHistory = qt.QTextEdit()
        self.chatHistory.setReadOnly(True)
        self.chatHistory.setMinimumHeight(300)
        mainLayout.addWidget(self.chatHistory)

        codeLabel = qt.QLabel("Generated Code:")
        mainLayout.addWidget(codeLabel)

        self.codeDisplay = qt.QTextEdit()
        self.codeDisplay.setReadOnly(True)
        self.codeDisplay.setMinimumHeight(150)
        self.codeDisplay.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace;")
        mainLayout.addWidget(self.codeDisplay)

        codeButtonLayout = qt.QHBoxLayout()
        self.executeButton = qt.QPushButton("Execute Code")
        self.executeButton.setEnabled(False)
        self.copyButton = qt.QPushButton("Copy to Clipboard")
        self.clearChatButton = qt.QPushButton("Clear Chat")
        codeButtonLayout.addWidget(self.executeButton)
        codeButtonLayout.addWidget(self.copyButton)
        codeButtonLayout.addWidget(self.clearChatButton)
        mainLayout.addLayout(codeButtonLayout)

        inputLayout = qt.QHBoxLayout()
        self.promptInput = qt.QTextEdit()
        self.promptInput.setPlaceholderText("Type your request here... (e.g., 'Load a sample volume and create a volume rendering')")
        self.promptInput.setMaximumHeight(80)
        self.sendButton = qt.QPushButton("Send")
        self.sendButton.setMinimumHeight(80)
        inputLayout.addWidget(self.promptInput, stretch=1)
        inputLayout.addWidget(self.sendButton)
        mainLayout.addLayout(inputLayout)

        self.statusLabel = qt.QLabel("Ready")
        mainLayout.addWidget(self.statusLabel)

        self.tokenLabel = qt.QLabel("Tokens: 0 | Cost: $0.000")
        mainLayout.addWidget(self.tokenLabel)

        self.setupConnections()

    def setupConnections(self):
        if hasattr(self, 'sendButton') and self.sendButton is not None:
            self.sendButton.clicked.connect(self.onSendButtonClicked)
        if hasattr(self, 'promptInput') and self.promptInput is not None:
            self.promptInput.textChanged.connect(self.onPromptTextChanged)
            # Add Ctrl+Enter shortcut for sending
            self.sendShortcut = qt.QShortcut(qt.QKeySequence("Ctrl+Return"), self.promptInput)
            self.sendShortcut.connect('activated()', self.onSendButtonClicked)
        # Note: Button connections removed - code is auto-executed
        if hasattr(self, 'clearChatButton') and self.clearChatButton is not None:
            self.clearChatButton.clicked.connect(self.onClearChatButtonClicked)
        if hasattr(self, 'saveSettingsButton') and self.saveSettingsButton is not None:
            self.saveSettingsButton.clicked.connect(self.onSaveSettings)
        if hasattr(self, 'testConnectionButton') and self.testConnectionButton is not None:
            self.testConnectionButton.clicked.connect(self.onTestConnection)
        if hasattr(self, 'providerSelector') and self.providerSelector is not None:
            self.providerSelector.currentTextChanged.connect(self.onProviderChanged)

    def disconnect(self):
        self.removeObservers()
        try:
            if hasattr(self, 'sendButton') and self.sendButton is not None:
                self.sendButton.clicked.disconnect()
            if hasattr(self, 'promptInput') and self.promptInput is not None:
                self.promptInput.textChanged.disconnect()
            if hasattr(self, 'sendShortcut') and self.sendShortcut is not None:
                self.sendShortcut.disconnect()
            if hasattr(self, 'clearChatButton') and self.clearChatButton is not None:
                self.clearChatButton.clicked.disconnect()
        except RuntimeError:
            pass

    def cleanup(self):
        self.disconnect()
        if self._streamPollTimer:
            self._streamPollTimer.stop()
        if self.logic:
            self.logic.cleanup()
        logger.info("SlicerAIAgent widget cleaned up")

    def enter(self):
        if (hasattr(self, 'chatHistory') and self.chatHistory is not None and
            self.logic and not self.logic.hasApiKey()):
            self.appendToChat("System", "Please configure your API key in Settings before using the agent.")

    def exit(self):
        pass

    def onSceneStartClose(self, caller, event):
        if self.logic:
            self.logic.pauseProcessing()

    def onSceneEndClose(self, caller, event):
        if self.logic:
            self.logic.resumeProcessing()

    # ------------------------------------------------------------------
    # Streaming chat display helpers
    # ------------------------------------------------------------------
    def _setChatHtml(self, html):
        """Replace the chat box contents and keep it scrolled to the bottom."""
        self.chatHistory.setHtml(html)
        scrollbar = self.chatHistory.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum)

    def _buildStreamingEntryHtml(self):
        """Build HTML for the current streaming assistant entry."""
        timestamp = getattr(self, '_streamTimestamp', '')
        parts = []
        if self._streamReasoning:
            escaped_reasoning = self.escapeHtml(self._streamReasoning).replace(chr(10), '<br>')
            parts.append(
                f'<div style="margin-left: 10px; margin-top: 5px; color: #666; '
                f'border-left: 3px solid #ccc; padding-left: 8px; font-style: italic;">'
                f'{escaped_reasoning}</div>'
            )
        if self._streamContent:
            escaped_content = self.escapeHtml(self._streamContent).replace(chr(10), '<br>')
            parts.append(
                f'<div style="margin-left: 10px; margin-top: 5px;">{escaped_content}</div>'
            )

        if not parts:
            parts.append('<div style="margin-left: 10px; margin-top: 5px; color: #aaa;">...</div>')

        body = ''.join(parts)
        return (
            f'<div style="margin: 10px 0;">'
            f'<span style="color: #999; font-size: 10px;">[{timestamp}]</span> '
            f'<span style="color: #009900; font-weight: bold;">Assistant:</span>'
            f'{body}'
            f'</div>'
            f'<hr style="border: none; border-top: 1px solid #eee; margin: 5px 0;">'
        )

    def _renderStreamingEntry(self):
        """Re-render the current streaming assistant entry in the chat box."""
        if not hasattr(self, 'chatHistory') or self.chatHistory is None:
            return
        self._setChatHtml(''.join(self._chatEntriesHtml) + self._buildStreamingEntryHtml())

    def _finalizeStreamingEntry(self):
        """Commit the current streaming assistant entry into chat history."""
        if self._streaming or self._streamReasoning or self._streamContent:
            self._chatEntriesHtml.append(self._buildStreamingEntryHtml())
            self._setChatHtml(''.join(self._chatEntriesHtml))

    def _drainStreamQueue(self):
        """Drain queued streaming events on the Qt main thread."""
        while True:
            try:
                event_type, payload = self._streamQueue.get_nowait()
            except queue.Empty:
                break

            if event_type == 'delta':
                self._onStreamDelta(payload)
            elif event_type == 'complete':
                self._onStreamComplete(payload)
            elif event_type == 'error':
                self._onStreamError(payload)

    def _onStreamDelta(self, delta):
        """Apply one streamed delta on the main thread."""
        if delta.get('round'):
            self._updateToolProgress(delta)
        else:
            self._streamReasoning += delta.get('reasoning_content', '')
            self._streamContent += delta.get('content', '')
            self._renderStreamingEntry()
        slicer.app.processEvents()

    def _updateToolProgress(self, delta):
        """Display tool execution progress as a separate committed entry."""
        progress_text = delta.get('reasoning_content', '').strip()
        if not progress_text:
            return

        timestamp = qt.QDateTime.currentDateTime().toString("hh:mm:ss")
        html = (
            f'<div style="margin: 5px 0; padding: 5px 10px; background-color: #f5f5f5; border-left: 3px solid #999;">'
            f'<span style="color: #999; font-size: 10px;">[{timestamp}]</span> '
            f'<span style="color: #666; font-weight: bold;">Search:</span>'
            f'<div style="margin-left: 10px; margin-top: 3px; white-space: pre-wrap; color: #555;">{self.escapeHtml(progress_text).replace(chr(10), "<br>")}</div>'
            f'</div>'
        )
        self._chatEntriesHtml.append(html)
        self._setChatHtml(''.join(self._chatEntriesHtml) + self._buildStreamingEntryHtml())

    def _onStreamComplete(self, response):
        """Called on the main thread when streaming finishes successfully."""
        self._streaming = False
        self._finalizeStreamingEntry()

        # Display generated code if any and auto-execute
        if response.get("code"):
            self.currentCode = response["code"]
            self.codeDisplay.setPlainText(response["code"])
            self._saveGeneratedCodeToFile(response["code"])
            # Auto-execute the generated code
            self._autoExecuteCode()

        # Update token usage
        if response.get("tokens"):
            tokens = response["tokens"]
            cost = response.get("cost", 0)
            self.tokenLabel.text = f"Tokens: {tokens} | Cost: ${cost:.4f}"

        self.statusLabel.text = "Ready"
        self.sendButton.setEnabled(True)

    def _onStreamError(self, error_msg):
        """Called on the main thread when the streaming request fails."""
        self._streaming = False
        self._finalizeStreamingEntry()
        logger.error(f"Error generating response: {error_msg}")

        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            self.appendToChat("Error",
                f"Request timed out.\n\n"
                f"Please check:\n"
                f"1. Your network connection\n"
                f"2. The model name is correct (use 'kimi-k2.5')\n"
                f"3. Your API key has access to K2.5 models\n\n"
                f"Technical details: {error_msg}")
        else:
            self.appendToChat("Error", f"Failed to generate response: {error_msg}")

        self.statusLabel.text = "Ready"
        self.sendButton.setEnabled(True)

    def onSendButtonClicked(self):
        prompt = self.promptInput.toPlainText().strip()
        if not prompt:
            return

        self.promptInput.clear()
        self.appendToChat("You", prompt)

        self.statusLabel.text = "Generating..."
        self.sendButton.setEnabled(False)
        slicer.app.processEvents()

        # Reset streaming accumulators
        self._streamReasoning = ""
        self._streamContent = ""
        self._streaming = True
        self._streamTimestamp = qt.QDateTime.currentDateTime().toString("hh:mm:ss")
        self._renderStreamingEntry()

        # Build context on the main thread (it reads the MRML scene)
        context = {"scene": self.logic._buildSceneContext()} if self.logic else None

        # Launch the streaming request in a background thread
        def _backgroundStream():
            try:
                def _onDelta(delta):
                    self._streamQueue.put(('delta', dict(delta)))

                response = self.logic.generateResponseStream(prompt, context, _onDelta)
                self._streamQueue.put(('complete', dict(response)))
            except Exception as e:
                self._streamQueue.put(('error', str(e)))

        thread = threading.Thread(target=_backgroundStream, daemon=True)
        thread.start()

    def onPromptTextChanged(self):
        hasText = bool(self.promptInput.toPlainText().strip())
        self.sendButton.setEnabled(hasText)

    def generateResponse(self, prompt):
        """Legacy non-streaming path (kept for backward compatibility)."""
        try:
            response = self.logic.generateResponse(prompt)
            self.appendToChat("Assistant", response["message"])

            if response.get("code"):
                self.currentCode = response["code"]
                self.codeDisplay.setPlainText(response["code"])
                self._saveGeneratedCodeToFile(response["code"])
                # Auto-execute the generated code
                self._autoExecuteCode()

            if response.get("tokens"):
                tokens = response["tokens"]
                cost = response.get("cost", 0)
                self.tokenLabel.text = f"Tokens: {tokens} | Cost: ${cost:.4f}"

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.appendToChat("Error", f"Failed to generate response: {str(e)}")
        finally:
            self.statusLabel.text = "Ready"
            self.sendButton.setEnabled(True)

    def _saveGeneratedCodeToFile(self, code):
        """Save the generated code to a local text file for user reference."""
        try:
            moduleDir = os.path.dirname(__file__)
            turn_number = 1
            if self.logic and hasattr(self.logic, 'llmClient') and self.logic.llmClient:
                turn_number = getattr(self.logic.llmClient, 'turn_number', 1)
            # turn_number is incremented right after the assistant response finishes,
            # so the code generated in turn N is saved when turn_number equals N+1.
            turn_number = max(1, turn_number - 1)
            latestPath = os.path.join(moduleDir, f'{turn_number}_latest_generated_code.txt')
            with open(latestPath, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            logger.warning(f"Failed to save generated code to file: {e}")

    def _autoExecuteCode(self, attempt=1, max_attempts=5):
        """Auto-execute generated code with pre-validation and self-correction on failure."""
        if not hasattr(self, 'currentCode') or not self.currentCode:
            return

        # Pre-validation: check for syntax errors and common issues
        if self.logic and hasattr(self.logic, 'codeValidator'):
            validation = self.logic.codeValidator.validate(self.currentCode)
            if not validation['valid']:
                # Syntax error detected before execution
                error_msg = validation['reason']
                self.appendToChat("System", 
                    f"Pre-validation failed (attempt {attempt}/{max_attempts}).\n"
                    f"Error: {error_msg}\n"
                    f"Auto-correcting...")
                if self.logic:
                    self.logic.addExecutionFeedback(
                        f"Code pre-validation failed (attempt {attempt}/{max_attempts}):\n"
                        f"Error: {error_msg}\n"
                        "The code had syntax errors or violated safety rules before execution."
                    )
                self._selfCorrectCode(error_msg, attempt, max_attempts)
                return

        self.statusLabel.text = f"Executing (attempt {attempt}/{max_attempts})..."
        slicer.app.processEvents()
        
        def onExecutionComplete(result):
            feedback_lines = []
            output_has_errors = False
            if result.get("timed_out", False):
                self.statusLabel.text = "Ready"
                output = result.get('output', 'No output')
                exec_time = result.get('execution_time', 30)
                msg = f"Code execution timed out after {exec_time:.1f}s."
                if output:
                    msg += f"\nOutput: {output}"
                self.appendToChat("Warning", msg)
                feedback_lines.append(f"Status: timed_out\nExecution time: {exec_time:.1f}s\nOutput: {output}")
            elif result["success"]:
                self.statusLabel.text = "Ready"
                output = result.get('output', 'No output')
                execution_time = result.get('execution_time', 0)
                msg = f"Code executed successfully in {execution_time:.2f}s."
                if output:
                    msg += f"\nOutput: {output}"
                self.appendToChat("System", msg)
                feedback_lines.append(f"Status: success\nExecution time: {execution_time:.2f}s\nOutput: {output}")
                # Detect errors printed but caught by the script itself
                lower_output = output.lower()
                if any(k in lower_output for k in ('error:', 'traceback', 'exception', 'failed')):
                    output_has_errors = True
                    feedback_lines.append("Warning: execution output contains error indicators even though no uncaught exception was raised.")
            else:
                # Execution failed
                error_msg = result.get('error', 'Unknown error')
                execution_time = result.get('execution_time', 0)
                msg = f"Execution failed (attempt {attempt}/{max_attempts}).\nError: {error_msg[:200]}"
                self.appendToChat("System", msg)
                feedback_lines.append(f"Status: failed\nExecution time: {execution_time:.2f}s\nError: {error_msg[:500]}")
            
            # Add execution feedback to conversation history
            if self.logic:
                feedback_text = "Code execution result:\n" + "\n".join(feedback_lines) + "\nThe MRML scene has been updated. Refer to the CURRENT SLICER SCENE in the next system prompt for the complete raw MRML."
                self.logic.addExecutionFeedback(feedback_text)
            
            # Self-correction for failures or suspicious outputs (but not timeouts)
            if not result.get("timed_out", False) and (not result["success"] or output_has_errors):
                if attempt < max_attempts:
                    self.appendToChat("System", "Auto-correcting...")
                    self._selfCorrectCode("", attempt, max_attempts)
                else:
                    self.statusLabel.text = "Ready"
                    final_error = result.get('error', 'Unknown error') if not result["success"] else "Output contains errors"
                    self.appendToChat("Error", 
                        f"Execution failed after {max_attempts} attempts.\n"
                        f"Final error: {final_error}")
        
        # Execute asynchronously
        self.logic.executeCodeAsync(self.currentCode, onExecutionComplete)
    
    def _selfCorrectCode(self, error_msg, attempt, max_attempts):
        """Generate corrected code based on execution feedback in conversation history."""
        if not self.currentCode:
            return
        
        # Lightweight correction prompt; detailed error and previous code
        # are already available in the conversation history via addExecutionFeedback.
        correction_prompt = (
            "The previous code execution encountered issues. "
            "Please refer to the execution feedback in the conversation history and provide the corrected code. "
            "Analyze the error carefully and make minimal changes to fix it. "
            "Only output the complete corrected Python code in a single code block."
        )

        self.appendToChat("You", f"[Auto-correction attempt {attempt+1}]")
        
        # Generate corrected code in background
        def generateCorrection():
            try:
                context = {"scene": self.logic._buildSceneContext()} if self.logic else None
                response = self.logic.generateResponse(correction_prompt, context=context)
                
                if response.get("code"):
                    self.currentCode = response["code"]
                    self.codeDisplay.setPlainText(response["code"])
                    self._saveGeneratedCodeToFile(response["code"])
                    # Recursively execute with incremented attempt counter
                    self._autoExecuteCode(attempt + 1, max_attempts)
                else:
                    self.appendToChat("Error", "No corrected code generated.")
                    self.statusLabel.text = "Ready"
            except Exception as e:
                self.appendToChat("Error", f"Self-correction failed: {str(e)}")
                self.statusLabel.text = "Ready"
        
        # Run correction in background thread
        import threading
        threading.Thread(target=generateCorrection, daemon=True).start()

    # Note: onCopyButtonClicked removed - copy functionality not needed with auto-execution

    def onClearChatButtonClicked(self):
        self.chatHistory.clear()
        self._chatEntriesHtml = []
        if self.logic:
            self.logic.clearConversation()
        self.codeDisplay.clear()
        self.currentCode = None

    def appendToChat(self, sender, message):
        if not hasattr(self, 'chatHistory') or self.chatHistory is None:
            logger.warning(f"Chat history not ready, message from {sender} discarded")
            return

        timestamp = qt.QDateTime.currentDateTime().toString("hh:mm:ss")

        if sender == "You":
            color = "#0066cc"
        elif sender == "Assistant":
            color = "#009900"
        elif sender == "System":
            color = "#666666"
        else:
            color = "#cc0000"

        html = f"""
        <div style="margin: 10px 0;">
            <span style="color: #999; font-size: 10px;">[{timestamp}]</span>
            <span style="color: {color}; font-weight: bold;">{sender}:</span>
            <div style="margin-left: 10px; margin-top: 5px;">{self.escapeHtml(message).replace(chr(10), '<br>')}</div>
        </div>
        <hr style="border: none; border-top: 1px solid #eee; margin: 5px 0;">
        """

        self._chatEntriesHtml.append(html)
        self._setChatHtml(''.join(self._chatEntriesHtml))

    def escapeHtml(self, text):
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;"))

    def _defaultModelsForProvider(self, provider: str) -> List[str]:
        if provider == "Claude":
            return [
                # Claude 4.6 Sonnet variants
                "claude-sonnet-4-6",
                "claude-sonnet-4-6-high",
                "claude-sonnet-4-6-low",
                "claude-sonnet-4-6-max",
                "claude-sonnet-4-6-medium",
                "claude-sonnet-4-6-thinking",
                # Claude 4.6 Opus variants
                "claude-opus-4-6",
                "claude-opus-4-6-high",
                "claude-opus-4-6-low",
                "claude-opus-4-6-max",
                "claude-opus-4-6-medium",
                "claude-opus-4-6-thinking",
                # Claude 4.5 Haiku variants
                "claude-haiku-4-5-20251001",
                "claude-haiku-4-5-20251001-thinking",
            ]
        return ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview", "kimi-k2-0905-preview", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]

    def _defaultBaseUrlForProvider(self, provider: str) -> str:
        if provider == "Claude":
            return "https://api.anthropic.com/v1"
        return "https://api.moonshot.cn/v1"

    def onProviderChanged(self, provider: str):
        if not hasattr(self, 'modelSelector') or self.modelSelector is None:
            return
        self.modelSelector.clear()
        self.modelSelector.addItems(self._defaultModelsForProvider(provider))
        if hasattr(self, 'baseUrlInput') and self.baseUrlInput is not None:
            self.baseUrlInput.text = self._defaultBaseUrlForProvider(provider)

    def onSaveSettings(self):
        if not hasattr(self, 'apiKeyInput') or self.apiKeyInput is None:
            return

        settings = qt.QSettings()
        settings.beginGroup("SlicerAIAgent")
        settings.setValue("apiKey", self.apiKeyInput.text)
        if hasattr(self, 'providerSelector') and self.providerSelector is not None:
            settings.setValue("provider", self.providerSelector.currentText)
        if hasattr(self, 'modelSelector') and self.modelSelector is not None:
            settings.setValue("model", self.modelSelector.currentText)
        if hasattr(self, 'baseUrlInput') and self.baseUrlInput is not None:
            settings.setValue("baseUrl", self.baseUrlInput.text)
        settings.endGroup()

        if self.logic:
            self.logic.setApiKey(self.apiKeyInput.text)
            if hasattr(self, 'providerSelector') and self.providerSelector is not None:
                self.logic.setProvider(self.providerSelector.currentText)
            if hasattr(self, 'modelSelector') and self.modelSelector is not None:
                self.logic.setModel(self.modelSelector.currentText)
            if hasattr(self, 'baseUrlInput') and self.baseUrlInput is not None:
                self.logic.setBaseUrl(self.baseUrlInput.text)

        slicer.util.infoDisplay("Settings saved successfully!")

    def onTestConnection(self):
        if not self.logic:
            slicer.util.warningDisplay("Logic not initialized")
            return

        if not self.logic.llmClient:
            init_error = getattr(self.logic, '_initError', None)
            if init_error:
                slicer.util.warningDisplay(
                    f"LLM client failed to initialize:\n\n{init_error}\n\n"
                    "Check the Slicer Python console for the full traceback."
                )
            else:
                slicer.util.warningDisplay(
                    "LLM client not initialized.\n\n"
                    "This usually means a required Python package failed to import.\n"
                    "Check the Slicer Python console for import errors."
                )
            return

        apiKey = self.apiKeyInput.text if hasattr(self, 'apiKeyInput') else ""
        model = self.modelSelector.currentText if hasattr(self, 'modelSelector') else "kimi-k2.5"
        baseUrl = self.baseUrlInput.text if hasattr(self, 'baseUrlInput') else ""
        provider = self.providerSelector.currentText if hasattr(self, 'providerSelector') else "Kimi"

        if not apiKey:
            slicer.util.warningDisplay("Please enter an API key first")
            return

        originalKey = self.logic.apiKey
        originalModel = self.logic.model
        originalBaseUrl = self.logic.baseUrl if hasattr(self.logic, 'baseUrl') else ""
        originalProvider = self.logic.llmClient.provider

        self.logic.setApiKey(apiKey)
        self.logic.setModel(model)
        self.logic.setProvider(provider)
        if baseUrl:
            self.logic.setBaseUrl(baseUrl)

        self.statusLabel.text = "Testing connection..."
        slicer.app.processEvents()

        try:
            result = self.logic.llmClient.testConnection()
            if result.get('success'):
                available = result.get('models', [])
                if available:
                    # Proxy returned a model list — check if our model is in it
                    if model in available:
                        msg = f"Connection successful!\n\nModel '{model}' is available."
                    else:
                        top = '\n'.join(f"  • {m}" for m in available[:15])
                        more = f"\n  ... and {len(available)-15} more" if len(available) > 15 else ""
                        msg = (
                            f"Connection successful, but model '{model}' was NOT found.\n\n"
                            f"Models available on this endpoint:\n{top}{more}\n\n"
                            f"Select one of the above models from the dropdown."
                        )
                else:
                    # /models not supported — confirmed via chat probe
                    msg = (
                        f"Connection successful!\n\n"
                        f"Model '{model}' is accessible.\n"
                        f"(This endpoint does not expose a model list.)"
                    )
                slicer.util.infoDisplay(msg)
            else:
                error = result.get('error', 'Unknown error')
                slicer.util.warningDisplay(f"Connection failed:\n{error}")
        except Exception as e:
            slicer.util.warningDisplay(f"Connection failed:\n{e}")
        finally:
            self.statusLabel.text = "Ready"
            self.logic.setApiKey(originalKey)
            self.logic.setModel(originalModel)
            self.logic.setProvider(originalProvider)
            # Always restore base URL (even if it was empty — use provider default)
            self.logic.setBaseUrl(originalBaseUrl if originalBaseUrl else self._defaultBaseUrlForProvider(originalProvider))

    def loadSettings(self):
        settings = qt.QSettings()
        settings.beginGroup("SlicerAIAgent")

        apiKey = settings.value("apiKey", "")
        provider = settings.value("provider", "Kimi")
        model = settings.value("model", "kimi-k2.5")
        baseUrl = settings.value("baseUrl", "")

        if hasattr(self, 'apiKeyInput') and self.apiKeyInput is not None:
            self.apiKeyInput.text = apiKey
        if hasattr(self, 'providerSelector') and self.providerSelector is not None:
            self.providerSelector.setCurrentText(provider)
            self.onProviderChanged(provider)
        if hasattr(self, 'modelSelector') and self.modelSelector is not None:
            self.modelSelector.setCurrentText(model)
        if hasattr(self, 'baseUrlInput') and self.baseUrlInput is not None:
            if baseUrl:
                self.baseUrlInput.text = baseUrl
            else:
                self.baseUrlInput.text = self._defaultBaseUrlForProvider(provider)

        settings.endGroup()

        if self.logic:
            self.logic.setApiKey(apiKey)
            self.logic.setModel(model)
            self.logic.setProvider(provider)
            if baseUrl:
                self.logic.setBaseUrl(baseUrl)
            else:
                self.logic.setBaseUrl(self._defaultBaseUrlForProvider(provider))

#------------------------------------------------------------------
# Logic Class
#------------------------------------------------------------------
class SlicerAIAgentLogic(ScriptedLoadableModuleLogic):
    """
    Business logic for SlicerAIAgent.
    Handles AI interactions, code generation, and execution.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.apiKey = None
        self.model = "kimi-k2.5"
        self.baseUrl = ""
        self.llmClient = None
        self.skill_path = None
        self.skill_mode = "unknown"
        self.codeValidator = None
        self.executor = None
        self.conversationStore = None
        self._processing = False
        self._initializeComponents()

    def _initializeComponents(self):
        try:
            from SlicerAIAgentLib import LLMClient, CodeValidator, SafeExecutor, ConversationStore, SkillTools

            self.conversationStore = ConversationStore()
            self.llmClient = LLMClient()

            # Resolve skill path relative to this module
            self.skill_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'Resources', 'Skills', 'slicer-skill-full'
            ))
            self.skill_mode = self._detectSkillMode()

            # Initialize tool executor for skill searching
            self.toolExecutor = SkillTools.SkillToolExecutor(self.skill_path)
            self.skillTools = SkillTools.get_skill_tools()
            self.codeValidator = CodeValidator()
            self.executor = SafeExecutor()

            logger.info("SlicerAIAgent logic components initialized")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Failed to initialize components: {e}\n{tb}")
            # Store the error so the UI can surface it to the user
            self._initError = str(e)

    def _detectSkillMode(self):
        """Detect skill mode from .setup-stamp.json."""
        stamp_path = os.path.join(self.skill_path, ".setup-stamp.json")
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, 'r', encoding='utf-8') as f:
                    stamp = json.load(f)
                return stamp.get("mode", "unknown")
            except Exception as e:
                logger.warning(f"Failed to read setup stamp: {e}")
        return "unknown"

    def setApiKey(self, apiKey):
        self.apiKey = apiKey
        if self.llmClient:
            self.llmClient.setApiKey(apiKey)

    def setModel(self, model):
        self.model = model
        if self.llmClient:
            self.llmClient.setModel(model)

    def setBaseUrl(self, baseUrl):
        self.baseUrl = baseUrl
        if self.llmClient:
            self.llmClient.setBaseUrl(baseUrl)

    def setProvider(self, provider):
        if self.llmClient:
            self.llmClient.setProvider(provider)

    def hasApiKey(self):
        return bool(self.apiKey)

    def generateResponse(self, prompt):
        """
        Generate AI response (non-streaming).

        Args:
            prompt: User's natural language request

        Returns:
            dict with keys: message, reasoning_content, code, tokens, cost
        """
        if not self.llmClient:
            raise RuntimeError("LLM client not initialized")
        if not self.apiKey:
            raise RuntimeError("API key not configured")

        context = {"scene": self._buildSceneContext()}
        response = self.llmClient.chat(prompt, context=context)
        self.conversationStore.addExchange(prompt, response)
        return response

    def generateResponseStream(self, prompt, context=None, on_delta=None, use_tools=True):
        """
        Generate AI response using streaming with optional tool calling.

        This runs the actual HTTP request (blocking I/O).  Callers should
        invoke this from a background thread.

        Args:
            prompt: User's natural language request
            context: Pre-built skill context (or None to build here)
            on_delta: Callback for incremental updates
            use_tools: Whether to use tool calling for skill search

        Returns:
            dict with keys: message, reasoning_content, code, tokens, cost
        """
        if not self.llmClient:
            raise RuntimeError("LLM client not initialized")
        if not self.apiKey:
            raise RuntimeError("API key not configured")

        if context is None:
            context = {"scene": self._buildSceneContext()}

        if use_tools and self.toolExecutor and self.skillTools:
            # Use tool calling for skill search
            try:
                # Progress callback to show tool execution in real-time
                def _on_progress(progress):
                    if on_delta:
                        on_delta(progress)
                
                response = self.llmClient.chatWithTools(
                    prompt,
                    tools=self.skillTools,
                    tool_executor=self._executeTool,
                    context=context,
                    on_progress=_on_progress,
                )
                
                # Stream the final response in small chunks for a smooth UI effect
                if on_delta and response:
                    reasoning = response.get('reasoning_content', '')
                    message = response.get('message', '')
                    chunk_size = 8
                    import time
                    if reasoning:
                        for i in range(0, len(reasoning), chunk_size):
                            chunk = reasoning[i:i+chunk_size]
                            on_delta({'reasoning_content': chunk, 'content': ''})
                            time.sleep(0.015)
                    if message:
                        for i in range(0, len(message), chunk_size):
                            chunk = message[i:i+chunk_size]
                            on_delta({'content': chunk, 'reasoning_content': ''})
                            time.sleep(0.015)
            except Exception as e:
                logger.warning(f"Tool calling failed, falling back to regular chat: {e}")
                response = self.llmClient.chatStream(prompt, context=context, on_delta=on_delta)
        else:
            # Fallback to regular streaming
            response = self.llmClient.chatStream(prompt, context=context, on_delta=on_delta)
        
        self.conversationStore.addExchange(prompt, response)
        return response

    def addExecutionFeedback(self, feedback_text):
        """
        Append code execution feedback to the LLM conversation history
        so the model can reason about previous execution outcomes.
        """
        if self.llmClient:
            self.llmClient.conversation_history.append({
                "role": "system",
                "content": feedback_text,
            })
    
    def _executeTool(self, tool_name, tool_args):
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool (Grep, ReadFile, Glob)
            tool_args: Tool arguments dict
            
        Returns:
            Tool execution result dict
        """
        if not self.toolExecutor:
            return {"error": "Tool executor not initialized"}
        
        try:
            result = self.toolExecutor.execute(tool_name, tool_args)
            # Log tool execution for debugging
            logger.info(f"Tool {tool_name} executed: {tool_args.get('pattern', tool_args.get('path', 'N/A'))}")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {"error": str(e)}

    def _buildSceneContext(self):
        """
        Build raw context about the current Slicer MRML scene.

        Returns:
            Dictionary with the full raw MRML XML string, or None.
        """
        try:
            scene = slicer.mrmlScene
            scene.SetSaveToXMLString(1)
            scene.Commit()
            raw_mrml = scene.GetSceneXMLString()
            if raw_mrml:
                return {"raw_mrml": raw_mrml}
            else:
                logger.warning("GetSceneXMLString() returned empty string")
        except Exception as e:
            logger.warning(f"Failed to get scene XML: {e}")

        return None

    def executeCode(self, code):
        validation = self.codeValidator.validate(code)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Code validation failed: {validation['reason']}"
            }
        return self.executor.execute(code)
    
    def executeCodeAsync(self, code, callback=None):
        """
        Execute code asynchronously without blocking the UI.
        
        Note: Due to Qt thread constraints, execution happens in the main thread
        but is scheduled via QTimer to allow the current event loop to process.
        
        Args:
            code: Python code to execute
            callback: Function to call with result dict when complete
        """
        validation = self.codeValidator.validate(code)
        if not validation["valid"]:
            if callback:
                callback({
                    "success": False,
                    "error": f"Code validation failed: {validation['reason']}"
                })
            return
        
        self.executor.executeAsync(code, callback)

    def clearConversation(self):
        if self.conversationStore:
            self.conversationStore.clear()
        if self.llmClient:
            self.llmClient.clearHistory()

    def pauseProcessing(self):
        self._processing = False

    def resumeProcessing(self):
        self._processing = True

    def cleanup(self):
        if self.llmClient:
            self.llmClient.cleanup()
        if self.executor:
            self.executor.cleanup()

#------------------------------------------------------------------
# Test Class
#------------------------------------------------------------------
class SlicerAIAgentTest(ScriptedLoadableModuleTest):
    """Unit tests for SlicerAIAgent."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_ModuleImport()
        self.test_CodeValidator()
        self.test_SafeExecutor()
        self.test_SkillPath()

    def test_ModuleImport(self):
        try:
            from SlicerAIAgentLib import LLMClient, CodeValidator, SafeExecutor, ConversationStore
            self.delayDisplay("Module import test passed")
        except Exception as e:
            self.delayDisplay(f"Module import test failed: {e}")
            raise

    def test_CodeValidator(self):
        from SlicerAIAgentLib import CodeValidator

        validator = CodeValidator.CodeValidator()

        safe_code = "volume = slicer.util.loadVolume('test.nrrd')"
        result = validator.validate(safe_code)
        self.assertTrue(result["valid"], "Safe code should pass validation")

        unsafe_code = "import os; os.system('rm -rf /')"
        result = validator.validate(unsafe_code)
        self.assertFalse(result["valid"], "Unsafe code should fail validation")

        self.delayDisplay("Code validator test passed")

    def test_SafeExecutor(self):
        from SlicerAIAgentLib import SafeExecutor

        executor = SafeExecutor.SafeExecutor()

        code = "result = 2 + 2"
        result = executor.execute(code)
        self.assertTrue(result["success"], "Simple code should execute successfully")

        self.delayDisplay("Safe executor test passed")

    def test_SkillPath(self):
        logic = SlicerAIAgentLogic()

        self.assertIsNotNone(logic.skill_path, "Skill path should be set")
        self.assertIn(logic.skill_mode, ["full", "lightweight", "web", "unknown"])
        logic.cleanup()
        self.delayDisplay("Skill path test passed")
