import json
import os
import queue
import threading
import unittest
import logging
from typing import Dict, List, Optional
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
        # Thinking display state (shown during streaming, hidden after)
        self._thinkingDisplayText = ""
        self._thinkingDisplayed = False
        # Thread-safe queue for streaming events (filled by worker, drained on main thread)
        self._streamQueue = queue.Queue()
        self._streamPollTimer = None
        # Timing data for performance analysis
        self._timing = None
        self.currentAgentPlan = None
        self._pendingConfirmation = None
        self._roleTrace = []
        self._currentLogDir = None
        self._currentAgentRole = "Idle"
        self._lastExecutionResult = None
        self._lastVerificationResult = None
        self._lastSceneAfter = None
        self._lastOutputHasErrors = False
        # Interactive workflow state
        self._workflowOrchestrator = None
        self._activeWorkflowId = None
        self._waitingForUser = False
        self._workflowBannerLabel = None
        self._autoAdvanceWorkflowStep = None
        self._workflowInstructionsLabel = None
        self._workflowDoneButton = None
        self._workflowCancelButton = None

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

        # Extension CLI Generator UI (insert after Settings, before Conversation History)
        self._setupExtensionCLIGenerator()

        # Interactive workflow UI
        self._setupWorkflowUI()

        self._streamPollTimer = qt.QTimer()
        self._streamPollTimer.setInterval(50)
        self._streamPollTimer.timeout.connect(self._drainStreamQueue)
        self._streamPollTimer.start()

        # Index status: one-time console log only (no UI label)
        self._indexStatusTimer = None
        self._updateIndexStatus()

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
        self.thinkingTimerLabel = self.ui.findChild(qt.QLabel, "thinkingTimerLabel")
        # Vector index status label — hidden, only console logging is used
        self.indexStatusLabel = self.ui.findChild(qt.QLabel, "indexStatusLabel")
        if self.indexStatusLabel:
            self.indexStatusLabel.setVisible(False)
        else:
            self._injectIndexStatusLabel()

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
        self.providerSelector.addItems(["Kimi", "DeepSeek", "Claude", "OpenAI", "Qwen"])
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

        statusTimerLayout = qt.QHBoxLayout()
        self.statusLabel = qt.QLabel("Ready")
        self.thinkingTimerLabel = qt.QLabel("")
        self.thinkingTimerLabel.setStyleSheet("color: #666; font-size: 11px;")
        self.thinkingTimerLabel.setAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
        statusTimerLayout.addWidget(self.statusLabel, stretch=1)
        statusTimerLayout.addWidget(self.thinkingTimerLabel)
        mainLayout.addLayout(statusTimerLayout)

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
        
        # Thinking timer
        self._thinkingTimer = qt.QTimer()
        self._thinkingTimer.setInterval(100)
        self._thinkingTimer.timeout.connect(self._updateThinkingTimer)
        self._thinkingStartTime = None
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
        if hasattr(self, '_indexStatusTimer') and self._indexStatusTimer:
            self._indexStatusTimer.stop()
        if self.logic:
            self.logic.cleanup()
        logger.info("SlicerAIAgent widget cleaned up")

    def _injectIndexStatusLabel(self):
        """No-op: index status is logged to console only, not shown in UI."""
        self.indexStatusLabel = None

    def _updateIndexStatus(self):
        """Log vector index status to console only; no UI indicator."""
        if not self.logic:
            logger.info("Vector index status: Not initialized")
            return

        status = self.logic.get_index_status()
        if status != "Ready":
            logger.info(f"Vector index status: {status} — pre-retrieval will be skipped. Run 'python scripts/build_rag.py' to build the index.")

    # ------------------------------------------------------------------
    # Extension CLI Generator UI
    # ------------------------------------------------------------------
    def _setupExtensionCLIGenerator(self):
        """Set up the Extension CLI Generator collapsible UI section."""
        # Create collapsible group box
        self._cliGeneratorGroup = ctk.ctkCollapsibleGroupBox()
        self._cliGeneratorGroup.title = "Extension CLI Generator"
        self._cliGeneratorGroup.collapsed = True
        # Insert after Settings, before Conversation History
        settingsGroup = self.ui.findChild(ctk.ctkCollapsibleGroupBox, "settingsGroupBox")
        if settingsGroup:
            parentLayout = settingsGroup.parent().layout()
            idx = parentLayout.indexOf(settingsGroup)
            parentLayout.insertWidget(idx + 1, self._cliGeneratorGroup)
        else:
            self.layout.insertWidget(1, self._cliGeneratorGroup)

        cliLayout = qt.QVBoxLayout(self._cliGeneratorGroup)

        # Row 1: Source selector + Refresh button
        sourceLayout = qt.QHBoxLayout()
        sourceLabel = qt.QLabel("Source:")
        sourceLayout.addWidget(sourceLabel)

        self._sourceSelector = qt.QComboBox()
        self._sourceSelector.setToolTip("Select the extension source to browse")
        self._sourceSelector.setMinimumWidth(200)
        self._sourceSelector.addItems(["Extension Manager", "Additional Module Paths", "Loaded Modules"])
        sourceLayout.addWidget(self._sourceSelector, 1)

        self._refreshExtensionsButton = qt.QPushButton("Refresh")
        self._refreshExtensionsButton.setToolTip("Re-scan extensions from the selected source")
        self._refreshExtensionsButton.setMaximumWidth(80)
        sourceLayout.addWidget(self._refreshExtensionsButton)

        cliLayout.addLayout(sourceLayout)

        # Row 2: Extension selector (populated based on source)
        extLayout = qt.QHBoxLayout()
        extLabel = qt.QLabel("Extension:")
        extLayout.addWidget(extLabel)

        self._extensionSelector = qt.QComboBox()
        self._extensionSelector.setToolTip("Select an extension from the chosen source")
        self._extensionSelector.setMinimumWidth(200)
        extLayout.addWidget(self._extensionSelector, 1)

        cliLayout.addLayout(extLayout)

        # Store extension data separately (keyed by label text)
        self._extensionDataMap = {}
        self._discoveredExtensions = []

        # Row 3: Analyze & Generate button
        self._analyzeGenerateButton = qt.QPushButton("Analyze & Generate CLI")
        self._analyzeGenerateButton.setToolTip(
            "Analyze the selected extension and generate operation CLI tools"
        )
        self._analyzeGenerateButton.setEnabled(False)
        cliLayout.addWidget(self._analyzeGenerateButton)

        # Row 4: Progress display
        self._cliProgressDisplay = qt.QTextEdit()
        self._cliProgressDisplay.setReadOnly(True)
        self._cliProgressDisplay.setMaximumHeight(150)
        self._cliProgressDisplay.setFont(qt.QFont("Monospace", 9))
        self._cliProgressDisplay.setPlaceholderText("Progress will appear here...")
        cliLayout.addWidget(self._cliProgressDisplay)

        # Row 5: Status indicator
        self._cliStatusLabel = qt.QLabel("Ready")
        self._cliStatusLabel.setStyleSheet("font-weight: bold;")
        cliLayout.addWidget(self._cliStatusLabel)

        # Row 6: Result group (hidden until generation completes)
        self._cliResultGroup = qt.QGroupBox("Result")
        self._cliResultGroup.setVisible(False)
        resultLayout = qt.QVBoxLayout(self._cliResultGroup)

        self._cliResultSummary = qt.QLabel("")
        self._cliResultSummary.setWordWrap(True)
        resultLayout.addWidget(self._cliResultSummary)

        cliLayout.addWidget(self._cliResultGroup)

        # Row 7: Action buttons
        actionLayout = qt.QHBoxLayout()
        self._testCliButton = qt.QPushButton("Test CLI")
        self._testCliButton.setToolTip("Validate generated templates with CodeValidator")
        self._testCliButton.setEnabled(False)
        actionLayout.addWidget(self._testCliButton)

        self._deleteCliButton = qt.QPushButton("Delete CLI")
        self._deleteCliButton.setToolTip("Remove the generated CLI files")
        self._deleteCliButton.setEnabled(False)
        actionLayout.addWidget(self._deleteCliButton)

        self._reviseCliButton = qt.QPushButton("Revise with LLM")
        self._reviseCliButton.setToolTip("Fix failed validation using LLM revision")
        self._reviseCliButton.setEnabled(False)
        actionLayout.addWidget(self._reviseCliButton)

        cliLayout.addLayout(actionLayout)

        # Internal state
        self._cliGeneratorRunning = False

        # Connect signals
        self._refreshExtensionsButton.clicked.connect(self._onRefreshExtensionsClicked)
        self._sourceSelector.currentIndexChanged.connect(self._onSourceSelectionChanged)
        self._extensionSelector.currentIndexChanged.connect(self._onExtensionSelectionChanged)
        self._analyzeGenerateButton.clicked.connect(self._onAnalyzeGenerateClicked)
        self._testCliButton.clicked.connect(self._onTestCliClicked)
        self._deleteCliButton.clicked.connect(self._onDeleteCliClicked)
        self._reviseCliButton.clicked.connect(self._onReviseCliClicked)

        # Populate initial extension list
        self._onRefreshExtensionsClicked()

    def _onRefreshExtensionsClicked(self):
        """Re-scan extensions from all sources and populate based on current source selection."""
        try:
            from SlicerAIAgentLib.ExtensionCLILoader import discover_installed_extensions
            self._discoveredExtensions = discover_installed_extensions()
        except Exception:
            self._discoveredExtensions = []

        self._populateExtensionSelector()

    def _onSourceSelectionChanged(self, index):
        """When the source combo changes, repopulate the extension list."""
        self._populateExtensionSelector()

    # Map from source combo label to source_type tag in discovered extensions
    _SOURCE_TYPE_MAP = {
        "Extension Manager": "extension_manager",
        "Additional Module Paths": "additional_paths",
        "Loaded Modules": "loaded_modules",
    }

    def _populateExtensionSelector(self):
        """Populate the extension combo box based on the selected source."""
        self._extensionSelector.clear()
        self._extensionDataMap.clear()

        source_label = self._sourceSelector.currentText
        source_type = self._SOURCE_TYPE_MAP.get(source_label, "")
        if not source_type:
            return

        for ext in self._discoveredExtensions:
            if ext.get("source_type") != source_type:
                continue

            name = ext["name"]
            label = name
            if ext.get("cli_status"):
                label += f" [{ext['cli_status']}]"
            if not ext.get("has_python"):
                label += " (no Python)"
            self._extensionDataMap[label] = {
                "type": "installed",
                "name": name,
                "path": ext.get("source_path", ext.get("install_path", "")),
                "source_type": ext.get("source_type", ""),
            }
            self._extensionSelector.addItem(label)

        # Enable the button if a valid selection exists after population
        has_selection = self._extensionSelector.currentIndex >= 0
        self._analyzeGenerateButton.setEnabled(has_selection and not self._cliGeneratorRunning)

    def _onExtensionSelectionChanged(self, index):
        """Enable/disable the Analyze button based on selection."""
        has_selection = index >= 0
        self._analyzeGenerateButton.setEnabled(has_selection and not self._cliGeneratorRunning)

    def _onAnalyzeGenerateClicked(self):
        """Start the analysis pipeline in a background thread."""
        data = self._getSelectedExtensionData()
        if not data:
            return

        ext_name = data["name"]
        source_path = data["path"]

        if not source_path or not os.path.isdir(source_path):
            self._cliProgressDisplay.append(f"Error: Source path not found: {source_path}")
            return

        if self._cliGeneratorRunning:
            return

        self._cliGeneratorRunning = True
        self._analyzeGenerateButton.setEnabled(False)
        self._cliStatusLabel.setText("Analyzing...")
        self._cliStatusLabel.setStyleSheet("font-weight: bold; color: orange;")
        self._cliProgressDisplay.clear()
        self._cliResultGroup.setVisible(False)
        self._reviseCliButton.setEnabled(False)

        self._cliProgressDisplay.append(f"Starting analysis of '{ext_name}'...")
        self._cliProgressDisplay.append(f"Source: {source_path}")

        # Run in background thread
        import threading

        def _run_analysis():
            try:
                from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
                from SlicerAIAgentLib.CodeValidator import CodeValidator

                analyzer = ExtensionCLIAnalyzer(
                    llm_client=self.logic.llmClient,
                    code_validator=CodeValidator(),
                    on_progress=lambda n, s, d: self._streamQueue.put(
                        ('cli_progress', {'stage': n, 'name': s, 'detail': d})
                    ),
                    on_error=lambda e: self._streamQueue.put(('cli_error', e)),
                )

                result = analyzer.analyze_and_generate(
                    extension_name=ext_name,
                    source_path=source_path,
                    source_type=data.get("source_type", ""),
                    force_overwrite=False,
                )
                self._streamQueue.put(('cli_complete', result))

            except Exception as e:
                self._streamQueue.put(('cli_error', str(e)))

        thread = threading.Thread(target=_run_analysis, daemon=True)
        thread.start()

    def _onTestCliClicked(self):
        """Validate generated templates with CodeValidator."""
        data = self._getSelectedExtensionData()
        if not data:
            return

        ext_name = data["name"]
        from SlicerAIAgentLib.ExtensionCLILoader import get_cli_base_dir
        cli_dir = os.path.join(get_cli_base_dir(), ext_name)
        if not os.path.isdir(cli_dir):
            self._cliProgressDisplay.append(f"No CLI directory found for {ext_name}")
            return

        # Load generators to find template files
        gen_path = os.path.join(cli_dir, "code_generators.json")
        if not os.path.isfile(gen_path):
            self._cliProgressDisplay.append("No code_generators.json found")
            return

        with open(gen_path, "r") as f:
            generators = json.load(f)

        from SlicerAIAgentLib.CodeValidator import CodeValidator
        validator = CodeValidator()
        from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer

        self._cliProgressDisplay.append("Running validation tests...")
        all_pass = True

        for gen in generators:
            tpl_file = gen.get("template_file", "")
            tpl_path = os.path.join(cli_dir, tpl_file)
            if not os.path.isfile(tpl_path):
                self._cliProgressDisplay.append(f"  SKIP: {tpl_file} not found")
                continue

            with open(tpl_path, "r") as f:
                content = f.read()

            sample = content.replace(
                "{vol_lookup}",
                "inputVolume = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')"
            )
            sample = ExtensionCLIAnalyzer._fill_remaining_placeholders(sample)
            result = validator.validate(sample)

            status = "PASS" if result.get("valid", True) else "FAIL"
            self._cliProgressDisplay.append(f"  {tpl_file}: {status}")
            if not result.get("valid", True):
                all_pass = False
                self._cliProgressDisplay.append(f"    Error: {result.get('reason', 'unknown')}")
            for w in result.get("warnings", []):
                self._cliProgressDisplay.append(f"    Warning: {w}")

        if all_pass:
            self._cliStatusLabel.setText("Validated")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: green;")
            self._reviseCliButton.setEnabled(False)
        else:
            self._cliStatusLabel.setText("Validation Failed")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
            self._reviseCliButton.setEnabled(True)

    def _onDeleteCliClicked(self):
        """Delete the CLI for the selected extension."""
        data = self._getSelectedExtensionData()
        if not data:
            return

        ext_name = data["name"]

        # Confirm
        reply = qt.QMessageBox.question(
            None, "Delete CLI",
            f"Delete the generated CLI for '{ext_name}'?",
            qt.QMessageBox.Yes | qt.QMessageBox.No,
        )
        if reply != qt.QMessageBox.Yes:
            return

        from SlicerAIAgentLib.ExtensionCLILoader import delete_cli
        if delete_cli(ext_name):
            self._cliProgressDisplay.append(f"Deleted CLI for '{ext_name}'")
            self._cliStatusLabel.setText("Ready")
            self._cliStatusLabel.setStyleSheet("font-weight: bold;")
            self._cliResultGroup.setVisible(False)
            self._testCliButton.setEnabled(False)
            self._deleteCliButton.setEnabled(False)
            self._reviseCliButton.setEnabled(False)
            self._onRefreshExtensionsClicked()
        else:
            self._cliProgressDisplay.append(f"No CLI found for '{ext_name}'")

    def _onReviseCliClicked(self):
        """Revise failed templates using LLM feedback."""
        data = self._getSelectedExtensionData()
        if not data:
            return

        ext_name = data["name"]

        if self._cliGeneratorRunning:
            return

        self._cliGeneratorRunning = True
        self._cliStatusLabel.setText("Revising...")
        self._cliStatusLabel.setStyleSheet("font-weight: bold; color: orange;")
        self._reviseCliButton.setEnabled(False)

        # Collect errors from the progress display
        progress_text = self._cliProgressDisplay.toPlainText()
        errors = [
            line.strip() for line in progress_text.split("\n")
            if "Error:" in line or "FAIL" in line
        ]

        import threading

        def _run_revision():
            try:
                from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
                from SlicerAIAgentLib.CodeValidator import CodeValidator

                analyzer = ExtensionCLIAnalyzer(
                    llm_client=self.logic.llmClient,
                    code_validator=CodeValidator(),
                    on_progress=lambda n, s, d: self._streamQueue.put(
                        ('cli_progress', {'stage': n, 'name': s, 'detail': d})
                    ),
                    on_error=lambda e: self._streamQueue.put(('cli_error', e)),
                )

                result = analyzer.revise(ext_name, errors)
                self._streamQueue.put(('cli_revision_complete', result))

            except Exception as e:
                self._streamQueue.put(('cli_error', str(e)))

        thread = threading.Thread(target=_run_revision, daemon=True)
        thread.start()

    def _autoReviseCli(self, generation_result):
        """Automatically trigger LLM revision after generation validation fails."""
        data = self._getSelectedExtensionData()
        if not data:
            self._reviseCliButton.setEnabled(True)
            return

        ext_name = data["name"]
        self._cliGeneratorRunning = True
        self._cliStatusLabel.setText("Auto-revising...")
        self._cliStatusLabel.setStyleSheet("font-weight: bold; color: orange;")

        # Extract validation errors
        val_result = generation_result.get("validation_result", {})
        errors = val_result.get("errors", [])
        if not errors:
            error = val_result.get("reason", "Unknown validation error")
            errors = [error]

        import threading

        def _run_revision():
            try:
                from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
                from SlicerAIAgentLib.CodeValidator import CodeValidator

                analyzer = ExtensionCLIAnalyzer(
                    llm_client=self.logic.llmClient,
                    code_validator=CodeValidator(),
                    on_progress=lambda n, s, d: self._streamQueue.put(
                        ('cli_progress', {'stage': n, 'name': s, 'detail': d})
                    ),
                    on_error=lambda e: self._streamQueue.put(('cli_error', e)),
                )

                result = analyzer.revise(ext_name, errors)
                self._streamQueue.put(('cli_revision_complete', result))

            except Exception as e:
                self._streamQueue.put(('cli_error', str(e)))

        thread = threading.Thread(target=_run_revision, daemon=True)
        thread.start()

    def _getSelectedExtensionData(self):
        """Get the data dict for the currently selected extension."""
        current_text = self._extensionSelector.currentText
        if not current_text:
            return None
        return self._extensionDataMap.get(current_text)

    # ------------------------------------------------------------------
    # Interactive Workflow UI
    # ------------------------------------------------------------------
    def _setupWorkflowUI(self):
        """Set up UI components for guided interactive workflows."""
        from SlicerAIAgentLib.WorkflowOrchestrator import WorkflowOrchestrator
        from SlicerAIAgentLib.InteractionManager import InteractionManager

        self._interactionManager = InteractionManager()
        self._workflowOrchestrator = WorkflowOrchestrator(
            interaction_manager=self._interactionManager,
        )

        # Workflow wait banner (hidden by default)
        self._workflowFrame = qt.QFrame()
        self._workflowFrame.setStyleSheet(
            "QFrame { background-color: #FFF3E0; border: 1px solid #FF9800; "
            "border-radius: 4px; padding: 4px; }"
        )
        workflowLayout = qt.QVBoxLayout(self._workflowFrame)
        workflowLayout.setContentsMargins(8, 4, 8, 4)

        self._workflowBannerLabel = qt.QLabel()
        self._workflowBannerLabel.setStyleSheet("font-weight: bold; color: #E65100;")
        workflowLayout.addWidget(self._workflowBannerLabel)

        self._workflowInstructionsLabel = qt.QLabel()
        self._workflowInstructionsLabel.setWordWrap(True)
        self._workflowInstructionsLabel.setStyleSheet("color: #333;")
        workflowLayout.addWidget(self._workflowInstructionsLabel)

        buttonLayout = qt.QHBoxLayout()
        self._workflowDoneButton = qt.QPushButton("Done")
        self._workflowDoneButton.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 3px;"
        )
        self._workflowDoneButton.connect("clicked()", self._onWorkflowDoneClicked)
        buttonLayout.addWidget(self._workflowDoneButton)

        self._workflowCancelButton = qt.QPushButton("Cancel Workflow")
        self._workflowCancelButton.setStyleSheet(
            "background-color: #f44336; color: white; "
            "padding: 6px 16px; border-radius: 3px;"
        )
        self._workflowCancelButton.connect("clicked()", self._onWorkflowCancelClicked)
        buttonLayout.addWidget(self._workflowCancelButton)
        buttonLayout.addStretch()

        workflowLayout.addLayout(buttonLayout)
        self._workflowFrame.setVisible(False)

        # Insert after Extension CLI Generator panel, before stream poll timer setup
        self.layout.addWidget(self._workflowFrame)

    def _enterWorkflowWait(self, step_info):
        """
        Enter wait state for an interactive workflow step.
        Shows the workflow banner with instructions and enables Done/Cancel buttons.
        """
        # Support both nested dict format and flat dispatch_workflow_step format
        interaction = step_info.get("interaction", {})
        step_desc = (
            step_info.get("explanation")
            or step_info.get("step_info", {}).get("description")
            or "Interactive step"
        )
        instructions = (
            interaction.get("placement_instructions", "")
            or step_info.get("interaction_instructions", "")
        )
        interaction_type = (
            interaction.get("interaction_type", "")
            or step_info.get("interaction_type", "")
        )

        self._waitingForUser = True
        self._currentWorkflowStepInfo = step_info

        self._workflowBannerLabel.text = (
            f"Waiting for your interaction: {step_desc}"
        )
        self._workflowInstructionsLabel.text = (
            f"Interaction type: {interaction_type}\n"
            f"{instructions}\n\n"
            f"Complete the interaction in the 3D view, then click 'Done' to continue."
        )
        self._workflowFrame.setVisible(True)
        self._workflowDoneButton.setEnabled(True)
        self.sendButton.setEnabled(False)

        self._setAgentStatus("Workflow", f"Waiting: {step_desc}")
        logger.info(f"[Workflow] Entered wait state for step: {step_desc}")

    def _exitWorkflowWait(self):
        """Hide workflow UI and restore normal state."""
        self._waitingForUser = False
        self._workflowFrame.setVisible(False)
        self._currentWorkflowStepInfo = None
        self._setReadyStatus()

    def _onWorkflowDoneClicked(self):
        """User clicked Done after completing an interactive step."""
        if not self._workflowOrchestrator or not self._activeWorkflowId:
            self._exitWorkflowWait()
            return

        self._workflowDoneButton.setEnabled(False)
        self._setAgentStatus("Workflow", "Processing interaction...")

        result = self._workflowOrchestrator.complete_interaction(
            self._activeWorkflowId,
            template_filler=self._getWorkflowTemplateFiller(),
        )

        if result.get("type") == "error" and result.get("retry"):
            # Validation failed — tell the user and keep waiting
            self.appendToChat("System", f"Validation failed: {result['error']}")
            self._workflowDoneButton.setEnabled(True)
            self._workflowOrchestrator._get_state(self._activeWorkflowId).status = "waiting_for_user"
            return

        if result.get("type") == "error":
            self.appendToChat("System", f"Workflow error: {result.get('error', 'Unknown error')}")
            self._exitWorkflowWait()
            return

        # Sync the completed interactive step to ExtensionCLILoader's tracking
        # so _find_next_step_local doesn't re-discover it on auto-advance.
        completed_step_id = result.get("step_id")
        if completed_step_id:
            wf_state = self._workflowOrchestrator._get_state(self._activeWorkflowId)
            if wf_state:
                from SlicerAIAgentLib.ExtensionCLILoader import _workflow_completed_steps
                done = _workflow_completed_steps.setdefault(wf_state.extension_name, set())
                done.add(completed_step_id)

        # Show post-interaction result
        post_result = result.get("post_result")
        if post_result:
            if post_result.get("success"):
                output = post_result.get("output", "")
                if output:
                    self.appendToChat("System", f"Step completed. Output: {output}")
            else:
                self.appendToChat("System", f"Post-processing error: {post_result.get('error', '')}")
                self._exitWorkflowWait()
                return

        self._exitWorkflowWait()

        if result.get("workflow_completed"):
            self.appendToChat("System", "Workflow completed successfully!")
            self._activeWorkflowId = None
            self.sendButton.setEnabled(True)
            return

        # Auto-proceed to next step
        next_step = result.get("next_step")
        if next_step:
            self._autoProceedWorkflowStep(next_step)
        else:
            self.sendButton.setEnabled(True)

    def _onWorkflowCancelClicked(self):
        """User clicked Cancel Workflow."""
        if self._workflowOrchestrator and self._activeWorkflowId:
            self._workflowOrchestrator.cancel_workflow(self._activeWorkflowId)
            self._activeWorkflowId = None
        self._exitWorkflowWait()
        self.appendToChat("System", "Workflow cancelled.")
        self.sendButton.setEnabled(True)

    def _autoProceedWorkflowStep(self, next_step_info):
        """Automatically prompt the LLM to execute the next workflow step."""
        step_id = next_step_info.get("step_id")
        step_type = next_step_info.get("step_type")
        description = next_step_info.get("description", "")

        if step_type == "branch":
            # Ask the user about the branch decision
            self.appendToChat(
                "System",
                f"Next step is optional: {description}. "
                f"Would you like to proceed with this step?"
            )
            self.sendButton.setEnabled(True)
            return

        # Auto-send a prompt to proceed with the next step
        self.appendToChat("System", f"Proceeding to next step: {description}")
        self.promptInput.setPlainText(f"Proceed with step '{step_id}': {description}")
        self.onSendButtonClicked()

    def _getWorkflowTemplateFiller(self):
        """
        Return a template filler callable that reads and fills .py.tpl files
        from the active workflow's CLI directory.
        """
        if not self._workflowOrchestrator or not self._activeWorkflowId:
            return None

        state = self._workflowOrchestrator._get_state(self._activeWorkflowId)
        if not state:
            return None

        ext_name = state.extension_name
        from SlicerAIAgentLib.ExtensionCLILoader import _fill_template, _ensure_cache
        _ensure_cache()
        import os
        cli_dir = os.path.join(
            os.path.dirname(__file__), "Resources", "extension_CLI", ext_name
        )

        def filler(template_path, args):
            full_path = os.path.join(cli_dir, template_path)
            if not os.path.exists(full_path):
                return None
            with open(full_path, "r") as f:
                template_text = f.read()
            return _fill_template(template_text, args)

        return filler

    def _handleCliProgress(self, stage_num, stage_name, detail):
        """Handle CLI generator progress updates on the main thread."""
        self._cliProgressDisplay.append(f"  Stage {stage_num}: {stage_name} — {detail}")

    def _handleCliComplete(self, result):
        """Handle CLI generator completion on the main thread."""
        self._cliGeneratorRunning = False
        self._analyzeGenerateButton.setEnabled(True)

        if result.get("success"):
            self._cliStatusLabel.setText("Validated")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: green;")
            self._cliResultGroup.setVisible(True)
            self._testCliButton.setEnabled(True)
            self._deleteCliButton.setEnabled(True)
            self._reviseCliButton.setEnabled(False)

            manifest = result.get("manifest", {})
            stages = manifest.get("stages", [])
            self._cliResultSummary.setText(
                f"Generated CLI for {manifest.get('extension_name', '?')} "
                f"(stages: {', '.join(stages)}). "
                f"Saved to: {result.get('cli_dir', '?')}"
            )
            self._cliProgressDisplay.append("CLI generation complete and validated!")

            # Refresh the extension selector to show updated status, preserving selection
            ext_name = manifest.get("extension_name", "")
            self._onRefreshExtensionsClicked()
            if ext_name:
                for i in range(self._extensionSelector.count):
                    if ext_name in self._extensionSelector.itemText(i):
                        self._extensionSelector.setCurrentIndex(i)
                        break

        else:
            self._cliStatusLabel.setText("Failed")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
            error = result.get("error", "Unknown error")
            self._cliProgressDisplay.append(f"FAILED: {error}")

            stages = result.get("stages_completed", [])
            if stages:
                self._cliProgressDisplay.append(
                    f"Completed stages: {stages}"
                )

            # Auto-revise if templates were generated but validation failed
            if result.get("validation_result") and not result["validation_result"].get("valid"):
                self._cliProgressDisplay.append("Auto-revising with LLM...")
                self._autoReviseCli(result)

    def _handleCliRevisionComplete(self, result):
        """Handle CLI revision completion on the main thread."""
        self._cliGeneratorRunning = False
        self._reviseCliButton.setEnabled(False)
        self._analyzeGenerateButton.setEnabled(True)

        if result.get("success"):
            self._cliStatusLabel.setText("Revised & Validated")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: green;")
            self._cliProgressDisplay.append(
                f"Revision succeeded after {result.get('attempts', '?')} attempts."
            )
            self._testCliButton.setEnabled(True)
            self._deleteCliButton.setEnabled(True)
        else:
            self._cliStatusLabel.setText("Revision Failed")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
            self._cliProgressDisplay.append(
                f"Revision failed: {result.get('error', 'unknown')}"
            )
            self._reviseCliButton.setEnabled(True)

    def _handleCliError(self, error_msg):
        """Handle CLI generator error on the main thread."""
        self._cliGeneratorRunning = False
        self._analyzeGenerateButton.setEnabled(True)
        self._cliStatusLabel.setText("Error")
        self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
        self._cliProgressDisplay.append(f"ERROR: {error_msg}")

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

        # Thinking section — shown during streaming, hidden after thinking_done
        if getattr(self, '_thinkingDisplayed', False) and self._thinkingDisplayText:
            escaped = self.escapeHtml(self._thinkingDisplayText).replace(chr(10), '<br>')
            # Truncate display to last ~2000 chars to avoid UI lag
            if len(escaped) > 2000:
                escaped = '...' + escaped[-2000:]
            parts.append(
                f'<div style="margin-left: 10px; margin-top: 5px; padding: 8px; '
                f'background-color: #f5f5f0; border-left: 3px solid #ccc; '
                f'color: #888; font-style: italic; max-height: 300px; overflow-y: auto;">'
                f'{escaped}</div>'
            )

        # Content section
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

    def _updateThinkingTimer(self):
        """Update the thinking timer display every 100ms."""
        if self._thinkingStartTime is not None:
            import time
            elapsed = time.time() - self._thinkingStartTime
            self.thinkingTimerLabel.text = f"⏱ {elapsed:.1f}s"
    
    def _startThinkingTimer(self):
        """Start the thinking timer."""
        import time
        self._thinkingStartTime = time.time()
        self.thinkingTimerLabel.text = "⏱ 0.0s"
        self._thinkingTimer.start()
    
    def _stopThinkingTimer(self, final_status=None):
        """Stop the thinking timer and show final elapsed time."""
        self._thinkingTimer.stop()
        if self._thinkingStartTime is not None:
            import time
            elapsed = time.time() - self._thinkingStartTime
            if final_status:
                self.thinkingTimerLabel.text = f"⏱ {final_status} {elapsed:.1f}s"
            else:
                self.thinkingTimerLabel.text = f"⏱ {elapsed:.1f}s"
        self._thinkingStartTime = None

    def _setAgentStatus(self, role, status):
        """Show current role-composed agent phase in the existing status label."""
        self._currentAgentRole = role or "Agent"
        if hasattr(self, 'statusLabel') and self.statusLabel is not None:
            self.statusLabel.text = f"{self._currentAgentRole}: {status}"

    def _setReadyStatus(self):
        """Reset status label after a turn finishes or is cancelled."""
        self._currentAgentRole = "Idle"
        if hasattr(self, 'statusLabel') and self.statusLabel is not None:
            self.statusLabel.text = "Ready"

    def _autoAdvanceNextStep(self, next_step):
        """Auto-advance to the next workflow step after an automated step completes."""
        step_id = next_step.get("step_id", "")
        is_optional = next_step.get("is_optional", False)
        if is_optional:
            # For optional steps, just ask the user
            self.appendToChat("System",
                f"Optional step available: '{step_id}' — {next_step.get('description', '')}. "
                f"Type 'proceed' to continue or 'skip' to skip.")
            self._setReadyStatus()
            return
        # For required steps, put the prompt into the input and send
        self.promptInput.setPlainText(
            f"Proceed with workflow step '{step_id}'"
        )
        self.onSendButtonClicked()

    def _roleForStatus(self, status_text):
        """Map low-level API status messages to the composed role shown in the UI."""
        normalized = str(status_text or "").lower()
        if "retriev" in normalized or "search" in normalized or "read" in normalized or "tool" in normalized:
            return "Retriever"
        if "generat" in normalized:
            return "Planner/Programmer"
        if "think" in normalized:
            return "Retriever"
        if "validat" in normalized:
            return "Safety Critic"
        if "execut" in normalized:
            return "Executor"
        if "verify" in normalized:
            return "Verifier"
        if "correct" in normalized:
            return "Repairer"
        return self._currentAgentRole or "Agent"
    
    def _finalizeStreamingEntry(self):
        """Commit the current streaming assistant entry into chat history."""
        if self._streaming or self._streamReasoning or self._streamContent:
            self._chatEntriesHtml.append(self._buildStreamingEntryHtml())
            self._setChatHtml(''.join(self._chatEntriesHtml))

    def _drainStreamQueue(self):
        """Drain queued streaming events on the Qt main thread.
        
        Batches consecutive streaming deltas to avoid calling setHtml() hundreds
        of times per second, which blocks the main thread and delays complete/error
        events by tens of seconds.
        """
        # Collect all events currently in the queue
        events = []
        while True:
            try:
                events.append(self._streamQueue.get_nowait())
            except queue.Empty:
                break
        
        if not events:
            return
        
        # Batch consecutive non-round deltas into a single render pass
        i = 0
        while i < len(events):
            event_type, payload = events[i]
            
            if event_type == 'delta':
                if payload.get('round'):
                    # Tool progress deltas are committed entries, process immediately
                    self._onStreamDelta(payload)
                    i += 1
                else:
                    # Batch consecutive streaming deltas (content only)
                    batched_content = ""
                    batch_start = i
                    while i < len(events):
                        et, ep = events[i]
                        if et != 'delta' or ep.get('round'):
                            break
                        # reasoning_content is intentionally suppressed from the chat UI
                        batched_content += ep.get('content', '')
                        i += 1
                    # Apply batched deltas in one go
                    if batched_content:
                        self._streamContent += batched_content
                        self._renderStreamingEntry()
                    slicer.app.processEvents()
            elif event_type == 'complete':
                self._onStreamComplete(payload)
                i += 1
            elif event_type == 'error':
                self._onStreamError(payload)
                i += 1
            elif event_type == 'correction_complete':
                self._handleCorrectionResult(**payload)
                i += 1
            elif event_type == 'correction_error':
                self._handleCorrectionError(**payload)
                i += 1
            elif event_type == 'status':
                self._setAgentStatus(self._roleForStatus(payload), payload)
                i += 1
            elif event_type == 'role_trace':
                self._recordRoleEvent(
                    payload.get('role', 'Unknown'),
                    payload.get('event', 'event'),
                    payload.get('details', {})
                )
                i += 1
            elif event_type == 'cli_progress':
                self._handleCliProgress(payload['stage'], payload['name'], payload['detail'])
                i += 1
            elif event_type == 'cli_complete':
                self._handleCliComplete(payload)
                i += 1
            elif event_type == 'cli_revision_complete':
                self._handleCliRevisionComplete(payload)
                i += 1
            elif event_type == 'cli_error':
                self._handleCliError(payload)
                i += 1
            elif event_type == 'thinking_delta':
                self._thinkingDisplayText += payload
                self._thinkingDisplayed = True
                self._renderStreamingEntry()
                i += 1
            elif event_type == 'thinking_done':
                self._thinkingDisplayed = False
                self._thinkingDisplayText = ""
                self._renderStreamingEntry()
                i += 1
            elif event_type == 'workflow_wait':
                self._enterWorkflowWait(payload)
                i += 1
            else:
                i += 1

    def _onStreamDelta(self, delta):
        """Apply one streamed delta on the main thread."""
        if delta.get('round'):
            self._updateToolProgress(delta)
        else:
            # reasoning_content is intentionally not accumulated into the chat UI
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
        self._thinkingDisplayed = False
        self._thinkingDisplayText = ""
        self._finalizeStreamingEntry()

        # Record LLM internal timing and token usage
        if self._timing:
            self._timing['llm_timing'] = response.get('timing_report', {})
            if 'retrieval_timing' in response:
                self._timing['retrieval_timing'] = response['retrieval_timing']
            import time
            self._timing['generation_complete'] = time.time()
            if response.get('tokens'):
                self._timing['tokens'] = response['tokens']
            if response.get('cost') is not None:
                self._timing['cost'] = response['cost']

        # Thinking is already persisted per-round via on_reasoning callback — no need to write again here

        # Display generated code if any and auto-execute
        if response.get("code"):
            self.currentCode = response["code"]
            self.currentAgentPlan = response.get("agent_plan")

            # Transfer interactive workflow step info from Logic to Widget
            _stepInfoFromInteractive = False
            if hasattr(self.logic, '_lastInteractiveStep') and self.logic._lastInteractiveStep:
                step_info = self.logic._lastInteractiveStep
                self._currentWorkflowStepInfo = step_info
                self.logic._lastInteractiveStep = None
                _stepInfoFromInteractive = True

                # Start or update the workflow if needed
                if self._workflowOrchestrator and step_info.get("step_id"):
                    if not self._activeWorkflowId:
                        # Auto-start workflow for the extension
                        ext_name = step_info.get("tool", "")
                        try:
                            from SlicerAIAgentLib.ExtensionCLILoader import _ensure_cache, get_cli_base_dir
                            _ensure_cache()
                            import os, json
                            wf_path = os.path.join(get_cli_base_dir(), ext_name, "workflow.json")
                            if os.path.isfile(wf_path):
                                with open(wf_path, "r") as f:
                                    wf_graph = json.load(f)
                                self._workflowOrchestrator.load_workflow_graph(ext_name, wf_graph)
                                state = self._workflowOrchestrator.start_workflow(ext_name)
                                self._activeWorkflowId = state.workflow_id
                        except Exception as e:
                            logger.warning(f"Failed to start workflow: {e}")

                    # Set the orchestrator's current step to waiting
                    if self._activeWorkflowId:
                        state = self._workflowOrchestrator._get_state(self._activeWorkflowId)
                        if state:
                            state.current_step = step_info.get("step_id")
                            state.status = "running"

            # Handle automated (non-interactive) workflow steps the same way
            if (not _stepInfoFromInteractive
                and hasattr(self.logic, '_lastWorkflowStep')
                and self.logic._lastWorkflowStep):
                step_info = self.logic._lastWorkflowStep
                self._currentWorkflowStepInfo = step_info
                self.logic._lastWorkflowStep = None

                if step_info.get("step_id"):
                    if not self._activeWorkflowId:
                        ext_name = step_info.get("tool", "")
                        try:
                            from SlicerAIAgentLib.ExtensionCLILoader import _ensure_cache, get_cli_base_dir
                            _ensure_cache()
                            import os, json
                            wf_path = os.path.join(get_cli_base_dir(), ext_name, "workflow.json")
                            if os.path.isfile(wf_path):
                                with open(wf_path, "r") as f:
                                    wf_graph = json.load(f)
                                self._workflowOrchestrator.load_workflow_graph(ext_name, wf_graph)
                                state = self._workflowOrchestrator.start_workflow(ext_name)
                                self._activeWorkflowId = state.workflow_id
                        except Exception as e:
                            logger.warning(f"Failed to start workflow for automated step: {e}")

                    if self._activeWorkflowId:
                        state = self._workflowOrchestrator._get_state(self._activeWorkflowId)
                        if state:
                            state.current_step = step_info.get("step_id")
                            state.status = "running"

            self._recordRoleEvent("Planner", "agent_plan_received", {
                "has_plan": bool(self.currentAgentPlan),
                "steps": len(self.currentAgentPlan.get("steps", [])) if isinstance(self.currentAgentPlan, dict) else 0,
                "risk_level": self.currentAgentPlan.get("risk_level") if isinstance(self.currentAgentPlan, dict) else None,
            })
            self._recordRoleEvent("Programmer", "code_received", {
                "code_chars": len(self.currentCode or ""),
            })
            self.codeDisplay.setPlainText(response["code"])
            self._displayAgentPlanSummary(self.currentAgentPlan)
            self._saveAgentPlanToFile(self.currentAgentPlan)
            self._saveGeneratedCodeToFile(response["code"])
            # Auto-execute the generated code
            if self._timing:
                self._timing['autoexecute_start'] = time.time()
            self._autoExecuteCode()

        # Update per-turn cumulative token usage
        if response.get("tokens"):
            self._currentTurnTokens += response["tokens"]
            self._currentTurnCost += response.get("cost", 0)
            self._updateTokenLabel()

        self._stopThinkingTimer("Done")
        self._setReadyStatus()
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
                f"2. The model name is correct (e.g. 'kimi-k2.5', 'deepseek-v4-pro')\n"
                f"3. Your API key has access to K2.5 models\n\n"
                f"Technical details: {error_msg}")
        else:
            self.appendToChat("Error", f"Failed to generate response: {error_msg}")

        self._stopThinkingTimer("Error")
        self._setReadyStatus()
        self.sendButton.setEnabled(True)

    def onSendButtonClicked(self):
        prompt = self.promptInput.toPlainText().strip()
        if not prompt:
            return

        self.promptInput.clear()
        self.appendToChat("You", prompt)
        self._lastUserPrompt = prompt  # Save for isolated self-correction context
        
        # Record the current turn number for consistent debug file naming
        if self.logic and hasattr(self.logic, 'llmClient') and self.logic.llmClient:
            self._currentTurn = getattr(self.logic.llmClient, 'turn_number', 1)
        else:
            self._currentTurn = 1
        
        # Reset per-turn cumulative token/cost counters
        self._currentTurnTokens = 0
        self._currentTurnCost = 0.0

        self.sendButton.setEnabled(False)
        slicer.app.processEvents()

        # Reset streaming accumulators
        self._streamReasoning = ""
        self._streamContent = ""
        self._streaming = True
        self._thinkingDisplayText = ""
        self._thinkingDisplayed = False
        self._streamTimestamp = qt.QDateTime.currentDateTime().toString("hh:mm:ss")
        self._renderStreamingEntry()

        # Initialize timing
        import time
        self._timing = {
            'turn_start': time.time(),
            'prompt': prompt,
        }
        self._currentLogDir = self._createRunLogDir(getattr(self, "_currentTurn", 1))
        if self.logic and self.logic.llmClient:
            self.logic.llmClient.setDebugOutputDir(self._currentLogDir)
        self._roleTrace = []
        self._setAgentStatus("Observer", "Reading request...")
        self._recordRoleEvent("Observer", "received_prompt", {
            "prompt_length": len(prompt),
            "turn": getattr(self, "_currentTurn", 1),
        })
        
        # Start real-time thinking timer
        self._startThinkingTimer()

        # Build context on the main thread (it reads the MRML scene)
        import time as _time
        ctx_start = _time.time()
        context = {"scene": self.logic._buildSceneContext()} if self.logic else None

        # Inject active workflow state into context
        if self._workflowOrchestrator and self._activeWorkflowId:
            wf_fragment = self._workflowOrchestrator.get_active_workflow_prompt_fragment()
            if wf_fragment and context:
                context["workflow_state"] = wf_fragment

        if self._timing:
            self._timing['context_build_time'] = _time.time() - ctx_start
        self._recordRoleEvent("Observer", "captured_scene_context", {
            "has_scene_context": bool(context and context.get("scene")),
            "elapsed_sec": round(_time.time() - ctx_start, 3),
        })
        self._setAgentStatus("Retriever", "Retrieving...")

        # Launch the streaming request in a background thread
        def _backgroundStream():
            try:
                def _onDelta(delta):
                    self._streamQueue.put(('delta', dict(delta)))

                def _onStatus(status_text):
                    self._streamQueue.put(('status', status_text))

                def _onReasoning(reasoning_text, round_num):
                    self._appendThinkingToFile(reasoning_text, turn=getattr(self, '_currentTurn', 1))
                    # Hide thinking from UI when the round completes
                    self._streamQueue.put(('thinking_done', None))

                def _onReasoningDelta(chunk):
                    self._streamQueue.put(('thinking_delta', chunk))

                import time as _time
                gen_start = _time.time()
                if self._timing:
                    self._timing['generation_start'] = gen_start
                response = self.logic.generateResponseStream(prompt, context, _onDelta, on_status=_onStatus, on_reasoning=_onReasoning, on_reasoning_delta=_onReasoningDelta)
                if self._timing:
                    self._timing['generation_end'] = _time.time()
                self._streamQueue.put(('role_trace', {
                    'role': 'Retriever',
                    'event': 'retrieval_and_tool_loop_completed',
                    'details': {
                        'has_retrieval_timing': bool(response.get('retrieval_timing')),
                        'tool_rounds': response.get('timing_report', {}).get('tool_rounds', 0),
                        'api_calls': response.get('timing_report', {}).get('api_calls', 0),
                    },
                }))
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
                self.currentAgentPlan = response.get("agent_plan")
                self._recordRoleEvent("Planner", "agent_plan_received", {
                    "has_plan": bool(self.currentAgentPlan),
                    "steps": len(self.currentAgentPlan.get("steps", [])) if isinstance(self.currentAgentPlan, dict) else 0,
                    "risk_level": self.currentAgentPlan.get("risk_level") if isinstance(self.currentAgentPlan, dict) else None,
                })
                self._recordRoleEvent("Programmer", "code_received", {
                    "code_chars": len(self.currentCode or ""),
                })
                self.codeDisplay.setPlainText(response["code"])
                self._displayAgentPlanSummary(self.currentAgentPlan)
                self._saveAgentPlanToFile(self.currentAgentPlan)
                self._saveGeneratedCodeToFile(response["code"])
                # Auto-execute the generated code
                self._autoExecuteCode()

            if response.get("tokens"):
                self._currentTurnTokens += response["tokens"]
                self._currentTurnCost += response.get("cost", 0)
                self._updateTokenLabel()
                if self._timing:
                    self._timing['tokens'] = self._currentTurnTokens
                    self._timing['cost'] = self._currentTurnCost

            # Persist thinking/reasoning content to file (non-streaming path)
            if response.get('reasoning_content'):
                self._appendThinkingToFile(response['reasoning_content'], turn=getattr(self, '_currentTurn', 1))

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.appendToChat("Error", f"Failed to generate response: {str(e)}")
        finally:
            self._setReadyStatus()
            self.sendButton.setEnabled(True)

    def _saveGeneratedCodeToFile(self, code, suffix=""):
        """Save the generated code to a local text file for user reference.
        
        Args:
            code: The generated Python code string.
            suffix: Optional suffix for the filename (e.g. '_correction_1').
        """
        try:
            turn_number = getattr(self, '_currentTurn', 1)
            latestPath = os.path.join(self._getCurrentLogDir(), f'{turn_number}{suffix}_code.txt')
            with open(latestPath, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            logger.warning(f"Failed to save generated code to file: {e}")

    def _saveAgentPlanToFile(self, plan, suffix=""):
        """Persist the parsed agent_plan JSON as a standalone debug artifact."""
        if not isinstance(plan, dict):
            return
        try:
            turn_number = getattr(self, '_currentTurn', 1)
            path = os.path.join(self._getCurrentLogDir(), f'{turn_number}{suffix}_agent_plan.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save agent plan to file: {e}")

    def _createRunLogDir(self, turn_number):
        """Create logs/YYYYmmdd_HHMMSS_turnN for all artifacts from one run."""
        from datetime import datetime
        moduleDir = os.path.dirname(__file__)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(moduleDir, "logs", f"{stamp}_turn{turn_number}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _getCurrentLogDir(self):
        """Return current run log directory, creating a fallback if needed."""
        if not self._currentLogDir:
            self._currentLogDir = self._createRunLogDir(getattr(self, "_currentTurn", 1))
            if self.logic and self.logic.llmClient:
                self.logic.llmClient.setDebugOutputDir(self._currentLogDir)
        os.makedirs(self._currentLogDir, exist_ok=True)
        return self._currentLogDir

    def _recordRoleEvent(self, role, event, details=None):
        """Record a structured event for the role-composed agent trace."""
        import time
        entry = {
            "time": round(time.time(), 3),
            "role": role,
            "event": event,
            "details": details or {},
        }
        self._roleTrace.append(entry)
        if self._timing is not None:
            self._timing["role_trace"] = list(self._roleTrace)

    def _saveRoleTraceToFile(self, suffix=""):
        """Persist the current role trace for academic/debug inspection."""
        try:
            turn_number = getattr(self, '_currentTurn', 1)
            path = os.path.join(self._getCurrentLogDir(), f'{turn_number}{suffix}_role_trace.json')
            payload = {
                "turn": turn_number,
                "prompt": getattr(self, "_lastUserPrompt", ""),
                "events": self._roleTrace,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save role trace: {e}")

    def _displayAgentPlanSummary(self, plan):
        """Show a compact plan summary in chat without adding a new UI surface."""
        if not plan:
            self.appendToChat("System", "No valid agent_plan block was returned. Auto-execution will be blocked.")
            return
        try:
            summary = plan.get("task_summary", "No task summary")
            overall_confidence = plan.get("overall_confidence") or self._deriveOverallPlanConfidence(plan)
            steps = plan.get("steps", [])
            step_lines = []
            for i, step in enumerate(steps[:5], start=1):
                if isinstance(step, dict):
                    action = step.get("action", "Unnamed step")
                    confidence = step.get("confidence", "unknown")
                    step_lines.append(f"{i}. {action} [{confidence}]")
            extra = "" if len(steps) <= 5 else f"\n... {len(steps) - 5} more step(s)"
            self.appendToChat(
                "System",
                "Verified plan received.\n"
                f"Task: {summary}\n"
                f"Overall confidence: {overall_confidence}\n"
                + ("\n".join(step_lines) if step_lines else "No steps listed")
                + extra
            )
        except Exception as e:
            logger.warning(f"Failed to display agent plan: {e}")

    def _deriveOverallPlanConfidence(self, plan):
        """Estimate whole-plan confidence for display when the plan lacks the field."""
        if not isinstance(plan, dict):
            return "unknown"
        steps = plan.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return "unknown"
        confidence_values = [
            str(step.get("confidence", "unknown")).lower()
            for step in steps
            if isinstance(step, dict)
        ]
        if not confidence_values:
            return "unknown"
        if any(value in ("low", "needs_lookup", "unknown") for value in confidence_values):
            return "low"
        if any(value == "medium" for value in confidence_values):
            return "medium"
        return "high"

    def _validateAgentPlan(self, plan):
        """Validate the structured plan enough to prevent blind execution."""
        errors = []
        warnings = []
        if not isinstance(plan, dict):
            return {
                "valid": False,
                "requires_confirmation": False,
                "errors": ["Missing or invalid agent_plan JSON block"],
                "warnings": [],
            }

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append("agent_plan.steps must be a non-empty list")

        risk = str(plan.get("risk_level", "unknown")).lower()
        plan_requires_confirmation = bool(plan.get("requires_confirmation")) or risk == "high"
        unresolved = plan.get("unverified_assumptions", [])
        if unresolved is None:
            unresolved = []
        if not isinstance(unresolved, list):
            warnings.append("agent_plan.unverified_assumptions should be a list")
            unresolved = [str(unresolved)]

        high_conf_steps = 0
        high_conf_without_evidence = 0
        needs_lookup = []
        if isinstance(steps, list):
            for idx, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    errors.append(f"Step {idx} must be an object")
                    continue
                if not step.get("action"):
                    errors.append(f"Step {idx} is missing action")
                confidence = str(step.get("confidence", "unknown")).lower()
                if confidence == "needs_lookup":
                    needs_lookup.append(step.get("action", f"step {idx}"))
                if confidence == "high":
                    high_conf_steps += 1
                    evidence = step.get("evidence")
                    if not evidence:
                        high_conf_without_evidence += 1
                # Validate expected_scene_change
                expected = step.get("expected_scene_change")
                if not isinstance(expected, dict) or not expected.get("type"):
                    errors.append(f"Step {idx} is missing required 'expected_scene_change'. Use 'not_checked' if this step has no scene effect.")
                else:
                    valid_types = {"node_count_delta", "node_exists", "node_modified",
                                   "node_has_display", "node_name_matches", "layout_changed",
                                   "selection_changed", "module_entered", "property_true", "not_checked"}
                    if str(expected.get("type")).lower() not in valid_types:
                        warnings.append(f"Step {idx} has unsupported expected_scene_change type '{expected.get('type')}'")

        if needs_lookup:
            errors.append("Plan still contains needs_lookup step(s): " + ", ".join(needs_lookup[:3]))
        if high_conf_steps and high_conf_without_evidence == high_conf_steps and high_conf_steps >= 2:
            warnings.append("Most high-confidence API steps have no explicit evidence")

        return {
            "valid": len(errors) == 0,
            "requires_confirmation": plan_requires_confirmation,
            "errors": errors,
            "warnings": warnings,
        }

    def _askExecutionConfirmation(self, reason, resume_callback):
        """Ask the user before executing high-risk or destructive code."""
        message_box = qt.QMessageBox(slicer.util.mainWindow())
        message_box.setWindowTitle("Confirm Agent Execution")
        message_box.setText("The generated plan or code requires confirmation before execution.")
        message_box.setInformativeText(reason)
        message_box.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        message_box.setDefaultButton(qt.QMessageBox.No)
        result = message_box.exec_()
        if result == qt.QMessageBox.Yes:
            self._recordRoleEvent("SafetyCritic", "execution_confirmed_by_user", {
                "reason": reason,
            })
            resume_callback()
        else:
            self._setReadyStatus()
            self.sendButton.setEnabled(True)
            self._recordRoleEvent("SafetyCritic", "execution_cancelled_by_user", {
                "reason": reason,
            })
            self._saveRoleTraceToFile()
            self.appendToChat("System", "Execution cancelled by user.")

    def _autoExecuteCode(self, attempt=1, max_attempts=5):
        """Auto-execute generated code with pre-validation and self-correction on failure."""
        if not hasattr(self, 'currentCode') or not self.currentCode:
            return

        self._setAgentStatus("Safety Critic", "Validating plan...")
        plan_validation = self._validateAgentPlan(getattr(self, "currentAgentPlan", None))
        self._recordRoleEvent("SafetyCritic", "plan_validation_completed", {
            "valid": plan_validation.get("valid"),
            "requires_confirmation": plan_validation.get("requires_confirmation"),
            "errors": plan_validation.get("errors", []),
            "warnings": plan_validation.get("warnings", []),
        })
        if not plan_validation["valid"]:
            error_msg = "; ".join(plan_validation["errors"])
            if attempt >= max_attempts:
                self._setReadyStatus()
                self.sendButton.setEnabled(True)
                self._recordRoleEvent("Repairer", "max_attempts_reached", {
                    "attempts": max_attempts,
                    "final_error": error_msg,
                    "stage": "plan_validation",
                })
                self._saveRoleTraceToFile()
                self.appendToChat(
                    "Error",
                    f"Plan validation failed after {max_attempts} attempts.\nFinal error: {error_msg}"
                )
                return
            self.appendToChat(
                "System",
                f"Plan validation failed (attempt {attempt}/{max_attempts}).\n"
                f"Error: {error_msg}\n"
                "Auto-correcting..."
            )
            if self.logic:
                self.logic.addExecutionFeedback(
                    f"Agent plan validation failed (attempt {attempt}/{max_attempts}):\n"
                    f"Error: {error_msg}\n"
                    "The response must include a valid ```agent_plan JSON block before the Python code."
                )
            self._selfCorrectCode(error_msg, attempt, max_attempts)
            return

        # Pre-validation: check for syntax errors and common issues
        import time
        if self._timing:
            self._timing['validation_start'] = time.time()
        
        validation = None
        if self.logic and hasattr(self.logic, 'codeValidator'):
            self._setAgentStatus("Safety Critic", "Validating code...")
            validation = self.logic.codeValidator.validate(self.currentCode)
            self._recordRoleEvent("SafetyCritic", "code_validation_completed", {
                "valid": validation.get("valid"),
                "requires_confirmation": validation.get("requires_confirmation"),
                "destructive_ops": validation.get("destructive_ops", []),
                "warnings": validation.get("warnings", []),
                "reason": validation.get("reason"),
            })
            if self._timing:
                self._timing['validation_end'] = time.time()
            if not validation['valid']:
                # Syntax error detected before execution
                error_msg = validation['reason']
                if attempt >= max_attempts:
                    self._setReadyStatus()
                    self.sendButton.setEnabled(True)
                    self._recordRoleEvent("Repairer", "max_attempts_reached", {
                        "attempts": max_attempts,
                        "final_error": error_msg,
                        "stage": "code_validation",
                    })
                    self._saveRoleTraceToFile()
                    self.appendToChat(
                        "Error",
                        f"Pre-validation failed after {max_attempts} attempts.\nFinal error: {error_msg}"
                    )
                    return
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
        else:
            if self._timing:
                self._timing['validation_end'] = time.time()

        requires_confirmation = plan_validation.get("requires_confirmation", False)
        confirmation_reasons = []
        if plan_validation.get("warnings"):
            for warning in plan_validation["warnings"]:
                logger.warning(f"Agent plan warning: {warning}")
        if requires_confirmation:
            confirmation_reasons.append(
                f"Plan risk level: {getattr(self, 'currentAgentPlan', {}).get('risk_level', 'unknown')}"
            )
        if validation and validation.get("requires_confirmation"):
            destructive_ops = ", ".join(validation.get("destructive_ops", []))
            confirmation_reasons.append(f"Potentially destructive operation(s): {destructive_ops}")
        if confirmation_reasons:
            reason = "\n".join(confirmation_reasons)
            logger.info(f"Auto-approving execution with warnings: {reason}")
            self.appendToChat("Warning", f"Executing with potentially destructive operations: {reason}")
            self._recordRoleEvent("SafetyCritic", "confirmation_required", {
                "reason": reason,
            })

        self._autoExecuteCodeConfirmed(attempt, max_attempts)

    def _autoExecuteCodeConfirmed(self, attempt=1, max_attempts=5):
        """Run already-validated code after optional confirmation has passed."""
        import time

        self._setAgentStatus("Executor", f"Executing (attempt {attempt}/{max_attempts})...")
        slicer.app.processEvents()

        if self._timing and 'execution_start' not in self._timing:
            self._timing['execution_start'] = time.time()
        if self._timing:
            self._timing['execution_async_call'] = time.time()

        self._recordRoleEvent("Executor", "execution_scheduled", {
            "attempt": attempt,
            "code_chars": len(self.currentCode or ""),
        })

        # Capture scene state BEFORE execution for semantic verification
        before_snapshot = None
        if hasattr(self, 'buildSceneSnapshot'):
            try:
                before_snapshot = self.buildSceneSnapshot()
            except Exception as e:
                logger.warning(f"Failed to build pre-execution scene snapshot: {e}")

        def onExecutionComplete(result):
            if self._timing:
                self._timing['execution_callback_start'] = time.time()
            feedback_lines = []
            output_has_errors = False
            self._recordRoleEvent("Executor", "execution_completed", {
                "success": result.get("success"),
                "timed_out": result.get("timed_out", False),
                "execution_time": result.get("execution_time", 0),
                "error": result.get("error"),
            })
            if result.get("timed_out", False):
                self._setReadyStatus()
                output = result.get('output', 'No output')
                exec_time = result.get('execution_time', 30)
                msg = f"Code execution timed out after {exec_time:.1f}s."
                if output:
                    msg += f"\nOutput: {output}"
                self.appendToChat("Warning", msg)
                feedback_lines.append(f"Status: timed_out\nExecution time: {exec_time:.1f}s\nOutput: {output}")
            elif result["success"]:
                output = result.get('output', 'No output')
                execution_time = result.get('execution_time', 0)
                msg = f"Code executed successfully in {execution_time:.2f}s."
                if output:
                    msg += f"\nOutput: {output}"
                self.appendToChat("System", msg)
                feedback_lines.append(f"Status: success\nExecution time: {execution_time:.2f}s\nOutput: {output}")
                # Detect actual errors (excluding VTK warnings which are often benign)
                lower_output = output.lower()
                # Strip VTK output lines — they frequently contain words like "error"
                # in benign messages (e.g., "Decimation completed without errors")
                non_vtk_output = "\n".join(
                    line for line in output.split("\n")
                    if not line.strip().startswith("[VTK")
                ).lower()
                if any(k in non_vtk_output for k in ('traceback', 'exception', 'failed', 'error')):
                    output_has_errors = True
                    feedback_lines.append("Warning: execution output contains error indicators even though no uncaught exception was raised.")
                self._lastExecutionResult = dict(result)
                self._lastOutputHasErrors = output_has_errors
            else:
                # Execution failed
                error_msg = result.get('error', 'Unknown error')
                execution_time = result.get('execution_time', 0)
                msg = f"Execution failed (attempt {attempt}/{max_attempts}).\nError: {error_msg[:200]}"
                self.appendToChat("System", msg)
                feedback_lines.append(f"Status: failed\nExecution time: {execution_time:.2f}s\nError: {error_msg[:500]}")
                self._lastExecutionResult = dict(result)
                self._lastOutputHasErrors = True

            # Semantic scene verification: compare before/after snapshots against agent_plan expectations
            if result.get("success") and not result.get("timed_out", False) and before_snapshot:
                try:
                    if hasattr(self, 'buildSceneSnapshot') and hasattr(self, 'verifySceneAgainstPlan'):
                        after_snapshot = self.buildSceneSnapshot()
                        plan = getattr(self, 'currentAgentPlan', None)
                        if plan:
                            verification = self.verifySceneAgainstPlan(before_snapshot, after_snapshot, plan)
                            if not verification.get("valid", True):
                                output_has_errors = True
                                verr = "; ".join(verification.get("errors", []))
                                feedback_lines.append(f"Scene verification failed: {verr}")
                                self.appendToChat("System", f"Scene verification failed: {verr}")
                                self._recordRoleEvent("SafetyCritic", "scene_verification_failed", {
                                    "errors": verification.get("errors", []),
                                    "warnings": verification.get("warnings", []),
                                    "diagnostics": verification.get("diagnostics", []),
                                })
                            elif verification.get("warnings"):
                                self._recordRoleEvent("SafetyCritic", "scene_verification_warnings", {
                                    "warnings": verification.get("warnings", []),
                                })
                except Exception as e:
                    logger.warning(f"Scene verification raised exception: {e}")

            # Add execution feedback to conversation history
            if self.logic:
                feedback_text = "Code execution result:\n" + "\n".join(feedback_lines) + "\nThe MRML scene has been updated. Refer to the CURRENT SLICER SCENE in the next system prompt for the complete raw MRML."
                self.logic.addExecutionFeedback(feedback_text)

            # Record execution timing
            if self._timing:
                self._timing['execution_end'] = time.time()
                self._timing['execution_result'] = 'success' if result.get('success') else 'failed'
                # Record executor internal timing
                if 'executor_scheduled' in result:
                    self._timing['executor_scheduled'] = result['executor_scheduled']
                if 'executor_actual_start' in result:
                    self._timing['executor_actual_start'] = result['executor_actual_start']
                self._writeTimingReport()

            # Interactive workflow detection: if an interactive step just executed,
            # transition to waiting_for_user and enter wait mode.
            if (result.get("success")
                and self._workflowOrchestrator
                and self._activeWorkflowId
                and self._workflowOrchestrator.is_workflow_active()):
                wf_state = self._workflowOrchestrator._get_state(self._activeWorkflowId)
                if wf_state and wf_state.status == "running":
                    step_info = getattr(self, '_currentWorkflowStepInfo', None)
                    if step_info and step_info.get("type") == "interactive":
                        wf_state.status = "waiting_for_user"
                        self._recordRoleEvent("Workflow", "entering_wait", {
                            "step_id": wf_state.current_step,
                        })
                        self._streamQueue.put(('workflow_wait', step_info))
                        return
                    # Automated step completed — auto-advance to next step
                    if step_info and step_info.get("type") == "automated":
                        completed_step_id = step_info.get("step_id", "")
                        # Mark current step complete in both tracking systems
                        from SlicerAIAgentLib.ExtensionCLILoader import _find_next_step_local, _workflow_completed_steps
                        done = _workflow_completed_steps.setdefault(step_info.get("tool", ""), set())
                        done.add(completed_step_id)
                        # Also mark in WorkflowOrchestrator's state so
                        # complete_interaction()/_find_next_step stays in sync
                        if self._activeWorkflowId:
                            wf_st = self._workflowOrchestrator._get_state(self._activeWorkflowId)
                            if wf_st and completed_step_id not in wf_st.completed_steps:
                                wf_st.completed_steps.append(completed_step_id)
                        # Load workflow graph for next-step lookup
                        try:
                            import os, json
                            from SlicerAIAgentLib.ExtensionCLILoader import get_cli_base_dir
                            ext_name = step_info.get("tool", "")
                            wf_path = os.path.join(get_cli_base_dir(), ext_name, "workflow.json")
                            if os.path.isfile(wf_path):
                                with open(wf_path) as wf_f:
                                    wf_graph = json.load(wf_f)
                                next_step = _find_next_step_local(wf_graph, done)
                                if next_step:
                                    self.appendToChat("System",
                                        f"Step '{completed_step_id}' completed. "
                                        f"Proceeding to next step: '{next_step['step_id']}'...")
                                    # Trigger a new turn with the next step
                                    self._autoAdvanceWorkflowStep = next_step
                        except Exception as e:
                            logger.warning(f"Workflow auto-advance failed: {e}")

            # Self-correction for failures or suspicious outputs (but not timeouts)
            # Skip self-correction during active workflow steps that succeeded —
            # VTK noise should not derail the workflow sequence.
            workflow_active = (
                self._workflowOrchestrator
                and self._activeWorkflowId
                and self._workflowOrchestrator.is_workflow_active()
            )
            if not result.get("timed_out", False) and (not result["success"] or (output_has_errors and not workflow_active)):
                if attempt < max_attempts:
                    self.appendToChat("System", "Auto-correcting...")
                    error_for_correction = result.get('error', '')
                    if not error_for_correction and output_has_errors:
                        # success=True but output contains clear failure keywords
                        error_for_correction = "\n".join(feedback_lines) or output
                    self._recordRoleEvent("Repairer", "correction_requested", {
                        "attempt": attempt + 1,
                        "reason": error_for_correction[:1000] if error_for_correction else "",
                    })
                    self._selfCorrectCode(error_for_correction, attempt, max_attempts)
                else:
                    self._setReadyStatus()
                    final_error = result.get('error', 'Unknown error') if not result["success"] else "Output contains errors"
                    self.appendToChat("Error", 
                        f"Execution failed after {max_attempts} attempts.\n"
                        f"Final error: {final_error}")
                    self._recordRoleEvent("Repairer", "max_attempts_reached", {
                        "attempts": max_attempts,
                        "final_error": final_error,
                    })
                    self._saveRoleTraceToFile()
            else:
                self._recordRoleEvent("Executor", "task_completed", {
                    "execution_success": result.get("success"),
                })
                self._saveRoleTraceToFile()

                # Auto-advance workflow: if an automated step just completed and
                # there's a next step queued, trigger it automatically.
                next_step = getattr(self, '_autoAdvanceWorkflowStep', None)
                if next_step and result.get("success"):
                    self._autoAdvanceWorkflowStep = None
                    tool_name = next_step.get("step_id", "")
                    # Use a timer to schedule the next step as a new user-like turn
                    step_id = next_step["step_id"]
                    self.appendToChat("System",
                        f"Auto-advancing to workflow step: {step_id}")
                    # Schedule the auto-advance on the Qt main thread
                    qt.QTimer.singleShot(100, lambda: self._autoAdvanceNextStep(next_step))
                else:
                    self._setReadyStatus()
        
        # Execute asynchronously
        self.logic.executeCodeAsync(self.currentCode, onExecutionComplete)
    
    def _selfCorrectCode(self, error_msg, attempt, max_attempts):
        """Generate corrected code with isolated context (no conversation history bloat)."""
        if not self.currentCode:
            return

        import time
        if self._timing:
            corrections = self._timing.setdefault('corrections', [])
            corrections.append({'attempt': attempt + 1, 'start': time.time()})

        error_detail = error_msg if error_msg else "Unknown error"
        self._recordRoleEvent("Repairer", "correction_started", {
            "attempt": attempt + 1,
            "error": error_detail[:1000],
        })
        self.appendToChat("You", f"[Auto-correction attempt {attempt+1}]")
        self._setAgentStatus("Repairer", "Correcting...")
        self._startThinkingTimer()
        if self.logic and self.logic.llmClient:
            self.logic.llmClient.debug_suffix = "_correction"
        
        # Build system prompt on the main thread BEFORE starting the background thread.
        # _buildSceneContext() calls slicer.mrmlScene.GetSceneXMLString() which is a Qt
        # method and must run on the main thread.
        system_content = "You are an expert 3D Slicer Python coding assistant."
        if self.logic and self.logic.llmClient and hasattr(self.logic.llmClient, '_buildSystemPrompt'):
            try:
                context = {"scene": self.logic._buildSceneContext()} if self.logic else None
                system_content = self.logic.llmClient._buildSystemPrompt(context)
            except Exception:
                if hasattr(self.logic.llmClient, '_loadSystemPromptTemplate'):
                    system_content = self.logic.llmClient._loadSystemPromptTemplate()
        elif self.logic and self.logic.llmClient and hasattr(self.logic.llmClient, '_loadSystemPromptTemplate'):
            system_content = self.logic.llmClient._loadSystemPromptTemplate()
        
        # Capture state needed by the background thread
        _logic = self.logic
        _currentCode = self.currentCode
        _lastUserPrompt = getattr(self, '_lastUserPrompt', '')
        _currentTurn = getattr(self, '_currentTurn', 1)
        _system_content = system_content
        
        def _run_correction():
            """Run chatWithToolsIsolated in a background thread so the main Qt event loop
            stays alive and _processStreamQueue can consume progress deltas in real time."""
            try:
                isolated_messages = [{'role': 'system', 'content': _system_content}]
                
                if _lastUserPrompt:
                    isolated_messages.append({'role': 'user', 'content': _lastUserPrompt})
                
                # Inject prior tool trajectory so LLM doesn't fix blind
                prior_tool_messages = []
                if _logic and _logic.llmClient:
                    for msg in _logic.llmClient.conversation_history:
                        if msg.get('role') in ('assistant', 'tool'):
                            prior_tool_messages.append(msg)
                if prior_tool_messages:
                    isolated_messages.extend(prior_tool_messages)
                    isolated_messages.append({
                        'role': 'system',
                        'content': 'The messages above show the tool searches and file reads from the original attempt. Use them for reference.'
                    })
                
                isolated_messages.append({
                    'role': 'assistant',
                    'content': (
                        f"Previous agent_plan:\n"
                        f"```json\n{json.dumps(getattr(self, 'currentAgentPlan', None), ensure_ascii=False, indent=2)}\n```\n\n"
                        f"Previous Python code:\n"
                        f"```python\n{_currentCode}\n```"
                    )
                })

                user_content = (
                    f"CRITICAL: The previous Python code execution failed with this error:\n"
                    f"{error_detail}\n\n"
                    "You have SearchSymbol, Grep, ReadFile, and VectorSearch tools available. "
                    "If the error is caused by an incorrect API signature, missing parameter, or wrong module path, "
                    "use the tools to verify the correct usage before fixing. "
                    "Do NOT search unnecessarily — if you are confident in the fix, apply it directly.\n\n"
                    "Your task is to fix the error and output the COMPLETE corrected Python code in ONE ```python block."
                    " Also output a corrected ```agent_plan JSON block before the Python block."
                )

                isolated_messages.append({
                    'role': 'user',
                    'content': user_content
                })
                
                # Save isolated prompt to debug file before sending
                try:
                    suffix = f"_correction_{attempt}"
                    first_debug = os.path.join(self._getCurrentLogDir(), f'{_currentTurn}{suffix}_first_prompt_debug.txt')
                    with open(first_debug, 'w', encoding='utf-8') as f:
                        for i, msg in enumerate(isolated_messages):
                            f.write(f"{'='*60}\n")
                            f.write(f"MESSAGE {i+1} | role: {msg.get('role', 'unknown')}\n")
                            f.write(f"{'='*60}\n")
                            f.write(f"{msg.get('content', '')}\n\n")
                except Exception:
                    pass
                
                # Use tool-calling isolated chat so LLM can re-search if needed
                def _on_correction_progress(progress):
                    self._streamQueue.put(('delta', dict(progress)))

                def _on_correction_status(status_text):
                    self._streamQueue.put(('status', status_text))

                def _on_correction_reasoning(reasoning_text, round_num):
                    self._appendThinkingToFile(reasoning_text, turn=getattr(self, '_currentTurn', 1))
                    self._streamQueue.put(('thinking_done', None))

                def _on_correction_reasoning_delta(chunk):
                    self._streamQueue.put(('thinking_delta', chunk))

                response = _logic.llmClient.chatWithToolsIsolated(
                    messages=isolated_messages,
                    tools=_logic.skillTools,
                    tool_executor=_logic._executeTool,
                    max_tool_rounds=5,
                    on_progress=_on_correction_progress,
                    on_status=_on_correction_status,
                    on_reasoning=_on_correction_reasoning,
                    on_reasoning_delta=_on_correction_reasoning_delta,
                )
                
                # Dispatch result handling via _streamQueue so it runs on the main thread
                self._streamQueue.put(('correction_complete', {
                    'response': response,
                    'attempt': attempt,
                    'max_attempts': max_attempts,
                    'error_detail': error_detail,
                    'original_prompt': _lastUserPrompt,
                }))
            except Exception as e:
                self._streamQueue.put(('correction_error', {
                    'error_msg': str(e),
                }))
        
        thread = threading.Thread(target=_run_correction, daemon=True)
        thread.start()
    
    def _handleCorrectionResult(self, response, attempt, max_attempts, error_detail, original_prompt):
        """Handle successful self-correction response on the main thread."""
        # Save correction timing report and token usage
        if self._timing:
            corrections = self._timing.get('corrections', [])
            if corrections:
                if response.get('timing_report'):
                    corrections[-1]['timing_report'] = response['timing_report']
                if response.get('tokens'):
                    corrections[-1]['tokens'] = response['tokens']
                if response.get('cost') is not None:
                    corrections[-1]['cost'] = response['cost']
        
        # Accumulate per-turn token/cost from self-correction
        if response.get('tokens'):
            self._currentTurnTokens += response['tokens']
            self._currentTurnCost += response.get('cost', 0)
            self._updateTokenLabel()
        
        if self.logic and self.logic.llmClient:
            self.logic.llmClient.debug_suffix = ""
        
        if response.get("code"):
            self.currentCode = response["code"]
            self.currentAgentPlan = response.get("agent_plan")
            self._recordRoleEvent("Repairer", "correction_received", {
                "attempt": attempt + 1,
                "has_plan": bool(self.currentAgentPlan),
                "code_chars": len(self.currentCode or ""),
            })
            self.codeDisplay.setPlainText(response["code"])
            self._displayAgentPlanSummary(self.currentAgentPlan)
            self._saveAgentPlanToFile(self.currentAgentPlan, suffix=f"_correction_{attempt}")
            self._saveGeneratedCodeToFile(response["code"], suffix=f"_correction_{attempt}")
            self._stopThinkingTimer("Corrected")
            
            # Update conversation_history: replace wrong code with corrected code
            if self.logic and self.logic.llmClient:
                history = self.logic.llmClient.conversation_history
                # Find last assistant message containing a code block and replace it
                for i in range(len(history) - 1, -1, -1):
                    msg = history[i]
                    if msg.get('role') == 'assistant' and '```' in msg.get('content', ''):
                        history[i] = {
                            'role': 'assistant',
                            'content': response.get('message', f"```python\n{response['code']}\n```")
                        }
                        if response.get('reasoning_content'):
                            history[i]['reasoning_content'] = response['reasoning_content']
                        break
                
                # Append correction marker
                history.append({
                    'role': 'system',
                    'content': (
                        f"CORRECTION: The previous code failed with: {error_detail}. "
                        f"After correction attempt {attempt + 1}, the working version is above. "
                        f"The original search results remain valid."
                    )
                })
                
                # Append compressed correction-phase tool results
                correction_messages = response.get('intermediate_messages', [])
                if correction_messages:
                    compressed = self.logic.llmClient._compressToolResultsForHistory(
                        correction_messages, user_prompt=original_prompt
                    )
                    history.extend(compressed)
            
            self._autoExecuteCode(attempt + 1, max_attempts)
        else:
            raw_msg = response.get('message', '')[:300]
            self.appendToChat("System", f"Correction response contained no code block. Raw response preview:\n{raw_msg}")
            self._stopThinkingTimer("Failed")
            self._setReadyStatus()
    
    def _appendThinkingToFile(self, reasoning_content: str, turn: int = 1):
        """Append LLM thinking/reasoning content to a local text file."""
        from datetime import datetime
        try:
            filepath = os.path.join(self._getCurrentLogDir(), f'{turn}_thinking_history.txt')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"Turn {turn} | {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{reasoning_content}\n\n")
        except Exception as e:
            logger.warning(f"Failed to write thinking history: {e}")
    
    def _handleCorrectionError(self, error_msg):
        """Handle self-correction error on the main thread."""
        self._stopThinkingTimer("Error")
        if self.logic and self.logic.llmClient:
            self.logic.llmClient.debug_suffix = ""
        self.appendToChat("Error", f"Self-correction failed: {error_msg}")
        self._setReadyStatus()


    def _updateTokenLabel(self):
        """Update the token/cost label with per-turn cumulative usage."""
        turn = getattr(self, '_currentTurn', 1)
        tokens = getattr(self, '_currentTurnTokens', 0)
        cost = getattr(self, '_currentTurnCost', 0.0)
        self.tokenLabel.text = f"Turn {turn} | Cumulative: {tokens} tokens | ${cost:.4f}"

    def _writeTimingReport(self):
        """Write detailed performance timing to a text file."""
        import time
        if not self._timing:
            return
        try:
            turn_number = 1
            if self.logic and hasattr(self.logic, 'llmClient') and self.logic.llmClient:
                turn_number = getattr(self.logic.llmClient, 'turn_number', 1)
                suffix = getattr(self.logic.llmClient, 'debug_suffix', "")
            else:
                suffix = ""
            # turn_number is already incremented after response, so current turn is turn_number-1
            turn_number = max(1, turn_number - 1)
            logPath = os.path.join(self._getCurrentLogDir(), f'{turn_number}_performance_log{suffix}.txt')

            t = self._timing
            lines = ["="*50, "Performance Timing Report", "="*50, ""]

            # ---- Compute phase times so the overview adds up ----
            phase1_scene = t.get('context_build_time', 0.0)

            phase2_decompose = 0.0
            phase2_faiss = 0.0
            phase2_retrieval = 0.0
            rt = t.get('retrieval_timing', {})
            if rt:
                phase2_decompose = rt.get('decompose_time', 0.0)
                phase2_faiss = sum(pq.get('time', 0.0) for pq in rt.get('retrieval_per_query', []))
                phase2_retrieval = phase2_decompose + phase2_faiss

            phase3_api = 0.0
            phase3_tool = 0.0
            phase3_other = 0.0
            phase3_tools = 0.0
            lt = t.get('llm_timing', {})
            if lt:
                phase3_api = lt.get('total_api_time', 0.0)
                phase3_tool = lt.get('total_tool_time', 0.0)
                phase3_other = lt.get('total_other_time', 0.0)
                phase3_tools = phase3_api + phase3_tool + phase3_other

            phase4_validation = 0.0
            if 'validation_start' in t and 'validation_end' in t:
                phase4_validation = t['validation_end'] - t['validation_start']
            phase4_exec = 0.0
            if 'execution_start' in t and 'execution_end' in t:
                phase4_exec = t['execution_end'] - t['execution_start']
            phase4_total = phase4_validation + phase4_exec

            # Totals
            total_to_generation = 0.0
            total_with_execution = 0.0
            if 'turn_start' in t:
                if 'generation_complete' in t:
                    total_to_generation = t['generation_complete'] - t['turn_start']
                if 'execution_end' in t:
                    total_with_execution = t['execution_end'] - t['turn_start']

            # Unmeasured overhead = remainder not captured by the explicit phases above
            measured_sum = phase1_scene + phase2_retrieval + phase3_tools + phase4_total
            unmeasured = max(0.0, total_with_execution - measured_sum)

            # ---- Timeline Overview (numbers must add up) ----
            lines.append("=== Timeline Overview ===")
            lines.append(f"Total turn wall-clock (including execution): {total_with_execution:.3f}s")
            if total_to_generation > 0:
                lines.append(f"Total up to code generation: {total_to_generation:.3f}s")
            lines.append("")
            lines.append("Phase breakdown:")
            lines.append(f"  1. Scene context build: {phase1_scene:.3f}s")
            lines.append(f"  2. Pre-retrieval (query decomposition + vector search): {phase2_retrieval:.3f}s")
            lines.append(f"  3. Autonomous tool-calling & code generation: {phase3_tools:.3f}s")
            lines.append(f"  4. Code validation & execution: {phase4_total:.3f}s")
            if unmeasured > 0.001:
                lines.append(f"  5. Unmeasured overhead (prompt formatting, UI handoff, etc.): {unmeasured:.3f}s")
            lines.append("")
            lines.append("Verification: " + " + ".join([
                f"{phase1_scene:.3f}", f"{phase2_retrieval:.3f}", f"{phase3_tools:.3f}",
                f"{phase4_total:.3f}", f"{unmeasured:.3f}"
            ]) + f" = {measured_sum + unmeasured:.3f}s")
            lines.append("")

            if 'tokens' in t:
                lines.append(f"Main generation tokens: {t['tokens']}")
            if 'cost' in t:
                lines.append(f"Main generation cost: ${t['cost']:.4f}")
            lines.append("")

            # ---- Role Trace ----
            if t.get('role_trace'):
                lines.append("-" * 40)
                lines.append("Role-Composed Agent Trace")
                lines.append("-" * 40)
                for event in t.get('role_trace', []):
                    role = event.get('role', 'Unknown')
                    name = event.get('event', 'event')
                    details = event.get('details', {})
                    detail_text = json.dumps(details, ensure_ascii=False, default=str)
                    if len(detail_text) > 240:
                        detail_text = detail_text[:240] + "... "
                    lines.append(f"{role}: {name} | {detail_text}")
                lines.append("")

            # ---- Phase 1: Scene Context (detail) ----
            if 'context_build_time' in t:
                lines.append("-" * 40)
                lines.append("Phase 1 — Scene Context Build")
                lines.append("-" * 40)
                lines.append(f"Scene context build: {t['context_build_time']:.3f}s")
                lines.append("")

            # ---- Phase 2: Pre-Retrieval (detail) ----
            if rt:
                lines.append("-" * 40)
                lines.append("Phase 2 — Dense Pre-Retrieval (Decompose + Vector Search)")
                lines.append("-" * 40)
                if 'decompose_time' in rt:
                    thinking_flag = "ON" if rt.get('decompose_thinking') else "OFF"
                    lines.append(f"Query decomposition: {rt['decompose_time']:.3f}s (thinking={thinking_flag})")
                if 'sub_queries' in rt:
                    for i, sq in enumerate(rt['sub_queries'], 1):
                        lines.append(f"  Sub-query {i}: {sq}")
                if 'retrieval_count' in rt:
                    lines.append(f"Retrieval calls: {rt['retrieval_count']}")
                if 'concatenated_count' in rt:
                    lines.append(f"Total chunks in context: {rt['concatenated_count']}")
                if 'retrieval_per_query' in rt:
                    lines.append(f"Vector search total: {phase2_faiss:.3f}s")
                    for pq in rt['retrieval_per_query']:
                        lines.append(f"  - {pq.get('query', '')}: {pq.get('count', 0)} chunks in {pq.get('time', 0):.3f}s")
                lines.append("")

            # ---- Phase 3: Tool-Calling Loop (detail) ----
            if lt:
                lines.append("-" * 40)
                lines.append("Phase 3 — Autonomous Tool-Calling Loop")
                lines.append("-" * 40)
                rounds = lt.get('rounds', [])
                grep_count = sum(1 for r in rounds if 'Grep' in r.get('tools', []))
                readfile_count = sum(1 for r in rounds if 'ReadFile' in r.get('tools', []))
                vectorsearch_count = sum(1 for r in rounds if 'VectorSearch' in r.get('tools', []))
                lines.append(f"API calls: {lt.get('api_calls', 0)}")
                lines.append(f"Tool rounds: {lt.get('tool_rounds', 0)}")
                lines.append(f"Grep calls: {grep_count}")
                lines.append(f"ReadFile calls: {readfile_count}")
                lines.append(f"VectorSearch calls: {vectorsearch_count}")
                lines.append("")
                lines.append(f"Time inside this phase:")
                lines.append(f"  LLM API wait time: {phase3_api:.3f}s")
                lines.append(f"  Tool execution time: {phase3_tool:.3f}s")
                lines.append(f"  Overhead (JSON parse, prompt rebuild, etc.): {phase3_other:.3f}s")
                lines.append("")
                if rounds:
                    lines.append("Per-round breakdown:")
                    for r in rounds:
                        tools = ', '.join(r.get('tools', [])) or 'done'
                        tok = r.get('tokens', 0)
                        tok_str = f" tokens={tok}" if tok else ""
                        thinking_flag = "ON" if r.get('thinking') else "OFF"
                        lines.append(
                            f"  Round {r['round']} | "
                            f"api={r['api_time']:.3f}s tool={r.get('tool_time', 0):.3f}s "
                            f"other={r.get('other_time', 0):.3f}s total={r['round_time']:.3f}s | "
                            f"thinking={thinking_flag} | tools=[{tools}]{tok_str}"
                        )
                    lines.append("")

            # ---- Phase 4: Execution (detail) ----
            has_exec = 'execution_start' in t or 'autoexecute_start' in t
            if has_exec:
                lines.append("-" * 40)
                lines.append("Phase 4 — Code Validation & Execution")
                lines.append("-" * 40)
                if 'validation_start' in t and 'validation_end' in t:
                    v_t = t['validation_end'] - t['validation_start']
                    lines.append(f"Syntax validation: {v_t:.3f}s")
                if 'execution_async_call' in t and 'autoexecute_start' in t:
                    async_t = t['execution_async_call'] - t['autoexecute_start']
                    lines.append(f"Pre-execution overhead: {async_t:.3f}s")
                if 'executor_scheduled' in t and 'execution_async_call' in t:
                    sched_t = t['executor_scheduled'] - t['execution_async_call']
                    lines.append(f"Executor scheduling delay: {sched_t:.3f}s")
                if 'executor_actual_start' in t and 'executor_scheduled' in t:
                    actual_delay = t['executor_actual_start'] - t['executor_scheduled']
                    lines.append(f"Qt event-loop delay (singleShot→run): {actual_delay:.3f}s")
                if 'execution_start' in t and 'executor_actual_start' in t:
                    exec_startup = t['execution_start'] - t['executor_actual_start']
                    lines.append(f"Executor startup overhead: {exec_startup:.3f}s")
                if 'execution_callback_start' in t and 'execution_end' in t:
                    cb_t = t['execution_callback_start'] - t['execution_end']
                    lines.append(f"Callback dispatch delay: {cb_t:.3f}s")
                if 'execution_start' in t:
                    if 'execution_end' in t:
                        exec_t = t['execution_end'] - t['execution_start']
                        lines.append(f"Code execution (exec() only): {exec_t:.3f}s (result: {t.get('execution_result', 'unknown')})")
                    else:
                        lines.append("Code execution: started but not finished yet")
                lines.append("")

            # ---- Self-corrections ----
            lines.append("-" * 40)
            lines.append("Self-Correction")
            lines.append("-" * 40)
            if 'corrections' in t:
                lines.append(f"Attempts: {len(t['corrections'])}")
                for corr in t['corrections']:
                    lines.append(f"  Attempt {corr['attempt']}: start={corr['start']:.3f}s")
                    if 'tokens' in corr:
                        lines.append(f"    Tokens: {corr['tokens']}")
                    if 'cost' in corr:
                        lines.append(f"    Cost: ${corr['cost']:.4f}")
                    if 'timing_report' in corr:
                        ct = corr['timing_report']
                        lines.append(f"    API calls: {ct.get('api_calls', 0)}")
                        lines.append(f"    Total API time: {ct.get('total_api_time', 0):.3f}s")
                        lines.append(f"    Total tool time: {ct.get('total_tool_time', 0):.3f}s")
                        lines.append(f"    Tool rounds: {ct.get('tool_rounds', 0)}")
                        rounds = ct.get('rounds', [])
                        if rounds:
                            lines.append(f"    Rounds: {len(rounds)}")
            else:
                lines.append("Attempts: 0")
            lines.append("")

            # ---- Token & Cost summary ----
            lines.append("-" * 40)
            lines.append("Token & Cost Summary")
            lines.append("-" * 40)
            total_tokens = t.get('tokens', 0)
            total_cost = t.get('cost', 0.0)
            if 'corrections' in t:
                for corr in t['corrections']:
                    total_tokens += corr.get('tokens', 0)
                    total_cost += corr.get('cost', 0.0)
            lines.append(f"TOTAL TOKENS (main + corrections): {total_tokens}")
            lines.append(f"TOTAL COST (main + corrections): ${total_cost:.4f}")
            lines.append("")
            lines.append("="*50)

            with open(logPath, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to write timing report: {e}")

    # Note: onCopyButtonClicked removed - copy functionality not needed with auto-execution

    def onClearChatButtonClicked(self):
        self.chatHistory.clear()
        self._chatEntriesHtml = []
        if self.logic:
            self.logic.clearConversation()
        self.codeDisplay.clear()
        self.currentCode = None
        self.currentAgentPlan = None

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
        if provider == "OpenAI":
            return [
                "gpt-5.5",
                "gpt-5.5-mini",
                "gpt-5.5-turbo",
                "gpt-4.5",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "o3",
                "o3-mini",
                "o1",
                "o1-mini",
            ]
        if provider == "DeepSeek":
            return [
                "deepseek-v4-pro",
                "deepseek-v4-flash",
            ]
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
        if provider == "Qwen":
            return [
                # Qwen3.6 series (use snapshot IDs that the API recognizes)
                "qwen3.6-plus-2026-04-02",
                "qwen3.6-flash-2026-04-16",
                "qwen3.6-flash",
                "qwen3.6-35b-a3b",
                # Qwen3.5 series
                "qwen3.5-plus",
                "qwen3.5-flash",
                "qwen3.5-flash-2026-02-23",
                "qwen3.5-122b-a10b",
                "qwen3.5-397b-a17b",
                "qwen3.5-35b-a3b",
                "qwen3.5-27b",
                # Qwen3 series
                "qwen3-max-2026-01-23",
                "qwen3-max-preview",
                "qwen3-coder-plus",
                "qwen3-coder-flash",
                # Legacy aliases with latest routing
                "qwen-plus",
                "qwen-plus-latest",
                "qwen-turbo",
                "qwen-turbo-latest",
                "qwen-flash",
                "qwen-flash-latest",
                "qwen-long-latest",
                # QwQ reasoning models
                "qwq-plus",
                "qwq-32b",
            ]
        return ["kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview", "kimi-k2-0905-preview", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]

    def _defaultBaseUrlForProvider(self, provider: str) -> str:
        if provider == "OpenAI":
            return "https://api.openai.com/v1"
        if provider == "DeepSeek":
            return "https://api.deepseek.com"
        if provider == "Claude":
            return "https://api.anthropic.com/v1"
        if provider == "Qwen":
            return "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
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
        self._lastInteractiveStep = None
        self._initializeComponents()

    def _initializeComponents(self):
        try:
            from SlicerAIAgentLib import LLMClient, CodeValidator, SafeExecutor, ConversationStore, SkillTools, SceneTools

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

            # Register extension source directories as searchable roots
            from SlicerAIAgentLib.ExtensionCLILoader import get_validated_extensions
            for ext_name, ext_data in get_validated_extensions().items():
                source_path = ext_data["manifest"].get("source_path", "")
                if source_path and os.path.isdir(source_path):
                    self.toolExecutor.extra_roots[ext_name] = source_path

            self.skillTools = SkillTools.get_skill_tools() + SceneTools.get_scene_tools()
            self.codeValidator = CodeValidator()
            self.executor = SafeExecutor()

            # Vector index background check
            self._indexBuilder = None
            self._index_status = "Unknown"
            self._start_index_background_check()

            # Background pre-load: import transformers and create ONNX session
            # so the first prompt doesn't block on the 4-minute import
            self._warmup_thread = self._start_vector_warmup()

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

    def _getVectorRetriever(self):
        """Get the cached vector retriever if available."""
        if self.toolExecutor and self.toolExecutor.has_vector_index():
            return self.toolExecutor._vector_retriever
        if self._indexBuilder and self._indexBuilder.index_exists():
            return self._indexBuilder.load_retriever()
        return None

    def _buildRetrievalContext(self, prompt: str, timing: Optional[Dict] = None) -> str:
        """
        Perform dense vector pre-retrieval with query decomposition.
        Breaks the request into sub-task queries, runs a separate semantic
        search for each, then merges and formats the results.
        """
        import time
        t_total0 = time.time()
        try:
            t_get0 = time.time()
            retriever = self._getVectorRetriever()
            t_get = time.time() - t_get0
            if not retriever or not retriever.is_ready():
                return ""

            import time
            # Step 1: Decompose into sub-task queries
            t0 = time.time()
            sub_queries = self.llmClient.decomposeQuery(prompt)
            t1 = time.time()

            # Step 2: Multi-retrieval using sub-queries (top-10 per sub-query)
            # Run searches sequentially so each query is fully independent.
            all_results = []
            per_query = []
            for sq in sub_queries:
                q_start = time.time()
                results = retriever.search(sq, top_k=10)
                q_end = time.time()
                all_results.append(results)
                per_query.append({
                    'query': sq,
                    'count': len(results),
                    'time': round(q_end - q_start, 3)
                })

            # Step 3: Concatenate all per-query results and format
            t_merge0 = time.time()
            concatenated: List[Any] = []
            for res in all_results:
                concatenated.extend(res)

            # ---- Full-file inclusion for heavily-referenced .md files ----
            # Count chunks per file path
            from collections import Counter
            file_counts = Counter(rc.chunk.file_path for rc in concatenated)
            # Identify .md files with >= 3 chunks
            full_md_files = [
                fp for fp, cnt in file_counts.items()
                if cnt >= 3 and fp.lower().endswith('.md')
            ]
            if full_md_files and self.toolExecutor:
                from SlicerAIAgentLib.SkillIndexer import CodeChunk, RetrievedChunk
                skill_root = self.toolExecutor.skill_path
                for fp in full_md_files:
                    abs_path = os.path.join(skill_root, fp)
                    if not os.path.exists(abs_path):
                        continue
                    try:
                        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                            full_content = f.read()
                        total_lines = full_content.count('\n') + 1
                        # Determine dominant source_type from existing chunks
                        source_types = [rc.chunk.source_type for rc in concatenated if rc.chunk.file_path == fp]
                        dominant_source = Counter(source_types).most_common(1)[0][0] if source_types else 'doc_example'
                        # Remove all individual chunks from this file
                        concatenated = [rc for rc in concatenated if rc.chunk.file_path != fp]
                        # Create synthetic whole-file chunk
                        synthetic_chunk = CodeChunk(
                            chunk_id=f"{fp}#full",
                            file_path=fp,
                            start_line=1,
                            end_line=total_lines,
                            content=full_content,
                            embedding_text="",
                            chunk_type="whole_file",
                            source_type=dominant_source,
                            language="markdown",
                        )
                        synthetic_rc = RetrievedChunk(
                            chunk=synthetic_chunk,
                            vector_score=1.0,
                            final_score=1.5,
                        )
                        concatenated.append(synthetic_rc)
                        retriever.full_file_paths.add(fp)
                    except Exception as e:
                        logger.warning(f"Failed to include full file {fp}: {e}")
                # Re-sort by final_score descending so full files appear near top
                concatenated.sort(key=lambda rc: rc.final_score, reverse=True)

            t_merge = time.time() - t_merge0
            formatted = retriever.format_for_prompt(concatenated)
            # Prepend the sub-queries so the LLM knows what topics were already searched
            if sub_queries:
                search_note = (
                    "## Pre-retrieval search coverage\n"
                    "The following topics were already searched automatically. "
                    "You do NOT need to call VectorSearch for these same topics again. "
                    "Only use tools for gaps not covered below.\n"
                    + '\n'.join(f"- {sq}" for sq in sub_queries)
                    + "\n\n"
                )
                formatted = search_note + formatted
            total_t = time.time() - t_total0

            if timing is not None:
                timing['decompose_time'] = round(t1 - t0, 3)
                timing['decompose_thinking'] = False
                timing['sub_queries'] = sub_queries
                timing['retrieval_count'] = len(sub_queries)
                timing['retrieval_per_query'] = per_query
                timing['concatenated_count'] = len(concatenated)

            return formatted
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"Dense pre-retrieval failed: {e}")
            return ""

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
        retrieval_timing = {}
        retrieval = self._buildRetrievalContext(prompt, retrieval_timing)
        if retrieval:
            context["retrieval_results"] = retrieval
        response = self.llmClient.chat(prompt, context=context)
        response['retrieval_timing'] = retrieval_timing
        self.conversationStore.addExchange(prompt, response)
        return response

    def generateResponseStream(self, prompt, context=None, on_delta=None, use_tools=True, on_status=None, on_reasoning=None, on_reasoning_delta=None):
        """
        Generate AI response using streaming with optional tool calling.

        This runs the actual HTTP request (blocking I/O).  Callers should
        invoke this from a background thread.

        Args:
            prompt: User's natural language request
            context: Pre-built skill context (or None to build here)
            on_delta: Callback for incremental updates
            use_tools: Whether to use tool calling for skill search
            on_status: Callback for status updates (str) — 'Retrieving...', 'Thinking...', etc.

        Returns:
            dict with keys: message, reasoning_content, code, tokens, cost
        """
        if not self.llmClient:
            raise RuntimeError("LLM client not initialized")
        if not self.apiKey:
            raise RuntimeError("API key not configured")

        if context is None:
            context = {"scene": self._buildSceneContext()}

        # Inject dense vector retrieval results if not already present
        retrieval_timing = {}
        if "retrieval_results" not in context:
            if on_status:
                on_status("Retrieving...")
            retrieval = self._buildRetrievalContext(prompt, retrieval_timing)
            if retrieval:
                context["retrieval_results"] = retrieval
        retrieval_timing["used_tool_loop"] = bool(use_tools and self.toolExecutor and self.skillTools)

        if use_tools and self.toolExecutor and self.skillTools:
            # Use tool calling for skill search
            try:
                if on_status:
                    on_status("Thinking...")
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
                    on_status=on_status,
                    on_reasoning=on_reasoning,
                    on_reasoning_delta=on_reasoning_delta,
                )
                
                # Tool calling returns complete response (no streaming during tool rounds).
                # Final code generation is non-streaming; the UI displays the complete result.
            except Exception as e:
                logger.warning(f"Tool calling failed, falling back to regular chat: {e}")
                response = self.llmClient.chatStream(prompt, context=context, on_delta=on_delta, on_status=on_status)
        else:
            # Fallback to regular streaming
            response = self.llmClient.chatStream(prompt, context=context, on_delta=on_delta, on_status=on_status)
        
        response['retrieval_timing'] = retrieval_timing
        self.conversationStore.addExchange(prompt, response)
        return response

    def addExecutionFeedback(self, feedback_text):
        """
        Append code execution feedback to the LLM conversation history.
        Only keeps the most recent 2 feedback messages to prevent context bloat.
        """
        if not self.llmClient:
            return
        
        history = self.llmClient.conversation_history
        MAX_FEEDBACK = 2
        
        # Find indices of existing execution feedback messages
        feedback_indices = [
            i for i, msg in enumerate(history)
            if msg.get('role') == 'system'
            and msg.get('content', '').startswith('Code execution result:')
        ]
        
        # Remove oldest ones, keeping at most (MAX_FEEDBACK - 1) recent ones
        # We delete from back to front so indices don't shift underneath us
        to_remove = feedback_indices[:-(MAX_FEEDBACK - 1)] if len(feedback_indices) >= MAX_FEEDBACK else []
        for idx in reversed(to_remove):
            history.pop(idx)
        
        history.append({
            "role": "system",
            "content": feedback_text,
        })
    
    def _executeTool(self, tool_name, tool_args):
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool (SearchSymbol, Grep, ReadFile, VectorSearch, GetNodeProperties)
            tool_args: Tool arguments dict
            
        Returns:
            Tool execution result dict
        """
        # Scene introspection tools (do not require skill executor)
        if tool_name == "GetNodeProperties":
            try:
                from SlicerAIAgentLib import SceneTools
                ids = tool_args.get("ids", [])
                if isinstance(ids, str):
                    ids = [ids]
                if not ids:
                    return {
                        "error": (
                            "GetNodeProperties was called with an empty ids list. "
                            "Do NOT call this tool unless you have specific node IDs from the scene summary. "
                            "If the scene summary is missing, proceed with the information you have."
                        )
                    }
                results = {}
                for node_id in ids:
                    results[node_id] = SceneTools.getNodeProperties(node_id)
                return {
                    "tool": "GetNodeProperties",
                    "results": results,
                    "count": len(results),
                }
            except Exception as e:
                logger.error(f"GetNodeProperties failed: {e}")
                return {"error": str(e)}

        if not self.toolExecutor:
            return {"error": "Tool executor not initialized"}
        
        try:
            result = self.toolExecutor.execute(tool_name, tool_args)
            # Log tool execution for debugging
            logger.info(f"Tool {tool_name} executed: {tool_args.get('pattern', tool_args.get('path', tool_args.get('ids', 'N/A')))}")

            # Track interactive workflow dispatch results for the Widget
            if isinstance(result, dict) and result.get("type") == "interactive":
                self._lastInteractiveStep = result
            # Track all workflow step dispatches (including automated) for auto-advance
            if isinstance(result, dict) and result.get("step_id") and result.get("tool"):
                self._lastWorkflowStep = result

            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {"error": str(e)}

    def _buildSceneContext(self):
        """
        Build structured context about the current Slicer MRML scene.

        Returns:
            Dictionary with a structured scene summary, or None.
        """
        try:
            from SlicerAIAgentLib import SceneTools
            summary = SceneTools.buildSceneSummary(max_nodes=50)
            if summary.get("scene_summary") is not None:
                return summary
        except Exception as e:
            logger.warning(f"Failed to build scene context: {e}")

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

    def buildSceneSnapshot(self):
        """Capture lightweight scene state for semantic verification."""
        snapshot = {
            "counts": {},
            "nodes": [],
            "segmentations": [],
            "models": [],
            "transforms": [],
            "volumes": [],
            "visible_nodes": 0,
            "layout": None,
            "active_module": None,
            "active_volume_id": None,
            "active_label_volume_id": None,
            "active_place_node_id": None,
        }
        try:
            scene = slicer.mrmlScene
            try:
                layout_node = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
                if layout_node:
                    snapshot["layout"] = layout_node.GetViewArrangement()
            except Exception:
                pass
            try:
                module_manager = slicer.app.moduleManager()
                active_module = module_manager.activeModule() if module_manager else None
                snapshot["active_module"] = active_module.name if active_module else None
            except Exception:
                pass
            try:
                selection_node = slicer.app.applicationLogic().GetSelectionNode()
                if selection_node:
                    snapshot["active_volume_id"] = selection_node.GetActiveVolumeID()
                    snapshot["active_label_volume_id"] = selection_node.GetActiveLabelVolumeID()
                    snapshot["active_place_node_id"] = selection_node.GetActivePlaceNodeID()
            except Exception:
                pass
            for i in range(scene.GetNumberOfNodes()):
                node = scene.GetNthNode(i)
                if not node:
                    continue
                class_name = node.GetClassName()
                snapshot["counts"][class_name] = snapshot["counts"].get(class_name, 0) + 1

                display_node = node.GetDisplayNode() if hasattr(node, "GetDisplayNode") else None
                node_info = {
                    "id": node.GetID(),
                    "name": node.GetName(),
                    "class": class_name,
                    "mtime": node.GetMTime() if hasattr(node, "GetMTime") else None,
                    "has_display": bool(display_node),
                    "visible": bool(display_node and display_node.GetVisibility()),
                }
                snapshot["nodes"].append(node_info)
                if display_node and display_node.GetVisibility():
                    snapshot["visible_nodes"] += 1

                if class_name == "vtkMRMLSegmentationNode":
                    segmentation_info = {
                        "id": node.GetID(),
                        "name": node.GetName(),
                        "segments": 0,
                        "has_closed_surface": False,
                        "total_voxels": 0,
                    }
                    try:
                        segmentation = node.GetSegmentation()
                        segmentation_info["segments"] = segmentation.GetNumberOfSegments()
                        if hasattr(segmentation, "ContainsRepresentation"):
                            segmentation_info["has_closed_surface"] = bool(segmentation.ContainsRepresentation("Closed surface"))
                        # Count total voxels across all segments
                        import vtkSegmentationCorePython as vtkSegmentationCore
                        for seg_idx in range(segmentation.GetNumberOfSegments()):
                            segment = segmentation.GetNthSegment(seg_idx)
                            if segment:
                                labelmap = segment.GetRepresentation("Binary labelmap")
                                if labelmap:
                                    segmentation_info["total_voxels"] += labelmap.GetNumberOfPoints()
                    except Exception:
                        pass
                    snapshot["segmentations"].append(segmentation_info)

                if class_name == "vtkMRMLModelNode":
                    model_info = {
                        "id": node.GetID(),
                        "name": node.GetName(),
                        "points": 0,
                        "cells": 0,
                        "has_polydata": False,
                    }
                    try:
                        polydata = node.GetPolyData()
                        if polydata:
                            model_info["points"] = polydata.GetNumberOfPoints()
                            model_info["cells"] = polydata.GetNumberOfCells()
                            model_info["has_polydata"] = True
                    except Exception:
                        pass
                    snapshot["models"].append(model_info)

                if class_name == "vtkMRMLLinearTransformNode":
                    transform_info = {
                        "id": node.GetID(),
                        "name": node.GetName(),
                        "is_identity": True,
                    }
                    try:
                        matrix = vtk.vtkMatrix4x4()
                        node.GetMatrixTransformToParent(matrix)
                        transform_info["is_identity"] = matrix.IsIdentity()
                    except Exception:
                        pass
                    snapshot["transforms"].append(transform_info)

                if class_name == "vtkMRMLScalarVolumeNode":
                    volume_info = {
                        "id": node.GetID(),
                        "name": node.GetName(),
                        "dimensions": [0, 0, 0],
                        "voxel_count": 0,
                    }
                    try:
                        image_data = node.GetImageData()
                        if image_data:
                            dims = image_data.GetDimensions()
                            volume_info["dimensions"] = list(dims)
                            volume_info["voxel_count"] = dims[0] * dims[1] * dims[2]
                    except Exception:
                        pass
                    snapshot["volumes"].append(volume_info)
        except Exception as e:
            logger.warning(f"Failed to build scene snapshot: {e}")
        return snapshot

    def getSceneCheckRegistry(self):
        """Return supported deterministic checks for optional plan verification."""
        return {
            "node_count_delta": self._checkNodeCountDelta,
            "node_exists": self._checkNodeExists,
            "node_modified": self._checkNodeModified,
            "node_has_display": self._checkNodeHasDisplay,
            "node_name_matches": self._checkNodeNameMatches,
            "node_has_content": self._checkNodeHasContent,
            "layout_changed": self._checkLayoutChanged,
            "selection_changed": self._checkSelectionChanged,
            "module_entered": self._checkModuleEntered,
            "property_true": self._checkPropertyTrue,
            "not_checked": self._checkNotChecked,
        }

    def verifySceneAgainstPlan(self, before, after, plan):
        """Compare optional machine-checkable expectations against scene snapshots."""
        result = {"valid": True, "errors": [], "warnings": [], "diagnostics": []}
        if not isinstance(plan, dict):
            return result

        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            return result

        registry = self.getSceneCheckRegistry()

        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            expected = step.get("expected_scene_change")
            if not isinstance(expected, dict):
                continue

            change_type = str(expected.get("type", "")).lower()
            if not change_type:
                change_type = self._inferLegacySceneCheckType(expected)
            check = registry.get(change_type)
            if not check:
                result["warnings"].append(
                    f"Step {idx} has unsupported expected_scene_change type '{change_type}' and was not checked"
                )
                continue
            check_result = check(before, after, expected)
            for warning in check_result.get("warnings", []):
                result["warnings"].append(f"Step {idx} {warning}")
            for error in check_result.get("errors", []):
                result["errors"].append(f"Step {idx} {error}")
            diag = check_result.get("_diagnostic")
            if diag:
                result["diagnostics"].append({"step": idx, **diag})

        if result["errors"]:
            result["valid"] = False
        return result

    def _inferLegacySceneCheckType(self, expected):
        if expected.get("node_class") or expected.get("class"):
            return "node_count_delta"
        if expected.get("property"):
            return "property_true"
        return ""

    def _matchSnapshotNodes(self, snapshot, node_class=None, name_contains=None):
        nodes = snapshot.get("nodes", []) if isinstance(snapshot, dict) else []
        name_filter = str(name_contains or "").lower()
        matches = []
        for node in nodes:
            if node_class and node.get("class") != node_class:
                continue
            if name_filter and name_filter not in str(node.get("name") or "").lower():
                continue
            matches.append(node)
        return matches

    def _collectSnapshotCandidates(self, snapshot, node_class=None, name_contains=None):
        """Collect near-miss node candidates when an exact match fails.

        Returns up to 5 candidate dicts with keys: id, name, class, score, match_notes.
        """
        nodes = snapshot.get("nodes", []) if isinstance(snapshot, dict) else []
        name_filter = str(name_contains or "").lower()
        class_filter = str(node_class or "").lower()
        candidates = []
        seen_ids = set()
        for node in nodes:
            nid = node.get("id")
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            actual_class = str(node.get("class") or "").lower()
            actual_name = str(node.get("name") or "").lower()
            score = 0
            notes = []
            if class_filter:
                if actual_class == class_filter:
                    score += 3
                    notes.append("class_exact")
                elif class_filter in actual_class or actual_class in class_filter:
                    score += 1
                    notes.append("class_partial")
            if name_filter:
                if actual_name == name_filter:
                    score += 3
                    notes.append("name_exact")
                elif name_filter in actual_name or actual_name in name_filter:
                    score += 1
                    notes.append("name_partial")
            if score > 0:
                candidates.append({
                    "id": nid,
                    "name": node.get("name"),
                    "class": node.get("class"),
                    "score": score,
                    "match_notes": notes,
                })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:5]

    def _checkNotChecked(self, before, after, expected):
        return {"errors": [], "warnings": []}

    def _checkNodeCountDelta(self, before, after, expected):
        before_counts = before.get("counts", {}) if isinstance(before, dict) else {}
        after_counts = after.get("counts", {}) if isinstance(after, dict) else {}
        node_class = expected.get("node_class") or expected.get("class")
        count_delta = expected.get("min_delta", expected.get("count_delta"))
        if not node_class or not isinstance(count_delta, int) or count_delta <= 0:
            return {"errors": [], "warnings": ["node_count_delta is missing node_class or positive min_delta"]}
        observed_delta = after_counts.get(node_class, 0) - before_counts.get(node_class, 0)
        if observed_delta < count_delta:
            return {"errors": [f"expected {node_class} count_delta >= {count_delta}, observed {observed_delta}"], "warnings": []}
        # If content assertions are specified, also verify the new nodes have actual data
        content_errors = self._checkContentAssertions(after, node_class, expected)
        if content_errors:
            return {"errors": content_errors, "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkNodeExists(self, before, after, expected):
        node_class = expected.get("node_class") or expected.get("class")
        matches = self._matchSnapshotNodes(after, node_class, expected.get("name_contains"))
        if not matches:
            label = node_class or "node"
            name = expected.get("name_contains")
            suffix = f" with name containing '{name}'" if name else ""
            error_msg = f"expected {label}{suffix} to exist"
            candidates = self._collectSnapshotCandidates(after, node_class, expected.get("name_contains"))
            return {
                "errors": [error_msg],
                "warnings": [],
                "_diagnostic": {
                    "check_type": "node_exists",
                    "expected": {"node_class": node_class, "name_contains": name},
                    "actual_candidates": candidates,
                }
            }
        # If content assertions are specified, also verify matching nodes have actual data
        content_errors = self._checkContentAssertions(after, node_class, expected)
        if content_errors:
            return {"errors": content_errors, "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkNodeModified(self, before, after, expected):
        node_class = expected.get("node_class") or expected.get("class")
        before_nodes = {
            node.get("id"): node
            for node in self._matchSnapshotNodes(before, node_class, expected.get("name_contains"))
            if node.get("id")
        }
        after_nodes = [
            node for node in self._matchSnapshotNodes(after, node_class, expected.get("name_contains"))
            if node.get("id")
        ]
        for node in after_nodes:
            before_node = before_nodes.get(node.get("id"))
            if before_node and node.get("mtime") and before_node.get("mtime") and node.get("mtime") > before_node.get("mtime"):
                return {"errors": [], "warnings": []}
        return {"errors": [f"expected an existing {node_class or 'node'} to be modified"], "warnings": []}

    def _checkNodeHasDisplay(self, before, after, expected):
        node_class = expected.get("node_class") or expected.get("class")
        matches = self._matchSnapshotNodes(after, node_class, expected.get("name_contains"))
        if not any(node.get("has_display") for node in matches):
            return {"errors": [f"expected {node_class or 'node'} to have a display node"], "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkContentAssertions(self, snapshot, node_class, expected):
        """Verify that matching nodes in the snapshot have actual content (points, voxels, etc.).
        
        Supported assertions:
        - min_points (for vtkMRMLModelNode)
        - min_cells (for vtkMRMLModelNode)
        - min_voxels (for vtkMRMLSegmentationNode)
        - min_segments (for vtkMRMLSegmentationNode)
        - min_voxel_count (for vtkMRMLScalarVolumeNode)
        - is_non_identity (for vtkMRMLLinearTransformNode)
        
        Returns a list of error strings (empty if all pass).
        """
        errors = []
        assertions = expected.get("content_assertions", {})
        if not assertions:
            # Also check legacy top-level keys for backward compat
            if expected.get("min_points") is not None:
                assertions["min_points"] = expected["min_points"]
            if expected.get("min_cells") is not None:
                assertions["min_cells"] = expected["min_cells"]
            if expected.get("min_voxels") is not None:
                assertions["min_voxels"] = expected["min_voxels"]
            if expected.get("min_segments") is not None:
                assertions["min_segments"] = expected["min_segments"]
            if expected.get("is_non_identity") is not None:
                assertions["is_non_identity"] = expected["is_non_identity"]
            if not assertions:
                return errors

        # Match nodes by class and optional name filter
        name_filter = expected.get("name_contains")
        if node_class == "vtkMRMLModelNode":
            models = snapshot.get("models", [])
            if name_filter:
                models = [m for m in models if name_filter.lower() in m.get("name", "").lower()]
            if assertions.get("min_points") is not None:
                min_p = assertions["min_points"]
                if not any(m.get("points", 0) >= min_p for m in models):
                    actual = ", ".join(f"{m.get('name', '?')}:{m.get('points', 0)}pts" for m in models) if models else "none"
                    errors.append(f"expected model with >= {min_p} points, found: {actual}")
            if assertions.get("min_cells") is not None:
                min_c = assertions["min_cells"]
                if not any(m.get("cells", 0) >= min_c for m in models):
                    actual = ", ".join(f"{m.get('name', '?')}:{m.get('cells', 0)}cells" for m in models) if models else "none"
                    errors.append(f"expected model with >= {min_c} cells, found: {actual}")

        elif node_class == "vtkMRMLSegmentationNode":
            segmentations = snapshot.get("segmentations", [])
            if name_filter:
                segmentations = [s for s in segmentations if name_filter.lower() in s.get("name", "").lower()]
            if assertions.get("min_voxels") is not None:
                min_v = assertions["min_voxels"]
                if not any(s.get("total_voxels", 0) >= min_v for s in segmentations):
                    actual = ", ".join(f"{s.get('name', '?')}:{s.get('total_voxels', 0)}voxels" for s in segmentations) if segmentations else "none"
                    errors.append(f"expected segmentation with >= {min_v} voxels, found: {actual}")
            if assertions.get("min_segments") is not None:
                min_s = assertions["min_segments"]
                if not any(s.get("segments", 0) >= min_s for s in segmentations):
                    actual = ", ".join(f"{s.get('name', '?')}:{s.get('segments', 0)}segs" for s in segmentations) if segmentations else "none"
                    errors.append(f"expected segmentation with >= {min_s} segments, found: {actual}")

        elif node_class == "vtkMRMLScalarVolumeNode":
            volumes = snapshot.get("volumes", [])
            if name_filter:
                volumes = [v for v in volumes if name_filter.lower() in v.get("name", "").lower()]
            if assertions.get("min_voxel_count") is not None:
                min_vc = assertions["min_voxel_count"]
                if not any(v.get("voxel_count", 0) >= min_vc for v in volumes):
                    actual = ", ".join(f"{v.get('name', '?')}:{v.get('voxel_count', 0)}voxels" for v in volumes) if volumes else "none"
                    errors.append(f"expected volume with >= {min_vc} voxels, found: {actual}")

        elif node_class == "vtkMRMLLinearTransformNode":
            transforms = snapshot.get("transforms", [])
            if name_filter:
                transforms = [t for t in transforms if name_filter.lower() in t.get("name", "").lower()]
            if assertions.get("is_non_identity"):
                if not any(not t.get("is_identity", True) for t in transforms):
                    actual = ", ".join(f"{t.get('name', '?')}:identity={t.get('is_identity', True)}" for t in transforms) if transforms else "none"
                    errors.append(f"expected non-identity transform, found: {actual}")

        return errors

    def _checkNodeHasContent(self, before, after, expected):
        """Verify that matching nodes have actual content (points, voxels, etc.)."""
        node_class = expected.get("node_class") or expected.get("class")
        if not node_class:
            return {"errors": [], "warnings": ["node_has_content is missing node_class"]}
        content_errors = self._checkContentAssertions(after, node_class, expected)
        if content_errors:
            return {"errors": content_errors, "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkNodeNameMatches(self, before, after, expected):
        name_contains = expected.get("name_contains")
        if not name_contains:
            return {"errors": [], "warnings": ["node_name_matches is missing name_contains"]}
        return self._checkNodeExists(before, after, expected)

    def _checkLayoutChanged(self, before, after, expected):
        if before.get("layout") == after.get("layout"):
            return {"errors": [f"expected layout to change from {before.get('layout')}"], "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkSelectionChanged(self, before, after, expected):
        before_selection = (
            before.get("active_volume_id"),
            before.get("active_label_volume_id"),
            before.get("active_place_node_id"),
        )
        after_selection = (
            after.get("active_volume_id"),
            after.get("active_label_volume_id"),
            after.get("active_place_node_id"),
        )
        if before_selection == after_selection:
            return {"errors": [f"expected active selection to change from {before_selection}"], "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkModuleEntered(self, before, after, expected):
        module_name = expected.get("module") or expected.get("module_name")
        active_module = after.get("active_module")
        if module_name and str(module_name).lower() != str(active_module).lower():
            return {"errors": [f"expected active module '{module_name}', observed '{active_module}'"], "warnings": []}
        if not module_name and before.get("active_module") == active_module:
            return {"errors": [f"expected active module to change from '{active_module}'"], "warnings": []}
        return {"errors": [], "warnings": []}

    def _checkPropertyTrue(self, before, after, expected):
        prop = str(expected.get("property", "")).lower()
        expected_value = expected.get("expected", True)
        if expected_value is not True:
            return {"errors": [], "warnings": ["property_true only checks expected=true properties"]}
        if prop in ("segmentation_has_segments", "segmentation_contains_segment"):
            if not any(s.get("segments", 0) > 0 for s in after.get("segmentations", [])):
                return {"errors": ["expected a segmentation with at least one segment"], "warnings": []}
        elif prop in ("segmentation_has_closed_surface", "closed_surface"):
            if not any(s.get("has_closed_surface") for s in after.get("segmentations", [])):
                return {"errors": ["expected a closed surface segmentation representation"], "warnings": []}
        elif prop in ("segmentation_has_voxels", "segmentation_voxels"):
            if not any(s.get("total_voxels", 0) > 0 for s in after.get("segmentations", [])):
                return {"errors": ["expected a segmentation with at least one voxel"], "warnings": []}
        elif prop in ("model_has_polydata", "model_polydata"):
            if not any(m.get("points", 0) > 0 and m.get("cells", 0) > 0 for m in after.get("models", [])):
                return {"errors": ["expected a model node with valid polydata"], "warnings": []}
        elif prop in ("model_has_points", "model_points"):
            if not any(m.get("points", 0) > 0 for m in after.get("models", [])):
                return {"errors": ["expected a model node with at least one point"], "warnings": []}
        elif prop in ("transform_is_non_identity", "non_identity_transform"):
            if not any(not t.get("is_identity", True) for t in after.get("transforms", [])):
                return {"errors": ["expected a non-identity transform"], "warnings": []}
        elif prop in ("display_visibility", "visible"):
            if after.get("visible_nodes", 0) <= 0:
                return {"errors": [], "warnings": ["expected at least one visible display node"]}
        else:
            return {"errors": [], "warnings": [f"unsupported property_true property '{prop}' was not checked"]}
        return {"errors": [], "warnings": []}

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

    def _start_index_background_check(self):
        """
        Check whether a vector index exists on disk.
        Does NOT auto-build or auto-update — the user must run scripts/build_rag.py manually.
        """
        try:
            from SlicerAIAgentLib.SkillIndexer import IndexBuilder

            # Use the same root directory as SlicerAIAgent.py so that the index
            # path is deterministic regardless of how SkillIndexer.py resolves __file__.
            self._index_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Resources", "Code_RAG", "v1"
            )
            self._indexBuilder = IndexBuilder(self.skill_path, index_dir=self._index_dir)
            if self._indexBuilder.index_exists():
                self._index_status = "Ready"
                logger.info(f"Vector index found: {self._index_dir}")
            else:
                self._index_status = "Missing"
                logger.info(f"Vector index not found at {self._index_dir}. Run scripts/build_rag.py to create it.")
        except Exception as e:
            logger.warning(f"Could not check index status: {e}")
            self._index_status = "Error"

    def _start_vector_warmup(self):
        """
        Start a background thread that eagerly loads the tokenizer and ONNX session.
        This prevents the ~4-minute first-prompt delay caused by transformers import.
        """
        import threading
        def _warmup():
            try:
                if not self.toolExecutor:
                    return
                if not self.toolExecutor.has_vector_index():
                    return
                retriever = self.toolExecutor._vector_retriever
                if retriever is None:
                    return
                vector = retriever.vector
                if vector is None:
                    return
                # Trigger tokenizer loading + ONNX session creation
                import time
                t0 = time.time()
                vector._ensure_model_files()
                t1 = time.time()
                vector._load_model()
                t2 = time.time()
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Background vector warmup failed: {e}")
        thread = threading.Thread(target=_warmup, daemon=True)
        thread.start()
        return thread

    def get_index_status(self) -> str:
        """Return current vector index status for UI display."""
        return getattr(self, '_index_status', 'Unknown')

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
