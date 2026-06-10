from .common import *
from .logic import SlicerAIAgentLogic
from .widget_core import WidgetCoreMixin
from .widget_cli import WidgetCLIMixin
from .widget_workflow import WidgetWorkflowMixin
from .widget_streaming import WidgetStreamingMixin
from .widget_execution import WidgetExecutionMixin
from .widget_settings import WidgetSettingsMixin


class SlicerAIAgentWidget(
    WidgetCoreMixin,
    WidgetCLIMixin,
    WidgetWorkflowMixin,
    WidgetStreamingMixin,
    WidgetExecutionMixin,
    WidgetSettingsMixin,
    ScriptedLoadableModuleWidget,
    VTKObservationMixin,
):
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
        self._workflowRuntime = None
        self._activeWorkflowId = None
        self._waitingForUser = False
        self._workflowBannerLabel = None
        self._autoAdvanceWorkflowStep = None
        self._workflowInstructionsLabel = None
        self._nodeChoiceResolver = None
        self._workflowChoiceButtons = []
        self._workflowChoiceInput = None
        self._workflowChoiceSubmitButton = None
        self._currentWorkflowUiState = {"active": False}
        self._taskWorkflowPanelActive = False
        self._announcedWorkflowIds = set()
        self._currentWorkflowStepInfo = None
        self._lastInjectedPreludeKeys = []
