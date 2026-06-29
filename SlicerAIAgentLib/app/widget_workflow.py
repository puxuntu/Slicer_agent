from .common import *


class WidgetWorkflowMixin:
    def _setupWorkflowUI(self):
        """Set up UI components for guided interactive workflows."""
        from SlicerAIAgentLib.WorkflowOrchestrator import WorkflowOrchestrator
        from SlicerAIAgentLib.InteractionManager import InteractionManager
        from SlicerAIAgentLib.WorkflowRuntime import WorkflowRuntime

        self._interactionManager = InteractionManager()
        self._workflowOrchestrator = WorkflowOrchestrator(
            interaction_manager=self._interactionManager,
        )
        self._workflowRuntime = WorkflowRuntime()

        self._setupWorkflowUserPanel()

    def _setupWorkflowUserPanel(self):
        """Create or connect the user-facing workflow panel controls."""
        if not getattr(self, "_workflowUserFrame", None):
            self._workflowUserFrame = qt.QFrame()
            self._workflowUserFrame.setObjectName("workflowUserFrame")
            self._workflowUserFrame.setStyleSheet(
                "QFrame#workflowUserFrame { background-color: #f7fbff; "
                "border: 1px solid #b8d7f2; border-radius: 4px; }"
            )
            layout = qt.QVBoxLayout(self._workflowUserFrame)
            layout.setContentsMargins(10, 8, 10, 8)

            header = qt.QHBoxLayout()
            self._workflowTitleLabel = qt.QLabel("Workflow")
            self._workflowTitleLabel.setStyleSheet("font-weight: bold; font-size: 14px; color: #1f3b57;")
            self._workflowStatusLabel = qt.QLabel("Idle")
            self._workflowStatusLabel.setStyleSheet("font-weight: bold; color: #3b6f9e;")
            header.addWidget(self._workflowTitleLabel, 1)
            header.addWidget(self._workflowStatusLabel)
            layout.addLayout(header)

            self._workflowProgressBar = qt.QProgressBar()
            self._workflowProgressBar.setMinimum(0)
            self._workflowProgressBar.setMaximum(1)
            self._workflowProgressBar.setValue(0)
            layout.addWidget(self._workflowProgressBar)

            self._workflowStepLabel = qt.QLabel("Step 0 of 0")
            self._workflowActionLabel = qt.QLabel("")
            self._workflowActionLabel.setWordWrap(True)
            self._workflowActionLabel.setStyleSheet("font-weight: bold; color: #222;")
            self._workflowInstructionLabel = qt.QLabel("")
            self._workflowInstructionLabel.setWordWrap(True)
            layout.addWidget(self._workflowStepLabel)
            layout.addWidget(self._workflowActionLabel)
            layout.addWidget(self._workflowInstructionLabel)

            # "Show brief" toggle + the terse instruction body, hidden until the
            # toggle is clicked. The primary label shows the detailed (clinical)
            # instruction by default; this reveals the terse "what to do" version.
            self._workflowDetailToggle = qt.QToolButton()
            self._workflowDetailToggle.setText("Show brief ▸")
            self._workflowDetailToggle.setAutoRaise(True)
            self._workflowDetailToggle.setStyleSheet("color: #3b6f9e; border: none; padding: 0;")
            self._workflowDetailToggle.clicked.connect(self._onToggleWorkflowDetails)
            self._workflowDetailLabel = qt.QLabel("")
            self._workflowDetailLabel.setWordWrap(True)
            self._workflowDetailLabel.setStyleSheet("color: #444;")
            layout.addWidget(self._workflowDetailToggle)
            layout.addWidget(self._workflowDetailLabel)

            self._workflowChoiceContainer = qt.QWidget()
            self._workflowChoiceLayout = qt.QHBoxLayout(self._workflowChoiceContainer)
            self._workflowChoiceLayout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._workflowChoiceContainer)

            controls = qt.QHBoxLayout()
            self._workflowDoneButton = qt.QPushButton("Done")
            self._workflowSkipButton = qt.QPushButton("Skip")
            self._workflowCancelButton = qt.QPushButton("Cancel")
            controls.addWidget(self._workflowDoneButton)
            controls.addWidget(self._workflowSkipButton)
            controls.addWidget(self._workflowCancelButton)
            controls.addStretch(1)
            layout.addLayout(controls)
            self.layout.addWidget(self._workflowUserFrame)

        self._workflowBannerLabel = self._workflowTitleLabel
        self._workflowInstructionsLabel = self._workflowInstructionLabel
        self._workflowChoiceButtons = []
        self._workflowChoiceInput = None
        self._workflowChoiceSubmitButton = None
        self._workflowNodeTree = None
        self._workflowNodeTreeSelectButton = None
        self._workflowNodeTreeContainer = None
        self._workflowSegmentsTable = None
        self._workflowSegmentsCombo = None
        self._workflowSegmentsContainer = None

        if getattr(self, "_workflowDoneButton", None):
            self._workflowDoneButton.clicked.connect(self._onWorkflowDoneClicked)
        if getattr(self, "_workflowSkipButton", None):
            self._workflowSkipButton.clicked.connect(self._onWorkflowSkipClicked)
        if getattr(self, "_workflowCancelButton", None):
            self._workflowCancelButton.clicked.connect(self._onWorkflowCancelClicked)

        # Build the replay stepper unconditionally: the workflow frame can be
        # loaded from the .ui file (widget_core.py) instead of built here, in
        # which case the programmatic block above is skipped. This wraps the
        # progress bar with the Back / Forward / Run-from-here buttons.
        self._setupReplayControls()

        self._positionWorkflowUserPanel()
        self._clearWorkflowPanel()
        self._applyWidthSafeLabels()

    def _applyWidthSafeLabels(self):
        """Stop variable-length labels from forcing the module panel wider.

        A QLabel with word-wrap OFF reports its full text width as its minimum
        size hint, and Slicer's module panel grows (and then locks) to satisfy
        the widest child's minimum. The workflow header's title + status are the
        offenders: when the status becomes its longest value ("Waiting for your
        interaction" at the first interaction step, right after step 9) the panel
        jumps wider and can no longer be dragged narrower. Word-wrapping drops
        each label's minimum width to its widest single word, so the panel stops
        auto-widening and the width the user set sticks. Applied at runtime so it
        covers both the .ui-loaded and the programmatic-fallback widgets.
        """
        labels = [
            getattr(self, "_workflowTitleLabel", None),
            getattr(self, "_workflowStatusLabel", None),
            getattr(self, "_workflowStepLabel", None),
            getattr(self, "statusLabel", None),  # main agent status row
        ]
        for label in labels:
            if label is None:
                continue
            try:
                label.setWordWrap(True)
                label.setMinimumWidth(0)
            except Exception:
                logger.debug("width-safe label setup failed", exc_info=True)

    def _positionWorkflowUserPanel(self):
        """Place the workflow panel below Debug and above the prompt input area."""
        if not getattr(self, "_workflowUserFrame", None):
            return
        debug_group = self.ui.findChild(ctk.ctkCollapsibleGroupBox, "debugGroupBox") if getattr(self, "ui", None) else None
        if not debug_group:
            return
        parent = debug_group.parent()
        parent_layout = parent.layout() if parent else None
        if not parent_layout:
            return
        workflow_index = parent_layout.indexOf(self._workflowUserFrame)
        debug_index = parent_layout.indexOf(debug_group)
        if debug_index < 0:
            return
        if workflow_index >= 0:
            parent_layout.removeWidget(self._workflowUserFrame)
            if workflow_index < debug_index:
                debug_index -= 1
        parent_layout.insertWidget(debug_index + 1, self._workflowUserFrame)

    def _updateWorkflowPanel(self, result_or_state=None):
        """Render generated-CLI or traditional task state into the workflow panel."""
        state = result_or_state if isinstance(result_or_state, dict) else {}
        should_map_runtime_result = (
            not state
            or any(key in state for key in ("type", "step_id", "tool", "next_step"))
        )
        if self._workflowRuntime and self._workflowRuntime.session and should_map_runtime_result:
            state = self._workflowRuntime.state_for_ui(state)
        elif should_map_runtime_result and not state.get("active"):
            state = self._workflowUiStateFromStepResult(state)

        self._currentWorkflowUiState = dict(state or {"active": False})
        if not getattr(self, "_workflowUserFrame", None):
            return

        if not self._currentWorkflowUiState.get("active"):
            self._workflowUserFrame.setVisible(False)
            return

        self._workflowUserFrame.setVisible(True)
        title = (
            self._currentWorkflowUiState.get("workflow_title")
            or self._currentWorkflowUiState.get("extension_name")
            or "Task"
        )
        status = self._currentWorkflowUiState.get("status") or "Running"
        total = int(self._currentWorkflowUiState.get("total_steps") or 0)
        completed = int(self._currentWorkflowUiState.get("completed_steps") or 0)
        current_index = int(self._currentWorkflowUiState.get("current_index") or 0)

        self._workflowTitleLabel.setText(str(title))
        self._workflowStatusLabel.setText(str(status))
        if total > 0:
            self._workflowProgressBar.setRange(0, total)
            self._workflowProgressBar.setValue(max(0, min(completed, total)))
            self._workflowProgressBar.setFormat(f"{completed}/{total}")
            step_text = f"Step {current_index or completed} of {total}"
            repeat_progress = self._currentWorkflowUiState.get("repeat_progress") or {}
            repeat_total = int(repeat_progress.get("total") or 0)
            repeat_current = int(repeat_progress.get("current") or 0)
            if repeat_total > 0 and repeat_current > 0:
                object_label = self._currentWorkflowUiState.get("object_label") or "Item"
                step_text += f" - {str(object_label).title()} {repeat_current} of {repeat_total}"
            elif repeat_current > 0:
                step_text += f" - Repeat iteration {repeat_current}"
            self._workflowStepLabel.setText(step_text)
            self._workflowStepLabel.setVisible(True)
        else:
            self._workflowProgressBar.setRange(0, 1)
            self._workflowProgressBar.setValue(0)
            self._workflowProgressBar.setFormat("")
            self._workflowStepLabel.setVisible(False)

        description = self._currentWorkflowUiState.get("description") or ""
        simple = self._currentWorkflowUiState.get("instructions") or ""
        detailed = self._currentWorkflowUiState.get("instructions_detailed") or ""
        # Show the detailed (clinical) instruction as the primary text by default;
        # fall back to the terse simple text when a step has no detailed version.
        primary = detailed or simple
        # Offer the terse "brief" version behind the toggle, only when it adds
        # something (a detailed version exists and the simple text differs).
        brief = simple if (detailed and simple and simple.strip() != detailed.strip()) else ""
        self._workflowActionLabel.setText(str(description))
        self._workflowActionLabel.setVisible(bool(description))
        self._workflowInstructionLabel.setText(str(primary))
        self._workflowInstructionLabel.setVisible(bool(primary))
        self._renderWorkflowDetails(brief)

        self._renderWorkflowChoices(self._currentWorkflowUiState)
        self._updateReplayControls(self._currentWorkflowUiState)

        self._workflowDoneButton.setVisible(bool(self._currentWorkflowUiState.get("can_done")))
        self._workflowSkipButton.setVisible(bool(self._currentWorkflowUiState.get("can_skip")))
        self._workflowCancelButton.setVisible(bool(self._currentWorkflowUiState.get("can_cancel")))
        self._workflowDoneButton.setEnabled(bool(self._currentWorkflowUiState.get("can_done")))
        self._workflowSkipButton.setEnabled(bool(self._currentWorkflowUiState.get("can_skip")))
        self._workflowCancelButton.setEnabled(bool(self._currentWorkflowUiState.get("can_cancel")))
        done_label = self._currentWorkflowUiState.get("done_label") or "Done"
        self._workflowDoneButton.setText(str(done_label) if self._currentWorkflowUiState.get("can_done") else "Done")

    def _workflowUiStateFromStepResult(self, result):
        """Fallback panel state for workflow results not tracked by WorkflowRuntime."""
        if not isinstance(result, dict) or not result:
            return {"active": False}
        result_type = result.get("type", "")
        choices = []
        for choice in result.get("choices") or []:
            if isinstance(choice, dict):
                label = choice.get("label") or choice.get("value") or "Choice"
                value = choice.get("value", label)
                choices.append({"label": label, "value": value})
        status = "Running"
        guidance = result.get("ui_guidance") if isinstance(result.get("ui_guidance"), dict) else {}
        if result_type in ("interactive", "mixed"):
            status = "Waiting for your interaction"
        elif result_type == "user_choice":
            status = "Waiting for your choice"
        elif result.get("workflow_completed"):
            status = "Completed"
        is_repeat_decision = result_type == "user_choice" and bool(result.get("repeat_decision"))
        if is_repeat_decision:
            # Loop continue/exit decision: show its own question/instruction so
            # the Yes/No buttons are unambiguous (not the step's guidance).
            description = result.get("question") or guidance.get("title") or ""
        elif result_type == "user_choice":
            description = guidance.get("title") or result.get("question") or ""
        else:
            description = guidance.get("title") or result.get("explanation") or result.get("instruction") or ""
        if is_repeat_decision:
            instructions = result.get("instruction") or guidance.get("instruction") or ""
        else:
            instructions = (
                guidance.get("instruction")
                or (result.get("interaction") or {}).get("placement_instructions")
                or result.get("interaction_instructions")
                or ""
            )
        return {
            "active": True,
            "workflow_title": result.get("tool", "Workflow"),
            "current_step": result.get("step_id"),
            "current_index": 0,
            "completed_steps": 0,
            "total_steps": 0,
            "status": status,
            "description": description,
            "instructions": instructions,
            "choices": choices,
            "default_value": result.get("default_value"),
            "parameter_name": result.get("parameter_name", ""),
            "choice_label": guidance.get("choice_label", ""),
            "input_label": guidance.get("input_label", ""),
            "done_label": guidance.get("done_label", "Done") or "Done",
            "object_label": guidance.get("object_label", ""),
            "repeat_progress": result.get("repeat_progress") or {},
            "needs_choice_input": result_type == "user_choice" and not choices,
            "can_done": result_type in ("interactive", "mixed"),
            "can_skip": bool(result.get("is_optional")),
            "can_cancel": not result.get("workflow_completed"),
        }

    # Qt/Slicer selection-widget class -> renderer family. Keyed purely on the
    # class the *original* extension uses (recorded by the pipeline), so the
    # reproduced panel matches the source UI. Only the ``segments_table`` family
    # changes dispatch precedence; ``node_tree`` / ``choice`` intentionally defer
    # to the existing heuristic (which already produces the same widget for those
    # steps), so node-selection extensions (Orbit/BRP) are byte-identical.
    _WORKFLOW_WIDGET_FAMILIES = {
        "qMRMLSegmentsTableView": "segments_table",
        "qMRMLNodeComboBox": "node_tree",
        "qMRMLSubjectHierarchyComboBox": "node_tree",
        "qMRMLSubjectHierarchyTreeView": "node_tree",
        "qMRMLCheckableNodeComboBox": "node_tree",
        "QComboBox": "choice",
        "ctkComboBox": "choice",
    }

    @staticmethod
    def _workflowWidgetFamily(widget_class):
        """Render family for the original selection widget class, or "" (unknown)
        to defer to the heuristic. Generic: no extension/step-specific names."""
        return WidgetWorkflowMixin._WORKFLOW_WIDGET_FAMILIES.get(
            str(widget_class or "").strip(), ""
        )

    def _renderWorkflowChoices(self, state):
        """Render choice buttons for generated CLI user_choice steps."""
        if getattr(self, "_workflowChoiceInput", None) is not None:
            if self._workflowChoiceLayout is not None:
                self._workflowChoiceLayout.removeWidget(self._workflowChoiceInput)
            self._workflowChoiceInput.setParent(None)
            self._workflowChoiceInput = None
        if getattr(self, "_workflowChoiceSubmitButton", None) is not None:
            if self._workflowChoiceLayout is not None:
                self._workflowChoiceLayout.removeWidget(self._workflowChoiceSubmitButton)
            self._workflowChoiceSubmitButton.setParent(None)
            self._workflowChoiceSubmitButton = None
        if getattr(self, "_workflowNodeTree", None) is not None:
            # Drop the selection-change observer before destroying the tree, so it
            # cannot dangle / accumulate across steps (the combo had no signal to
            # leak; the tree does).
            try:
                self._workflowNodeTree.currentItemChanged.disconnect(self._onWorkflowNodeTreeSelectionChanged)
            except Exception:
                pass
            self._workflowNodeTree = None
        self._workflowNodeTreeSelectButton = None
        if getattr(self, "_workflowNodeTreeContainer", None) is not None:
            # The container owns the tree + Select button; reparenting it to None
            # destroys all three together.
            if self._workflowChoiceLayout is not None:
                self._workflowChoiceLayout.removeWidget(self._workflowNodeTreeContainer)
            self._workflowNodeTreeContainer.setParent(None)
            self._workflowNodeTreeContainer = None
        if getattr(self, "_workflowSegmentsCombo", None) is not None:
            # Drop the segmentation-picker observer before destroying it (mirrors
            # the node-tree teardown).
            try:
                self._workflowSegmentsCombo.currentNodeChanged.disconnect(self._onWorkflowSegmentsComboChanged)
            except Exception:
                pass
            self._workflowSegmentsCombo = None
        self._workflowSegmentsTable = None
        if getattr(self, "_workflowSegmentsContainer", None) is not None:
            # The container owns the segments table (+ optional combo + Done button);
            # reparenting it to None destroys them together.
            if self._workflowChoiceLayout is not None:
                self._workflowChoiceLayout.removeWidget(self._workflowSegmentsContainer)
            self._workflowSegmentsContainer.setParent(None)
            self._workflowSegmentsContainer = None
        self._workflowSegmentNameCombo = None
        if getattr(self, "_workflowSegmentNameContainer", None) is not None:
            # The container owns the name combobox + Select button; reparenting to
            # None destroys them together.
            if self._workflowChoiceLayout is not None:
                self._workflowChoiceLayout.removeWidget(self._workflowSegmentNameContainer)
            self._workflowSegmentNameContainer.setParent(None)
            self._workflowSegmentNameContainer = None
        for button in getattr(self, "_workflowChoiceButtons", []):
            if self._workflowChoiceLayout is not None:
                self._workflowChoiceLayout.removeWidget(button)
            button.setParent(None)
        self._workflowChoiceButtons = []

        choices = state.get("choices") or []
        step_id = state.get("current_step")
        needs_input = bool(state.get("needs_choice_input"))
        if self._workflowChoiceContainer is not None:
            self._workflowChoiceContainer.setVisible(bool(choices) or needs_input)
        if self._workflowChoiceLayout is None:
            return
        if not choices and needs_input:
            default_value = state.get("default_value")
            # The original extension's selection widget class (recorded by the
            # pipeline) is authoritative for the render family, so the reproduced
            # panel matches the source UI rather than a node_class-inferred guess.
            family = self._workflowWidgetFamily(state.get("source_widget_class"))
            if family == "segments_table":
                # Source widget was a qMRMLSegmentsTableView: reproduce it so the
                # user unticks individual segments/fragments exactly like the
                # extension. Authoritative — if no segmentation resolves we fall to
                # the free-text box below, never the generic node tree (which would
                # silently substitute whole-node selection for segment selection).
                if self._renderWorkflowSegmentsTable(state):
                    return
            else:
                # Segment-selection step (original extension used a qMRMLSegmentsTableView):
                # let the user untick individual segments/fragments on a segmentation node
                # exactly like the source. More specific than the generic node tree, so it
                # takes precedence; falls through if no segmentation resolves.
                if state.get("segment_selection") and self._renderWorkflowSegmentsTable(state):
                    return
                # Segment-NAME selection step (the source used a content combobox of
                # a segmentation's segment names, e.g. a "Fragment" box): offer a
                # single-pick dropdown of those names instead of a scene-node tree.
                # More specific than the node tree, so it takes precedence; falls
                # through (to free-text, never the node tree) if none resolves.
                if state.get("segment_name_selection") and self._renderWorkflowSegmentNamePicker(state):
                    return
                # Node-selection step: offer a Data-module-style subject-hierarchy tree
                # of the matching scene nodes (with the native eye / opacity / color
                # controls) instead of a free-text box, so the user can identify which
                # node is which before picking. Falls back to the text box only if no
                # matching node exists. (The LLM auto-match still runs first and may
                # auto-advance before this UI is ever shown.)
                node_class = state.get("node_class")
                if node_class and self._renderWorkflowNodeTree(state, node_class, default_value):
                    return
            self._workflowChoiceInput = qt.QLineEdit()
            input_label = state.get("input_label") or state.get("choice_label") or "value"
            self._workflowChoiceInput.setPlaceholderText(f"Enter {input_label}")
            if default_value is not None:
                self._workflowChoiceInput.setText(str(default_value))
            self._workflowChoiceInput.returnPressed.connect(self._onWorkflowChoiceInputSubmitted)

            self._workflowChoiceSubmitButton = qt.QPushButton("Set")
            self._workflowChoiceSubmitButton.setToolTip("Use this value")
            self._workflowChoiceSubmitButton.clicked.connect(self._onWorkflowChoiceInputSubmitted)

            self._workflowChoiceLayout.addWidget(self._workflowChoiceInput, 1)
            self._workflowChoiceLayout.addWidget(self._workflowChoiceSubmitButton)
            self._workflowChoiceInput.setVisible(True)
            self._workflowChoiceSubmitButton.setVisible(True)
            return
        if not choices:
            return

        for choice in choices:
            label = str(choice.get("label") or choice.get("value") or "Choice")
            value = choice.get("value", label)
            # Cap the visible label so a long choice (e.g. a long node name) can't
            # make a wide button that forces the whole module panel wider; the
            # full text stays available in the tooltip.
            display = label if len(label) <= 40 else (label[:39] + "…")
            button = qt.QPushButton(display)
            button.setToolTip(label if display != label else f"Select {label}")
            button.clicked.connect(lambda checked=False, sid=step_id, val=value: self._onWorkflowChoiceClicked(sid, val))
            self._workflowChoiceLayout.addWidget(button)
            button.setVisible(True)
            self._workflowChoiceButtons.append(button)

    def _renderWorkflowDetails(self, brief):
        """Show the 'Show brief' toggle when a terse instruction is available.

        The primary instruction label now shows the detailed (clinical) text by
        default; this toggle reveals the terse "what to do now" version.
        """
        toggle = getattr(self, "_workflowDetailToggle", None)
        label = getattr(self, "_workflowDetailLabel", None)
        if toggle is None or label is None:
            return
        brief = str(brief or "")
        self._workflowDetailText = brief
        if not brief:
            toggle.setVisible(False)
            label.setVisible(False)
            return
        # New step: show the toggle collapsed (brief hidden) by default.
        toggle.setVisible(True)
        toggle.setText("Show brief ▸")
        label.setText(brief)
        label.setVisible(False)

    def _onToggleWorkflowDetails(self):
        label = getattr(self, "_workflowDetailLabel", None)
        toggle = getattr(self, "_workflowDetailToggle", None)
        if label is None or toggle is None:
            return
        expanded = not label.visible
        label.setVisible(expanded)
        toggle.setText("Hide brief ▾" if expanded else "Show brief ▸")

    def _showWorkflowInteraction(self, result):
        """Show an interactive or mixed workflow wait state."""
        self._updateWorkflowPanel(result)

    def _showWorkflowChoice(self, result):
        """Show a user-choice workflow step as buttons when choices are known."""
        self._updateWorkflowPanel(result)

    def _clearWorkflowPanel(self):
        """Hide and reset the user-facing workflow panel."""
        self._currentWorkflowUiState = {"active": False}
        self._taskWorkflowPanelActive = False
        if not getattr(self, "_workflowUserFrame", None):
            return
        self._workflowUserFrame.setVisible(False)
        self._workflowTitleLabel.setText("Workflow")
        self._workflowStatusLabel.setText("Idle")
        self._workflowProgressBar.setRange(0, 1)
        self._workflowProgressBar.setValue(0)
        self._workflowStepLabel.setText("Step 0 of 0")
        self._workflowActionLabel.setText("")
        self._workflowInstructionLabel.setText("")
        self._renderWorkflowDetails("")
        self._renderWorkflowChoices({})
        self._updateReplayControls({})
        self._workflowDoneButton.setVisible(False)
        self._workflowSkipButton.setVisible(False)
        self._workflowCancelButton.setVisible(False)
        self._workflowDoneButton.setText("Done")

    def _onWorkflowDoneClicked(self):
        self._closeFloatingWorkflowControl()
        current_step = self._currentWorkflowUiState.get("current_step")
        if current_step:
            self.sendButton.setEnabled(False)
            # The interaction for this step is over: drop any observers and
            # debounce timers the runtime registered, so they cannot
            # accumulate across repeat iterations and consecutive
            # interactive steps (a generic cause of degraded/locked
            # interaction state after several placements).
            try:
                self._interactionManager.cleanup()
            except Exception:
                logger.debug("Interaction cleanup on Done failed", exc_info=True)
            self._runWorkflowStepDirect(current_step, "proceed")

    def _onWorkflowSkipClicked(self):
        current_step = self._currentWorkflowUiState.get("current_step")
        if current_step:
            self.sendButton.setEnabled(False)
            self._runWorkflowStepDirect(current_step, "skip")

    def _onWorkflowCancelClicked(self):
        self._closeFloatingWorkflowControl()
        current_step = self._currentWorkflowUiState.get("current_step")
        self.sendButton.setEnabled(False)
        self._runWorkflowStepDirect(current_step, "cancel")

    def _onWorkflowChoiceClicked(self, step_id, value):
        # While scrubbing the replay, the choices belong to a past step: clicking
        # one re-runs the workflow from that step with the chosen value.
        if self._currentWorkflowUiState.get("replay_previewing"):
            index = self._currentWorkflowUiState.get("preview_index")
            if index is not None:
                self._rerunFromCheckpoint(index, {"choice_value": value})
            return
        if step_id:
            self.sendButton.setEnabled(False)
            self._runWorkflowStepDirect(step_id, "choice_made", args={"choice_value": value})

    def _onWorkflowChoiceInputSubmitted(self):
        step_id = self._currentWorkflowUiState.get("current_step")
        if not step_id or self._workflowChoiceInput is None:
            return
        value = self._workflowChoiceInput.text.strip()
        if not value:
            value = str(self._currentWorkflowUiState.get("default_value") or "").strip()
        if not value:
            return
        if self._currentWorkflowUiState.get("replay_previewing"):
            index = self._currentWorkflowUiState.get("preview_index")
            if index is not None:
                self._rerunFromCheckpoint(index, {"choice_value": value})
            return
        self.sendButton.setEnabled(False)
        self._runWorkflowStepDirect(step_id, "choice_made", args={"choice_value": value})

    def _renderWorkflowNodeTree(self, state, node_class, default_value):
        """Show a Data-module-style qMRMLSubjectHierarchyTreeView filtered to
        ``node_class`` (native eye/visibility column + right-click opacity/color),
        with a Select button beneath it.

        This is the literal widget the Data module embeds, so it reproduces the
        full node-identification experience: the user can toggle a node's
        visibility (and adjust opacity/colour via right-click) to see which scene
        object a name refers to before committing. The tree's proxy model applies
        the same exclusions the old qMRMLNodeComboBox did -- it filters to
        ``node_class`` (subclass-inclusive, like the combo's
        ``showChildNodeTypes=True``) and automatically hides ``HideFromEditors``/
        internal helper nodes (e.g. a ``parameterNodeWrapper`` placeholder model),
        so no extra filtering code is needed.

        Returns True if it rendered (>=1 selectable node of this class exists), or
        False to let the caller fall back to the free-text box.
        """
        # Candidate list = scene nodes of node_class, minus HideFromEditors. This
        # both replicates the old combo's "no node -> free text" empty gate and
        # supplies the candidates for the _bestNodeMatchIndex default guess. We
        # mirror the tree's own HideFromEditors exclusion here so the emptiness
        # check matches exactly what the tree will actually display.
        candidates = []
        try:
            for node in slicer.util.getNodesByClass(node_class):
                if node is None:
                    continue
                try:
                    if node.GetHideFromEditors():
                        continue
                except Exception:
                    pass
                candidates.append({"id": node.GetID(), "name": node.GetName(), "node": node})
        except Exception:
            logger.debug("Enumerating node candidates failed", exc_info=True)
            candidates = []
        if not candidates:
            return False

        # Container stacks the tree above its Select button (the choice layout is a
        # QHBoxLayout; a tall tree reads better over the button than beside it).
        container = qt.QWidget()
        vbox = qt.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        tree = slicer.qMRMLSubjectHierarchyTreeView()
        tree.setMRMLScene(slicer.mrmlScene)     # scene BEFORE filtering
        tree.nodeTypes = [node_class]           # exact class, subclass-inclusive
        # Trim columns the narrow module panel has no room for; keep the eye
        # (visibility) column on (its default). Colour stays reachable via
        # right-click. For a vtkMRMLSegmentationNode the eye toggles whole-
        # segmentation visibility (per-segment is not exposed in this row), which
        # is fine for telling which node is which.
        for _attr in ("idColumnVisible", "transformColumnVisible", "descriptionColumnVisible"):
            try:
                setattr(tree, _attr, False)
            except Exception:
                logger.debug("Tree column setup (%s) failed", _attr, exc_info=True)
        # Bound the height so the tree does not swallow the panel (it scrolls
        # internally); expand horizontally only to the available width -- never
        # demand more, so a long node name cannot force the panel wider (see
        # _applyWidthSafeLabels).
        tree.setMinimumHeight(140)
        tree.setMaximumHeight(220)
        tree.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self._workflowNodeTree = tree

        # Default to the best keyword/name match (only a guess; user can change).
        try:
            idx = self._bestNodeMatchIndex(
                candidates, default_value, state.get("node_keywords") or []
            )
            if 0 <= idx < len(candidates):
                tree.setCurrentNode(candidates[idx]["node"])
        except Exception:
            logger.debug("Defaulting node tree selection failed", exc_info=True)

        button = qt.QPushButton("Select")
        button.setToolTip("Use the selected node")
        button.clicked.connect(self._onWorkflowNodeTreeSelected)
        self._workflowNodeTreeSelectButton = button

        # Enable Select only while a real data node of node_class is current; a
        # folder/patient/study row yields currentNode() == None.
        tree.currentItemChanged.connect(self._onWorkflowNodeTreeSelectionChanged)
        self._updateNodeTreeSelectButtonEnabled()

        vbox.addWidget(tree, 1)
        vbox.addWidget(button)
        self._workflowNodeTreeContainer = container
        self._workflowChoiceLayout.addWidget(container, 1)
        container.setVisible(True)
        return True

    def _nodeTreeValidCurrentNode(self):
        """Return the tree's current node iff it is a real data node of the step's
        ``node_class`` (not a folder/hierarchy row), else None."""
        tree = getattr(self, "_workflowNodeTree", None)
        if tree is None:
            return None
        try:
            node = tree.currentNode()
        except Exception:
            return None
        if node is None:
            return None
        node_class = (self._currentWorkflowUiState or {}).get("node_class")
        if node_class:
            try:
                if not node.IsA(node_class):
                    return None
            except Exception:
                return None
        return node

    def _updateNodeTreeSelectButtonEnabled(self):
        button = getattr(self, "_workflowNodeTreeSelectButton", None)
        if button is not None:
            button.setEnabled(self._nodeTreeValidCurrentNode() is not None)

    def _onWorkflowNodeTreeSelectionChanged(self, *args):
        self._updateNodeTreeSelectButtonEnabled()

    def _bestNodeMatchIndex(self, candidates, default_value, keywords):
        """Best candidate index: exact recorded name → keyword substring → first.

        Keyword scoring uses substring containment (case-insensitive) so prefix
        keywords like 'mandib' match the node name 'MandibleSegmentation'; the
        distinctive keywords then break the tie against shared words like
        'segmentation'. It is only a default guess — the user picks from the
        node tree.
        """
        names = [str(c.get("name") or "") for c in candidates]
        dv = str(default_value or "").strip()
        if dv:
            for i, name in enumerate(names):
                if name == dv:
                    return i
        kws = [str(k).strip().lower() for k in (keywords or []) if len(str(k).strip()) >= 3]
        if kws:
            best_i, best_score = 0, 0
            for i, name in enumerate(names):
                low = name.lower()
                score = sum(1 for k in kws if k in low)
                if score > best_score:
                    best_i, best_score = i, score
            if best_score > 0:
                return best_i
        return 0

    def _onWorkflowNodeTreeSelected(self):
        node = self._nodeTreeValidCurrentNode()
        if node is None:
            return
        name = ""
        try:
            name = str(node.GetName() or "").strip()
        except Exception:
            name = ""
        if not name:
            return
        if self._currentWorkflowUiState.get("replay_previewing"):
            index = self._currentWorkflowUiState.get("preview_index")
            if index is not None:
                self._rerunFromCheckpoint(index, {"choice_value": name})
            return
        step_id = self._currentWorkflowUiState.get("current_step")
        if step_id:
            self.sendButton.setEnabled(False)
            self._runWorkflowStepDirect(step_id, "choice_made", args={"choice_value": name})

    # ------------------------------------------------------------------
    # Segment-selection step (qMRMLSegmentsTableView)
    # ------------------------------------------------------------------
    def _resolveParamNodeFieldNodeID(self, field):
        """Live MRML node ID held by a parameterNodeWrapper ``field`` on the active
        workflow extension's parameter node, or "" if it cannot be resolved.

        Used to bind a segments table to the exact segmentation the source binds it
        to (e.g. ``OutputFracSeg``), captured by the pipeline as
        ``segmentation_target_param``. Best-effort: runs in the agent's own Python
        (not the sandbox), so attribute access is fine; any failure returns "" and
        the caller falls back to keyword best-match.
        """
        field = str(field or "").strip()
        if not field:
            return ""
        runtime = getattr(self, "_workflowRuntime", None)
        session = getattr(runtime, "session", None) if runtime is not None else None
        ext = getattr(session, "extension_name", None) if session is not None else None
        if not ext:
            return ""
        module_name = ext
        try:
            from SlicerAIAgentLib.ExtensionCLILoader import get_validated_extensions
            meta = (get_validated_extensions().get(ext) or {}).get("workflow_metadata", {}) or {}
            module_name = meta.get("extension_module_name") or ext
        except Exception:
            module_name = ext
        widget = None
        try:
            widget = slicer.util.getModuleWidget(module_name)
        except Exception:
            widget = None
        if widget is None:
            return ""
        paramNode = None
        for getter in (
            lambda: widget.logic.getParameterNode(),
            lambda: widget._parameterNode,
            lambda: widget.getParameterNode(),
        ):
            try:
                pn = getter()
            except Exception:
                pn = None
            if pn is not None:
                paramNode = pn
                break
        if paramNode is None:
            return ""
        try:
            node = getattr(paramNode, field, None)
            if node is not None and hasattr(node, "GetID"):
                return str(node.GetID() or "")
        except Exception:
            return ""
        return ""

    def _preferredSegmentationIndex(self, candidates, state):
        """Index of the segmentation to default-select among ``candidates``:
        the exact pipeline-captured target field first, then name/keyword
        best-match, then the first candidate."""
        target_param = str(state.get("segmentation_target_param") or "").strip()
        if target_param:
            try:
                target_id = self._resolveParamNodeFieldNodeID(target_param)
                if target_id:
                    for i, c in enumerate(candidates):
                        if c.get("id") == target_id:
                            return i
            except Exception:
                logger.debug("Resolving segmentation target_param failed", exc_info=True)
        try:
            return self._bestNodeMatchIndex(
                candidates, state.get("default_value"), state.get("segmentation_keywords") or []
            )
        except Exception:
            logger.debug("Segmentation best-match failed", exc_info=True)
        return 0

    def _renderWorkflowSegmentsTable(self, state):
        """Show a real qMRMLSegmentsTableView so the user can untick individual
        segments/fragments on a segmentation node, exactly like the original
        extension's selector. Toggling the eye column sets per-segment visibility
        (``vtkMRMLSegmentationDisplayNode.SetSegmentVisibility``), which is the
        same state the extension's downstream code reads; clicking Done simply
        advances (no choice value is invented).

        Returns True if it rendered (>=1 selectable segmentation exists), or False
        to let the caller fall back to the node tree / free-text box.
        """
        node_class = state.get("segmentation_node_class") or "vtkMRMLSegmentationNode"
        # Candidate segmentations = scene nodes of node_class, minus HideFromEditors
        # (mirrors the node tree's exclusion / empty gate).
        candidates = []
        try:
            for node in slicer.util.getNodesByClass(node_class):
                if node is None:
                    continue
                try:
                    if node.GetHideFromEditors():
                        continue
                except Exception:
                    pass
                candidates.append({"id": node.GetID(), "name": node.GetName(), "node": node})
        except Exception:
            logger.debug("Enumerating segmentation candidates failed", exc_info=True)
            candidates = []
        if not candidates:
            return False

        container = qt.QWidget()
        vbox = qt.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        # Which segmentation to show. Prefer the exact target the source binds the
        # table to (segmentation_target_param -> a parameterNodeWrapper field,
        # captured by the pipeline); else default-guess by name keywords (e.g.
        # 'fracture' vs 'pelvis'); else the first. With more than one, also offer a
        # combo so the user can switch.
        idx = self._preferredSegmentationIndex(candidates, state)
        if not (0 <= idx < len(candidates)):
            idx = 0
        seg_node = candidates[idx]["node"]
        if len(candidates) > 1:
            combo = slicer.qMRMLNodeComboBox()
            combo.nodeTypes = [node_class]
            combo.addEnabled = False
            combo.removeEnabled = False
            combo.renameEnabled = False
            combo.noneEnabled = False
            combo.showHidden = False
            combo.setMRMLScene(slicer.mrmlScene)
            try:
                combo.setCurrentNodeID(candidates[idx]["id"])
                seg_node = candidates[idx]["node"]
            except Exception:
                logger.debug("Defaulting segmentation combo failed", exc_info=True)
            cur = combo.currentNode()
            if cur is not None:
                seg_node = cur
            combo.currentNodeChanged.connect(self._onWorkflowSegmentsComboChanged)
            self._workflowSegmentsCombo = combo
            vbox.addWidget(combo)

        table = slicer.qMRMLSegmentsTableView()
        try:
            table.setMRMLScene(slicer.mrmlScene)
        except Exception:
            logger.debug("Segments table setMRMLScene failed", exc_info=True)
        # Reproduce the source selector's columns: per-segment eye (visibility),
        # colour and opacity (the original qMRMLSegmentsTableView shows all three).
        # Keep the segmentation-editor status column hidden — it is not part of the
        # extension's fragment selector. Wrapped individually because the available
        # column setters vary across Slicer builds.
        for _meth, _arg in (
            ("setVisibilityColumnVisible", True),
            ("setColorColumnVisible", True),
            ("setOpacityColumnVisible", True),
            ("setStatusColumnVisible", False),
            ("setHeaderVisible", True),
        ):
            try:
                getattr(table, _meth)(_arg)
            except Exception:
                logger.debug("Segments table %s failed", _meth, exc_info=True)
        try:
            table.setSelectionMode(qt.QAbstractItemView.NoSelection)
        except Exception:
            logger.debug("Segments table selection-mode setup failed", exc_info=True)
        table.setMinimumHeight(140)
        table.setMaximumHeight(260)
        table.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self._workflowSegmentsTable = table
        self._bindSegmentsTable(seg_node)

        button = qt.QPushButton(state.get("done_label") or "Done")
        button.setToolTip("Finish selecting and continue")
        button.clicked.connect(self._onWorkflowSegmentsDone)

        vbox.addWidget(table, 1)
        vbox.addWidget(button)
        self._workflowSegmentsContainer = container
        self._workflowChoiceLayout.addWidget(container, 1)
        container.setVisible(True)
        return True

    def _renderWorkflowSegmentNamePicker(self, state):
        """Show a single-pick combobox of the SEGMENT NAMES of a segmentation, so
        the user picks one fragment/segment by name exactly like the extension's
        own content combobox (e.g. the "Fragment" selector), instead of a tree of
        whole scene nodes. The picked name flows through choice_made -> choice_value
        identically to a literal-choice step (and Part 4 mirrors it onto the live
        source combobox so its connected handler fires).

        Returns True if it rendered (a segmentation with >=1 segment resolved), or
        False so the caller falls back to the free-text box (never the node tree).
        """
        node_class = state.get("segmentation_node_class") or "vtkMRMLSegmentationNode"
        candidates = []
        try:
            for node in slicer.util.getNodesByClass(node_class):
                if node is None:
                    continue
                try:
                    if node.GetHideFromEditors():
                        continue
                except Exception:
                    pass
                candidates.append({"id": node.GetID(), "name": node.GetName(), "node": node})
        except Exception:
            logger.debug("Enumerating segmentation candidates failed", exc_info=True)
            candidates = []
        if not candidates:
            return False
        # Safety for the deterministic content-combobox signal: with more than one
        # segmentation and no way to pick the right one (no bound target field and
        # no widget-name keyword match), do NOT guess -- fall through to the
        # free-text box so a non-segment content combobox isn't shown a random
        # segmentation's segments.
        if len(candidates) > 1:
            target = str(state.get("segmentation_target_param") or "").strip()
            kws = [str(k).lower() for k in (state.get("segmentation_keywords") or []) if len(str(k)) >= 3]
            resolved = bool(target and self._resolveParamNodeFieldNodeID(target))
            if not resolved and kws:
                resolved = any(
                    any(k in str(c.get("name") or "").lower() for k in kws) for c in candidates
                )
            if not resolved:
                return False
        idx = self._preferredSegmentationIndex(candidates, state)
        if not (0 <= idx < len(candidates)):
            idx = 0
        seg_node = candidates[idx]["node"]
        # Segment names in segment order (the source combobox lists them all).
        names = []
        try:
            seg = seg_node.GetSegmentation()
            for i in range(seg.GetNumberOfSegments()):
                sid = seg.GetNthSegmentID(i)
                nm = seg.GetSegment(sid).GetName() if sid else ""
                if nm:
                    names.append(str(nm))
        except Exception:
            logger.debug("Enumerating segment names failed", exc_info=True)
            names = []
        if not names:
            return False

        container = qt.QWidget()
        vbox = qt.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        combo = qt.QComboBox()
        for nm in names:
            combo.addItem(nm)
        combo.setCurrentIndex(0)
        button = qt.QPushButton(state.get("choice_label") or "Select")
        button.setToolTip("Pick this item and continue")
        button.clicked.connect(self._onWorkflowSegmentNameSelected)
        vbox.addWidget(combo)
        vbox.addWidget(button)
        self._workflowSegmentNameCombo = combo
        self._workflowSegmentNameContainer = container
        self._workflowChoiceLayout.addWidget(container, 1)
        container.setVisible(True)
        return True

    def _onWorkflowSegmentNameSelected(self):
        combo = getattr(self, "_workflowSegmentNameCombo", None)
        if combo is None:
            return
        try:
            name = str(combo.currentText or "").strip()
        except Exception:
            name = ""
        if not name:
            return
        if self._currentWorkflowUiState.get("replay_previewing"):
            index = self._currentWorkflowUiState.get("preview_index")
            if index is not None:
                self._rerunFromCheckpoint(index, {"choice_value": name})
            return
        step_id = self._currentWorkflowUiState.get("current_step")
        if step_id:
            self.sendButton.setEnabled(False)
            self._runWorkflowStepDirect(step_id, "choice_made", args={"choice_value": name})

    def _bindSegmentsTable(self, seg_node):
        """Bind the segments table to ``seg_node``, ensuring a display node exists
        so the eye column can write per-segment visibility."""
        table = getattr(self, "_workflowSegmentsTable", None)
        if table is None or seg_node is None:
            return
        try:
            seg_node.CreateDefaultDisplayNodes()
        except Exception:
            logger.debug("CreateDefaultDisplayNodes failed", exc_info=True)
        try:
            table.setSegmentationNode(seg_node)
        except Exception:
            logger.debug("setSegmentationNode failed", exc_info=True)

    def _onWorkflowSegmentsComboChanged(self, *args):
        # Read the node from the combo (currentNodeChanged has overloaded
        # signatures; the positional arg may be a bool).
        combo = getattr(self, "_workflowSegmentsCombo", None)
        if combo is not None:
            try:
                self._bindSegmentsTable(combo.currentNode())
            except Exception:
                logger.debug("Segments combo change rebind failed", exc_info=True)

    def _onWorkflowSegmentsDone(self):
        # The user edited per-segment visibility directly on the scene; advance the
        # step with no fabricated choice value (downstream code reads visibility).
        if self._currentWorkflowUiState.get("replay_previewing"):
            index = self._currentWorkflowUiState.get("preview_index")
            if index is not None:
                self._rerunFromCheckpoint(index, {})
            return
        step_id = self._currentWorkflowUiState.get("current_step")
        if step_id:
            self.sendButton.setEnabled(False)
            self._runWorkflowStepDirect(step_id, "proceed")

    def _enterWorkflowWait(self, step_info):
        """
        Enter wait state for an interactive workflow step.
        Shows user instructions in the Workflow panel. Text commands remain a fallback.
        """
        step_desc = (
            step_info.get("explanation")
            or step_info.get("step_info", {}).get("description")
            or "Interactive step"
        )

        self._waitingForUser = True
        self._currentWorkflowStepInfo = step_info
        self._showWorkflowInteraction(step_info)
        # The interaction is shown ONLY in the inline workflow panel (module UI).
        # The separate floating "AI Agent — workflow step" pop-up was removed at
        # the user's request; the panel's Done/Cancel buttons drive the workflow.

        # Placement guard: shortly after the wait state is shown (so the pre
        # code has settled), verify Slicer is actually armed for the
        # interaction; re-arm if anything upstream dropped placement mode.
        try:
            qt.QTimer.singleShot(600, lambda: self._ensurePlacementArmed(step_info))
        except Exception:
            logger.debug("Placement guard scheduling failed", exc_info=True)

        self._setAgentStatus("Workflow", f"Waiting: {step_desc}")
        logger.info(f"[Workflow] Entered wait state for step: {step_desc}")

    def _ensurePlacementArmed(self, step_info):
        """Re-arm markup placement if the waited-on step lost it.

        Generic guard for interactive markup steps: a post template, an
        extension callback, or a layout rebuild can leave Slicer outside
        place mode while the workflow is waiting for the user to place
        points — the visible symptom is "interaction tools hidden". When the
        runtime knows the step's interaction node, placement is restored
        deterministically (active list + single place mode). Fail-open.
        """
        try:
            if not self._waitingForUser or not isinstance(step_info, dict):
                return
            current = getattr(self, "_currentWorkflowStepInfo", None)
            if current is not step_info:
                return  # a newer step superseded this wait
            interaction = step_info.get("interaction") or {}
            node_class = str(interaction.get("node_class") or "")
            if not node_class.startswith("vtkMRMLMarkups") and step_info.get(
                "interaction_type", ""
            ) not in ("plane", "curve", "line", "fiducial"):
                return

            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            in_place_mode = bool(
                interactionNode
                and interactionNode.GetCurrentInteractionMode() == interactionNode.Place
            )
            has_active_place_node = bool(
                selectionNode and selectionNode.GetActivePlaceNodeID()
            )
            if in_place_mode and has_active_place_node:
                return  # armed — nothing to do

            from SlicerAIAgentLib.workflow_state import latest_interaction_node_for_step
            node = latest_interaction_node_for_step(step_info.get("step_id", ""))
            if node is None:
                logger.info(
                    "[Workflow] Placement not armed for step %s and no remembered "
                    "node to re-arm with", step_info.get("step_id", "?"),
                )
                return
            slicer.modules.markups.logic().SetActiveListID(node)
            if interactionNode:
                interactionNode.SwitchToSinglePlaceMode()
            self._recordRoleEvent("Executor", "placement_rearmed", {
                "step_id": step_info.get("step_id", ""),
                "node": node.GetName(),
            })
            logger.info(
                "[Workflow] Re-armed placement for step %s on node %s",
                step_info.get("step_id", "?"), node.GetName(),
            )
        except Exception:
            logger.debug("Placement guard failed open", exc_info=True)

    def _exitWorkflowWait(self):
        """Exit workflow wait state and restore normal state."""
        self._waitingForUser = False
        self._currentWorkflowStepInfo = None
        self._closeFloatingWorkflowControl()
        self._setReadyStatus()

    def _closeFloatingWorkflowControl(self):
        window = getattr(self, "_floatingWorkflowControl", None)
        self._floatingWorkflowControl = None
        if window is not None:
            try:
                window.close()
                window.deleteLater()
            except Exception:
                logger.debug("Floating workflow control close failed", exc_info=True)

    def _autoProceedWorkflowStep(self, next_step_info):
        """Automatically prompt the LLM to execute the next workflow step."""
        step_id = next_step_info.get("step_id")
        step_type = next_step_info.get("step_type")
        description = next_step_info.get("description", "")

        if step_type == "branch":
            # Ask the user about the branch decision
            self._updateWorkflowPanel({
                "active": True,
                "workflow_title": self._currentWorkflowUiState.get("workflow_title", "Workflow"),
                "status": "Waiting for your choice",
                "current_step": step_id,
                "current_index": self._currentWorkflowUiState.get("current_index", 0),
                "completed_steps": self._currentWorkflowUiState.get("completed_steps", 0),
                "total_steps": self._currentWorkflowUiState.get("total_steps", 0),
                "description": description,
                "instructions": "This step is optional.",
                "can_done": True,
                "can_skip": True,
                "can_cancel": True,
            })
            self.sendButton.setEnabled(True)
            return

        # Auto-send a prompt to proceed with the next step
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
            SLICER_AI_AGENT_ROOT, "Resources", "extension_CLI", ext_name
        )

        def filler(template_path, args):
            full_path = os.path.join(cli_dir, template_path)
            if not os.path.exists(full_path):
                return None
            with open(full_path, "r") as f:
                template_text = f.read()
            return _fill_template(template_text, args)

        return filler

    def _applyWorkflowDisplayProperties(self, step_info):
        """Apply display properties from the workflow step to newly created markup nodes."""
        display_props = step_info.get("display_properties")
        if not display_props:
            return

        try:
            # Find recently created markup nodes matching the step's node class
            node_class = step_info.get("node_class", "")
            interaction_type = step_info.get("interaction_type", "")
            step_id = step_info.get("step_id", "")

            # Try to find the node by the template variable convention
            ext_name = step_info.get("tool", "")
            node_var = f"_{ext_name.lower()}_{step_id}_id"
            import __main__
            node_id = getattr(__main__, node_var, None)

            if node_id:
                node = slicer.mrmlScene.GetNodeByID(node_id)
            else:
                # Fallback: find the most recently added markup node of the expected class
                node = None
                all_nodes = slicer.mrmlScene.GetNodesByClass(node_class) if node_class else None
                if all_nodes:
                    all_nodes.UnRegister(None)
                    count = all_nodes.GetNumberOfItems()
                    if count > 0:
                        node = all_nodes.GetItemAsObject(count - 1)

            if not node:
                logger.warning(
                    f"[Workflow] Could not find markup node for step '{step_id}' "
                    f"to apply display properties"
                )
                return

            # Check for unresolved symbolic view tags and attempt fallback resolution
            view_ids = display_props.get("addViewNodeIDs", [])
            resolved_ids = []
            unresolved = []
            for ref in view_ids:
                vid = self._interactionManager._resolve_view_node_ref(ref)
                if vid:
                    resolved_ids.append(vid)
                else:
                    tag = ref.get("tag", "?")
                    unresolved.append(tag)
                    # Fallback: resolve symbolic tag via slicer module globals
                    # (set by the create_bone_models template or extension init)
                    if ref.get("symbolic"):
                        try:
                            tag_val = getattr(slicer, ref["tag"], None)
                            if tag_val is not None:
                                cls = ref.get("class", "vtkMRMLViewNode")
                                vn = slicer.mrmlScene.GetSingletonNode(str(tag_val), cls)
                                if vn:
                                    resolved_ids.append(vn.GetID())
                                    unresolved.pop()
                        except Exception:
                            pass

            if resolved_ids:
                # Replace symbolic refs with resolved IDs
                display_props = dict(display_props)
                display_props["addViewNodeIDs"] = resolved_ids

            if unresolved:
                logger.warning(
                    f"[Workflow] Could not resolve view tags for step '{step_id}': {unresolved}. "
                    f"Markup will be visible in all views."
                )

            display_node = node.GetDisplayNode()
            if display_node:
                self._interactionManager._apply_display_properties(display_node, display_props)

                # Verify: log what was actually applied
                applied_views = []
                if hasattr(display_node, "GetNumberOfViewNodeIDs"):
                    for i in range(display_node.GetNumberOfViewNodeIDs()):
                        applied_views.append(display_node.GetNthViewNodeID(i))
                logger.info(
                    f"[Workflow] Applied display properties to '{node.GetName()}' "
                    f"for step '{step_id}': views={applied_views}, "
                    f"props={list(display_props.keys())}"
                )

            # Apply locked state on the node itself
            if display_props.get("locked") is not None:
                node.SetLocked(display_props["locked"])

        except Exception as e:
            logger.warning(f"[Workflow] Failed to apply display properties: {e}")

    def _handleCliProgress(self, stage_num, stage_name, detail):
        """Handle CLI generator progress updates on the main thread."""
        self._cliProgressDisplay.append(f"  Phase {stage_num}: {stage_name} — {detail}")

    def _handleCliProbeRequest(self, payload):
        """Execute a CLI live-API probe on the Qt/Slicer main thread."""
        response_queue = payload.get('response_queue')
        probe_code = payload.get('probe_code', '')
        try:
            from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
            result = ExtensionCLIAnalyzer._execute_probe(probe_code)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        if response_queue is not None:
            response_queue.put(result)

    def _handleCliComplete(self, result):
        """Handle CLI generator completion on the main thread."""
        self._cliGeneratorRunning = False
        self._analyzeGenerateButton.setEnabled(True)

        if result.get("success"):
            # Generation is pure background analysis — it must NOT touch the live
            # MRML scene/viewport. Runtime (live-execution) validation is deferred
            # to the Repair button, which runs the steps deliberately and invisibly.
            self._finalizeCliValidation(result, {})

        else:
            self._cliStatusLabel.setText("Failed")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
            error = result.get("error", "Unknown error")
            self._cliProgressDisplay.append(f"FAILED: {error}")

            stages = result.get("phases_completed") or result.get("stages_completed", [])
            if stages:
                self._cliProgressDisplay.append(
                    f"Completed phases: {stages}"
                )

            # Auto-revise if templates were generated but validation failed
            if result.get("validation_result") and not result["validation_result"].get("valid"):
                self._cliProgressDisplay.append("Auto-revising with LLM...")
                self._autoReviseCli(result)

    # Maximum live-exec → repair → re-exec passes before giving up on a defect.
    _MAX_LIVE_REPAIR_PASSES = 3

    def _runCliLiveValidation(self, result, outer_iter=0):
        """Live-execute generated steps; repair runtime defects; re-validate.

        Runs on the Qt main thread (SafeExecutor needs slicer.mrmlScene). Each
        pass executes every runnable step's template for real; any step that
        raises (status failed_bug) is sent to a targeted LLM repair, then this
        method is re-entered to confirm the fix. Precondition-skipped steps never
        block. Bounded by _MAX_LIVE_REPAIR_PASSES.
        """
        cli_dir = result.get("cli_dir")
        if not cli_dir or not os.path.isdir(cli_dir):
            self._finalizeCliValidation(result, {})
            return

        try:
            from SlicerAIAgentLib.SafeExecutor import SafeExecutor
            from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
            from SlicerAIAgentLib.CodeValidator import CodeValidator
            executor = SafeExecutor()
            analyzer = ExtensionCLIAnalyzer(
                llm_client=self.logic.llmClient,
                output_base_dir=os.path.join(
                    SLICER_AI_AGENT_ROOT, "Resources", "extension_CLI"
                ),
                code_validator=CodeValidator(),
            )
            self._cliProgressDisplay.append(
                f"Live-validating generated steps (pass {outer_iter + 1})..."
            )
            live_results = analyzer.live_validate_templates(cli_dir, executor)
        except Exception as e:
            self._cliProgressDisplay.append(f"Live validation error: {e}")
            self._finalizeCliValidation(result, {})
            return

        self._writeLiveExecutionArtifact(cli_dir, live_results, outer_iter)

        failures = [
            r for r in live_results.values()
            if r.get("status") == "failed_bug"
        ]
        n_valid = sum(1 for r in live_results.values() if r.get("status") == "live_valid")
        n_skip = sum(
            1 for r in live_results.values()
            if str(r.get("status", "")).startswith("skipped")
        )
        self._cliProgressDisplay.append(
            f"Live validation: {n_valid} ran clean, {len(failures)} failed, "
            f"{n_skip} skipped (preconditions/interaction)."
        )

        if failures and outer_iter < self._MAX_LIVE_REPAIR_PASSES:
            self._cliLiveResult = result
            self._cliLiveIter = outer_iter
            self._cliLiveResults = live_results
            self._cliGeneratorRunning = True
            self._cliStatusLabel.setText("Repairing live failures...")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: orange;")

            ext_name = (result.get("manifest", {}) or {}).get("extension_name", "")
            data = self._getSelectedExtensionData() or {}
            source_path = data.get("path", "")

            def _run_live_repair():
                try:
                    from SlicerAIAgentLib.ExtensionCLIAnalyzer import ExtensionCLIAnalyzer
                    from SlicerAIAgentLib.CodeValidator import CodeValidator
                    repair_analyzer = ExtensionCLIAnalyzer(
                        llm_client=self.logic.llmClient,
                        output_base_dir=os.path.join(
                            SLICER_AI_AGENT_ROOT, "Resources", "extension_CLI"
                        ),
                        code_validator=CodeValidator(),
                        on_progress=lambda n, s, d: self._streamQueue.put(
                            ('cli_progress', {'stage': n, 'name': s, 'detail': d})
                        ),
                        on_error=lambda e: self._streamQueue.put(('cli_error', e)),
                    )
                    repair = repair_analyzer.repair_live_failures(
                        ext_name, failures, source_path=source_path
                    )
                    self._streamQueue.put(('cli_live_repair_complete', repair))
                except Exception as e:
                    self._streamQueue.put(('cli_error', str(e)))

            threading.Thread(target=_run_live_repair, daemon=True).start()
            return

        # No failures, or repair budget exhausted: finalize on these results.
        self._finalizeCliValidation(result, live_results)

    def _handleCliLiveRepairComplete(self, repair):
        """Resume the live-validation loop after a targeted live repair."""
        self._cliGeneratorRunning = False
        result = getattr(self, "_cliLiveResult", None)
        outer_iter = getattr(self, "_cliLiveIter", 0)
        if not result:
            return
        if repair.get("success"):
            self._cliProgressDisplay.append(
                f"Live repair rewrote {len(repair.get('repaired', []))} "
                f"template(s); re-validating..."
            )
            self._runCliLiveValidation(result, outer_iter + 1)
        else:
            # No change produced — re-validating again would loop on the same
            # defect. Finalize honestly with the last known live results.
            self._cliProgressDisplay.append(
                f"Live repair produced no change ({repair.get('error', 'unknown')})."
            )
            self._finalizeCliValidation(
                result, getattr(self, "_cliLiveResults", {})
            )

    def _writeLiveExecutionArtifact(self, cli_dir, live_results, outer_iter):
        """Persist a per-step live-execution report under debug/ for transparency."""
        try:
            debug_dir = os.path.join(cli_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            steps = []
            for tpl_key, rec in sorted(live_results.items()):
                steps.append({
                    "template_key": tpl_key,
                    "step_id": rec.get("step_id", ""),
                    "operation_type": rec.get("operation_type", ""),
                    "status": rec.get("status", ""),
                    "error": rec.get("error"),
                    "execution_time": rec.get("execution_time", 0),
                })
            payload = {
                "pass": outer_iter + 1,
                "summary": {
                    "live_valid": sum(1 for s in steps if s["status"] == "live_valid"),
                    "failed_bug": sum(1 for s in steps if s["status"] == "failed_bug"),
                    "skipped": sum(
                        1 for s in steps if str(s["status"]).startswith("skipped")
                    ),
                },
                "steps": steps,
            }
            with open(
                os.path.join(debug_dir, "live_execution.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(payload, f, indent=2)
        except Exception:
            logger.debug("Failed to write live_execution.json", exc_info=True)

    def _persistLiveValidationStatus(self, result, failed_keys, n_valid, n_skipped):
        """Reflect the live-execution outcome in the manifest status gate.

        A package with any failed_bug step is downgraded to validation_failed so
        the runtime loader does not surface a step that crashes on use. Skipped
        (precondition/interaction) steps never block — they are validated by
        static checks and recorded for transparency.
        """
        cli_dir = result.get("cli_dir")
        if not cli_dir:
            return
        manifest_path = os.path.join(cli_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            return
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["live_validation"] = {
                "passed": not failed_keys,
                "live_valid_count": n_valid,
                "failed_steps": sorted(failed_keys),
                "skipped_count": n_skipped,
            }
            if failed_keys:
                # Preserve the generation status once so a later clean pass can
                # restore it rather than leaving the package stuck as failed.
                if manifest.get("status") != "validation_failed":
                    manifest["_pre_live_status"] = manifest.get("status")
                manifest["status"] = "validation_failed"
            else:
                # Clean live pass: restore the generation status if a prior live
                # run had downgraded it.
                if manifest.get("status") == "validation_failed":
                    manifest["status"] = manifest.pop(
                        "_pre_live_status", "validated_with_warnings"
                    )
                else:
                    manifest.pop("_pre_live_status", None)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            logger.debug("Failed to persist live validation status", exc_info=True)

    def _finalizeCliValidation(self, result, live_results):
        """Finalize the CLI generation UI after validation (static + live).

        Args:
            result: The original generation result dict from the analyzer.
            live_results: Dict from live_validate_templates, or {} if live
                validation was skipped. Only status == 'failed_bug' counts as a
                failure; 'skipped_*' statuses do not block validation.
        """
        failed_keys = [
            tpl_key for tpl_key, tpl_res in live_results.items()
            if tpl_res.get("status") == "failed_bug"
        ]
        n_valid = sum(
            1 for r in live_results.values() if r.get("status") == "live_valid"
        )
        n_skipped = sum(
            1 for r in live_results.values()
            if str(r.get("status", "")).startswith("skipped")
        )
        # Only stamp a live-validation outcome when live execution actually ran;
        # a background-only generation finalize (live_results == {}) must not claim
        # a live result.
        if live_results:
            self._persistLiveValidationStatus(result, failed_keys, n_valid, n_skipped)

        if failed_keys:
            self._cliStatusLabel.setText(
                f"Live validation: {len(failed_keys)} step(s) failing"
            )
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
            self._cliProgressDisplay.append(
                "Live validation could not repair: " + ", ".join(sorted(failed_keys))
            )
            self._cliResultGroup.setVisible(True)
            self._cliGeneratorRunning = False
            self._analyzeGenerateButton.setEnabled(True)
            self._refreshCliActionButtons()
            return

        # All validations passed
        has_live = bool(live_results)
        status_text = "Live-Validated ✓" if has_live else "Validated"
        self._cliStatusLabel.setText(status_text)
        self._cliStatusLabel.setStyleSheet("font-weight: bold; color: green;")
        self._cliResultGroup.setVisible(True)
        self._cliGeneratorRunning = False
        self._analyzeGenerateButton.setEnabled(True)
        self._refreshCliActionButtons()

        manifest = result.get("manifest", {})
        stages = manifest.get("stages", [])
        live_info = (
            f" ({n_valid} steps live-executed, {n_skipped} skipped)" if has_live else ""
        )
        self._cliResultSummary.setText(
            f"Generated CLI for {manifest.get('extension_name', '?')} "
            f"(workflow steps: {', '.join(stages)}).{live_info} "
            f"Saved to: {result.get('cli_dir', '?')}"
        )
        self._cliProgressDisplay.append(f"CLI generation complete: {status_text}!")

        # Refresh the extension selector to show updated status, preserving selection
        ext_name = manifest.get("extension_name", "")
        self._onRefreshExtensionsClicked()
        if ext_name:
            for i in range(self._extensionSelector.count):
                if ext_name in self._extensionSelector.itemText(i):
                    self._extensionSelector.setCurrentIndex(i)
                    break

    def _handleCliRevisionComplete(self, result):
        """Handle automatic CLI revision completion (generation-failure fallback)."""
        self._cliGeneratorRunning = False
        self._analyzeGenerateButton.setEnabled(True)

        if result.get("success"):
            self._cliStatusLabel.setText("Revised & Validated")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: green;")
            self._cliProgressDisplay.append(
                f"Revision succeeded after {result.get('attempts', '?')} attempts."
            )
        else:
            self._cliStatusLabel.setText("Revision Failed")
            self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
            self._cliProgressDisplay.append(
                f"Revision failed: {result.get('error', 'unknown')}"
            )
        self._refreshCliActionButtons()

    def _handleCliRepairComplete(self, repair):
        """Resume after the LLM template repair (Repair button), then live-validate.

        The off-thread `repair_generated_cli` has rewritten templates from recorded
        runtime API errors + the user's function-error descriptions. Now live-execute
        the rewritten templates to catch any API crash (introduced or pre-existing
        in a precondition-free step) and auto-repair it, then finalize.
        """
        result = getattr(self, "_cliRepairResult", None)
        if repair.get("repaired"):
            self._cliProgressDisplay.append(
                f"Repair rewrote {len(repair.get('repaired', []))} template(s) "
                f"(recorded API errors: {repair.get('runtime_error_count', 0)}, "
                f"function errors: {repair.get('function_error_count', 0)}). "
                "Live-validating…"
            )
            if getattr(self, "_cliFunctionErrorInput", None):
                self._cliFunctionErrorInput.clear()
        else:
            self._cliProgressDisplay.append(
                f"Repair made no template changes ({repair.get('error', 'nothing to fix')})."
            )
        self._cliGeneratorRunning = False
        if result:
            self._runCliLiveValidation(result, outer_iter=0)
        else:
            self._analyzeGenerateButton.setEnabled(True)
            self._refreshCliActionButtons()

    def _handleCliError(self, error_msg):
        """Handle CLI generator error on the main thread."""
        self._cliGeneratorRunning = False
        self._analyzeGenerateButton.setEnabled(True)
        self._cliStatusLabel.setText("Error")
        self._cliStatusLabel.setStyleSheet("font-weight: bold; color: red;")
        self._cliProgressDisplay.append(f"ERROR: {error_msg}")
        self._refreshCliActionButtons()

    def enter(self):
        if (hasattr(self, 'chatHistory') and self.chatHistory is not None and
            self.logic and not self.logic.hasApiKey()):
            self.appendToChat("System", "Please configure your API key in Settings before using the agent.")

    def exit(self):
        pass
