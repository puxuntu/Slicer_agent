### Interactive Workflow: PelvicFracturePlanning

**Tool name:** `PelvicFracturePlanning`
**Type:** Guided interactive workflow

This tool orchestrates a multi-step workflow where some steps require the user to
perform 3D interactions (drawing curves, positioning planes, placing fiducials).
Execute steps sequentially, ONE STEP PER TURN. After each interactive step, relay instructions to the user
and wait for them to complete the interaction before proceeding.

**Workflow Steps:**
1. `cb_step_1` [user_choice] — In the "Input CT Volume" option, choose the Pelvic Volume.
   - Ask user: Choose the Pelvic Volume from the Input CT Volume selector.
2. `cb_step_2` [extension_op] — Click "Run Step 1: Segment Pelvis" button.
3. `cb_step_3` [extension_op] — Click "Run Step 2: Segment Fractures" button.
4. `cb_step_4` [user_choice] — In the "Untick any fragments to exclude it from planning" section, untick these segments.
   - Ask user: Which fracture segments should be excluded from planning?
5. `cb_step_5` [extension_op] — Click "Run Step 3: Generate Template" button.
6. `cb_step_6` [extension_op] — Click "Run Step 4: Register _Reduce" button.
7. `cb_step_7` [extension_op] — Click "Run Step 4: Register _Reduce" button.
8. `cb_step_8` [user_choice] — If further adjustments are required, tick the "Manually adjust a fragment" checkbox. If not, jump to step 10.
   - Ask user: Do you want to manually adjust a fragment?
9. `cb_step_9` [extension_op] — Choose which fragment needs adjustment in the "Fragment" selection box.
10. `cb_step_10` [user_interaction] — Manually adjust the position and rotation of the selected fragment.
   - Interaction: generic
11. `cb_step_11` [extension_op] — Click the "Apply adjustments" button.
12. `cb_step_12` [extension_op] — Click the "Plan Screws" button.
13. `cb_step_13` [user_choice] — If further adjustments are required, tick the "Edit Screw trajectories" checkbox. If not, stop here.
   - Ask user: Do you want to edit screw trajectories?
14. `cb_step_14` [user_interaction] — Manually adjust the position and rotation of the screw trajectories.
   - Interaction: generic
15. `cb_step_15` [extension_op] — Click the "Regenerate screw from edited lines" button.
16. `cb_step_16` [extension_op] — Untick the "Edit Screw trajectories" checkbox.

**Protocol:**
1. Call `PelvicFracturePlanning` with `workflow_step='cb_step_1'` and `user_action='start'` to begin
2. For **extension_op** and **slicer_op** steps: output the returned `code` verbatim in a ```python block. Then call the next step.
3. For **user_interaction** steps: output the returned `pre_code` verbatim in a ```python block. Relay instructions to the user. Wait for them to click 'Done'.
4. For **user_choice** steps: ask the returned question. After the user answers, call the same step with `user_action='choice_made'` and `choice_value`.
5. After each step completes, call the tool with the NEXT step's `step_id` and `user_action='start'`.
6. Continue until all steps are done.

**CRITICAL RULES:**
- Execute ONE step per turn. Do NOT call multiple steps in a single turn.
- Do NOT skip extension_op or slicer_op steps. Their code MUST be output and executed.
- Always start from step 1 (`cb_step_1`) and proceed in order.