### Interactive Workflow: BoneReconstructionPlanner

**Tool name:** `BoneReconstructionPlanner`
**Type:** Guided interactive workflow

This tool orchestrates a multi-step workflow where some steps require the user to
perform 3D interactions (drawing curves, positioning planes, placing fiducials).
Execute steps sequentially, ONE STEP PER TURN. After each interactive step, relay instructions to the user
and wait for them to complete the interaction before proceeding.

**Workflow Steps:**
1. `create_bone_models` [automated] — Generate 3D surface models from mandible and fibula segmentations using the module's Logic.
2. `add_mandibular_curve` [interactive] — Place a markup curve along the ideal mandibular arch to guide initial cut plane positioning.
   - Interaction: curve
   - Tell user: Switch to curve placement mode. Click on the mandibular segmentation to define a curve. Use at least three points for an accurate curve representation.
3. `add_cut_plane` [interactive] — Place a markup plane on the mandible model; adjacent planes define bone pieces for reconstruction.
   - Interaction: plane
   - Tell user: Switch to persistent plane placement mode. Click three points on the mandibular model to define each cut plane. Continue placing planes until all resections are defined.
4. `add_fibula_line` [interactive] — Draw a markup line along the fibula diaphysis (distal to proximal) to define the fibula axis.
   - Interaction: line
   - Tell user: Switch to line placement mode. Click two points along the fibula segmentation to define the fibula axis.
5. `center_fibula_line` [automated] — Automatically adjust the fibula line to coincide with the anatomical axis of the fibula model.
6. `update_fibula_planes` [automated] — Compute fibula cut planes from mandible planes and fibula axis, create bone pieces, and transform them to the mandible position.
7. `hard_vsp_update` [automated] — Force a complete recomputation of the virtual surgical plan outputs without modifying input objects.
8. `create_reconstruction_3d_model` [automated] — Generate a combined STL model of the neomandible (mandible plus fibula pieces) for 3D printing.
9. `create_dental_implant_fiducials` [optional] — Place fiducial points on neomandible pieces where dental implant cylinders should be positioned.
10. `create_dental_implant_cylinders` [optional] — Generate cylindrical models representing dental implants at the fiducial locations on the neomandible.
11. `create_plate_curve` [optional] — Draw a markup curve along the neomandible surface to define the path of the custom titanium plate.
12. `create_custom_plate` [optional] — Generate a 3D model of a patient-specific titanium plate following the plate curve.
13. `create_miter_boxes` [optional] — Create miter box models for fibula osteotomies; each box includes a slit for the saw blade.
14. `create_fibula_screw_fiducials` [optional] — Place fiducial points on the fibula guide base where screw holes will be drilled.
15. `create_fibula_screw_cylinders` [optional] — Generate cylinder models for screw holes on the fibula guide base at the fiducial locations.
16. `boolean_fibula_guide` [optional] — Perform Boolean subtraction to combine screw holes, dental implant cylinders, and miter boxes into a final fibula surgical guide prototype.
17. `create_mandible_saw_boxes` [optional] — Create saw box models for mandible osteotomies at the first and last cut planes.
18. `create_mandible_screw_fiducials` [optional] — Place fiducial points on the mandible guide base where screw holes will be drilled.
19. `create_mandible_screw_cylinders` [optional] — Generate cylinder models for screw holes on the mandible guide base at the fiducial locations.
20. `boolean_mandible_guide` [optional] — Perform Boolean subtraction to combine screw holes, bridge model, and saw boxes into a final mandible surgical guide prototype.

**Protocol:**
1. Call `BoneReconstructionPlanner` with `workflow_step='create_bone_models'` and `user_action='start'` to begin
2. For **automated** steps: output the returned `code` verbatim in a ```python block. Then call the next step.
3. For **interactive** steps: output the returned `pre_code` verbatim in a ```python block. Relay instructions to the user. Wait for them to click 'Done'.
4. For **optional** steps: ask user if they want to proceed. If yes, call with `user_action='start'`. If no, call with `user_action='skip'`.
5. After each step completes, call the tool with the NEXT step's `step_id` and `user_action='start'`.
6. Continue until all steps are done.

**CRITICAL RULES:**
- Execute ONE step per turn. Do NOT call multiple steps in a single turn.
- Do NOT skip automated steps. Their code MUST be output and executed.
- Always start from step 1 (`create_bone_models`) and proceed in order.