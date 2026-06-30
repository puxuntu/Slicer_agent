## PelvicFracturePlanning

1. [op=user_choice] In the "Input CT Volume" option, choose the Pelvic Volume.
2. [op=extension_op] Click "Run Step 1: Segment Pelvis" button.
3. [op=extension_op] Click "Run Step 2: Segment Fractures" button.
4. [op=user_choice] In the "Untick any fragments to exclude it from planning" section, untick these segments.
5. [op=extension_op] Click "Run Step 3: Generate Template" button.
6. [op=extension_op] Click "Run Step 4: Register _Reduce" button.
7. [op=branch_op] If further adjustments are required, tick the "Manually adjust a fragment" checkbox. If not, jump to step 12.
8. [op=user_choice] In the "Fragment" option, Choose which fragment needs adjustment in the "Fragment" selection box.
9. [op=user_interaction] Manually adjust the position and rotation of the selected fragment.
10. [op=branch_op] If further adjustments are required, jump to step 8. If not, jump to step 11.
11. [op=extension_op] Click the "Apply adjustments" button.
12. [op=extension_op] Click the "Run Step 5: Plan Screws" button.
13. [op=branch_op] If further adjustments are required, tick the "Edit Screw trajectories" checkbox. If not, stop here.
14. [op=user_interaction] Manually adjust the position and rotation of the screw trajectories.
15. [op=extension_op] Click the "Regenerate screws from edited lines" button.
16. [op=extension_op] Untick the "Edit Screw trajectories" checkbox.