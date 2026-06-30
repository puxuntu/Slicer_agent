# --- PelvicFracturePlanning: If further adjustments are required, tick the "Manually adjust a fragment" checkbox. If not, jump to step 12. ---
import slicer
# precondition:begin
# Ensure the extension module is active so module.enter() has run.
_active_module_name = slicer.util.selectedModule()
if _active_module_name != 'PelvicFracturePlanning':
    try:
        slicer.util.selectModule('PelvicFracturePlanning')
    except Exception as _module_enter_error:
        print(f"Warning: could not activate module 'PelvicFracturePlanning': {_module_enter_error}")
# precondition:end

# Drive the extension's own widget handler on the live module widget:
# it performs the full action (reads selected nodes, creates the
# output nodes downstream steps depend on, toggles dependent UI).
_widget = None
try:
    _widget = slicer.util.getModuleWidget('PelvicFracturePlanning')
except Exception:
    _widget = None
if _widget is None:
    try:
        _widget = slicer.modules.pelvicfractureplanning.widgetRepresentation().self()
    except Exception:
        _widget = None
if _widget is None:
    raise RuntimeError("Could not obtain the PelvicFracturePlanning module widget for 'chkManualAdjust'.")
if not hasattr(_widget, 'onManualAdjustToggled'):
    raise RuntimeError("PelvicFracturePlanning widget has no handler 'onManualAdjustToggled' for 'chkManualAdjust'; regenerate the CLI.")
# Set the control's checked state (signals blocked to avoid a
# double-fire), then invoke the handler once. The handler may read
# the widget state (no arg) or accept the new bool — try the bool.
_ctrl = None
try:
    _ctrl = _widget.ui.chkManualAdjust
except Exception:
    _ctrl = None
if _ctrl is not None:
    try:
        _ctrl.blockSignals(True)
        _ctrl.checked = True
        _ctrl.blockSignals(False)
    except Exception:
        pass
try:
    _widget.onManualAdjustToggled(True)
except TypeError:
    _widget.onManualAdjustToggled()
print("[PelvicFracturePlanning] Step 'cb_step_7': set 'chkManualAdjust' = True via onManualAdjustToggled.")

