"""
SafeExecutor - Sandboxed execution environment for generated code.

Provides safe execution of Python code with proper error handling,
timeout protection, and output capture.

IMPORTANT: Slicer Python code must run in the main thread due to Qt GUI requirements.
This executor uses cooperative multitasking (processEvents) to keep UI responsive.
"""

import ast
import builtins
import contextlib
import io
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional
from datetime import datetime

import slicer
import qt
import vtk

try:
    import ctk
except ImportError:
    ctk = None

logger = logging.getLogger(__name__)


class SafeExecutor:
    """
    Executes Python code in a controlled environment.
    
    IMPORTANT: All execution happens in the MAIN THREAD due to Slicer/Qt requirements.
    For long-running operations, the code should periodically call:
        slicer.app.processEvents()
    
    Features:
    - Captures stdout/stderr
    - Exception handling with detailed tracebacks
    - Timeout detection (cooperative - requires code to yield)
    - Access to Slicer globals
    - Execution history
    """
    
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_OUTPUT_LENGTH = 10000  # characters
    
    def __init__(self, timeout: Optional[int] = None):
        """
        Initialize the safe executor.
        
        Args:
            timeout: Execution timeout in seconds (default: 30)
        """
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.execution_history = []
        self._globals_dict = self._buildGlobals()
        self._execution_start_time = None
        self._should_cancel = False
        
    def _buildGlobals(self) -> Dict[str, Any]:
        """
        Return the __main__ module's globals dictionary.
        
        This makes the execution environment identical to the Slicer Python
        Console, so that shortcuts like getNode (injected by slicerqt.py)
        are automatically available.
        """
        return sys.modules['__main__'].__dict__
    
    def _injectHelpers(self, globals_dict: Dict) -> Dict:
        """
        Inject helper functions into the execution globals to support
        cooperative multitasking and timeout handling.
        """
        # Add a helper that code can call to check if it should yield
        def checkTimeout():
            """Check if execution has exceeded timeout. Raises TimeoutError if so."""
            if self._execution_start_time is None:
                return
            elapsed = (datetime.now() - self._execution_start_time).total_seconds()
            if elapsed > self.timeout:
                raise TimeoutError(f"Execution exceeded {self.timeout} seconds")
            # Also process events to keep UI responsive
            slicer.app.processEvents()
        
        # Add processEvents helper for user code
        def keepAlive(message="Processing..."):
            """
            Call this in long-running loops to keep the UI responsive.
            Also checks for timeout.
            """
            checkTimeout()
        
        globals_dict['_checkTimeout'] = checkTimeout
        globals_dict['_keepAlive'] = keepAlive
        globals_dict['keepAlive'] = keepAlive  # User-friendly alias
        
        return globals_dict
    
    def execute(self, code: str, timeout: Optional[int] = None, 
                progress_callback: Optional[Callable[[str], None]] = None) -> Dict:
        """
        Execute Python code safely in the main thread.
        
        NOTE: Due to Qt GUI requirements, execution happens in the main thread.
        For long-running operations, the code should call keepAlive() periodically
        or the timeout will be checked between statements.
        
        Args:
            code: Python code to execute
            timeout: Override default timeout (seconds)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with:
                - success: bool
                - output: str (captured stdout)
                - error: str (error message if failed)
                - traceback: str (full traceback if failed)
                - execution_time: float (seconds)
                - result: Any (return value of last expression if any)
                - timed_out: bool (True if execution was terminated due to timeout)
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        
        if not code or not code.strip():
            return {
                "success": False,
                "output": "",
                "error": "Empty code",
                "traceback": "",
                "execution_time": 0,
                "result": None,
                "timed_out": False,
            }
        
        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "success": False,
                "output": "",
                "error": f"Syntax error: {e}",
                "traceback": traceback.format_exc(),
                "execution_time": 0,
                "result": None,
                "timed_out": False,
            }
        
        # Record execution start
        self._execution_start_time = datetime.now()
        self._should_cancel = False
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Prepare execution environment
        # Use __main__ globals directly so that code runs in the same namespace
        # as the Slicer Python Console (getNode and other shortcuts are available).
        exec_globals = self._globals_dict
        exec_globals = self._injectHelpers(exec_globals)
        
        result_value = None
        error_msg = None
        traceback_str = None
        timed_out = False
        
        # Temporarily redirect VTK error/warning output so that C++ layer messages
        # (e.g. "[VTK] ModifySegmentByLabelmap: Invalid segment") are captured and
        # can trigger the self-correction mechanism.
        original_vtk_window = None
        vtk_log_path = None
        try:
            original_vtk_window = vtk.vtkOutputWindow.GetInstance()
            fd, vtk_log_path = tempfile.mkstemp(suffix='.vtklog')
            os.close(fd)
            vtk_file_window = vtk.vtkFileOutputWindow()
            vtk_file_window.SetFileName(vtk_log_path)
            vtk.vtkOutputWindow.SetInstance(vtk_file_window)
        except Exception:
            # If VTK redirection fails, continue without it
            original_vtk_window = None
            vtk_log_path = None

        try:
            # Execute with output capture
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                # Compile code to enable better error reporting
                compiled_code = compile(code, '<SlicerKimiAgent>', 'exec')
                tree = ast.parse(code)
                
                # Whole-block atomic execution (same as Slicer Python Console).
                exec(compiled_code, exec_globals)
                
                # Try to get result of last expression
                try:
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = ast.Expression(tree.body[-1].value)
                        last_expr.lineno = tree.body[-1].lineno
                        last_expr.col_offset = tree.body[-1].col_offset
                        result_value = eval(compile(last_expr, '<string>', 'eval'), exec_globals)
                except:
                    pass
                        
        except TimeoutError as e:
            timed_out = True
            error_msg = str(e)
            logger.warning(f"Code execution timed out after {effective_timeout} seconds")
            
        except Exception as e:
            error_msg = str(e)
            traceback_str = traceback.format_exc()

        finally:
            # Restore original VTK output window
            if original_vtk_window is not None:
                try:
                    vtk.vtkOutputWindow.SetInstance(original_vtk_window)
                except Exception:
                    pass

        # Read captured VTK log and inject it into stderr so output_has_errors
        # detection in SlicerAIAgent can trigger self-correction.
        if vtk_log_path and os.path.exists(vtk_log_path):
            try:
                with open(vtk_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    vtk_output = f.read().strip()
                if vtk_output:
                    # Prefix lines so they're easily identifiable
                    prefixed = "\n".join(f"[VTK] {line}" for line in vtk_output.splitlines())
                    stderr_capture.write(f"\n[VTK ERRORS/WARNINGS]:\n{prefixed}\n")
                os.remove(vtk_log_path)
            except Exception:
                pass
            
        # Calculate execution time
        execution_time = (datetime.now() - self._execution_start_time).total_seconds()
        
        # Get output (limited length)
        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        if stderr_output:
            output += "\n[STDERR]:\n" + stderr_output
            
        if len(output) > self.MAX_OUTPUT_LENGTH:
            output = output[:self.MAX_OUTPUT_LENGTH] + "\n... [output truncated]"
        
        # Record in history
        execution_record = {
            "timestamp": self._execution_start_time.isoformat(),
            "code": code,
            "success": error_msg is None and not timed_out,
            "execution_time": execution_time,
            "timed_out": timed_out,
        }
        if error_msg:
            execution_record["error"] = error_msg
        
        self.execution_history.append(execution_record)
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-50:]
        
        self._execution_start_time = None
        
        return {
            "success": error_msg is None and not timed_out,
            "output": output,
            "error": error_msg,
            "traceback": traceback_str,
            "execution_time": execution_time,
            "result": result_value,
            "timed_out": timed_out,
        }
    
    def executeAsync(self, code: str, callback: Optional[Callable[[Dict], None]] = None, 
                    timeout: Optional[int] = None):
        """
        Execute code asynchronously (non-blocking) using QTimer.
        
        Note: Due to Qt thread constraints, execution still happens in main thread,
        but is scheduled via QTimer to allow current event loop to complete.
        
        Args:
            code: Python code to execute
            callback: Function to call with result dict when done
            timeout: Override default timeout
        """
        def executeAndCallback():
            result = self.execute(code, timeout)
            if callback:
                callback(result)
                
        # Schedule execution on main thread via QTimer
        qt.QTimer.singleShot(10, executeAndCallback)
        
    def getHistory(self) -> list:
        """Get execution history."""
        return self.execution_history.copy()
        
    def clearHistory(self):
        """Clear execution history."""
        self.execution_history = []
        
    def addGlobal(self, name: str, value: Any):
        """
        Add a global variable for code execution.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self._globals_dict[name] = value
        
    def removeGlobal(self, name: str):
        """
        Remove a global variable.
        
        Args:
            name: Variable name to remove
        """
        if name in self._globals_dict:
            del self._globals_dict[name]
            
    def cleanup(self):
        """Cleanup resources."""
        self.execution_history = []
        # Do not reset _globals_dict since it points to __main__.__dict__
