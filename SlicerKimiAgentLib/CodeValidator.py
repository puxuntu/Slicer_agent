"""
CodeValidator - Validates Python code for safety before execution.

Implements AST-based static analysis to detect and block unsafe operations.
"""

import ast
import builtins
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CodeValidator:
    """
    Validates Python code for safe execution within Slicer.
    
    Security layers:
    1. AST parsing to detect dangerous imports and calls
    2. Import whitelist/blacklist checking
    3. Function call analysis
    4. File operation restrictions
    """
    
    # Modules that are completely blocked
    BLOCKED_MODULES = {
        'os', 'subprocess', 'sys', 'socket', 'urllib', 'urllib2', 'http',
        'ftplib', 'telnetlib', 'ctypes', 'mmap', 'resource', 'signal',
        'pty', 'pickle', 'cPickle', 'shelve', 'marshal', 'imp',
        'importlib._bootstrap', 'importlib.machinery',
    }
    
    # Specific functions that are blocked (module.function)
    BLOCKED_FUNCTIONS = {
        'os.system', 'os.popen', 'os.spawn', 'os.exec', 'os.fork', 'os.kill',
        'os.remove', 'os.rmdir', 'os.unlink', 'os.rename', 'os.chmod',
        'eval', 'exec', 'execfile', 'compile', '__import__',
        'open', 'file', 'input', 'raw_input',
        'getattr', 'setattr', 'delattr',
        'globals', 'locals', 'vars', 'dir',
    }
    
    # Allowed modules for Slicer operations
    ALLOWED_MODULES = {
        'slicer', 'vtk', 'qt', 'ctk', 'numpy', 'SimpleITK', 'sitk',
        'math', 'random', 'datetime', 'collections', 'itertools',
        'functools', 'json', 're', 'string', 'hashlib', 'copy',
        'typing', 'abc', 'enum', 'decimal', 'fractions', 'numbers',
        'statistics', 'csv', 'io', 'base64', 'binascii', 'struct',
    }
    
    # Destructive operations that require confirmation
    DESTRUCTIVE_PATTERNS = {
        'RemoveNode', 'Clear', 'Delete', 'Remove',
        'saveNode', 'write', 'export',
    }
    
    def __init__(self):
        """Initialize the code validator."""
        self.warnings = []
        self.errors = []
        
    def validate(self, code: str) -> Dict:
        """
        Validate Python code for safe execution.
        
        Args:
            code: Python code string to validate
            
        Returns:
            Dictionary with:
                - valid: bool, whether code passes validation
                - reason: str, explanation if invalid
                - warnings: List[str], cautionary notes
                - requires_confirmation: bool, if destructive operations detected
                - destructive_ops: List[str], detected destructive operations
        """
        self.warnings = []
        self.errors = []
        destructive_ops = []
        
        # Check for empty code
        if not code or not code.strip():
            return {
                "valid": False,
                "reason": "Empty code",
                "warnings": [],
                "requires_confirmation": False,
                "destructive_ops": [],
            }
            
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "valid": False,
                "reason": f"Syntax error: {e}",
                "warnings": [],
                "requires_confirmation": False,
                "destructive_ops": [],
            }
            
        # Analyze AST
        visitor = CodeAnalysisVisitor(
            blocked_modules=self.BLOCKED_MODULES,
            blocked_functions=self.BLOCKED_FUNCTIONS,
            allowed_modules=self.ALLOWED_MODULES,
        )
        visitor.visit(tree)
        
        self.errors.extend(visitor.errors)
        self.warnings.extend(visitor.warnings)
        
        # Check for destructive operations
        destructive_ops = self._detectDestructiveOperations(code, tree)
        requires_confirmation = len(destructive_ops) > 0
        
        # Determine validity
        is_valid = len(self.errors) == 0
        
        reason = None
        if self.errors:
            reason = "; ".join(self.errors)
            
        return {
            "valid": is_valid,
            "reason": reason,
            "warnings": self.warnings,
            "requires_confirmation": requires_confirmation,
            "destructive_ops": destructive_ops,
        }
        
    def _detectDestructiveOperations(self, code: str, tree: ast.AST) -> List[str]:
        """Detect potentially destructive operations in the code."""
        destructive_ops = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check function name against destructive patterns
                func_name = self._getFunctionName(node.func)
                if func_name:
                    for pattern in self.DESTRUCTIVE_PATTERNS:
                        if pattern in func_name:
                            destructive_ops.append(func_name)
                            break
                            
        return list(set(destructive_ops))  # Deduplicate
        
    def _getFunctionName(self, node) -> Optional[str]:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None
        
    def sanitize(self, code: str) -> str:
        """
        Attempt to sanitize code by removing/replacing dangerous constructs.
        
        Args:
            code: Original code
            
        Returns:
            Sanitized code (or original if sanitization not possible)
        """
        # For now, just return original - sanitization is complex
        # Could implement: comment out imports, wrap in try-except, etc.
        return code


class CodeAnalysisVisitor(ast.NodeVisitor):
    """
    AST visitor that analyzes code for security issues.
    """
    
    def __init__(self, blocked_modules: Set[str], blocked_functions: Set[str], allowed_modules: Set[str]):
        self.blocked_modules = blocked_modules
        self.blocked_functions = blocked_functions
        self.allowed_modules = allowed_modules
        self.errors = []
        self.warnings = []
        self.imported_modules = set()
        
    def visit_Import(self, node):
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self._checkModule(module_name, node.lineno)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Check from...import statements."""
        if node.module:
            module_name = node.module.split('.')[0]
            self._checkModule(module_name, node.lineno)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        """Check function calls."""
        func_name = self._getCallName(node.func)
        
        if func_name:
            # Check against blocked functions
            if func_name in self.blocked_functions:
                self.errors.append(f"Blocked function call: {func_name} (line {node.lineno})")
            elif any(func_name.startswith(bf + '.') for bf in self.blocked_functions):
                self.errors.append(f"Blocked function call: {func_name} (line {node.lineno})")
                
            # Check for eval/exec with string arguments
            if func_name in ('eval', 'exec'):
                self.errors.append(f"Dangerous function call: {func_name} (line {node.lineno})")
                
        self.generic_visit(node)
        
    def visit_Expression(self, node):
        """Check expressions (catch bare calls)."""
        self.generic_visit(node)
        
    def _checkModule(self, module_name: str, lineno: int):
        """Check if a module is allowed."""
        self.imported_modules.add(module_name)
        
        if module_name in self.blocked_modules:
            self.errors.append(f"Blocked module import: {module_name} (line {lineno})")
        elif module_name not in self.allowed_modules:
            # Not in explicit allow list - warn but don't block
            # This allows user-defined modules and Slicer extensions
            if not module_name.startswith('slicer') and not module_name.startswith('vtk'):
                self.warnings.append(f"Unrecognized module: {module_name} (line {lineno})")
                
    def _getCallName(self, node) -> Optional[str]:
        """Get the full name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return None
