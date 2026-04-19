"""
SkillTools - Tool implementations for searching the Slicer skill.

Provides cross-platform search functionality:
- Windows: Uses PowerShell/findstr (if available) or Python fallback
- Linux/macOS: Uses grep/find
"""

import os
import re
import json
import subprocess
import platform
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to bundled ripgrep binary (Windows)
_RG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "rg.exe")


class SkillToolExecutor:
    """
    Executes tool calls for searching the Slicer skill.
    Handles platform differences between Windows and Unix.
    """
    
    def __init__(self, skill_path: str):
        self.skill_path = skill_path
        self.platform = platform.system().lower()  # 'windows', 'linux', 'darwin'
    
    def _relativize(self, path: str) -> str:
        """Convert an absolute path back to a relative forward-slash path."""
        try:
            rel = os.path.relpath(path, self.skill_path)
        except ValueError:
            # On Windows, relpath can fail if paths are on different drives
            rel = path
        return rel.replace(os.sep, '/')

    def execute(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool (Grep, ReadFile, Glob)
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        import time
        start = time.time()
        
        if tool_name == "Grep":
            result = self._grep(arguments.get("pattern", ""), arguments.get("path", ""))
        elif tool_name == "ReadFile":
            result = self._readfile(
                arguments.get("path", ""),
                arguments.get("start_line"),
                arguments.get("end_line")
            )
        elif tool_name == "Glob":
            result = self._glob(arguments.get("pattern", ""), arguments.get("path", ""))
        else:
            return {"error": f"Unknown tool: {tool_name}"}
        
        elapsed = time.time() - start
        if isinstance(result, dict):
            result["_tool_timing"] = f"{elapsed:.3f}s"
        
        # Normalize absolute paths in the result back to relative forward-slash paths
        if isinstance(result, dict):
            if "path" in result:
                result["path"] = self._relativize(result["path"])
            if "results" in result and isinstance(result["results"], list):
                for item in result["results"]:
                    if isinstance(item, dict) and "file" in item:
                        item["file"] = self._relativize(item["file"])
            if "file" in result:
                result["file"] = self._relativize(result["file"])
        return result
    
    def _find_rg(self) -> Optional[str]:
        """Return path to ripgrep binary if available, else None."""
        # 1. Windows bundled binary
        if os.path.isfile(_RG_PATH):
            return _RG_PATH
        # 2. System-installed rg (any platform: Linux, macOS, Windows with rg in PATH)
        try:
            result = subprocess.run(["rg", "--version"], capture_output=True, timeout=2)
            if result.returncode == 0:
                return "rg"
        except Exception:
            pass
        return None

    def _grep_rg(self, pattern: str, path: str) -> List[Dict]:
        """Universal ripgrep implementation (works on all platforms)."""
        results = []
        rg_exe = self._find_rg()
        if not rg_exe:
            return results

        try:
            cmd = [
                rg_exe,
                "-n",
                "-i",
                "-m", "20",
                "--max-columns", "200",
                "--no-heading",
                "--color", "never",
                pattern,
                path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ""

            if result.returncode in (0, 1):
                for line in stdout.strip().split('\n')[:20]:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r'^(.+?):(\d+):(.*)$', line)
                    if match:
                        results.append({
                            "file": match.group(1),
                            "line": int(match.group(2)),
                            "content": match.group(3)[:200],
                        })
        except Exception as e:
            logger.warning(f"ripgrep failed: {e}")
        return results

    def _grep(self, pattern: str, path: str) -> Dict:
        """
        Search for pattern in files.
        Cross-platform implementation: ripgrep > platform-native > Python fallback.
        """
        # Normalize path
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)

        if not os.path.exists(path):
            return {"error": f"Path not found: {path}"}

        results = []

        # Priority 1: ripgrep (fastest, all platforms)
        try:
            results = self._grep_rg(pattern, path)
        except Exception as e:
            logger.warning(f"rg search failed: {e}")

        # Priority 2: platform-native tools
        if not results:
            try:
                if self.platform == "windows":
                    results = self._grep_windows_native(pattern, path)
                else:
                    results = self._grep_unix(pattern, path)
            except Exception as e:
                logger.warning(f"Platform-native grep failed: {e}")

        # Priority 3: pure Python fallback
        if not results:
            results = self._grep_python(pattern, path)

        return {
            "tool": "Grep",
            "pattern": pattern,
            "path": path,
            "results": results[:20],
            "count": len(results)
        }

    def _grep_windows_native(self, pattern: str, path: str) -> List[Dict]:
        """Windows fallback: PowerShell Select-String."""
        results = []
        escaped_pattern = pattern.replace("'", "''")
        ps_cmd = rf'Select-String -Path "{path}\*" -Pattern \'{escaped_pattern}\' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 20 | ForEach-Object {{ "{{0}}:{{1}}:{{2}}" -f $_.Path,$_.LineNumber,$_.Line }}'

        try:
            result = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True,
                timeout=30
            )
            stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ""
            if result.returncode == 0 or stdout:
                for line in stdout.strip().split('\n'):
                    line = line.strip()
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            results.append({
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "content": parts[2][:200]
                            })
        except Exception as e:
            logger.warning(f"PowerShell grep failed: {e}")
        return results

    def _grep_unix(self, pattern: str, path: str) -> List[Dict]:
        """Unix fallback: system grep."""
        results = []
        try:
            result = subprocess.run(
                ["grep", "-r", "-n", "-i", pattern, path],
                capture_output=True,
                timeout=30
            )
            stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ""
            if result.returncode in [0, 1]:
                lines = stdout.strip().split('\n')[:20]
                for line in lines:
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            results.append({
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "content": parts[2][:200]
                            })
        except Exception as e:
            logger.warning(f"Unix grep failed: {e}")
        return results
    
    def _grep_python(self, pattern: str, path: str) -> List[Dict]:
        """Pure Python grep implementation (fallback)."""
        results = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(('.md', '.py', '.h', '.cxx', '.cpp', '.txt')):
                        files.append(os.path.join(root, filename))
                if len(files) > 100:  # Limit files
                    break
        
        for filepath in files[:50]:  # Limit files
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append({
                                "file": filepath,
                                "line": i,
                                "content": line.strip()[:200]  # Limit length
                            })
                            if len(results) >= 20:
                                break
                        if i > 1000:  # Limit lines per file
                            break
            except:
                continue
        
        return results
    
    def _readfile(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict:
        """Read the full file content. Line range parameters are ignored to ensure the LLM always receives complete context in the current turn."""
        # Normalize path
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)
        
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}
        
        if not os.path.isfile(path):
            return {"error": f"Path is not a file: {path}"}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {
                "tool": "ReadFile",
                "path": path,
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    def _glob(self, pattern: str, path: str) -> Dict:
        """Find files matching pattern."""
        # Normalize path
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)
        
        if not os.path.exists(path):
            return {"error": f"Path not found: {path}"}
        
        results = []
        
        try:
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if re.search(pattern.replace('*', '.*'), filename, re.IGNORECASE):
                        results.append(os.path.join(root, filename))
                        if len(results) >= 20:
                            break
                if len(results) >= 20:
                    break
        except Exception as e:
            return {"error": f"Failed to glob: {str(e)}"}
        
        return {
            "tool": "Glob",
            "pattern": pattern,
            "path": path,
            "results": results,
            "count": len(results)
        }


# Tool definitions for AI
def get_skill_tools() -> List[Dict]:
    """
    Get tool definitions for the AI.
    These are passed to the API to register available tools.
    
    IMPORTANT: Search the skill as needed for the task, but avoid repeated
    searches for the same topic. Once you find the relevant API, provide the
    final response with Python code.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "Grep",
                "description": "Quick search for API names or keywords. Search as needed for the task, but avoid repeated searches for the same topic. Once you find the relevant API, write the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for (e.g., 'loadVolume', 'downloadMRHead')"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path within skill (e.g., 'slicer-source/Docs/developer_guide/script_repository')"
                        }
                    },
                    "required": ["pattern", "path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ReadFile",
                "description": "Read the complete content of a file from the skill knowledge base. Always returns the full file so you have complete context. Use after Grep to confirm exact API signatures and usage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to file (e.g., 'slicer-source/Docs/developer_guide/script_repository/volumes.md')"
                        }
                    },
                    "required": ["path"]
                }
            }
        },

    ]
