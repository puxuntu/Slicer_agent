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
        
        if tool_name == "FindFile":
            result = self._findfile(arguments.get("pattern", ""), arguments.get("path", ""))
        elif tool_name == "SearchSymbol":
            result = self._search_symbol(
                arguments.get("pattern", ""),
                arguments.get("path", ""),
                arguments.get("type", "all")
            )
        elif tool_name == "Grep":
            result = self._grep(arguments.get("pattern", ""), arguments.get("path", ""))
        elif tool_name == "ReadFile":
            result = self._readfile(
                arguments.get("path", ""),
                arguments.get("query")
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
            # Legacy line-by-line results (FindFile / SearchSymbol / old Grep)
            if "results" in result and isinstance(result["results"], list):
                for item in result["results"]:
                    if isinstance(item, dict) and "file" in item:
                        item["file"] = self._relativize(item["file"])
            # Aggregated Grep results
            if "files" in result and isinstance(result["files"], list):
                for item in result["files"]:
                    if isinstance(item, dict) and "file" in item:
                        item["file"] = self._relativize(item["file"])
            if "representative_matches" in result and isinstance(result["representative_matches"], list):
                for item in result["representative_matches"]:
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

    def _grep_rg_aggregate(self, pattern: str, path: str) -> Dict:
        """Aggregate ripgrep: returns per-file summary instead of line-by-line matches."""
        rg_exe = self._find_rg()
        if not rg_exe:
            return {"error": "ripgrep not found"}

        # Step 1: Count matches per file
        count_cmd = [
            rg_exe, "-i", "--count-matches", "--no-heading",
            pattern, path,
        ]
        try:
            count_result = subprocess.run(count_cmd, capture_output=True, timeout=15)
        except Exception as e:
            return {"error": f"ripgrep count failed: {e}"}

        file_hits = {}
        total_hits = 0
        if count_result.returncode in (0, 1):
            stdout = count_result.stdout.decode('utf-8', errors='ignore')
            for line in stdout.strip().split('\n'):
                if not line:
                    continue
                if ':' in line:
                    file_path, count_str = line.rsplit(':', 1)
                    try:
                        count = int(count_str)
                        if count > 0:
                            rel_path = self._relativize(file_path)
                            file_hits[rel_path] = count
                            total_hits += count
                    except ValueError:
                        pass

        # Step 2: Get representative matches from top files
        sorted_files = sorted(file_hits.items(), key=lambda x: x[1], reverse=True)
        representative = []

        for file_path, _ in sorted_files[:5]:
            abs_path = os.path.join(self.skill_path, file_path) if not os.path.isabs(file_path) else file_path
            sample_cmd = [
                rg_exe, "-n", "-i", "-m", "3",
                "--max-columns", "300",
                "--no-heading", "--color", "never",
                pattern, abs_path,
            ]
            try:
                sample_result = subprocess.run(sample_cmd, capture_output=True, timeout=10)
                if sample_result.returncode in (0, 1):
                    for line in sample_result.stdout.decode('utf-8', errors='ignore').strip().split('\n'):
                        match = re.match(r'^(.+?):(\d+):(.*)$', line)
                        if match:
                            representative.append({
                                "file": self._relativize(match.group(1)),
                                "line": int(match.group(2)),
                                "content": match.group(3).strip(),
                            })
            except Exception:
                pass

        files_summary = [
            {"file": f, "hits": h}
            for f, h in sorted_files[:20]
        ]

        return {
            "total_hits": total_hits,
            "total_files": len(file_hits),
            "files": files_summary,
            "representative_matches": representative[:10],
        }

    def _grep(self, pattern: str, path: str) -> Dict:
        """
        Search for pattern in files using ripgrep.
        Returns aggregated summary (per-file hit counts + representative matches).
        """
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)

        if not os.path.exists(path):
            return {"error": f"Path not found: {path}"}

        if not self._find_rg():
            return {
                "error": (
                    "ripgrep (rg) is not installed. "
                    "Please install ripgrep and ensure it is in your PATH. "
                    "Download: https://github.com/BurntSushi/ripgrep#installation"
                )
            }

        result = self._grep_rg_aggregate(pattern, path)
        if "error" in result:
            return result

        return {
            "tool": "Grep",
            "pattern": pattern,
            "path": self._relativize(path),
            **result,
        }


    
    def _readfile(self, path: str, query: Optional[str] = None) -> Dict:
        """Read file content with smart slicing for large files."""
        # Normalize path
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)
        
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}
        
        if not os.path.isfile(path):
            return {"error": f"Path is not a file: {path}"}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            if total_lines < 500:
                content = ''.join(lines)
                strategy = "full"
            else:
                is_markdown = path.lower().endswith('.md')
                if query:
                    if is_markdown:
                        content = self._slice_markdown_by_query(lines, query)
                        strategy = "markdown_heading"
                    else:
                        content = self._slice_by_grep_context(lines, query)
                        strategy = "grep_context"
                else:
                    content = ''.join(lines[:500]) + "\n... [file truncated: provide query to locate specific section] ..."
                    strategy = "truncated"
            
            return {
                "tool": "ReadFile",
                "path": path,
                "content": content,
                "total_lines": total_lines,
                "strategy": strategy,
            }
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}

    def _slice_markdown_by_query(self, lines: List[str], query: str) -> str:
        """Slice markdown by heading boundaries, keeping sections that match query."""
        query_lower = query.lower()
        sections = []
        current_section = []
        current_heading = ""
        
        for line in lines:
            if re.match(r'^#{1,6}\s+', line):
                if current_section:
                    section_text = ''.join(current_section)
                    if query_lower in section_text.lower():
                        sections.append(section_text)
                current_heading = line
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            section_text = ''.join(current_section)
            if query_lower in section_text.lower():
                sections.append(section_text)
        
        if sections:
            return '\n\n... [other sections omitted] ...\n\n'.join(sections)
        else:
            return ''.join(lines[:300]) + "\n... [no matching section found, truncated] ..."

    def _slice_by_grep_context(self, lines: List[str], query: str) -> str:
        """Find query in lines and extract ±100 line context blocks."""
        query_lower = query.lower()
        match_lines = []
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                match_lines.append(i)
        
        if not match_lines:
            return ''.join(lines[:300]) + "\n... [query not found, truncated] ..."
        
        CONTEXT = 100
        blocks = []
        for line_no in match_lines:
            start = max(0, line_no - CONTEXT)
            end = min(len(lines), line_no + CONTEXT + 1)
            blocks.append((start, end))
        
        merged = []
        for start, end in sorted(blocks):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        parts = []
        for i, (start, end) in enumerate(merged):
            if i > 0:
                skipped = merged[i][0] - merged[i-1][1]
                parts.append(f"\n... [{skipped} lines skipped] ...\n")
            parts.append(''.join(lines[start:end]))
        
        return ''.join(parts)
    
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

    def _findfile(self, pattern: str, path: str) -> Dict:
        """Search for files by name pattern (case-insensitive)."""
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)
        
        if not os.path.exists(path):
            return {"error": f"Path not found: {path}"}
        
        results = []
        regex = re.compile(pattern.replace('*', '.*').replace('?', '.'), re.IGNORECASE)
        
        try:
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if regex.search(filename):
                        results.append(os.path.join(root, filename))
                        if len(results) >= 20:
                            break
                if len(results) >= 20:
                    break
        except Exception as e:
            return {"error": f"Failed to find files: {str(e)}"}
        
        return {
            "tool": "FindFile",
            "pattern": pattern,
            "path": path,
            "results": results,
            "count": len(results)
        }

    def _search_symbol(self, pattern: str, path: str, symbol_type: str = "all") -> Dict:
        """
        Search for symbol definitions (functions, classes, headings).
        Only matches definitions, not call sites or comments.
        """
        if not os.path.isabs(path):
            path = os.path.join(self.skill_path, path)
        
        if not os.path.exists(path):
            return {"error": f"Path not found: {path}"}
        
        results = []
        pattern_lower = pattern.lower()
        
        def _scan_file(filepath: str):
            ext = os.path.splitext(filepath)[1].lower()
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception:
                return
            
            if ext == '.py':
                # Python: def / class definitions
                for i, line in enumerate(lines, 1):
                    match = re.match(r'^(\s*)(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)', line)
                    if match:
                        name = match.group(3)
                        sym_type = "function" if match.group(2) == "def" else "class"
                        if symbol_type in ("all", sym_type) and pattern_lower in name.lower():
                            results.append({
                                "name": name,
                                "type": sym_type,
                                "file": filepath,
                                "line": i,
                                "signature": line.strip()[:120]
                            })
            
            elif ext in ('.cxx', '.cpp', '.h', '.hxx', '.c'):
                # C/C++: function/class/struct definitions (simplified regex)
                for i, line in enumerate(lines, 1):
                    # Match: return_type name(...) {  or  class/struct Name {
                    match = re.match(
                        r'^[\s\w:*&<>~]+\s+(\w+)\s*\([^)]*\)\s*\{'
                        r'|^(?:class|struct)\s+(\w+)'
                        r'|^\s*(\w+)\s*\([^)]*\)\s*;',
                        line
                    )
                    if match:
                        name = match.group(1) or match.group(2) or match.group(3)
                        if name:
                            if 'class' in line or 'struct' in line:
                                sym_type = "class"
                            else:
                                sym_type = "function"
                            if symbol_type in ("all", sym_type) and pattern_lower in name.lower():
                                results.append({
                                    "name": name,
                                    "type": sym_type,
                                    "file": filepath,
                                    "line": i,
                                    "signature": line.strip()[:120]
                                })
            
            elif ext == '.md':
                # Markdown: headings as symbols
                for i, line in enumerate(lines, 1):
                    match = re.match(r'^(#{1,6})\s+(.+)$', line)
                    if match:
                        name = match.group(2).strip()
                        level = len(match.group(1))
                        if symbol_type in ("all", "heading") and pattern_lower in name.lower():
                            results.append({
                                "name": name,
                                "type": f"heading_h{level}",
                                "file": filepath,
                                "line": i,
                                "signature": line.strip()[:120]
                            })
        
        if os.path.isfile(path):
            _scan_file(path)
        else:
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ('.py', '.cxx', '.cpp', '.h', '.hxx', '.c', '.md'):
                        _scan_file(os.path.join(root, filename))
                        if len(results) >= 20:
                            break
                if len(results) >= 20:
                    break
        
        return {
            "tool": "SearchSymbol",
            "pattern": pattern,
            "type": symbol_type,
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
                "name": "FindFile",
                "description": "Search for files by name pattern (case-insensitive, supports * and ? wildcards). Use this as the first step when you need to locate a specific file or type of file within the skill knowledge base. Returns up to 20 matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "File name pattern (e.g., 'volumes*', '*.py', 'qSlicerVolume*')"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path within skill to search under (e.g., 'slicer-source')"
                        }
                    },
                    "required": ["pattern", "path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "SearchSymbol",
                "description": "Search for symbol definitions (functions, classes, markdown headings) by name. Only matches actual definitions, not call sites or comments. Supports Python, C/C++, and Markdown files. Use this when you want to find where a specific function or class is defined.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Symbol name pattern (e.g., 'loadVolume', 'vtkMRML*', 'MyClass')"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path within skill (e.g., 'slicer-source/Libs/MRML')"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["all", "function", "class", "heading"],
                            "description": "Filter by symbol type: 'all' (default), 'function', 'class', or 'heading' (markdown only)"
                        }
                    },
                    "required": ["pattern", "path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Grep",
                "description": "Full-text search across files. Returns an aggregated summary (per-file hit counts + representative matches), not line-by-line results. Use after FindFile or SearchSymbol to confirm specific usage patterns.",
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
                "description": "Read the content of a file from the skill knowledge base. For files under 500 lines, returns the full content. For larger files, provide a 'query' parameter to extract only relevant sections (e.g., the function or heading matching your keyword).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to file (e.g., 'slicer-source/Docs/developer_guide/script_repository/volumes.md')"
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional keyword to locate a specific section in large files (>500 lines). For markdown files, matches headings. For code files, matches function names or keywords."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
    ]
