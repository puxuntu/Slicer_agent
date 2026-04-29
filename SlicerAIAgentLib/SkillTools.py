"""
SkillTools - Tool implementations for searching the Slicer skill.

Provides cross-platform search functionality:
- All platforms: Uses ripgrep (rg) for fast aggregated search.
  Windows falls back to a bundled rg.exe; Linux/macOS require system rg in PATH.
"""

import os
import re
import json
import subprocess
import platform
import logging
from typing import List, Dict, Optional, Any
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
        self._tree_sitter_available = self._ensure_tree_sitter()
        self._read_history = {}  # path -> {"query": str, "strategy": str}
        self._vector_retriever = self._init_vector_retriever()
    
    def _ensure_tree_sitter(self) -> bool:
        """
        Check if tree-sitter and language parsers are available.
        If not, attempt to install them automatically.
        Returns True if available (either already installed or successfully installed).
        """
        try:
            import tree_sitter
            import tree_sitter_python
            import tree_sitter_cpp
            return True
        except ImportError:
            pass
        
        packages = ["tree-sitter", "tree-sitter-python", "tree-sitter-cpp"]
        
        # Try Slicer's built-in pip_install first (if running inside Slicer)
        try:
            import slicer
            logger.info("Attempting to install tree-sitter via slicer.util.pip_install...")
            for pkg in packages:
                slicer.util.pip_install(pkg)
        except ImportError:
            # Not in Slicer; use subprocess pip
            logger.info("Attempting to install tree-sitter via pip subprocess...")
            import sys
            for pkg in packages:
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", pkg],
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {pkg}: {e}")
                    return False
        except Exception as e:
            logger.warning(f"Failed to install tree-sitter: {e}")
            return False
        
        # Verify installation
        try:
            import tree_sitter
            import tree_sitter_python
            import tree_sitter_cpp
            logger.info("tree-sitter successfully installed and available.")
            return True
        except ImportError:
            logger.warning("tree-sitter installation verification failed.")
            return False
    
    def _get_tree_sitter_parser(self, ext: str):
        """Get a tree-sitter parser for the given file extension."""
        try:
            from tree_sitter_languages import get_parser
            if ext == '.py':
                return get_parser('python')
            elif ext in ('.cxx', '.cpp', '.h', '.hxx', '.c'):
                return get_parser('cpp')
            else:
                return None
        except Exception:
            return None
    
    def _extract_symbols_from_ast(self, node, filepath: str, pattern_lower: str, symbol_type: str, results: List[Dict], lines: List[str]):
        """Recursively walk AST to find function/class definitions."""
        node_type = node.type
        
        # Python
        if node_type == 'function_definition':
            name_node = None
            for child in node.children:
                if child.type == 'identifier':
                    name_node = child
                    break
            if name_node and (symbol_type in ("all", "function")):
                name = name_node.text.decode('utf8')
                if pattern_lower in name.lower():
                    line_num = node.start_point[0] + 1
                    sig = lines[line_num - 1].strip()[:120] if line_num <= len(lines) else ""
                    results.append({
                        "name": name,
                        "type": "function",
                        "file": filepath,
                        "line": line_num,
                        "signature": sig,
                        "source_type": self._infer_source_type(filepath),
                    })
        
        elif node_type == 'class_definition':
            name_node = None
            for child in node.children:
                if child.type == 'identifier':
                    name_node = child
                    break
            if name_node and (symbol_type in ("all", "class")):
                name = name_node.text.decode('utf8')
                if pattern_lower in name.lower():
                    line_num = node.start_point[0] + 1
                    sig = lines[line_num - 1].strip()[:120] if line_num <= len(lines) else ""
                    results.append({
                        "name": name,
                        "type": "class",
                        "file": filepath,
                        "line": line_num,
                        "signature": sig,
                        "source_type": self._infer_source_type(filepath),
                    })
        
        # C/C++
        elif node_type in ('function_definition', 'declaration'):
            # For C++, we need to find the declarator to get the name
            # function_definition in C++ grammar may have nested structure
            # We'll do a simpler approach: look for function_declarator or identifier
            def _find_name_in_node(n):
                if n.type in ('identifier', 'field_identifier'):
                    return n.text.decode('utf8')
                for c in n.children:
                    found = _find_name_in_node(c)
                    if found:
                        return found
                return None
            
            # Only process if it looks like a function definition/declaration
            has_body = any(c.type == 'compound_statement' for c in node.children)
            is_declaration = node_type == 'declaration'
            
            if (has_body or is_declaration) and (symbol_type in ("all", "function")):
                name = _find_name_in_node(node)
                if name and pattern_lower in name.lower():
                    line_num = node.start_point[0] + 1
                    sig = lines[line_num - 1].strip()[:120] if line_num <= len(lines) else ""
                    results.append({
                        "name": name,
                        "type": "function",
                        "file": filepath,
                        "line": line_num,
                        "signature": sig,
                        "source_type": self._infer_source_type(filepath),
                    })
        
        elif node_type in ('class_specifier', 'struct_specifier'):
            name_node = None
            for child in node.children:
                if child.type == 'type_identifier':
                    name_node = child
                    break
            if name_node and (symbol_type in ("all", "class")):
                name = name_node.text.decode('utf8')
                if pattern_lower in name.lower():
                    line_num = node.start_point[0] + 1
                    sig = lines[line_num - 1].strip()[:120] if line_num <= len(lines) else ""
                    results.append({
                        "name": name,
                        "type": "class",
                        "file": filepath,
                        "line": line_num,
                        "signature": sig,
                        "source_type": self._infer_source_type(filepath),
                    })
        
        # Recurse into children
        for child in node.children:
            self._extract_symbols_from_ast(child, filepath, pattern_lower, symbol_type, results, lines)
    
    def _infer_source_type(self, file_path: str) -> str:
        """Infer the role of a file based on its path within the Slicer codebase."""
        path_lower = file_path.lower().replace('\\', '/')
        if '/testing/python/' in path_lower:
            return 'test_example'
        if '/docs/developer_guide/script_repository/' in path_lower:
            return 'doc_example'
        if '/editoreffects/' in path_lower or '/editor_effects/' in path_lower:
            return 'effect_implementation'
        if '/widgets/' in path_lower:
            return 'ui_implementation'
        if '/modules/cli/' in path_lower:
            return 'cli_module'
        if '/modules/scripted/' in path_lower:
            return 'scripted_module'
        if '/modules/loadable/' in path_lower:
            return 'loadable_module'
        if '/base/python/slicer/' in path_lower:
            return 'python_api'
        if '/libs/mrml/core/' in path_lower:
            return 'mrml_definition'
        return 'source'

    def _relativize(self, path: str) -> str:
        """Convert an absolute path back to a relative forward-slash path."""
        # If already relative, normalize separators only
        if not os.path.isabs(path):
            return path.replace(os.sep, '/')
        
        try:
            rel = os.path.relpath(path, self.skill_path)
        except ValueError:
            # On Windows, relpath can fail if paths are on different drives
            rel = path
        
        # If relpath escapes skill_path (starts with ..), the path is not under skill_path
        # or path resolution produced an anomaly. Try manual suffix extraction.
        if rel.startswith('..'):
            norm_path = os.path.normpath(path).replace(os.sep, '/').lower()
            norm_skill = os.path.normpath(self.skill_path).replace(os.sep, '/').lower()
            if norm_path.startswith(norm_skill):
                suffix = norm_path[len(norm_skill):].lstrip('/')
                return suffix
            # Path genuinely outside skill_path — return basename to avoid leaking
            # system directory structure to the LLM (which may reuse it in next round)
            base = os.path.basename(path.rstrip(os.sep))
            return base if base else path
        
        return rel.replace(os.sep, '/')

    def _init_vector_retriever(self) -> Any:
        """
        Attempt to load an existing vector index.
        Returns VectorRetriever if available, else None.
        """
        import time
        t0 = time.time()
        try:
            from .SkillIndexer import IndexBuilder, _get_model_cache_dir, _get_index_dir
            builder = IndexBuilder(self.skill_path)
            retriever = builder.load_retriever()
            t1 = time.time()
            if retriever and retriever.is_ready():
                logger.info("Vector retriever loaded successfully.")
                return retriever
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"Vector retriever initialization failed: {e}")
        return None

    def has_vector_index(self) -> bool:
        """Return True if a dense vector index is loaded and ready."""
        result = self._vector_retriever is not None and self._vector_retriever.is_ready()
        return result

    def execute(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool (FindFile, SearchSymbol, Grep, ReadFile, VectorSearch)
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
        elif tool_name == "VectorSearch":
            result = self._vector_search(
                arguments.get("query", ""),
                arguments.get("top_k", 10)
            )
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
    
    def _vector_search(self, query: str, top_k: int = 10) -> Dict:
        """Execute dense vector search against the pre-built index."""
        if not self._vector_retriever:
            return {
                "tool": "VectorSearch",
                "query": query,
                "error": "Vector index not available. Please build index first."
            }
        try:
            results = self._vector_retriever.search(query, top_k)
            # Filter out chunks from files already included as full files in Phase 2
            excluded = self._vector_retriever.full_file_paths
            if excluded:
                results = [rc for rc in results if rc.chunk.file_path not in excluded]
            formatted = self._vector_retriever.format_for_prompt(results)
            # Serialize results for JSON compatibility
            serializable = []
            for rc in results:
                c = rc.chunk
                serializable.append({
                    "chunk_id": c.chunk_id,
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "chunk_type": c.chunk_type,
                    "source_type": c.source_type,
                    "vector_score": round(rc.vector_score, 4),
                    "final_score": round(rc.final_score, 4),
                })
            return {
                "tool": "VectorSearch",
                "query": query,
                "results": serializable,
                "formatted_context": formatted,
            }
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return {
                "tool": "VectorSearch",
                "query": query,
                "error": str(e),
            }

    def _find_rg(self) -> Optional[str]:
        # 1. Windows bundled binary
        if self.platform == 'windows' and os.path.isfile(_RG_PATH):
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
                elif os.path.isfile(path):
                    # rg --count-matches --no-heading on a single file outputs just the number
                    try:
                        count = int(line)
                        if count > 0:
                            rel_path = self._relativize(path)
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
                rg_exe, "-H", "-n", "-i", "-B", "5", "-A", "5", "-m", "3",
                "--max-columns", "500",
                "--color", "never",
                pattern, abs_path,
            ]
            try:
                sample_result = subprocess.run(sample_cmd, capture_output=True, timeout=10)
                if sample_result.returncode in (0, 1):
                    stdout = sample_result.stdout.decode('utf-8', errors='ignore').strip()
                    if not stdout:
                        continue
                    # rg -B/-A output separates match blocks with '--'
                    blocks = []
                    current_block = []
                    for line in stdout.split('\n'):
                        if line == '--':
                            if current_block:
                                blocks.append(current_block)
                                current_block = []
                        else:
                            current_block.append(line)
                    if current_block:
                        blocks.append(current_block)

                    for block in blocks:
                        matched_line = None
                        for raw_line in block:
                            m = re.match(r'^(.+?):(\d+):(.*)$', raw_line)
                            if m:
                                line_content = m.group(3)
                                if pattern.lower() in line_content.lower():
                                    matched_line = m
                                    break
                        if not matched_line and block:
                            for raw_line in block:
                                m = re.match(r'^(.+?):(\d+):(.*)$', raw_line)
                                if m:
                                    matched_line = m
                                    break

                        if matched_line:
                            context_lines = []
                            for raw_line in block:
                                m = re.match(r'^(.+?):(\d+):(.*)$', raw_line)
                                if m:
                                    context_lines.append(m.group(3))
                                else:
                                    context_lines.append(raw_line)
                            representative.append({
                                "file": self._relativize(matched_line.group(1)),
                                "line": int(matched_line.group(2)),
                                "content": matched_line.group(3).strip(),
                                "context": '\n'.join(context_lines),
                            })
            except Exception:
                pass

        files_summary = [
            {
                "file": f,
                "hits": h,
                "source_type": self._infer_source_type(f),
            }
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
            is_markdown = path.lower().endswith('.md')
            
            # Duplicate read detection
            note = None
            rel_path = self._relativize(path)
            if rel_path in self._read_history:
                prev = self._read_history[rel_path]
                note = (
                    f"[Note: This file was previously read with query='{prev['query']}' "
                    f"(strategy={prev['strategy']}). "
                    f"If the current content is insufficient, consider searching a different file.]"
                )
            
            if total_lines < 500:
                content = ''.join(lines)
                strategy = "full"
            else:
                if query:
                    if is_markdown:
                        content = self._slice_markdown_by_query(lines, query)
                        strategy = "markdown_heading"
                    else:
                        # For test files, prefer test-method slicing over class-level AST
                        is_test_file = '/Testing/Python/' in path.replace('\\', '/')
                        if is_test_file:
                            test_slice = self._slice_test_method(lines, query)
                            if test_slice:
                                content = test_slice
                                strategy = "test_method"
                            else:
                                ext = os.path.splitext(path)[1].lower()
                                ast_slice = self._slice_by_ast_boundary(lines, query, ext)
                                if ast_slice:
                                    content = ast_slice
                                    strategy = "ast_boundary"
                                else:
                                    content = self._slice_by_grep_context(lines, query)
                                    strategy = "grep_context"
                        else:
                            # Try AST-based boundary slice first (exact function/class boundaries)
                            ext = os.path.splitext(path)[1].lower()
                            ast_slice = self._slice_by_ast_boundary(lines, query, ext)
                            if ast_slice:
                                content = ast_slice
                                strategy = "ast_boundary"
                            else:
                                content = self._slice_by_grep_context(lines, query)
                                strategy = "grep_context"
                else:
                    content = ''.join(lines[:500]) + "\n... [file truncated: provide query to locate specific section] ..."
                    strategy = "truncated"
            
            # Record this read for duplicate detection
            self._read_history[rel_path] = {"query": query or "(none)", "strategy": strategy}
            
            result = {
                "tool": "ReadFile",
                "path": path,
                "query": query,
                "content": content,
                "total_lines": total_lines,
                "strategy": strategy,
                "source_type": self._infer_source_type(path),
            }
            notes = []
            if note:
                notes.append(note)
            
            # Gist-link dead-end detection
            gist_links = re.findall(r'https?://gist\.github\.com/\S+', content)
            has_local_code = bool(re.search(r'```python\s*\n', content))
            if len(gist_links) > 2 and not has_local_code:
                notes.append(
                    "[Note: This section contains mainly external gist links with no local executable examples. "
                    "Search for specific function names in test files instead.]"
                )
            
            if notes:
                result["_note"] = "\n".join(notes)
            
            # For markdown files, list all available headings so the LLM can decide if further reads are needed
            if is_markdown:
                headings = []
                for line in lines:
                    m = re.match(r'^(#{1,6})\s+(.+)$', line)
                    if m:
                        level = len(m.group(1))
                        title = m.group(2).strip()
                        headings.append(f"{'  ' * (level - 1)}{title}")
                if headings:
                    MAX_HEADINGS = 30
                    if len(headings) > MAX_HEADINGS:
                        result["available_sections"] = headings[:MAX_HEADINGS] + [f"... {len(headings) - MAX_HEADINGS} more headings omitted ..."]
                    else:
                        result["available_sections"] = headings
            return result
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
    
    def _slice_test_method(self, lines: List[str], query: str) -> Optional[str]:
        """
        Slice Python test files by test method boundaries.
        Returns complete test method / TestSection bodies that match the query.
        More precise than class-level AST slicing for test files.
        """
        query_lower = query.lower()
        matches = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Match test methods: def test_xxx(...) or def TestSection_xxx(...)
            match = re.match(r'^(\s*)def\s+(test_\w+|TestSection_\w+)\s*\(', line)
            if match:
                indent = match.group(1)
                method_name = match.group(2)
                start = i
                i += 1
                # Capture until next method/class at same or less indentation
                while i < len(lines):
                    next_line = lines[i]
                    # Check if next line is a new def/class at same or less indent
                    if re.match(rf'^{re.escape(indent)}(?:def\s+|class\s+)', next_line):
                        break
                    i += 1
                end = i
                body = ''.join(lines[start:end])
                if query_lower in body.lower():
                    matches.append((method_name, start, end))
                continue
            i += 1
        
        if not matches:
            return None
        
        MAX_MATCHES = 3
        parts = []
        for i, (method_name, start, end) in enumerate(matches[:MAX_MATCHES]):
            parts.append(f"=== test method: {method_name} (lines {start+1}-{end}) ===\n")
            parts.append(''.join(lines[start:end]))
        
        if len(matches) > MAX_MATCHES:
            parts.append(f"\n... [{len(matches) - MAX_MATCHES} more test methods omitted, use more specific query] ...")
        
        return ''.join(parts)
    
    def _slice_by_ast_boundary(self, lines: List[str], query: str, ext: str) -> Optional[str]:
        """
        Slice code by AST boundaries: return complete function/class bodies that match query.
        Ensures function bodies are never truncated mid-definition.
        Falls back to None if tree-sitter unavailable or parse fails.
        """
        if not self._tree_sitter_available:
            return None
        
        parser = self._get_tree_sitter_parser(ext)
        if not parser:
            return None
        
        source = ''.join(lines)
        try:
            tree = parser.parse(bytes(source, 'utf8'))
        except Exception:
            return None
        
        query_lower = query.lower()
        matches = []
        
        def _extract_node_name(node):
            """Extract identifier name from function/class AST node."""
            target_types = {
                'function_definition': ('identifier', 'field_identifier'),
                'class_definition': ('identifier',),
                'class_specifier': ('type_identifier',),
                'struct_specifier': ('type_identifier',),
            }
            expected = target_types.get(node.type, ('identifier', 'type_identifier', 'field_identifier'))
            
            def _find(n):
                if n.type in expected:
                    return n.text.decode('utf8')
                for c in n.children:
                    found = _find(c)
                    if found:
                        return found
                return None
            
            return _find(node)
        
        def _walk(node):
            if node.type in ('function_definition', 'class_definition', 'class_specifier', 'struct_specifier'):
                name = _extract_node_name(node)
                
                # Strategy 1: name contains query
                if name and query_lower in name.lower():
                    matches.append((name, node))
                    return  # Don't recurse into children to avoid nested duplicates
                
                # Strategy 2: body contains query (but node has a name)
                if name:
                    body = node.text.decode('utf8').lower()
                    if query_lower in body:
                        matches.append((name, node))
                        return
            
            for child in node.children:
                _walk(child)
        
        _walk(tree.root_node)
        
        if not matches:
            return None
        
        # Deduplicate by line range
        seen = set()
        unique = []
        for name, node in matches:
            key = (node.start_point[0], node.end_point[0])
            if key not in seen:
                seen.add(key)
                unique.append((name, node))
        
        MAX_MATCHES = 3
        parts = []
        for i, (name, node) in enumerate(unique[:MAX_MATCHES]):
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            node_type = node.type.replace('_', ' ')
            parts.append(f"=== {node_type}: {name} (lines {start_line+1}-{end_line+1}) ===\n")
            parts.append(''.join(lines[start_line:end_line+1]))
        
        if len(unique) > MAX_MATCHES:
            parts.append(f"\n... [{len(unique) - MAX_MATCHES} more AST matches omitted, use more specific query] ...")
        
        return ''.join(parts)
    
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
            "count": len(results),
            "results_with_type": [
                {"file": self._relativize(r), "source_type": self._infer_source_type(r)}
                for r in results
            ],
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
            
            # Use AST-based parsing for Python/C++ if tree-sitter is available
            use_ast = self._tree_sitter_available and ext in ('.py', '.cxx', '.cpp', '.h', '.hxx', '.c')
            
            if use_ast:
                parser = self._get_tree_sitter_parser(ext)
                if parser:
                    try:
                        source = ''.join(lines)
                        tree = parser.parse(bytes(source, 'utf8'))
                        self._extract_symbols_from_ast(
                            tree.root_node, filepath, pattern_lower, symbol_type, results, lines
                        )
                        return  # AST handled everything for this file
                    except Exception:
                        # AST parse failed, fall through to regex
                        pass
            
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
                                    "signature": line.strip()[:120],
                                    "source_type": self._infer_source_type(filepath),
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
                                "signature": line.strip()[:120],
                                "source_type": self._infer_source_type(filepath),
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
        {
            "type": "function",
            "function": {
                "name": "VectorSearch",
                "description": "Dense vector search over the pre-indexed knowledge base. Returns the most relevant code snippets. Use this as a fast first step before using ReadFile or Grep.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what you need (e.g., 'load a volume and display it')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return (default 10)"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
    ]
