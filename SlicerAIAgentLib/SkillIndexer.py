"""
SkillIndexer - Dense vector retrieval index for Slicer skill knowledge base.

Implements pure dense vector (FAISS) retrieval with source-type weighting.
Indexes and model caches are stored under the project's Code_RAG/ directory.

Components:
- Chunker: splits knowledge-base files into semantic chunks
- VectorIndex: dense semantic retrieval via ONNX Runtime + FAISS
- VectorRetriever: similarity search + source_type weighting
- IndexBuilder: orchestrates incremental index building
"""

import hashlib
import json
import logging
import os
import ast
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Return the repository root directory (parent of SlicerAIAgentLib/)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root


def _get_index_dir() -> str:
    """Return the project-local index directory: <repo>/Resources/Code_RAG/v1/."""
    d = os.path.join(_get_project_root(), "Resources", "Code_RAG", "v1")
    return d


def _get_model_cache_dir() -> str:
    """Return the project-local model cache directory: <repo>/Resources/Code_RAG/models/."""
    d = os.path.join(_get_project_root(), "Resources", "Code_RAG", "models")
    return d


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single semantic chunk from a knowledge-base file."""
    chunk_id: str           # "{rel_path}#{start_line}-{end_line}"
    file_path: str          # relative to skill root
    start_line: int
    end_line: int
    content: str            # raw text
    embedding_text: str     # text fed to the embedding model
    chunk_type: str         # function | class | heading | code_block | whole_file
    source_type: str        # doc_example | python_api | effect_implementation | ...
    language: str           # python | cpp | markdown | other


@dataclass
class RetrievedChunk:
    """A chunk returned by the dense vector retriever."""
    chunk: CodeChunk
    vector_score: float = 0.0
    final_score: float = 0.0


# ---------------------------------------------------------------------------
# Dependency helpers (mirrors SkillTools._ensure_tree_sitter pattern)
# ---------------------------------------------------------------------------

def _ensure_packages(packages: List[str]) -> bool:
    """Ensure Python packages are importable, installing if necessary."""
    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_').split('[')[0])
        except ImportError:
            missing.append(pkg)
    if not missing:
        return True

    # Try Slicer's pip_install first
    try:
        import slicer
        for pkg in missing:
            logger.info(f"Installing {pkg} via slicer.util.pip_install...")
            slicer.util.pip_install(pkg)
        return True
    except ImportError:
        pass

    # Fallback to subprocess pip
    import subprocess
    import sys
    for pkg in missing:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {pkg}: {e}")
            return False
    return True


def _get_faiss() -> Any:
    """Lazy import faiss with auto-install."""
    try:
        import faiss
        return faiss
    except ImportError:
        if _ensure_packages(["faiss-cpu"]):
            import faiss
            return faiss
        raise ImportError("faiss-cpu is required but could not be installed.")


def _get_onnxruntime() -> Any:
    """Lazy import onnxruntime with auto-install."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        if _ensure_packages(["onnxruntime"]):
            import onnxruntime as ort
            return ort
        raise ImportError("onnxruntime is required but could not be installed.")


def _get_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError:
        if _ensure_packages(["numpy>=1.21.0"]):
            import numpy as np
            return np
        raise


# ---------------------------------------------------------------------------
# API description extraction (for embedding text enrichment)
# ---------------------------------------------------------------------------


def _extract_api_description(content: str) -> str:
    """Extract function signature + docstring from a Python code chunk."""
    import ast
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                sig_line = content.split('\n')[0].strip()
                doc = ast.get_docstring(node)
                parts = []
                if sig_line:
                    parts.append(f"API: {sig_line}")
                if doc:
                    parts.append(f"Description: {doc}")
                return "\n".join(parts) if parts else ""
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Split knowledge-base files into semantic chunks."""

    # P0 + P1 directories that we actually index
    INDEXED_PREFIXES = (
        # ── Cookbook examples (highest value for code gen) ──
        "slicer-source/Docs/developer_guide/script_repository",

        # ── Core Python API ──
        "slicer-source/Base/Python/slicer/util.py",
        "slicer-source/Base/Python/slicer/",

        # ── Scripted module reference implementations ──
        "slicer-source/Modules/Scripted/",

        # ── Segment Editor Python effects ──
        "slicer-source/Modules/Loadable/Segmentations/EditorEffects/Python/",

        # ── Loadable module Python tests (programmatic control APIs) ──
        "slicer-source/Modules/Loadable/Colors/Testing/Python/",
        "slicer-source/Modules/Loadable/CropVolume/Testing/Python/",
        "slicer-source/Modules/Loadable/Markups/Testing/Python/",
        "slicer-source/Modules/Loadable/Markups/Widgets/Testing/Python/",
        "slicer-source/Modules/Loadable/Plots/Testing/Python/",
        "slicer-source/Modules/Loadable/SceneViews/Testing/Python/",
        "slicer-source/Modules/Loadable/Segmentations/Testing/Python/",
        "slicer-source/Modules/Loadable/Sequences/Testing/Python/",
        "slicer-source/Modules/Loadable/SubjectHierarchy/Testing/Python/",
        "slicer-source/Modules/Loadable/SubjectHierarchy/Widgets/Python/",
        "slicer-source/Modules/Loadable/Tables/Testing/Python/",
        "slicer-source/Modules/Loadable/VolumeRendering/Testing/Python/",
        "slicer-source/Modules/Loadable/Volumes/Testing/Python/",

        # ── MRML node definitions (maps to Python MRML API) ──
        "slicer-source/Libs/MRML/Core/",
    )

    # Extensions we know how to chunk
    CHUNKABLE_EXTS = {'.py', '.cxx', '.cpp', '.h', '.hxx', '.c', '.md'}

    def __init__(self, tree_sitter_available: bool = True):
        self._tree_sitter_available = tree_sitter_available

    def should_index_file(self, rel_path: str) -> bool:
        """Only index files under P0/P1 prefixes with known extensions."""
        rel_unix = rel_path.replace('\\', '/')
        # Must match a prefix
        if not any(rel_unix.startswith(p) for p in self.INDEXED_PREFIXES):
            return False
        ext = os.path.splitext(rel_path)[1].lower()
        if ext not in self.CHUNKABLE_EXTS:
            return False
        return True

    def chunk_file(self, filepath: str, rel_path: str) -> List[CodeChunk]:
        """Dispatch to the appropriate chunking strategy."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Cannot read {filepath}: {e}")
            return []

        if not lines:
            return []

        ext = os.path.splitext(filepath)[1].lower()
        source_type = self._infer_source_type(rel_path)

        if ext == '.py':
            chunks = self._chunk_python_ast(filepath, lines, rel_path, source_type)
        elif ext in ('.cxx', '.cpp', '.h', '.hxx', '.c'):
            chunks = self._chunk_cpp_ast(filepath, lines, rel_path, source_type)
        elif ext == '.md':
            chunks = self._chunk_markdown(filepath, lines, rel_path, source_type)
        else:
            chunks = []

        if not chunks:
            # Fallback: whole-file chunk
            content = ''.join(lines)
            chunks = [CodeChunk(
                chunk_id=f"{rel_path}#1-{len(lines)}",
                file_path=rel_path,
                start_line=1,
                end_line=len(lines),
                content=content,
                embedding_text=self._enhance_code_embedding_text(
                    rel_path, content, "whole_file", source_type
                ),
                chunk_type="whole_file",
                source_type=source_type,
                language="other",
            )]

        return chunks

    def _infer_source_type(self, rel_path: str) -> str:
        """Mirror of SkillTools._infer_source_type."""
        path_lower = rel_path.lower().replace('\\', '/')
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

    def _get_tree_sitter_parser(self, ext: str):
        """Get a tree-sitter parser (same logic as SkillTools)."""
        if not self._tree_sitter_available:
            return None
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

    def _chunk_python_ast(self, filepath: str, lines: List[str],
                          rel_path: str, source_type: str) -> List[CodeChunk]:
        parser = self._get_tree_sitter_parser('.py')
        if not parser:
            return []

        source = ''.join(lines)
        try:
            tree = parser.parse(bytes(source, 'utf8'))
        except Exception:
            return []

        chunks = []

        def _walk(node):
            if node.type in ('function_definition', 'class_definition'):
                name = None
                for child in node.children:
                    if child.type == 'identifier':
                        name = child.text.decode('utf8')
                        break
                if name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    content = ''.join(lines[node.start_point[0]:node.end_point[0] + 1])
                    ctype = 'function' if node.type == 'function_definition' else 'class'
                    chunks.append(CodeChunk(
                        chunk_id=f"{rel_path}#{start_line}-{end_line}",
                        file_path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        embedding_text=self._enhance_code_embedding_text(
                            name, content, ctype, source_type
                        ),
                        chunk_type=ctype,
                        source_type=source_type,
                        language='python',
                    ))
                # Recurse into class bodies for nested methods
                if node.type == 'class_definition':
                    for child in node.children:
                        _walk(child)
            else:
                for child in node.children:
                    _walk(child)

        _walk(tree.root_node)
        return chunks

    def _chunk_cpp_ast(self, filepath: str, lines: List[str],
                       rel_path: str, source_type: str) -> List[CodeChunk]:
        parser = self._get_tree_sitter_parser(os.path.splitext(filepath)[1])
        if not parser:
            return []

        source = ''.join(lines)
        try:
            tree = parser.parse(bytes(source, 'utf8'))
        except Exception:
            return []

        chunks = []

        def _find_name(node):
            if node.type in ('identifier', 'field_identifier', 'type_identifier'):
                return node.text.decode('utf8')
            for c in node.children:
                found = _find_name(c)
                if found:
                    return found
            return None

        def _walk(node):
            if node.type in ('function_definition', 'declaration'):
                name = _find_name(node)
                if name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    content = ''.join(lines[node.start_point[0]:node.end_point[0] + 1])
                    chunks.append(CodeChunk(
                        chunk_id=f"{rel_path}#{start_line}-{end_line}",
                        file_path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        embedding_text=self._enhance_code_embedding_text(
                            name, content, 'function', source_type
                        ),
                        chunk_type='function',
                        source_type=source_type,
                        language='cpp',
                    ))
            elif node.type in ('class_specifier', 'struct_specifier'):
                name = _find_name(node)
                if name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    content = ''.join(lines[node.start_point[0]:node.end_point[0] + 1])
                    chunks.append(CodeChunk(
                        chunk_id=f"{rel_path}#{start_line}-{end_line}",
                        file_path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        embedding_text=self._enhance_code_embedding_text(
                            name, content, 'class', source_type
                        ),
                        chunk_type='class',
                        source_type=source_type,
                        language='cpp',
                    ))
            else:
                for child in node.children:
                    _walk(child)

        _walk(tree.root_node)
        return chunks

    def _chunk_markdown(self, filepath: str, lines: List[str],
                        rel_path: str, source_type: str) -> List[CodeChunk]:
        """Chunk markdown by headings; code blocks under a heading stay with it."""
        chunks = []
        current_heading = None
        current_lines: List[str] = []
        current_start = 1

        def _flush():
            nonlocal current_heading, current_lines, current_start
            if current_lines:
                content = ''.join(current_lines).strip()
                if content:
                    start = current_start
                    end = current_start + len(current_lines) - 1
                    title = current_heading or "(untitled)"
                    chunks.append(CodeChunk(
                        chunk_id=f"{rel_path}#{start}-{end}",
                        file_path=rel_path,
                        start_line=start,
                        end_line=end,
                        content=content,
                        embedding_text=f"heading {title}\n{content}",
                        chunk_type='heading',
                        source_type=source_type,
                        language='markdown',
                    ))
            current_lines = []
            current_heading = None

        for i, line in enumerate(lines):
            m = re.match(r'^(#{1,6})\s+(.+)$', line)
            if m:
                _flush()
                current_heading = m.group(2).strip()
                current_start = i + 1
                current_lines.append(line)
            else:
                current_lines.append(line)

        _flush()
        return chunks

    def _enhance_code_embedding_text(self, name: str, content: str,
                                     chunk_type: str, source_type: str) -> str:
        """Prefix code chunks with type hints and API descriptions for better embedding quality."""
        if chunk_type == 'function':
            prefix = f"function {name}: "
        elif chunk_type == 'class':
            prefix = f"class {name}: "
        else:
            prefix = ""
        # Include source_type context
        ctx = f"[{source_type}] "
        # NEW: extract signature + docstring for Python chunks
        api_desc = _extract_api_description(content)
        if api_desc:
            prefix = api_desc + "\n" + prefix
        return ctx + prefix + content


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------

class VectorIndex:
    """Dense semantic retrieval via ONNX Runtime + FAISS."""

    MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
    ONNX_SUBPATH = "onnx/model.onnx"
    DIMENSION = 768
    MAX_SEQ_LENGTH = 1024

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.session = None      # onnxruntime InferenceSession
        self.tokenizer = None    # tokenizers Tokenizer
        self.faiss_index = None
        self.chunk_ids: List[str] = []

    @staticmethod
    def _find_tokenizer_json(cache_dir: str) -> Optional[str]:
        """Find tokenizer.json in the HF Hub cache under cache_dir."""
        # Prefer the exact model repo cache
        model_cache = os.path.join(cache_dir, "models--jinaai--jina-embeddings-v2-base-code")
        if os.path.isdir(model_cache):
            for root, _, files in os.walk(model_cache):
                if "tokenizer.json" in files:
                    return os.path.join(root, "tokenizer.json")
        # Fallback: search entire cache_dir
        for root, _, files in os.walk(cache_dir):
            if "tokenizer.json" in files:
                return os.path.join(root, "tokenizer.json")
        return None

    def _ensure_model_files(self):
        """Download ONNX model and load lightweight tokenizer if not cached locally."""
        import time
        t0 = time.time()
        cache_dir = _get_model_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        # Ensure tokenizer is available
        if self.tokenizer is None:
            t1 = time.time()
            try:
                from tokenizers import Tokenizer
            except ImportError:
                _ensure_packages(["tokenizers"])
                from tokenizers import Tokenizer

            tokenizer_json = self._find_tokenizer_json(cache_dir)
            if tokenizer_json is None:
                raise RuntimeError(
                    f"tokenizer.json not found under {cache_dir}. "
                    "Please ensure the model tokenizer files are downloaded."
                )
            t_tok0 = time.time()
            self.tokenizer = Tokenizer.from_file(tokenizer_json)
            self.tokenizer.enable_truncation(max_length=self.MAX_SEQ_LENGTH)
            pad_id = self.tokenizer.token_to_id("<pad>")
            if pad_id is None:
                pad_id = 0
            self.tokenizer.enable_padding(
                length=self.MAX_SEQ_LENGTH,
                pad_id=pad_id,
                pad_token="<pad>",
            )
        # Ensure ONNX model is available
        onnx_path = os.path.join(cache_dir, "model.onnx")
        onnx_exists = os.path.exists(onnx_path)
        onnx_size = os.path.getsize(onnx_path) if onnx_exists else 0
        if not onnx_exists:
            self._download_onnx_model(onnx_path)

    def _download_onnx_model(self, onnx_path: str):
        """Download ONNX model from HuggingFace Hub with progress logging."""
        import urllib.request
        import time
        url = f"https://huggingface.co/{self.MODEL_NAME}/resolve/main/{self.ONNX_SUBPATH}"
        logger.info(f"Downloading ONNX model from {url} ...")
        logger.info(f"Target: {onnx_path} (this is a one-time ~640 MB download)")

        def _report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                if block_num % 50 == 0 or pct >= 100:
                    logger.info(f"  Download progress: {pct:.1f}% "
                               f"({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)")

        t0 = time.time()
        try:
            urllib.request.urlretrieve(url, onnx_path, reporthook=_report)
            elapsed = time.time() - t0
            final_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
            logger.info(f"ONNX model downloaded successfully: {onnx_path}")
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"Failed to download ONNX model: {e}")
            raise

    def _load_model(self):
        import time
        if self.session is None:
            t0 = time.time()
            _get_onnxruntime()
            self._ensure_model_files()

            cache_dir = _get_model_cache_dir()
            onnx_path = os.path.join(cache_dir, "model.onnx")

            import onnxruntime as ort
            logger.info(f"Loading ONNX model from {onnx_path} ...")

            # Auto-detect GPU providers; fallback to CPU
            available = ort.get_available_providers()
            preferred = []
            if 'CUDAExecutionProvider' in available:
                preferred.append('CUDAExecutionProvider')
            if 'DmlExecutionProvider' in available:
                preferred.append('DmlExecutionProvider')
            if 'ROCMExecutionProvider' in available:
                preferred.append('ROCMExecutionProvider')
            if 'CoreMLExecutionProvider' in available:
                preferred.append('CoreMLExecutionProvider')
            preferred.append('CPUExecutionProvider')

            if preferred[0] != 'CPUExecutionProvider':
                logger.info(f"GPU provider available: {preferred[0]}")
            else:
                logger.info("No GPU provider found; using CPU.")

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 0  # use all CPU cores
            sess_options.inter_op_num_threads = 0

            t_sess0 = time.time()
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=preferred,
            )
            t_sess = time.time() - t_sess0
            total = time.time() - t0
            provider = self.session.get_providers()[0]
            logger.info(f"ONNX model loaded successfully on provider: {provider}")
        return self.session

    @staticmethod
    def _mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
        """Mean pooling over token embeddings, masking padding tokens."""
        np = _get_numpy()
        mask = np.expand_dims(attention_mask.astype(np.float32), -1)  # [batch, seq_len, 1]
        sum_embeddings = np.sum(last_hidden_state * mask, axis=1)      # [batch, hidden_dim]
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    @staticmethod
    def _l2_normalize(vectors: Any) -> Any:
        """L2 normalize embeddings."""
        np = _get_numpy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, a_min=1e-12, a_max=None)

    def _encode(self, texts: List[str], batch_size: int = 8) -> Any:
        """Encode texts into embeddings using ONNX Runtime."""
        import time
        t0 = time.time()
        np = _get_numpy()
        t_load0 = time.time()
        session = self._load_model()
        t_load = time.time() - t_load0
        tokenizer = self.tokenizer

        all_embeddings = []
        total = len(texts)
        start = time.time()
        n_batches = (total + batch_size - 1) // batch_size
        last_logged_done = 0

        def _format_eta(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.0f}m {seconds%60:.0f}s"
            else:
                return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"

        for batch_idx, i in enumerate(range(0, total, batch_size)):
            batch = texts[i:i + batch_size]
            t_batch0 = time.time()

            # Use lightweight tokenizers library (Rust-based, no transformers import)
            encodings = tokenizer.encode_batch(batch)
            input_ids_list = [enc.ids for enc in encodings]
            attention_mask_list = [enc.attention_mask for enc in encodings]
            type_ids_list = [enc.type_ids for enc in encodings]

            inputs = {
                "input_ids": np.array(input_ids_list, dtype=np.int64),
                "attention_mask": np.array(attention_mask_list, dtype=np.int64),
            }
            # Only include token_type_ids if the tokenizer produced them
            has_type_ids = any(len(tids) > 0 and any(tids) for tids in type_ids_list)
            if has_type_ids:
                inputs["token_type_ids"] = np.array(type_ids_list, dtype=np.int64)

            t_tok = time.time() - t_batch0

            # Build ONNX inputs dynamically based on model's expected inputs
            ort_inputs = {}
            for inp in session.get_inputs():
                name = inp.name
                if name == "input_ids":
                    ort_inputs[name] = inputs["input_ids"]
                elif name == "attention_mask":
                    ort_inputs[name] = inputs["attention_mask"]
                elif name == "token_type_ids":
                    if "token_type_ids" in inputs:
                        ort_inputs[name] = inputs["token_type_ids"]
                    else:
                        ort_inputs[name] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

            t_inf0 = time.time()
            outputs = session.run(None, ort_inputs)
            t_inf = time.time() - t_inf0
            last_hidden_state = outputs[0]

            # Handle both pooled and hidden-state outputs
            if last_hidden_state.ndim == 2:
                embeddings = last_hidden_state
            else:
                embeddings = self._mean_pooling(last_hidden_state, inputs["attention_mask"])

            embeddings = self._l2_normalize(embeddings)
            all_embeddings.append(embeddings)
            t_batch = time.time() - t_batch0

            done = min(i + batch_size, total)
            # Log every time at least 5 more chunks have been processed since last log
            if done - last_logged_done >= 5 or done == total:
                elapsed = time.time() - start
                pct = done * 100.0 / total
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                logger.info(
                    f"[PROGRESS] {done:>5}/{total} ({pct:5.1f}%) | "
                    f"{rate:.1f} chunks/s | ETA: {_format_eta(remaining)}"
                )
                last_logged_done = done

        elapsed = time.time() - start
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(f"[PROGRESS] {total:>5}/{total} (100.0%) | {rate:.1f} chunks/s total | Done in {_format_eta(elapsed)}")
        total_t = time.time() - t0

        return np.concatenate(all_embeddings, axis=0)

    def build(self, chunks: List[CodeChunk], batch_size: int = 16):
        """Build FAISS index from chunks."""
        import time
        np = _get_numpy()

        texts = [c.embedding_text for c in chunks]
        session = self._load_model()
        provider = session.get_providers()[0] if session else "CPUExecutionProvider"
        logger.info(f"[EMBED] Encoding {len(texts)} chunks with {self.MODEL_NAME} (provider: {provider}) ...")
        logger.info(f"[EMBED] Max sequence length: {self.MAX_SEQ_LENGTH} tokens. Batch size: {batch_size}")

        start = time.time()
        embeddings = self._encode(texts, batch_size=batch_size)
        elapsed = time.time() - start
        speed = len(texts) / elapsed if elapsed > 0 else 0
        logger.info(f"[EMBED] Done: {len(texts)} chunks in {elapsed:.1f}s ({speed:.1f} chunks/s)")

        self.chunk_ids = [c.chunk_id for c in chunks]
        faiss = _get_faiss()
        self.faiss_index = faiss.IndexFlatIP(self.DIMENSION)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        self.save()

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Return (chunk_id, cosine_similarity) sorted descending."""
        import time
        t0 = time.time()
        if self.faiss_index is None:
            return []
        np = _get_numpy()
        t_enc0 = time.time()
        query_embedding = self._encode([query], batch_size=1)
        t_enc = time.time() - t_enc0
        t_faiss0 = time.time()
        scores, indices = self.faiss_index.search(
            np.array(query_embedding, dtype=np.float32), top_k
        )
        t_faiss = time.time() - t_faiss0
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append((self.chunk_ids[int(idx)], float(score)))
        total = time.time() - t0
        return results

    def save(self):
        if self.faiss_index is None:
            return
        faiss = _get_faiss()
        faiss.write_index(self.faiss_index, self.index_path)
        meta_path = self.index_path + ".meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'chunk_ids': self.chunk_ids}, f)

    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            return False
        try:
            faiss = _get_faiss()
            self.faiss_index = faiss.read_index(self.index_path)
            meta_path = self.index_path + ".meta.json"
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.chunk_ids = meta['chunk_ids']
            return True
        except Exception as e:
            logger.warning(f"Failed to load vector index: {e}")
            return False


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

_SOURCE_TYPE_WEIGHTS = {
    'doc_example': 1.3,
    'python_api': 1.2,
    'effect_implementation': 1.1,
    'scripted_module': 1.1,
    'test_example': 1.05,
    'source': 1.0,
}

class VectorRetriever:
    """Pure dense vector retrieval with source_type weighting."""

    def __init__(self, vector_index: VectorIndex,
                 chunks_metadata: Dict[str, CodeChunk]):
        self.vector = vector_index
        self.chunks = chunks_metadata

    def is_ready(self) -> bool:
        return self.vector.faiss_index is not None

    def search(self, query: str, top_k: int = 15) -> List[RetrievedChunk]:
        """Run pure vector search and return ranked chunks."""
        if not self.is_ready():
            return []

        # Over-fetch for deduplication
        vec_results = self.vector.search(query, top_k=top_k * 3)

        # Apply source_type weighting
        scored = []
        for cid, sim in vec_results:
            chunk = self.chunks.get(cid)
            if not chunk:
                continue
            weight = _SOURCE_TYPE_WEIGHTS.get(chunk.source_type, 1.0)
            final = sim * weight
            scored.append((cid, final, sim, chunk))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by file (max 3 chunks per file)
        file_counts: Dict[str, int] = {}
        results: List[RetrievedChunk] = []
        for cid, final, sim, chunk in scored:
            fp = chunk.file_path
            if file_counts.get(fp, 0) >= 3:
                continue
            file_counts[fp] = file_counts.get(fp, 0) + 1
            results.append(RetrievedChunk(
                chunk=chunk,
                vector_score=sim,
                final_score=final,
            ))
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def merge_results(result_lists: List[List[RetrievedChunk]], top_k: int = 15) -> List[RetrievedChunk]:
        """
        Merge multiple retrieval result lists, deduplicate by chunk_id,
        keep the highest final_score, and return top_k.
        """
        best: Dict[str, RetrievedChunk] = {}
        for results in result_lists:
            for rc in results:
                cid = rc.chunk.chunk_id
                if cid not in best or rc.final_score > best[cid].final_score:
                    best[cid] = rc
        merged = sorted(best.values(), key=lambda rc: rc.final_score, reverse=True)
        return merged[:top_k]

    @staticmethod
    def merge_results_with_quota(
        result_lists: List[List[RetrievedChunk]],
        quota_per_list: int = 2,
        total_slots: int = 15,
    ) -> List[RetrievedChunk]:
        """
        Merge retrieval results with per-sub-query quota guarantee.

        Phase 1: each sub-query gets `quota_per_list` top chunks (guaranteed).
        Phase 2: remaining slots filled from global pool by final_score.
        """
        seen: set = set()
        merged: List[RetrievedChunk] = []

        # Phase 1: guarantee quota for each sub-query
        for results in result_lists:
            count = 0
            for rc in sorted(results, key=lambda x: x.final_score, reverse=True):
                cid = rc.chunk.chunk_id
                if cid not in seen:
                    seen.add(cid)
                    merged.append(rc)
                    count += 1
                    if count >= quota_per_list:
                        break

        # Phase 2: fill remaining slots from global pool
        if len(merged) < total_slots:
            all_remaining: List[RetrievedChunk] = []
            for results in result_lists:
                for rc in results:
                    if rc.chunk.chunk_id not in seen:
                        all_remaining.append(rc)
            all_remaining.sort(key=lambda x: x.final_score, reverse=True)
            for rc in all_remaining:
                if len(merged) >= total_slots:
                    break
                cid = rc.chunk.chunk_id
                if cid not in seen:
                    seen.add(cid)
                    merged.append(rc)

        # Final sort by score descending
        merged.sort(key=lambda x: x.final_score, reverse=True)
        return merged

    def format_for_prompt(self, results: List[RetrievedChunk]) -> str:
        """Format retrieval results for injection into system prompt."""
        if not results:
            return ""
        lines = ["## Relevant code snippets from knowledge base:\n"]
        for i, rc in enumerate(results, 1):
            c = rc.chunk
            lines.append(
                f"[{i}] {c.file_path} (lines {c.start_line}-{c.end_line}) "
                f"[{c.source_type}] score:{rc.final_score:.3f}\n"
                f"```{c.language if c.language != 'other' else ''}\n"
                f"{c.content}\n"
                f"```\n"
            )
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# IndexBuilder
# ---------------------------------------------------------------------------

class IndexBuilder:
    """Orchestrate incremental index building / updating."""

    INDEX_VERSION = "1.0"
    INDEX_SUBDIR = "v1"

    def __init__(self, skill_path: str, index_dir: Optional[str] = None):
        self.skill_path = skill_path
        # Allow caller to explicitly specify index_dir so that __file__
        # resolution inside _get_index_dir() is not the only source of truth.
        if index_dir is not None:
            self.index_dir = index_dir
        else:
            self.index_dir = _get_index_dir()
        os.makedirs(self.index_dir, exist_ok=True)
        self._manifest_path = os.path.join(self.index_dir, "manifest.json")
        self._metadata_path = os.path.join(self.index_dir, "chunks_metadata.jsonl")

    def index_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.index_dir, "vector_index.faiss")) and
            os.path.exists(self._manifest_path)
        )

    def _file_fingerprint(self, filepath: str) -> str:
        """Return mtime + size as a quick fingerprint."""
        try:
            stat = os.stat(filepath)
            return f"{stat.st_mtime:.6f}:{stat.st_size}"
        except Exception:
            return ""

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": self.INDEX_VERSION,
            "model": VectorIndex.MODEL_NAME,
            "dimension": VectorIndex.DIMENSION,
            "skill_path": self.skill_path,
            "file_fingerprints": {},
            "total_chunks": 0,
            "build_time": "",
        }

    def _save_manifest(self, manifest: Dict[str, Any]):
        with open(self._manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _load_existing_chunks(self) -> Dict[str, CodeChunk]:
        """Load previously indexed chunks from metadata jsonl."""
        chunks: Dict[str, CodeChunk] = {}
        if not os.path.exists(self._metadata_path):
            return chunks
        try:
            with open(self._metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    chunk = CodeChunk(**obj)
                    chunks[chunk.chunk_id] = chunk
        except Exception as e:
            logger.warning(f"Could not load existing chunk metadata: {e}")
        return chunks

    def _scan_files(self) -> List[Tuple[str, str]]:
        """Return list of (abs_path, rel_path) for files to index."""
        chunker = Chunker()
        files = []
        for root, _, filenames in os.walk(self.skill_path):
            for filename in filenames:
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, self.skill_path).replace('\\', '/')
                if chunker.should_index_file(rel_path):
                    files.append((abs_path, rel_path))
        return files

    def build_or_update(self) -> bool:
        """
        Build a fresh index or incrementally update an existing one.
        Returns True on success.
        """
        import time
        t0 = time.time()
        manifest = self._load_manifest()
        old_fingerprints = manifest.get("file_fingerprints", {})
        existing_chunks = self._load_existing_chunks()

        files = self._scan_files()
        new_fingerprints: Dict[str, str] = {}

        # Determine which files changed
        changed_files = []
        unchanged_chunk_ids = set()
        for abs_path, rel_path in files:
            fp = self._file_fingerprint(abs_path)
            new_fingerprints[rel_path] = fp
            if old_fingerprints.get(rel_path) == fp and existing_chunks:
                # File unchanged — keep its chunks
                for cid, chunk in existing_chunks.items():
                    if chunk.file_path == rel_path:
                        unchanged_chunk_ids.add(cid)
            else:
                changed_files.append((abs_path, rel_path))

        # Detect deleted files
        current_rel_paths = {rp for _, rp in files}
        for cid in list(existing_chunks.keys()):
            if existing_chunks[cid].file_path not in current_rel_paths:
                # File was deleted — mark its chunks for removal
                pass

        if not changed_files and existing_chunks:
            logger.info("Index is up-to-date (no file changes detected).")
            return True

        logger.info(
            f"Index update: {len(changed_files)} changed/new files, "
            f"{len(unchanged_chunk_ids)} unchanged chunks."
        )

        # Re-chunk changed files
        t1 = time.time()
        chunker = Chunker()
        new_chunks_list: List[CodeChunk] = []
        for abs_path, rel_path in changed_files:
            chunks = chunker.chunk_file(abs_path, rel_path)
            new_chunks_list.extend(chunks)
        logger.info(f"[CHUNK] {len(changed_files)} files chunked in {time.time()-t1:.1f}s")

        # Assemble final chunk list: unchanged + new
        final_chunks = []
        kept = 0
        for cid, chunk in existing_chunks.items():
            if cid in unchanged_chunk_ids:
                final_chunks.append(chunk)
                kept += 1
        final_chunks.extend(new_chunks_list)

        logger.info(f"Total chunks after update: {len(final_chunks)} (kept {kept}, new {len(new_chunks_list)})")

        if not final_chunks:
            logger.warning("No chunks to index. Check skill_path and P0/P1 filters.")
            return False

        # Build metadata jsonl
        logger.info("[STAGE 2/4] Saving chunk metadata...")
        t2 = time.time()
        with open(self._metadata_path, 'w', encoding='utf-8') as f:
            for chunk in final_chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        logger.info(f"[META] Metadata saved in {time.time()-t2:.1f}s")

        # Build Vector
        logger.info("[STAGE 3/4] Encoding chunks and building FAISS index (this may take a few minutes)...")
        t4 = time.time()
        vector_path = os.path.join(self.index_dir, "vector_index.faiss")
        vector = VectorIndex(vector_path)
        vector.build(final_chunks)
        logger.info(f"[FAISS] Index saved in {time.time()-t4:.1f}s")

        logger.info(f"[TOTAL] Index build completed in {time.time()-t0:.1f}s")

        # Save manifest
        manifest.update({
            "version": self.INDEX_VERSION,
            "model": VectorIndex.MODEL_NAME,
            "dimension": VectorIndex.DIMENSION,
            "skill_path": self.skill_path,
            "file_fingerprints": new_fingerprints,
            "total_chunks": len(final_chunks),
            "build_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_manifest(manifest)
        logger.info("Index build/update completed successfully.")
        return True

    def load_retriever(self) -> Optional[VectorRetriever]:
        """Load vector index from disk and return a ready VectorRetriever."""
        if not self.index_exists():
            return None

        vector_path = os.path.join(self.index_dir, "vector_index.faiss")

        vector = VectorIndex(vector_path)
        if not vector.load():
            return None

        # Load metadata
        chunks: Dict[str, CodeChunk] = {}
        if os.path.exists(self._metadata_path):
            try:
                with open(self._metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        chunk = CodeChunk(**obj)
                        chunks[chunk.chunk_id] = chunk
            except Exception as e:
                logger.warning(f"Failed to load chunk metadata: {e}")
                return None

        if not chunks:
            return None

        return VectorRetriever(vector, chunks)
