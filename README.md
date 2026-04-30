# SlicerAIAgent

> **⚠️ Work in Progress — Rough Prototype**
>
> This project is actively under development. The current version is a rough prototype for proof-of-concept and rapid iteration. Expect breaking changes, incomplete features, and unpolished UI.

---

## Motivation

When a clinician opens 3D Slicer, the goal is already clear — for example, comparing a patient's pre-operative CT with a 6-month follow-up after mandible reconstruction to assess bone graft integration. But turning that intent into results means navigating a dense UI, hunting through documentation, switching between segmentation tools and 3D views, and manually tuning dozens of parameters. The input data and desired outcome are certain; it is the execution path that is burdensome.

SlicerAIAgent closes this gap by letting clinicians state their goal in plain language and handling the rest end to end: searching the Slicer knowledge base for the right APIs, confirming exact function signatures, generating executable Python code, and running it directly inside Slicer to manipulate the scene. No scripting, no menu diving, no parameter guessing. This points toward a broader shift in medical software — from learning complex interfaces to simply stating intent and letting the agent operate the application on the user's behalf.

---

## Demos

### Demo 1 — Volume Rendering

**Prompt:**
```
load an example volume and generate a volume rendering
```

https://github.com/user-attachments/assets/256e8913-5942-4db3-b5b9-7841796f4eb3

The agent searches for volume-loading APIs, reads the relevant documentation, generates code to download the MRHead sample volume, and creates a GPU volume rendering with a preset transfer function.

---

### Demo 2 — Segmentation, 3D Reconstruction, and Plane Cutting

**Prompt:**
```
Load a sample volume, segment it using a random threshold, and reconstruct it into a 3D shape. Then, cut the 3D shape with a random plane and display each part in a different color
```

https://github.com/user-attachments/assets/7cf2846d-887b-48db-95fb-342e4c79fc07

The agent chains multiple Slicer operations: loading data → threshold-based segmentation → 3D surface reconstruction (`vtkMarchingCubes`) → arbitrary plane clipping → multi-color display of the clipped parts.

---

## Technical Approach

SlicerAIAgent is not a simple "prompt → code" wrapper. It implements a **multi-phase autonomous pipeline** that combines dense vector retrieval, autonomous tool calling, AST-based security validation, safe execution inside Slicer's own Python interpreter, and automatic self-correction on failure.

### 1. Architecture Overview

The extension follows Slicer's standard `ScriptedLoadableModule` pattern with four cooperating classes living in a single module file:

| Class | Responsibility |
|-------|----------------|
| `SlicerAIAgent` | Module metadata (title, category, icon, help text). |
| `SlicerAIAgentWidget` | Qt UI setup, real-time streaming display, auto-execution orchestration, settings persistence, and thinking timer. |
| `SlicerAIAgentLogic` | Business logic: pre-retrieval, LLM client coordination, tool execution, code validation, and safe execution scheduling. |
| `SlicerAIAgentTest` | Smoke tests for imports, validator rules, executor behavior, and skill path resolution. |

Core library code lives under `SlicerAIAgentLib/` and is organized into focused submodules:

| Library | Role |
|---------|------|
| `LLMClient.py` | HTTP client for OpenAI-compatible (Kimi, DeepSeek) and native Anthropic (Claude) APIs. Handles streaming, tool-calling loops, conversation history trimming, token/cost tracking, and query decomposition. |
| `SkillTools.py` | Tool executor: `FindFile`, `SearchSymbol`, `Grep`, `ReadFile`, and `VectorSearch`. Uses ripgrep for fast aggregated search and tree-sitter for AST-based symbol extraction. |
| `SkillIndexer.py` | Dense vector retrieval backend: chunking, ONNX Runtime embedding, FAISS indexing, source-type weighting, and incremental index building. |
| `CodeValidator.py` | AST-based static analysis to block dangerous imports and calls before execution. |
| `SafeExecutor.py` | Runs generated code in `__main__.__dict__` (same namespace as the Slicer Python Console). Captures stdout/stderr, intercepts VTK C++ errors, and rolls back the MRML scene on failure. |
| `ConversationStore.py` | Persistent conversation history via Qt `QSettings`, with export/import and per-session stats. |
| `SlicerCodeTemplates.py` | Reusable Slicer Python snippets for common operations. |

---

### 2. The Four-Phase Request Pipeline

Every user prompt travels through four distinct phases before the final result is shown in the Slicer scene.

#### Phase 1 — Dense Vector Pre-Retrieval (RAG)

Before the LLM ever sees the prompt, the system performs an **intelligent multi-retrieval** pass over a local dense vector index built from the Slicer knowledge base (`Resources/Skills/slicer-skill-full/`).

**Step-by-step:**

1. **Query Decomposition** — The `LLMClient.decomposeQuery()` method analyzes the user request. Simple requests (fewer than 12 words, no commas, at most one "and") are kept as-is. Complex multi-step requests are broken into 2–5 independent sub-queries via a lightweight LLM call. This ensures each sub-task gets its own focused semantic search.

2. **Per-Sub-Query Vector Search** — For each sub-query, `VectorRetriever.search()` runs a pure dense vector search against a FAISS `IndexFlatIP` index. The embedding model is `jinaai/jina-embeddings-v2-base-code` (768-dim) running via ONNX Runtime with GPU provider auto-detection (CUDA, DirectML, ROCm, CoreML, or CPU fallback). Each sub-query retrieves the top-10 most relevant chunks.

3. **Source-Type Weighting** — Retrieved chunks are re-ranked by a source-type multiplier:
   - `doc_example` × 1.3 (official cookbook examples)
   - `python_api` × 1.2 (`slicer.util`, core Python API)
   - `effect_implementation` / `scripted_module` × 1.1
   - `test_example` × 1.05
   - `source` × 1.0

4. **Smart Full-File Inclusion** — If a markdown file contributes 5 or more chunks to the concatenated result, the system replaces all individual chunks from that file with a single synthetic "whole file" chunk. This avoids redundant snippet injection when an entire topic file (e.g., `segmentations.md`) is highly relevant.

5. **Context Injection** — The formatted retrieval results, along with a "search coverage" note listing the sub-queries, are injected into the system prompt under the heading `## RELEVANT KNOWLEDGE BASE SNIPPETS`. The LLM is instructed to treat these snippets as its first source of truth and to avoid re-searching the same topics.

If the vector index is missing or not ready, this phase silently skips itself and the system falls back to the traditional pure tool-calling workflow.

#### Phase 2 — Autonomous Tool-Calling Loop

After pre-retrieval, the LLM is given **five tools** from the start: `FindFile`, `SearchSymbol`, `Grep`, `ReadFile`, and `VectorSearch`. The LLM autonomously decides the best path to a solution — no manual phase orchestration is required.

**Workflow:**

1. **Search** — The LLM may call `Grep`, `FindFile`, or `SearchSymbol` to locate relevant APIs across the skill knowledge base. Multiple tool calls are executed in parallel via a `ThreadPoolExecutor`.

2. **Read** — Once promising files are identified, the LLM calls `ReadFile` to confirm exact function signatures and usage patterns. `ReadFile` implements **smart slicing** to keep token usage low:
   - Files under 500 lines → full content.
   - Markdown files ≥500 lines → heading-based query matching.
   - Code files ≥500 lines → AST boundary extraction (complete function/class bodies via tree-sitter) or ±100-line grep-context fallback.
   - Python test files → test-method slicing (precise `def test_*` / `def TestSection_*` boundaries).
   - Duplicate read detection warns the LLM if it re-reads the same file with the same query.
   - Gist dead-end detection flags sections that contain only external gist links with no local executable examples.

3. **Generate** — When the LLM has enough information, it outputs a single ` ```python ` code block. The tool loop terminates **immediately** upon detecting this block. If the LLM reaches the maximum of 10 tool rounds without generating code, the system appends a force-generate user message and makes one final API call.

**Conversation history management:** Full tool results are used within the current turn, but before persisting to `conversation_history`, they are compressed via `_compressToolResultsForHistory()`. ReadFile content is passed through (already sliced at the tool layer), Grep results are kept as-is, and VectorSearch drops the large `formatted_context` field. This prevents context bloat across multi-turn conversations. History is also subject to FIFO character-based trimming (500K character limit), dropping the oldest messages first.

#### Phase 3 — Code Validation & Safe Execution

Generated code is **auto-executed** without requiring a manual button press.

**Pre-validation:**
- `CodeValidator` parses the code with the Python `ast` module.
- It blocks imports from dangerous modules (`os`, `subprocess`, `sys`, `socket`, `urllib`, `ctypes`, `pickle`, etc.).
- It blocks dangerous function calls (`eval`, `exec`, `compile`, `open`, `getattr`, `globals`, etc.).
- It detects potentially destructive operations (`RemoveNode`, `Clear`, `Delete`, `saveNode`, etc.) and flags them for confirmation.
- If validation fails, the system enters self-correction directly without attempting execution.

**Execution:**
- `SafeExecutor.execute()` runs the code in `sys.modules['__main__'].__dict__`, which is the **exact same namespace** as the Slicer Python Console. This means shortcuts like `getNode` (injected by `slicerqt.py`) are automatically available.
- Execution is scheduled via `qt.QTimer.singleShot(10, ...)` to stay on the Qt main thread, satisfying MRML scene and GUI thread-safety requirements.
- stdout and stderr are captured with `contextlib.redirect_stdout/stderr`.
- VTK C++ errors are intercepted by temporarily replacing the global `vtkOutputWindow` with a `vtkFileOutputWindow` pointing to a temp file. After execution, the log is read back, prefixed with `[VTK ERROR]` or `[VTK WARNING]`, and injected into the captured stderr. This allows the self-correction mechanism to react to runtime VTK errors even when no Python exception was raised.

**Scene rollback on failure:**
- Before execution, the system calls `slicer.mrmlScene.SaveStateForUndo()`.
- It also snapshots the set of existing node IDs.
- If execution raises an exception or times out, it calls `slicer.mrmlScene.Undo()` and then deletes any nodes whose IDs did not exist before execution (catching display nodes, storage nodes, and subject-hierarchy items that `Undo()` may miss because their `UndoEnabled` flag is `False`).
- The original undo flag is restored regardless of outcome.

#### Phase 4 — Self-Correction & Recovery

If execution fails, times out, or produces error indicators in stdout/stderr (including `[VTK ERROR]`), the agent automatically enters **self-correction mode**.

**How it works:**

1. An **isolated retry** is launched in a background thread via `chatWithToolsIsolated()`. This method runs the full tool-calling loop but does **not** read from or write to the main `conversation_history`, and it does **not** increment `turn_number`. Failed attempts therefore never pollute the user's conversation context.

2. The isolated prompt is constructed from:
   - The original system prompt (with current MRML scene context).
   - The original user prompt.
   - The prior tool trajectory from the failed attempt (assistant + tool messages), so the LLM does not fix blind.
   - The failed code block.
   - A user message containing the exact error and instructions to fix it.

3. The LLM may use tools again during correction (up to 5 tool rounds) to verify API signatures if the error suggests an incorrect call.

4. If correction succeeds, the corrected code replaces the failed code in `conversation_history`, a correction marker is appended, and the new code is auto-executed.

5. This repeats up to **5 attempts total**. If all attempts fail, the agent reports the final error to the user.

---

### 3. Streaming & Real-Time Feedback

Because MRML scene access and all UI updates must happen on the Qt main thread, HTTP I/O runs in a background `threading.Thread`, while UI updates are marshaled back via a thread-safe `queue.Queue`.

**Mechanism:**

- A `_streamQueue` is filled by the worker thread with events: `delta`, `complete`, `error`, `correction_complete`, `correction_error`, and `status`.
- A `QTimer` polls the queue every **50 ms** on the main thread and drains all pending events in a batch.
- Consecutive streaming deltas (reasoning/content) are batched into a single render pass to avoid calling `setHtml()` hundreds of times per second, which would block the main thread.
- Tool progress deltas are committed immediately as separate chat entries so the user sees search activity in real time.

**UI elements:**
- **Thinking timer (⏱)** — Updates every 100 ms while the LLM is working.
- **Status label** — Shows current phase: "Retrieving...", "Thinking...", "Searching & reading...", "Executing...", etc.
- **Token/cost label** — Displays per-turn cumulative token usage and estimated cost (e.g., `Turn 3 | Cumulative: 4,231 tokens | $0.0123`).

**Debug artifacts** written to the module directory per turn:
- `{turn}_first_prompt_debug.txt` — Full prompt sent to the LLM.
- `{turn}_code.txt` — Generated Python code.
- `{turn}_performance_log.txt` — Detailed phase-by-phase timing breakdown (scene context, retrieval, API wait, tool execution, validation, execution, self-correction).
- `thinking_history.txt` — Accumulated reasoning content across all turns.

---

### 4. Dense Vector Retrieval Backend

The `SkillIndexer.py` module implements a complete dense retrieval pipeline that can be built and updated independently via `scripts/build_rag.py`.

**Chunking (`Chunker`):**
- Indexes only high-value prefixes: `script_repository/`, `Base/Python/slicer/`, `Modules/Scripted/`, `Modules/Loadable/*/Testing/Python/`, `Libs/MRML/Core/`, etc.
- Python and C++ files are chunked by AST boundaries (function and class definitions) using tree-sitter.
- Markdown files are chunked by headings; code blocks under a heading stay with it.
- Each chunk's **embedding text** is enriched with the function signature, docstring, source-type label, and a type prefix for better natural-language-to-code matching.

**Embedding (`VectorIndex`):**
- Model: `jinaai/jina-embeddings-v2-base-code` exported to ONNX (~640 MB, downloaded once).
- Runtime: ONNX Runtime with mean-pooling and L2 normalization.
- Tokenizer: Lightweight `tokenizers` library (Rust-based, no heavy `transformers` import).
- Sequence length capped at 1024 tokens.
- Batch encoding with progress logging and ETA estimation.

**Search (`VectorRetriever`):**
- Pure dense search via FAISS `IndexFlatIP` (inner product on normalized vectors = cosine similarity).
- Source-type weighting applied post-search.
- Returns ranked chunks with file path, line range, similarity score, and boosted score.

**Incremental building (`IndexBuilder`):**
- Compares file mtime+size fingerprints against a manifest.
- Re-chunks only changed or new files.
- Rebuilds the FAISS index from the merged chunk list (unchanged + new).
- Stores metadata as JSONL and the manifest as JSON under `Resources/Code_RAG/v1/`.

---

### 5. Security Model

Generated code is treated as untrusted until validated. The security layers are:

**Blocked modules:** `os`, `subprocess`, `sys`, `socket`, `urllib`, `ctypes`, `pickle`, `marshal`, `imp`, `importlib._bootstrap`, etc.

**Blocked functions:** `eval`, `exec`, `compile`, `open`, `file`, `input`, `getattr`, `setattr`, `globals`, `locals`, `vars`, `dir`, `__import__`, etc.

**Allowed modules:** `slicer`, `vtk`, `qt`, `ctk`, `numpy`, `json`, `re`, `math`, `random`, `datetime`, `collections`, `itertools`, `io`, `base64`, `hashlib`, `typing`, etc.

**Runtime constraints:**
- No file I/O (`open()` is blocked).
- No network calls.
- No system commands or subprocesses.
- Execution is bounded by a 30-second cooperative timeout.
- The MRML scene can be rolled back on failure.

---

## Setup & Installation

### 1. Install the Extension

Add the repository root to Slicer's additional module paths:

1. Open 3D Slicer.
2. Go to **View → Application Settings → Modules → Additional module paths**.
3. Add the root of this repository.
4. Restart Slicer. The module appears under the **AI** category.

### 2. Configure API Key

Enter your API key in the Settings panel of the module UI. Supported providers:

- **Kimi / Moonshot** (default): `https://api.moonshot.cn/v1`
- **DeepSeek**: `https://api.deepseek.com`
- **Claude (Anthropic native)**: `https://api.anthropic.com/v1`

### 3. Build the Dense Vector Index (Optional but Recommended)

The vector index enables fast pre-retrieval of relevant knowledge base snippets before each LLM call.

```bash
python scripts/build_rag.py
```

This scans `Resources/Skills/slicer-skill-full/`, chunks changed files, encodes them with the ONNX model, and writes the FAISS index to `Resources/Code_RAG/v1/`. The first run downloads the ~640 MB ONNX model.

### 4. Run Tests

Tests must run inside a live Slicer instance:

```python
import unittest
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('SlicerAIAgentTest')
unittest.TextTestRunner(verbosity=2).run(suite)
```

---

## Related Projects

* **[slicer-skill](https://github.com/pieper/slicer-skill)** — The foundational Claude skill for 3D Slicer that pioneered the MCP integration and local documentation indexing workflow.
* **[SlicerClaw](https://github.com/jumbojing/slicerClaw)** — A lightning-fast AI assistant natively integrated into 3D Slicer.
* **[mcp-slicer](https://github.com/zhaoyouj/mcp-slicer)** — A standalone MCP server for 3D Slicer by @zhaoyouj, installable via `pip` / `uvx`. It uses Slicer's built-in WebServer API as a bridge and can be launched outside of Slicer.
* **[SlicerDeveloperAgent](https://github.com/muratmaga/SlicerDeveloperAgent)** — A Slicer extension by Murat Maga that embeds an AI coding agent directly inside 3D Slicer using Gemini, letting users prompt, run, and iterate on scripts and modules without leaving the application. See the [Discourse discussion](https://discourse.slicer.org/t/developer-agent-for-slicer/44787) for background.
* **[NA-MIC Project Week 44 — Claude Scientific Skill for Imaging Data Commons](https://projectweek.na-mic.org/PW44_2026_GranCanaria/Projects/ClaudeScientificSkillForImagingDataCommons/)** — A project that developed a Claude skill for the [Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (IDC), published at [ImagingDataCommons/idc-claude-skill](https://github.com/ImagingDataCommons/idc-claude-skill).
* **[SlicerChat: Building a Local Chatbot for 3D Slicer](https://arxiv.org/abs/2407.11987)** (Barr, 2024) — Explores integrating a locally-run LLM (Code-Llama Instruct) into 3D Slicer to assist users with the software's steep learning curve, investigating the effects of fine-tuning, model size, and domain knowledge (Python samples, Markdown docs, Discourse forums) on answer quality.
* **[Talk2View](https://talk2view.com/)** — A platform for conversational interaction with medical imaging data and visualization tools.
