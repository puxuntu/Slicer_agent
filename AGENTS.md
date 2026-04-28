<!-- AGENTS.md — SlicerAIAgent -->
# SlicerAIAgent — Agent Guide

> **Project status:** Rough prototype. Expect breaking changes and incomplete features.

## Project Overview

SlicerAIAgent is a **3D Slicer extension** that embeds an AI coding assistant directly inside the Slicer application. Users control Slicer via natural language: the agent searches a local Slicer knowledge base (the "skill"), confirms API signatures, generates executable Python code, and runs it inside Slicer's Python interpreter to manipulate the MRML scene.

The extension is built as a standard Slicer **scripted module** and follows Slicer's `ScriptedLoadableModule` pattern.

## Technology Stack

| Layer | Technology |
|-------|------------|
| Host application | 3D Slicer (Python/Qt/VTK environment) |
| Module framework | Slicer `ScriptedLoadableModule` |
| UI toolkit | Qt / ctk (Slicer's bundled versions) |
| Build system | CMake (`CMakeLists.txt`) |
| LLM backend | OpenAI-compatible APIs (Kimi / Moonshot) and native Anthropic Messages API (Claude) |
| HTTP client | `urllib.request` (standard library); `httpx` declared in `requirements.txt` |
| Optional parsing | `tree-sitter` + `tree-sitter-python` + `tree-sitter-cpp` (auto-installed at runtime) |
| Dense retrieval | `faiss-cpu`, `onnxruntime`, `transformers` (auto-installed at runtime) |
| Other deps | `numpy`, `jsonschema` (from `requirements.txt`) |

## Project Structure

```
SlicerAIAgent.py              # Main module: Module, Widget, Logic, and Test classes
SlicerAIAgentLib/             # Core library package
├── __init__.py               # Package exports
├── LLMClient.py              # LLM API client: streaming, tool calling, history, pricing
├── SkillTools.py             # Tool executor: FindFile, SearchSymbol, Grep, ReadFile, VectorSearch
├── SkillIndexer.py           # Dense vector retrieval index: Chunker, VectorIndex, VectorRetriever, IndexBuilder
├── CodeValidator.py          # AST-based security validation before code execution
├── SafeExecutor.py           # Code execution in __main__.__dict__ with undo/rollback
├── ConversationStore.py      # Persistent conversation history via Qt QSettings
└── SlicerCodeTemplates.py    # Reusable Slicer Python code snippets
Resources/
├── Code_RAG/                 # Vector index artifacts (gitignored, created by scripts/build_rag.py)
├── Icons/                    # Extension icon
├── Info/
│   └── version_info.py       # Version metadata (__version__ = "0.1.0")
├── UI/
│   └── SlicerAIAgent.ui      # Qt Designer UI file (loaded at runtime)
└── Prompts/
    ├── README.md             # Prompt documentation
    └── system_prompt.md      # LLM system prompt template (loaded externally)
Testing/
└── SlicerAIAgentTest.py      # Comprehensive unit / integration test suite
scripts/build_rag.py          # Standalone script to build/update the dense vector retrieval index
CMakeLists.txt                # Slicer extension build configuration
requirements.txt              # Python dependencies for Slicer to install
```

### Key Files for Agents

- **`SlicerAIAgent.py`** — The monolithic module file containing all four Slicer scripted module classes: `SlicerAIAgent` (metadata), `SlicerAIAgentWidget` (UI setup, streaming display, auto-execution, self-correction, settings persistence), `SlicerAIAgentLogic` (orchestrates LLM client, tool executor, validator, executor, and dense vector retrieval), and `SlicerAIAgentTest` (smoke tests). The Widget implements a queue-based streaming UI (`_streamQueue` + `QTimer` polling at 50 ms) and auto-executes generated code with up to 5 self-correction attempts.
- **`SlicerAIAgentLib/LLMClient.py`** — Handles all LLM communication. Supports both OpenAI-compatible and native Anthropic Messages API formats. Implements streaming (`chatStream`), non-streaming (`chat`), tool-calling loops (`chatWithTools`, `chatWithToolsIsolated`), token/cost tracking, conversation history FIFO trimming (500K character limit), SSE parsing, query decomposition (`decomposeQuery`), and combined decomposition + HyDE (`decomposeQueryWithHyDE`) for multi-step retrieval. `_buildSystemPrompt` injects vector retrieval results and current MRML scene XML when available.
- **`SlicerAIAgentLib/SkillTools.py`** — Executes the five tools given to the LLM: `FindFile`, `SearchSymbol`, `Grep`, `ReadFile`, `VectorSearch`. Uses `ripgrep` (bundled `rg.exe` on Windows, or system `rg`) for fast aggregated grep. `ReadFile` implements smart slicing: AST boundary extraction (via tree-sitter), markdown heading queries, grep-context fallback, and test-method slicing. `VectorSearch` delegates to `SkillIndexer.VectorRetriever` if a local index is present.
- **`SlicerAIAgentLib/SkillIndexer.py`** — Dense vector retrieval backend. Contains:
  - `Chunker` — splits P0/P1 knowledge-base files (`.py`, `.cxx`/`.cpp`/`.h`, `.md`) into semantic chunks using tree-sitter AST boundaries or markdown headings. Embeds function signatures and docstrings into the embedding text for better natural-language-to-code matching.
  - `VectorIndex` — dense semantic retrieval via ONNX Runtime (`jinaai/jina-embeddings-v2-base-code` ONNX export, 768-dim) + FAISS (`IndexFlatIP`).
  - `VectorRetriever` — pure dense vector search with source-type weighting (`doc_example` ×1.3, `python_api` ×1.2, etc.). Deduplicates to max 3 chunks per file. Supports `merge_results_with_quota` for multi-query decomposition.
  - `IndexBuilder` — incremental index builder. Scans skill tree, compares file mtime fingerprints, re-chunks only changed files, then rebuilds the FAISS vector index. Stores everything under `<repo>/Resources/Code_RAG/v1/`; model cache lives under `<repo>/Resources/Code_RAG/models/`.
- **`SlicerAIAgentLib/SafeExecutor.py`** — Runs generated code in `sys.modules['__main__'].__dict__` (same namespace as the Slicer Python Console). Captures stdout/stderr, intercepts VTK errors via `vtkFileOutputWindow`, supports cooperative timeout, and rolls back the MRML scene on failure (`SaveStateForUndo` + node-ID-based cleanup). Execution is scheduled via `qt.QTimer.singleShot` to stay on the main thread.
- **`Resources/Prompts/system_prompt.md`** — The external system prompt loaded at runtime. Contains the LLM's search strategy, source tree map, critical rules, and code execution environment description.

## Build and Test Commands

### Build (inside Slicer build tree)

This is a standard Slicer extension. Build it by pointing CMake to the Slicer source/build tree:

```bash
mkdir build && cd build
cmake -DSlicer_DIR:PATH=/path/to/Slicer-SuperBuild-Build/Slicer-build ..
cmake --build . --config Release
```

Or load it directly into a Slicer installation for development:
1. Open 3D Slicer.
2. Go to **View → Application Settings → Modules → Additional module paths**.
3. Add the root of this repository.
4. Restart Slicer. The module appears under the **AI** category.

### Install Python Dependencies

Slicer's build system automatically installs packages listed in `requirements.txt` via `slicer_install_python_package`. For manual installation inside Slicer's Python console:

```python
slicer.util.pip_install("httpx>=0.24.0")
slicer.util.pip_install("numpy>=1.21.0")
slicer.util.pip_install("jsonschema>=4.0.0")
```

`tree-sitter` and its language parsers are auto-installed at runtime by `SkillTools.py` if missing.

### Build Dense Vector Retrieval Index

Run the standalone script to create or incrementally update the dense vector index:

```bash
python scripts/build_rag.py
```

This scans `Resources/Skills/slicer-skill-full/`, chunks changed files, and writes index artifacts to `Resources/Code_RAG/v1/`. The index directory and model cache are gitignored.

### Run Tests

Tests are Slicer `ScriptedLoadableModuleTest` classes and must run from **inside Slicer's Python console**:

```python
import unittest
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('SlicerAIAgentTest')
unittest.TextTestRunner(verbosity=2).run(suite)
```

The main test file is `Testing/SlicerAIAgentTest.py`. It tests module imports, LLMClient helpers, CodeValidator rules, SafeExecutor behavior, ConversationStore persistence, SlicerCodeTemplates, SkillIndexer basics, and an integration workflow.

## Code Style Guidelines

- **Language:** All code, comments, and docstrings are written in **English**.
- **Naming:** Follow Slicer/Qt conventions.
  - Methods and variables: `camelCase` (e.g., `onSendButtonClicked`, `_buildSceneContext`).
  - Classes: `PascalCase` (e.g., `SlicerAIAgentLogic`, `SafeExecutor`).
  - Private/protected members: leading underscore (e.g., `_streamQueue`, `_timing`).
- **Module pattern:** Slicer scripted modules use a 4-class structure:
  1. `SlicerAIAgent(ScriptedLoadableModule)` — module metadata
  2. `SlicerAIAgentWidget(ScriptedLoadableModuleWidget)` — UI setup and event handling
  3. `SlicerAIAgentLogic(ScriptedLoadableModuleLogic)` — business logic, LLM interaction, code execution
  4. `SlicerAIAgentTest(ScriptedLoadableModuleTest)` — unit tests
- **String formatting:** Prefer f-strings for readability. Multi-line HTML building uses f-strings with joined lists.
- **Logging:** Use the standard `logging` module. Loggers are module-level (`logger = logging.getLogger(__name__)`).

## Testing Instructions

- Tests **must** run inside a live Slicer instance because they import `slicer`, `qt`, `vtk`, and `ctk`.
- Each test calls `slicer.mrmlScene.Clear(0)` in `setUp` to ensure isolation.
- The test suite in `Testing/SlicerAIAgentTest.py` covers:
  - Module import and component initialization
  - LLMClient defaults, model normalization, history management, SSE parsing, code extraction
  - CodeValidator: safe code acceptance, syntax error rejection, blocked import rejection, destructive-op detection
  - SafeExecutor: simple execution, output capture, exception handling, global injection
  - Skill path resolution and mode detection
  - ConversationStore: add/get, stats, search, export/import round-trip
  - SlicerCodeTemplates: retrieval, tag/keyword search, prompt formatting
  - Integration: end-to-end component wiring

## Architecture and Runtime Behavior

### Dense Vector Pre-Retrieval (Phase 1)

Before each LLM request, `SlicerAIAgentLogic` automatically calls `_buildRetrievalContext`, which:
1. Decomposes complex user prompts into sub-queries via `llmClient.decomposeQueryWithHyDE`.
2. Runs `VectorRetriever.search` for each sub-query (top-15 per query).
3. Merges results with `merge_results_with_quota` (quota per sub-query = 3, total slots ≈ max(15, len(sub_queries) × 5)).
4. Formats the result and injects it into the system prompt under `## RELEVANT KNOWLEDGE BASE SNIPPETS`.

If the index is missing or not ready, the system silently falls back to the traditional pure tool-calling workflow.

### Autonomous Tool-Calling Workflow

The LLM is given five tools from the start (`FindFile`, `SearchSymbol`, `Grep`, `ReadFile`, `VectorSearch`) and decides autonomously how to use them:

1. **Search** — `Grep` or `SearchSymbol` to locate relevant APIs across the skill knowledge base.
2. **Read** — `ReadFile` to confirm exact function signatures and usage patterns.
3. **Generate** — When the LLM outputs a ` ```python ` block, the loop terminates and the code is extracted.

Tool results are compressed before being persisted to conversation history to prevent context bloat.

### Code Execution Pipeline

1. **Pre-validation** — `CodeValidator` performs AST analysis to catch syntax errors and blocked imports/calls.
2. **Execution** — `SafeExecutor` runs the code in `__main__.__dict__` (main thread, via `QTimer.singleShot`).
3. **Output capture** — stdout/stderr are captured. VTK C++ errors are intercepted and injected into stderr.
4. **Feedback** — Execution results are appended to conversation history (max 2 recent feedback messages kept).
5. **Self-correction** — On failure or detected errors in output, an isolated retry (`chatWithToolsIsolated`) is triggered. Failed attempts do **not** pollute the main `conversation_history`. Up to 5 attempts total.
6. **Scene rollback** — On execution failure or timeout, the MRML scene is rolled back via `Undo()` and node-ID-based cleanup.

### Streaming and UI

- LLM responses stream in a background `threading.Thread`. Deltas are queued to the Qt main thread via `queue.Queue` + `QTimer` polling (50ms interval).
- A thinking timer (`⏱`) shows elapsed LLM time in real time.
- Token usage and estimated cost are tracked per-turn and displayed in the UI.
- Debug artifacts are written to the module directory:
  - `{turn}_first_prompt_debug.txt` — full prompt sent to the LLM
  - `{turn}_code.txt` — generated code
  - `{turn}_performance_log.txt` — detailed timing breakdown
  - `thinking_history.txt` — accumulated reasoning content

## Security Model

- **Blocked modules:** `os`, `subprocess`, `sys`, `socket`, `urllib`, `ctypes`, `pickle`, etc.
- **Blocked functions:** `eval`, `exec`, `compile`, `open`, `getattr`, `globals`, etc.
- **Allowed modules:** `slicer`, `vtk`, `qt`, `ctk`, `numpy`, `json`, `re`, `math`, etc.
- Generated code cannot perform file I/O, network calls, or system commands.

## Deployment and Packaging

- The extension is packaged via Slicer's CMake/CPack machinery (`include(${Slicer_EXTENSION_CPACK})`).
- The skill knowledge base (`Resources/Skills/slicer-skill-full/`) is **not** version-controlled. It is a large external dependency that must be set up separately (see `.gitignore`).
- Runtime debug output files (`*_prompt_debug.txt`, `*_code.txt`, `*_performance_log.txt`, `thinking_history.txt`) are gitignored.
- Dense vector index artifacts (`Resources/Code_RAG/`) are gitignored.

## Important Notes for Agents

- **Do not assume the skill directory exists** in a fresh clone. The `skill_mode` will be `"unknown"` if the stamp file is missing.
- **Do not refactor the monolithic module file into multiple files** without also updating `CMakeLists.txt` and the Slicer macro calls. Slicer scripted modules expect the main `.py` file to declare the classes together.
- **Any change to the system prompt** should be made in `Resources/Prompts/system_prompt.md`, not hardcoded in `LLMClient.py`. The UI and logic rely on external loading.
- **The code is auto-executed** after generation. There is no manual "Execute" button in the current UI flow (the buttons were removed). Self-correction is automatic.
- **Retrieval dependencies** (`faiss-cpu`, `onnxruntime`, `transformers`) are commented out in `requirements.txt` and are auto-installed at runtime via `SkillIndexer._ensure_packages` (mirroring the tree-sitter installation pattern). This avoids breaking Slicer builds if the packages are not immediately available.
- **Qt thread safety:** MRML scene access and all UI updates must happen on the main thread. Background threads are used only for HTTP I/O. Use `qt.QTimer.singleShot(0, ...)` or the internal `_streamQueue` pattern to marshal results back to the main thread.
- **Do not push this repo to GitHub automatically unless the user asks you to do so.**
