# SlicerAIAgent

An AI-powered assistant for [3D Slicer](https://www.slicer.org/) that turns natural language into executable scene manipulation. Clinicians state their intent; the agent grounds it against the Slicer knowledge base, plans the steps, generates safe Python code, and runs it directly inside the application.

SlicerAIAgent operates in two complementary modes:

- **Autonomous Mode** — For open-ended requests. The agent interprets the goal, searches documentation and source code on the fly, produces a structured plan, generates executable Python, validates it, and executes it automatically. If something fails, it self-corrects in an isolated loop without polluting the conversation.
- **Guided Workflow Mode** — For complex, multi-step extension-based procedures. The system pre-generates validated operation templates from extension cookbooks and executes them deterministically, mixing automated code steps with interactive 3D operations where the user places curves, planes, or fiducials directly in the Slicer scene.

---

## Demos

### Demo 1 — Pelvic fracture reduction

1. load the CT volume  
2. segment the pelvic fracture
3. implement the screw placement planning

https://github.com/user-attachments/assets/ea27d290-b2c5-4be0-9954-8051650bda90

---

### Demo 2 — Voxtell Segmentation

1. load an example CT chest volume
2. segment the Spine  
3. segment the left lung with the red color
4. segment the right lung with the green color
5. segment the rib with the yellow color

https://github.com/user-attachments/assets/59d5265e-e488-413f-84b5-55eb2f8a2da9

---

### Demo 3 — surgical planning of mandibular reconstruction

1. If the fibula is from the right leg, tick the "Right side leg" checkbox.
2. In the "Select mandibular segmentation" section, choose the mandibular segmentation.
3. In the "Select fibula segmentation" section, choose the fibula segmentation.
4. For the "Current Scalar Volume" option, select the Mandible Volume.
5. Click "Create bone models from segmentations" button.
6. Change the layout to "Conventional".
7. For the R (red) view, toggle on "slice visibility in 3D view".
8. For the R (red) view, toggle on "FOV, Spacing match 2D" (adjusts slice resolution to match the 2D viewport pixel spacing).
9. In the toolbar, turn on "slice intersection visibility". In the slice intersection interaction options, turn on "set interaction", then enable both "Translate" and "Rotate".
10. Manually adjust the slice intersection position by holding Shift and moving the mouse in a view.
11. Click the "Add mandibular curve" button.
12. Configure the display settings of the mandibular curve created by the "Add mandibular curve" button so it is shown in both "View 1" and "Red".
13. Manually click and draw on the "Red" view to create a curve along the mandible.
14. Change the layout to "BoneReconstructionPlanner".
15. For the R (red) view, toggle off "slice visibility in 3D view".
16. Manually set how many cut planes you want.
17. Click "Add cut plane" button.
18. Place one mandibular cut plane using the extension's Add cut plane workflow. If the user requested N cut planes, repeat the Add cut plane + place plane
  interaction N times. Do not store these planes as a rotation plane; they are mandibular cut planes managed by the extension.
19. Click "Add fibula line" button.
20. Draw a line over the fibula in "3D View 2", starting with the first point distally and the last point proximally.
21. Click "Center fibula line using fibula model" button to align the line with the anatomical axis of the fibula.
22. Click "Update fibula planes over fibula line; update fibula bone pieces and transform them to mandible" to generate the reconstruction and create the fibula cut planes.

https://github.com/user-attachments/assets/9b92f27d-2393-41b0-8216-6fc3dcd95a02

---

## Technique Points

SlicerAIAgent is not a simple prompt-to-code wrapper. It is built around a clear separation between **Offline Stage** (everything that happens before the user types a prompt) and **Online Stage** (everything that happens during the conversation). This split is the key reason the system can remain responsive, deterministic, and safe even when driving complex clinical workflows.

### Offline Stage — Knowledge Preparation & Code Generation

The offline stage transforms raw Slicer source code, documentation, and installed extensions into structured, searchable, and executable assets.

#### Dense Vector Index Building

Slicer has thousands of APIs across core modules, scripted extensions, and C++ libraries. Memorizing them in an LLM's weights is brittle and incomplete. Instead, the system builds a local dense-vector index from the Slicer knowledge base:

- **Chunking**: Python and C++ source files are split at AST boundaries (functions and classes); markdown documentation is split by heading. Each chunk is enriched with its signature, docstring, and source-type label to improve natural-language-to-code matching.
- **Embedding**: A code-specific embedding model encodes every chunk into a 768-dimensional vector using ONNX Runtime (with GPU auto-detection).
- **Indexing**: Vectors are stored in a FAISS index for fast inner-product (cosine) search. A manifest tracks file fingerprints so the index can be updated incrementally when the knowledge base changes.

Running `python scripts/build_rag.py` rebuilds or refreshes this index. The first run downloads the ~640 MB ONNX model.

#### UI Pre-Analysis

A major challenge in medical imaging software is the gap between what a user says and what the API expects. A user might say "turn on slice intersections," but the executable API is `SetIntersectingSlicesEnabled(True)` on a `vtkMRMLCrosshairNode`. The UI pre-analysis pipeline (`scripts/build_ui_analysis.py`) closes this gap by scanning Slicer's UI definitions and mapping user-facing labels, actions, and tooltips to their nearby implementation and API evidence. At runtime, the agent can search this index to translate a UI description into the correct executable call.

#### Extension CLI Generator

For complex extensions with multi-step workflows (e.g., surgical planning tools), repeatedly asking an LLM to replan at every step is slow, expensive, and non-deterministic. The Extension CLI Generator solves this by pre-compiling extension workflows into validated code templates:

1. **Source Scanning** — Discovers installed extensions from the Extension Manager, additional module paths, or loaded modules.
2. **Cookbook Parsing** — Reads a human-written markdown cookbook that describes the extension's step-by-step workflow. The cookbook is the *contract*: without it, generation does not proceed.
3. **Logic Analysis** — The LLM analyzes the extension source code to identify operations, parameters, and dependencies.
4. **API Probing** — Live code probes run inside Slicer's Python environment to verify actual API signatures and runtime behavior.
5. **Template Generation** — Two types of templates are produced:
   - *Extension operations* (`extension_op`): calls into the extension's own Python API.
   - *Slicer operations* (`slicer_op`): calls into Slicer core APIs, generated via the same autonomous tool-calling loop the main agent uses.
6. **Validation** — Generated templates are checked with AST parsing and static validation rules.
7. **Auto-Revision** — If validation fails, the LLM receives the errors and the original analysis context and revises the templates automatically.

The output is a validated Extension CLI: tool schemas, code templates, a workflow graph, and a prompt fragment. Once generated, this CLI can be loaded at runtime and executed deterministically.

---

### Online Stage — Runtime Agent & Workflow Execution

The online stage handles every user turn. Its design principle is: **ground first, then generate; validate first, then execute; recover automatically on failure.**

#### Dense Vector Pre-Retrieval (RAG)

Before the LLM ever sees the user prompt, the system performs an intelligent multi-retrieval pass:

1. **Query Decomposition** — Complex multi-step requests are broken into 2–5 independent sub-queries. Simple requests stay as-is.
2. **Per-Sub-Query Vector Search** — Each sub-query searches the FAISS index. The top-10 most relevant chunks are retrieved per sub-query.
3. **Source-Type Weighting** — Results are re-ranked by provenance: official cookbook examples get the highest boost, followed by core Python APIs, then effect implementations and test examples.
4. **Smart Full-File Inclusion** — If a single markdown file contributes 5 or more chunks, the system replaces all its individual chunks with one synthetic "whole file" chunk to avoid redundant snippet injection.
5. **Context Injection** — The formatted results are injected into the system prompt as the LLM's first source of truth, with instructions to avoid re-searching the same topics.

If the vector index is missing, this step silently skips and the system falls back to the traditional tool-calling workflow.

#### Autonomous Tool-Calling Loop

After pre-retrieval, the LLM is given search tools (`VectorSearch`, `Grep`, `ReadFile`, `SearchSymbol`) and autonomously decides how to ground the request. The loop works as follows:

- **Search** — The LLM calls `Grep` or `SearchSymbol` to locate APIs, or `VectorSearch` for semantic matches. Multiple tool calls execute in parallel.
- **Read** — Once promising files are identified, `ReadFile` confirms exact function signatures and usage patterns. For large files, it uses smart slicing: heading-based extraction for markdown, AST boundary extraction for code, and test-method slicing for Python test files.
- **Generate** — When the LLM has enough evidence, it outputs a structured `agent_plan` JSON block followed by a complete `python` code block. The loop terminates when executable code is detected.

For anatomical segmentation requests, a dedicated `GenerateSegmentationCode` tool produces a ready-to-run VoxTell snippet with GPU detection and model-path resolution.

Conversation history is compressed before persistence: tool results are summarized, vector search drops large formatted context fields, and a FIFO character limit (500K) trims the oldest messages first. This prevents context bloat across long multi-turn sessions.

#### Role-Composed Agent Pipeline

Every user prompt travels through an explicit internal pipeline. These roles are encoded in the system prompt, status UI, debug logs, and role trace, but they do not require separate LLM calls for every role.

| Role | Function |
|------|----------|
| **Observer** | Reads the user request and current MRML scene context. |
| **Retriever** | Uses dense retrieval and search/read tools to ground Slicer APIs. |
| **Planner** | Produces a structured `agent_plan` with task summary, overall confidence, risk, and assumptions. Each step includes API evidence and may declare machine-checkable scene expectations. |
| **Programmer** | Produces the complete executable Python block. |
| **Safety Critic** | Validates the plan and code, blocks unsafe operations, and flags destructive actions for confirmation. |
| **Executor** | Runs the code on Slicer's Qt main thread. |
| **Repairer** | Performs isolated self-correction if validation or execution fails. |

The Planner's `agent_plan` is a first-class artifact. It includes confidence scores, risk levels, and `expected_scene_change` declarations (e.g., `node_exists`, `node_count_delta`). After execution, the agent verifies these expectations and enters self-correction if the scene does not match.

#### Security & Safe Execution

Generated code is treated as untrusted until validated. The security layers are:

- **Blocked modules**: `os`, `subprocess`, `sys`, `socket`, `urllib`, `ctypes`, `pickle`, `marshal`, etc.
- **Blocked functions**: `eval`, `exec`, `compile`, `open`, `getattr`, `globals`, `locals`, `__import__`, etc.
- **Allowed modules**: `slicer`, `vtk`, `qt`, `ctk`, `numpy`, `SimpleITK`, `json`, `re`, `math`, and standard safe libraries.
- **Runtime constraints**: No file I/O, no network calls, no subprocesses. Execution is bounded by a 30-second cooperative timeout.

`SafeExecutor.execute()` runs code in `sys.modules['__main__'].__dict__` — the exact same namespace as the Slicer Python Console — so shortcuts like `getNode` are automatically available. Execution is scheduled via `qt.QTimer.singleShot` to stay on the Qt main thread, satisfying MRML scene and GUI thread-safety requirements. stdout and stderr are captured, and VTK C++ errors are intercepted by temporarily redirecting the global `vtkOutputWindow` to a temp file, so the self-correction mechanism can react to runtime VTK errors even when no Python exception was raised.

Before execution, the system calls `slicer.mrmlScene.SaveStateForUndo()`. If execution raises an exception or times out, it calls `Undo()` and deletes any nodes whose IDs did not exist before execution (catching display nodes, storage nodes, and subject-hierarchy items that `Undo()` may miss).

#### Self-Correction

If execution fails, the agent automatically enters self-correction mode:

1. An **isolated retry** is launched in a background thread. It runs the full tool-calling loop but does **not** read from or write to the main conversation history, and it does **not** increment the turn number. Failed attempts never pollute the user's context.
2. The isolated prompt includes the original system prompt, the original user prompt, the full prior tool trajectory, the previous `agent_plan`, the failed code, and the exact error message.
3. The LLM may use tools again during correction (up to 5 tool rounds) to verify API signatures.
4. If correction succeeds, the corrected plan and code replace the failed versions in history, a correction marker is appended, and the new code is auto-executed.
5. This repeats up to **5 attempts total**. If all fail, the agent reports the final error to the user.

#### Guided Workflow Runtime

When a validated Extension CLI is active, the system switches from the autonomous LLM loop to a deterministic workflow runtime:

- **Turn Routing**: A lightweight router classifies every user prompt. Simple control words (`done`, `proceed`, `skip`, `cancel`) are routed directly to the workflow engine. New autonomous requests that conflict with an active workflow are queued until the workflow finishes.
- **WorkflowOrchestrator**: A state machine that tracks the current step, completed steps, and node registry. Steps can be:
  - *Automated* — code runs immediately.
  - *Interactive* — the system creates markup nodes, enters Slicer placement mode, and waits for the user to place points or draw curves.
  - *Mixed* — automated setup runs first, then the system pauses for user interaction.
  - *Branch* — conditional paths based on user choice.
  - *User Choice* — the LLM or a lightweight resolver selects from scene nodes or predefined options.
- **InteractionManager**: Handles low-level Slicer 3D coordination: markup node creation, placement mode entry/exit, debounced VTK observers, and validation (e.g., minimum control point counts).
- **NodeChoiceResolver**: When a workflow step needs to choose between multiple scene nodes (e.g., "which volume to use?"), a narrow LLM call resolves the ambiguity by matching node names against the step's described role.

This design means a complex 10-step surgical planning workflow can run with minimal LLM overhead: the planning and code generation happened offline; online execution is fast, deterministic, and interruptible.

#### Streaming & Real-Time Feedback

Because MRML scene access and all UI updates must happen on the Qt main thread, HTTP I/O runs in a background `threading.Thread`, while UI updates are marshaled back via a thread-safe `queue.Queue`:

- A `_streamQueue` is filled by the worker thread with events: `delta`, `complete`, `error`, `correction_complete`, `status`.
- A `QTimer` polls the queue every 50 ms on the main thread and drains all pending events in a batch.
- Consecutive streaming deltas are batched into a single render pass to avoid blocking the main thread.
- Tool progress deltas are committed immediately as separate chat entries so the user sees search activity in real time.

The UI shows a **thinking timer**, a **role-aware status label** (`Observer: Reading request...`, `Retriever: Searching...`, `Planner/Programmer: Generating...`, etc.), and a **per-turn token/cost label**.

Debug artifacts are written under timestamped run folders (`logs/YYYYMMDD_HHMMSS_turnN/`) and include the parsed plan, generated code, prompts, performance timing, role trace, and thinking history.

---

## Related Projects & Acknowledgments

- **[slicer-skill](https://github.com/pieper/slicer-skill)** — The foundational Claude skill for 3D Slicer that pioneered the MCP integration and local documentation indexing workflow.
- **[SlicerClaw](https://github.com/jumbojing/slicerClaw)** — A lightning-fast AI assistant natively integrated into 3D Slicer.
- **[mcp-slicer](https://github.com/zhaoyouj/mcp-slicer)** — A standalone MCP server for 3D Slicer by @zhaoyouj, installable via `pip` / `uvx`. It uses Slicer's built-in WebServer API as a bridge and can be launched outside of Slicer.
- **[SlicerDeveloperAgent](https://github.com/muratmaga/SlicerDeveloperAgent)** — A Slicer extension by Murat Maga that embeds an AI coding agent directly inside 3D Slicer using Gemini, letting users prompt, run, and iterate on scripts and modules without leaving the application. See the [Discourse discussion](https://discourse.slicer.org/t/developer-agent-for-slicer/44787) for background.
- **[NA-MIC Project Week 44 — Claude Scientific Skill for Imaging Data Commons](https://projectweek.na-mic.org/PW44_2026_GranCanaria/Projects/ClaudeScientificSkillForImagingDataCommons/)** — A project that developed a Claude skill for the [Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (IDC), published at [ImagingDataCommons/idc-claude-skill](https://github.com/ImagingDataCommons/idc-claude-skill).
- **[SlicerChat: Building a Local Chatbot for 3D Slicer](https://arxiv.org/abs/2407.11987)** (Barr, 2024) — Explores integrating a locally-run LLM (Code-Llama Instruct) into 3D Slicer to assist users with the software's steep learning curve, investigating the effects of fine-tuning, model size, and domain knowledge on answer quality.
- **[Talk2View](https://talk2view.com/)** — A platform for conversational interaction with medical imaging data and visualization tools.
- **[VoxTell](https://github.com/MIC-DKFZ/VoxTell)** — A Slicer extension for text-promptable AI segmentation of anatomical structures, enabling natural-language-driven organ and tissue segmentation.
