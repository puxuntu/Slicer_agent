# SlicerAIAgent

> **⚠️ Work in Progress — Rough Prototype**
>
> This project is actively under development. The current version is a rough prototype for proof-of-concept and rapid iteration. Expect breaking changes, incomplete features, and unpolished UI.

---

## Motivation

3D Slicer is a powerful open-source platform for medical image computing, but writing Python scripts to perform even simple tasks—loading volumes, segmenting, rendering—requires significant domain knowledge and API familiarity.

**SlicerAIAgent** aims to lower this barrier by letting users describe what they want in natural language, while an LLM (Kimi / Claude) reasons about the task, searches the Slicer knowledge base, and generates executable Python code automatically.

The agent doesn't just "hallucinate" code. It **actively searches** the Slicer source tree, documentation, and script repository using built-in tools (Grep + ReadFile), confirms exact API signatures, and then generates the final script.

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

| Component | Implementation |
|-----------|----------------|
| **LLM Backend** | OpenAI-compatible APIs (Kimi / Claude) |
| **Tool Calling** | Autonomous strategy: LLM decides when to `Grep`, when to `ReadFile`, and when to generate code |
| **Code Execution** | Runs in `__main__.__dict__` (same namespace as Slicer Python Console); VTK errors are intercepted and injected into stderr |
| **Self-Correction** | Isolated `chatIsolated()` calls on failure; failed attempts do **not** pollute `conversation_history` |
| **Real-time Feedback** | Thinking timer (⏱) shows elapsed LLM time; per-round performance logs written to `{turn}_performance_log.txt` |

### Autonomous Tool-Calling Workflow

**SlicerAIAgent gives the LLM both `Grep` and `ReadFile` tools from the start**, letting it autonomously decide the best path to a solution:

1. **Search** — Call `Grep` to locate files containing relevant APIs (e.g., `downloadMRHead`, `setSliceViewerLayers`). Multiple searches can be batched in parallel.
2. **Read** — Call `ReadFile` to read the full content of the most relevant files to confirm exact function signatures and usage patterns. Multiple files can be read in parallel.
3. **Generate** — When the LLM has enough information, it outputs a single ` ```python ` code block and the loop terminates immediately.

The system simply provides the tools and waits for the LLM to output a ` ```python` code block — no manual phase orchestration required.

### Performance Optimization

| Optimization | Description | Impact |
|-------------|-------------|--------|
| **Zero-wait tool orchestration** | The LLM controls the entire workflow — searching, reading, and generating without any system-imposed phase switches or idle API rounds | **~2× speedup** on typical tasks |
| **Parallel tool execution** | Grep and ReadFile calls within the same round are executed concurrently via `ThreadPoolExecutor` | Reduces wall-clock tool time |
| **Local result compression** | ReadFile results are compressed deterministically (keep code blocks, truncate prose) before persisting to history | Prevents context bloat without extra LLM calls |
| **Batched UI updates** | Streaming deltas are accumulated and flushed in batches rather than per-delta | Eliminates main-thread blocking |

### Safety & Reliability

- **Syntax pre-validation** — Generated code is checked with `ast.parse` before execution.
- **Forbidden module list** — Code using `os`, `subprocess`, `eval`, `pickle`, `urllib`, etc. is rejected.
- **Self-correction loop** — If execution fails (Python exception or VTK error), the agent builds an isolated prompt (original request + failed code + error) and retries up to 5 times.
- **Conversation history isolation** — Failed self-correction attempts never pollute the main conversation context.

---

## Related Projects

* **[slicer-skill](https://github.com/pieper/slicer-skill)** — The foundational Claude skill for 3D Slicer that pioneered the MCP integration and local documentation indexing workflow.
* **[SlicerClaw](https://github.com/jumbojing/slicerClaw)** — A lightning-fast AI assistant natively integrated into 3D Slicer.
* **[mcp-slicer](https://github.com/zhaoyouj/mcp-slicer)** — A standalone MCP server for 3D Slicer by @zhaoyouj, installable via `pip` / `uvx`. It uses Slicer's built-in WebServer API as a bridge and can be launched outside of Slicer.
* **[SlicerDeveloperAgent](https://github.com/muratmaga/SlicerDeveloperAgent)** — A Slicer extension by Murat Maga that embeds an AI coding agent directly inside 3D Slicer using Gemini, letting users prompt, run, and iterate on scripts and modules without leaving the application. See the [Discourse discussion](https://discourse.slicer.org/t/developer-agent-for-slicer/44787) for background.
* **[NA-MIC Project Week 44 — Claude Scientific Skill for Imaging Data Commons](https://projectweek.na-mic.org/PW44_2026_GranCanaria/Projects/ClaudeScientificSkillForImagingDataCommons/)** — A project that developed a Claude skill for the [Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (IDC), published at [ImagingDataCommons/idc-claude-skill](https://github.com/ImagingDataCommons/idc-claude-skill).
* **[SlicerChat: Building a Local Chatbot for 3D Slicer](https://arxiv.org/abs/2407.11987)** (Barr, 2024) — Explores integrating a locally-run LLM (Code-Llama Instruct) into 3D Slicer to assist users with the software's steep learning curve, investigating the effects of fine-tuning, model size, and domain knowledge (Python samples, Markdown docs, Discourse forums) on answer quality.
