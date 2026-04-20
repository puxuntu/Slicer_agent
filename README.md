# SlicerAIAgent

> **⚠️ Work in Progress — Rough Prototype**
>
> This project is actively under development. The current version is a rough prototype for proof-of-concept and rapid iteration. Expect breaking changes, incomplete features, and unpolished UI.

---

## Motivation

3D Slicer is a powerful open-source platform for medical image computing, but writing Python scripts to perform even simple tasks—loading volumes, segmenting, rendering—requires significant domain knowledge and API familiarity.

**SlicerAIAgent** aims to lower this barrier by letting users describe what they want in natural language, while an LLM (Kimi / Claude) reasons about the task, searches the Slicer knowledge base, and generates executable Python code automatically.

The agent doesn't just "hallucinate" code. It follows a three-phase workflow: search for relevant APIs → read source files to confirm exact signatures → generate the final script.

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
| **LLM Backend** | OpenAI-compatible APIs (Kimi / Claude) with streaming SSE support |
| **Tool Calling** | Three-phase strategy: `Grep` → `ReadFile` → `Generate` |
| **Code Execution** | Runs in `__main__.__dict__` (same namespace as Slicer Python Console); VTK errors are intercepted and injected into stderr |
| **Self-Correction** | Isolated `chatIsolated()` calls on failure; failed attempts do **not** pollute `conversation_history` |
| **Real-time Feedback** | Thinking timer (⏱) shows elapsed LLM time; per-round performance logs written to `{turn}_performance_log.txt` |
| **Streaming UI** | Deltas are batched before `setHtml()` to prevent main-thread blocking |

### Three-Phase Workflow

1. **Search Phase** — The LLM uses `Grep` to locate files containing relevant APIs (e.g., `downloadMRHead`, `setSliceViewerLayers`).
2. **ReadFile Phase** — The LLM reads the full content of the most relevant files to confirm exact function signatures and usage patterns.
3. **Generate Phase** — No tools available. The LLM writes a single ` ```python ` code block with the complete executable script.

If the generated code fails at runtime (Python exception or VTK error), the agent enters **self-correction mode**: it sends an isolated prompt (original request + failed code + error) and retries execution up to 5 times.
