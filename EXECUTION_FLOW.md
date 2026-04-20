# SlicerAgent 完整执行流程

> 本文档描述从用户在 UI 输入 prompt 到代码最终执行的完整时间线。

---

## 第一阶段：用户点击 Send

**执行线程：Slicer 主线程（Qt GUI 线程）**

1. 用户点击 Send 按钮，或按 Ctrl+Enter。
2. 程序读取输入框文字，清空输入框，将用户消息显示在 Chat 区域。
3. 界面状态变为 "Generating..."，Send 按钮变灰禁用。
4. 启动**实时计时器**（⏱ 0.0s），每 100ms 刷新，显示在状态栏右侧。
5. 保存当前回合编号（用于命名 debug 文件如 `1_code.txt`、`1_performance_log.txt`）。
6. 初始化**性能计时器**（`turn_start`），开始记录本轮各阶段耗时。
7. 读取当前 Slicer 场景的 MRML 数据，打包为上下文信息。
8. **构建完整 prompt**：system_prompt.md + SKILL.md + MRML 场景上下文 + 用户请求，写入 `1_first_prompt_debug.txt`。
9. **启动后台线程**，将完整 prompt 交给 LLM 处理。

> 从此刻起，工作分成两条线：后台线程负责与 LLM 服务器通信，主线程继续运行 UI，防止界面卡死。

---

## 第二阶段：后台线程与 LLM 对话（三阶段工具调用）

**执行线程：后台线程（网络 I/O）**

后台线程调用 `chatWithTools`，与 LLM 进行最多 20 轮交互，通常经历 3 个明确阶段。

> **可用工具**：`Grep`、`ReadFile`。`Glob` 已移除（Windows 环境下性能极差且结果不可靠）。

### 阶段 1：搜索（Search）

LLM 只能使用 **Grep** 工具。

**搜索策略（由 system prompt 强制规定）：**

1. **Analyze** — LLM 将用户请求拆解为子任务（如 "load volume | segment | reconstruct 3D | clip | color"）。
2. **Map** — 每个子任务映射到对应的 script repository topic 文件：
   - volumes → `script_repository/volumes.md`
   - segmentations/threshold → `script_repository/segmentations.md`
   - 3D models/display/color → `script_repository/models.md`
   - transforms/planes/clip → `script_repository/transforms.md`
3. **Parallel Grep** — 在**第一轮**就同时 Grep 所有相关 topic 文件，而非逐轮摸索。
4. **Expand if needed** — 若 topic 文件信息不足，按顺序扩展到：
   - `slicer-source/Base/Python/slicer/util.py`
   - `slicer-source/Modules/Scripted/<module>/`
   - `slicer-source/Modules/Loadable/<module>/`
   - `slicer-dependencies/VTK/`（仅限底层几何操作）

**关键约束：**
- 禁止第一轮就 grep 整个 `slicer-source` 树。
- ReadFile 在此阶段不可用。
- 若 LLM 认为已找到所需文件路径，则**立即停止调用 Grep**。

Grep 返回结果包含：**哪些文件的哪一行**匹配了关键词。

### 阶段 2：读文件（ReadFile）

- 系统发送过渡消息："Search phase complete. Now use ReadFile..."
- LLM 只能使用 **ReadFile** 工具。
- 读取第一阶段找到的最相关文件（如 `segmentations.md`、`SegmentEditorThresholdEffect.py`）的完整内容。
- **允许多个 ReadFile 并行调用。**
- 若 LLM 认为已掌握足够的 API 签名和使用方法，则停止读文件，进入下一阶段。

> **已知性能问题**：搜索→读文件、读文件→生成 之间的过渡轮会消耗 API 时间（约 1-3 秒/轮），期间没有实际的工具调用，仅为阶段切换。

### 阶段 3：生成代码（Generate）

- 系统发送过渡消息："File reading phase complete. Now write the final Python code..."
- LLM **没有任何工具可用**，只能直接输出最终的 Python 代码。
- 此时 LLM 的响应是**流式**的（SSE），逐字传输。

**历史消息压缩**：
- 此阶段结束后，工具结果会被压缩后存入 conversation history。
- 压缩方式：使用本地 `_fallbackCompressReadFile` 截断长文本（保留代码块，文本限制 500 字符）。
- **不再调用 LLM 进行总结**（早期版本会触发一次额外的 LLM API 调用，耗时 50-60 秒，已移除）。

---

## 第三阶段：流式输出 → 写入队列

**执行线程：后台线程 → 主线程的桥梁**

LLM 在第三阶段每输出一个字（或一小段 reasoning 文字），后台线程会：

1. 将该片段包装为一个 **delta 事件**。
2. 将 delta 事件扔进**线程安全队列**（`_streamQueue`）。

同时，阶段性进度（如 "[Search] Round 1: Grep volumes.md, segmentations.md, models.md"）也会作为 delta 事件写入同一队列。

当 LLM 的 SSE 流完全结束后，后台线程再往队列中放入一个 **complete 事件**，携带完整的 response（包含最终代码、token 用量、耗时统计）。

---

## 第四阶段：主线程消费队列（每 50ms 处理一次）

**执行线程：Slicer 主线程**

主线程中有一个**定时器**，每 50 毫秒触发一次，负责"消费"队列中的所有事件：

### 处理阶段性进度（Search / ReadFile / Transition）

若队列中是"搜索进度"或"阶段切换"事件，主线程会：

- 立即将其追加到 Chat 历史中（作为已提交的灰色条目）。
- 显示 `[Search] Round 1: Grep: volumes.md, segmentations.md, models.md` 等信息。

### 处理流式文字（Assistant 正在打字）

若队列中是一连串的"流式输出"片段，主线程会：

- 将所有片段**累积起来**（而非每收到一个字就刷新一次 UI）。
- 累积完成后，**一次性**更新 Chat 区域的 Assistant 消息。
- 调用 `slicer.app.processEvents()`，让 UI 有机会响应其他操作。

> **关键优化**：之前每个 delta 都单独调用 `setHtml()`，导致主线程阻塞数十秒（后台 SSE 流已结束，但主线程仍在处理积压的 delta，延迟 `complete` 事件处理 50-100 秒）。现已改为**批量刷新** —— 连续 delta 合并为单次 `setHtml()` 调用，工具进度 delta 则立即处理。

### 处理 complete 事件（流式输出结束）

当队列中出现 `complete` 事件时，主线程知道 LLM 已全部输出完毕，于是：

1. 将流式的 Assistant 消息"归档"到 Chat 历史中。
2. 提取 response 中的代码，显示到 **Generated Code** 区域。
3. 将代码保存到 `1_code.txt`。
4. 将最后阶段的 prompt 保存到 `1_last_prompt_debug.txt`。
5. **触发自动执行**。

---

## 第五阶段：自动执行代码

**执行线程：Slicer 主线程**

### 5.1 执行前检查

1. 进行**语法预检查**（使用 Python 的 `ast.parse`），检查代码是否存在明显语法错误。
2. 若无语法错误，界面状态变为 "Executing..."。
3. 记录多个时间戳：`autoexecute_start`、`validation_start/end`、`execution_async_call`。

### 5.2 调度执行

代码执行必须在**主线程**运行（因为 Slicer 的 MRML/VTK 不允许跨线程操作）。为避免阻塞 UI，执行按以下方式调度：

- 通过 `qt.QTimer.singleShot(10, executeAndCallback)` 调度。
- 10 毫秒后，在主线程上实际执行代码。
- 记录时间戳：`executor_scheduled`、`executor_actual_start`（由 SafeExecutor 内部记录）。

### 5.3 实际执行

代码在沙箱环境中运行：

- 捕获所有 `print()` 输出（stdout）。
- 捕获所有错误信息（stderr）。
- **拦截 VTK 的 C++ 错误输出**：重定向到临时文件，执行完毕后再读取回来。
- 在 `__main__.__dict__` 中执行代码（与 Slicer Python Console 同一命名空间，`slicer`、`qt` 等变量天然可用）。
- 若代码最后一行是表达式，还会记录其结果。

执行完成后返回结果字典：
- `success`：是否成功
- `output`：所有 print 输出的文字
- `error`：异常信息（如有）
- `execution_time`：实际执行耗时（秒）
- `executor_scheduled` / `executor_actual_start`：调度时间戳

### 5.4 执行后回调

执行结果通过回调函数传回 UI 层：

1. 记录时间戳：`execution_callback_start`、`execution_end`。
2. 在 Chat 中显示执行结果：`Code executed successfully in 0.29s. Output: ...`
3. 将执行结果追加到 conversation history（作为系统反馈，供下一轮对话参考）。
4. **写入性能报告文件** `1_performance_log.txt`，包含：
   - 总 wall-clock 时间（从 `turn_start` 到执行结束）
   - 每轮 API 调用耗时
   - 工具执行耗时（Grep、ReadFile）
   - 代码验证、调度、执行各阶段细粒度耗时
5. 停止实时计时器，状态栏显示 "Done"。

---

## 第六阶段：执行失败时的自纠正

**执行线程：Slicer 主线程**

若代码执行抛出异常，或输出中包含 `error:`、`traceback`、`[vtk error]` 等关键字，程序会尝试**自动修复**：

1. 构建**隔离对话**：系统提示 + 用户最初提问 + 失败代码 + 错误信息。
2. 发送给 LLM（**不写入 conversation history**，避免污染上下文）。
3. LLM 返回修复后的代码。
4. 更新 Generated Code 区域，保存为 `1_correction_1_code.txt`。
5. **再次自动执行**。
6. 最多尝试 5 次，若仍失败则放弃，在 Chat 中报告最终错误。

---

## 已知性能瓶颈

| 瓶颈 | 原因 | 状态 |
|---|---|---|
| **Transition rounds** | 阶段切换（Search→ReadFile、ReadFile→Generate）各消耗 1 轮 API 调用，无实际工具调用 | 未解决 |
| **Windows I/O 延迟** | `rg.exe` 在 Slicer Python 运行时内被 Windows Defender 扫描，0.8s → 6-7s | 未解决 |
| **Self-correction 硬提示** | LLM 偶尔在代码块外输出推理文字，导致提取的 `code` 字段可能不准确 | 未解决 |
