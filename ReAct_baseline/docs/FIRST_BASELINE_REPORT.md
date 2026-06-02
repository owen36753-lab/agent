# ALFWorld Plain ReAct Baseline 第一版报告

日期：2026-06-03

## 1. 第一版结论

第一版已经完成一个最小、可运行、可复现的 ALFWorld Plain ReAct baseline。

当前已经验证：

```text
Mock 环境回归测试
    -> ALFWorld 文字环境 reset() / step()
    -> ALFWorld 内置 expert 完整通关
    -> OpenAI-compatible LLM API 连通
    -> Ollama 本地模型连通
    -> 本地 qwen3:4b 执行真实 ALFWorld episode
    -> JSONL 日志记录
    -> 规则式失败分类
    -> 自动生成评估报告
```

需要区分两个结论：

1. 工程链路已经跑通。
2. 当前本地 `qwen3:4b` 首轮实验没有完成任务，效果仍需改进。

这符合 baseline 的用途：先建立稳定、透明、可测量的下限，再逐步增加变量。

## 2. 第一版范围

第一版只实现 Plain ReAct：

```text
Observation
    -> 构造 Prompt
    -> LLM 输出 Thought 和一条 Action
    -> 解析 Action: 后的第一行
    -> env.step(action)
    -> 记录新 Observation
    -> 继续下一步或结束 episode
```

第一版刻意不加入：

- LangGraph
- 长期记忆
- 反思
- 重规划
- Planner / Executor 分层
- 微调

原因是后续需要做清晰的对照实验。每次只增加一个变量，才能判断改动是否真正提升
任务成功率。

## 3. 技术路线

### 3.1 环境

- 操作系统：Windows + WSL Ubuntu
- 环境：ALFWorld `0.4.2`
- 交互模式：TextWorld 文字环境 `AlfredTWEnv`
- Python：WSL 中使用 Python `3.12`
- 数据集目录：`~/.cache/alfworld`

ALFWorld 数据集、虚拟环境和运行日志都保留在本地，不提交到 Git。

### 3.2 Agent

- 策略：Plain ReAct
- 每步 LLM 调用次数：一次
- 每步环境动作数：一个
- 输出格式：

```text
Thought: ...
Action: ...
```

`parse_action()` 只提取第一个 `Action:` 标记后的第一行，防止模型一次执行多个动作。
同时保存 `raw_response`，方便定位格式错误。

### 3.3 LLM 接入

项目使用最小 OpenAI-compatible `/chat/completions` 客户端，可以替换后端：

- Gemini `gemini-2.5-flash-lite`
- Ollama 本地 `qwen3:4b`
- 其他 OpenAI-compatible 服务

客户端包含：

- 最小请求间隔
- HTTP `429` 退避重试
- `Retry-After` 和错误体重试时间解析
- API Key 与模型名缺失检查

### 3.4 日志与评估

每一步写入 `results/step_logs.jsonl`：

```text
episode_id
step
goal
observation
thought
action
raw_response
new_observation
reward
done
success
action_valid
```

每个 episode 写入 `results/episode_summary.jsonl`：

```text
episode_id
goal
success
num_steps
total_reward
failure_reason
final_observation
```

评估器生成 `results/evaluation_report.json`：

```text
task success rate
average steps
average steps for successful episodes
average steps for failed episodes
action valid rate
failure reason distribution
```

### 3.5 成功与动作合法性

`done` 只表示 episode 结束，不一定表示成功。

成功判定顺序：

```text
优先读取 info["success"]
    -> ALFWorld info["won"] 规范化为 info["success"]
    -> 缺失时回退到 done and reward > 0
```

真实环境动作合法性使用执行动作前的 `admissible_commands` 判断。它表示命令是否属于
当前环境允许集合，不代表动作一定有助于完成任务。

## 4. 完整复现流程

### 4.1 安装文字版环境

在 WSL Ubuntu 中：

```bash
cd "/mnt/c/Users/Mr.Orange/Desktop/agent/git-workspace/agent/ReAct_baseline"
bash scripts/setup_text_baseline.sh
source ~/.venvs/react-baseline/bin/activate
```

脚本会：

1. 选择 Python `3.9` 至 `3.12`。
2. 创建 `~/.venvs/react-baseline`。
3. 安装 ALFWorld 与 PyYAML。
4. 执行 `alfworld-download`。
5. 将数据保存到 `~/.cache/alfworld`。

### 4.2 逐层验证

先执行离线 mock：

```bash
python main.py --mock --clear-results
```

再验证真实环境单步调用：

```bash
python real_env_smoke_test.py
```

再验证真实环境完整任务：

```bash
python expert_env_smoke_test.py
```

这个脚本调用 ALFWorld 内置 handcoded expert，不调用 LLM。它的作用是证明数据集、
环境状态变化和成功判定都能工作。

### 4.3 使用 Gemini API

```bash
cp .env.example .env.local
```

在本地填写 Gemini API Key 后：

```bash
source .env.local
python api_smoke_test.py
python main.py --episodes 1 --max-steps 30 --clear-results
```

`.env.local` 不进入 Git。

### 4.4 使用 Ollama 本地模型

在 Windows 安装 Ollama 并拉取模型：

```powershell
ollama pull qwen3:4b
```

WSL 默认不能直接访问 Windows 的 `localhost:11434`。使用管理员 PowerShell 将
Ollama 的本地端口只转发到 WSL 虚拟网卡：

```powershell
netsh interface portproxy add v4tov4 listenaddress=<WSL网卡地址> listenport=11434 connectaddress=127.0.0.1 connectport=11434
```

不应为了方便将 Ollama 直接监听到 `0.0.0.0`。

WSL 本地 `.env.local` 使用默认网关动态读取 Windows 主机地址：

```bash
export LLM_PROVIDER="openai_compatible"
export LLM_MODEL="qwen3:4b"
export LLM_API_KEY="ollama"
export LLM_BASE_URL="http://$(ip route show default | awk '{print $3}'):11434/v1"
export LLM_MIN_REQUEST_INTERVAL_SECONDS="0"
```

然后：

```bash
source .env.local
python api_smoke_test.py
python main.py --episodes 1 --max-steps 30 --clear-results
```

## 5. 已完成验证

### 5.1 Mock 闭环

结果：

```text
num_episodes: 1
task_success_rate: 1.0
average_steps: 2.0
action_valid_rate: 1.0
```

说明主循环、日志和评估器可以完整工作。

### 5.2 真实环境 reset / step

结果：

```text
Overall we have 134 games in split=eval_out_of_distribution
Real ALFWorld reset/step smoke test passed.
Action: go to cabinet 1
Reward: 0.0
Done: False
```

说明 ALFWorld 文字环境和数据集已经可用。

### 5.3 内置 expert 完整通关

已完成一个真实任务：

```text
Your task is to: put a soapbottle in toilet.
SUCCESS: True | steps: 13 | reward: 1.0
Final observation: You move the soapbottle 1 to the toilet 1.
```

说明多步状态变化、奖励和成功判定可以正确工作。

### 5.4 LLM 连通

Gemini `gemini-2.5-flash-lite` 与 Ollama `qwen3:4b` 均已通过单次
`api_smoke_test.py`。

`qwen3:4b` 返回：

```text
Thought: I should inspect the room.
Action: look
```

## 6. 首轮本地模型实验

模型：

```text
Ollama qwen3:4b
```

任务：

```text
heat some apple and put it in fridge
```

实验参数：

```text
episodes: 1
max_steps: 30
```

首轮历史输出：

```text
task_success_rate: 0.0
average_steps: 30.0
average_steps_failed: 30.0
total_reward: 0.0
failure_reason: invalid_action
```

首轮日志曾显示 `action_valid_rate: 0.8`。审查发现旧实现只将空字符串视为无效动作，
而真实环境没有直接提供 `action_valid` 时会将其他非空动作默认计为有效。第一版提交前
已经修复：现在使用动作执行前的 `admissible_commands` 判断合法性。因此历史 `0.8`
不能作为正式结论，需要在后续重新运行后更新。

轨迹中的典型动作：

```text
open cabinet 1
open cabinet 2
...
open microwave 1
open fridge 1
```

环境持续返回：

```text
Nothing happens.
```

原因是模型跳过了导航步骤。ALFWorld 中通常需要：

```text
go to cabinet 1
open cabinet 1
```

模型还在部分步骤输出长篇解释而没有提供严格的 `Action:` 行。解析器将这些动作记录为
空字符串，并通过 `raw_response` 保留原始输出，便于分析。

## 7. 遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `wsl --install` 报错 `14098` | Windows 组件存储损坏 | 管理员 PowerShell 执行 DISM 与 SFC 修复，重启后重新启用组件 |
| TextWorld 在 reset 时出现 `NameError: name 'r' is not defined` | WSL 默认 Python `3.14` 不兼容 TextWorld | 使用 Python `3.12`，安装脚本限制为 `3.9` 至 `3.12` |
| 内置 expert 某些任务提前结束 | handcoded expert 在少数任务上不稳定 | smoke test 限定较稳定任务类型，最多尝试 5 次并输出原因 |
| Gemini 返回 HTTP `429` | 免费层请求或日额度受限 | 增加节流、退避重试和错误信息输出；完整实验可切换付费 API 或本地模型 |
| WSL 访问 Ollama 报 `Connection refused` | Ollama 只监听 Windows `127.0.0.1` | 使用 WSL 虚拟网卡 `portproxy` 安全转发，不监听 `0.0.0.0` |
| `llama3.2:3b` 输出格式不稳定 | 小模型指令遵循能力有限 | 本机 baseline 改用 `qwen3:4b` |
| 历史 `action_valid_rate` 偏高 | 非空动作被默认视为有效 | 使用执行前的 `admissible_commands` 判断真实环境动作合法性 |

## 8. 首轮结果分析

首轮实验不是“环境安装失败”，而是“Agent 策略失败”：

```text
环境可用
    + 模型可调用
    + ReAct 循环可执行
    + 日志可追踪
    + 评估器可输出
    - 模型不了解 ALFWorld 命令约束
    - 模型未稳定遵循 Action 输出格式
```

这正是第一版 baseline 需要暴露的信息。它给后续实验提供了可观察的起点。

## 9. 后续预期方向

按优先级建议：

1. 将当前状态的 `admissible_commands` 提供给 LLM。
2. 在 Prompt 中加入少量 ALFWorld 命令示例，例如先 `go to` 再 `open`。
3. 对比本地 `qwen3:4b` 与更强云端模型，重新运行多个 episode。
4. 固定数据 split、模型、温度和最大步数，记录可比较的指标。
5. Plain ReAct 稳定后，再单独增加 memory、reflection 或 LangGraph 流程控制。

其中第 1、2 项仍然属于 Plain ReAct baseline 的基础提示设计，不属于复杂优化。

## 10. 第一版验收标准

第一版的成功标准不是要求小模型立即获得高任务成功率，而是：

- 可以离线运行 mock 闭环。
- 可以加载真实 ALFWorld 文字环境。
- 可以用内置 expert 完成真实任务。
- 可以替换 OpenAI-compatible LLM 后端。
- 可以运行真实 LLM episode。
- 可以保存完整 step-level 与 episode-level 日志。
- 可以生成规定的评估指标。
- 可以从日志解释失败原因。
- Git 仓库不包含 API Key、数据集、虚拟环境或生成日志。

以上条件已经满足。后续工作应以该版本为基线，逐项提高真实任务成功率。
