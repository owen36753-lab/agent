# ALFWorld Plain ReAct Baseline

用于 ALFWorld 文字环境的最小可运行 LLM ReAct Agent。

这个项目刻意保持简单：先把 `Observation -> Thought -> Action -> Environment`
闭环、日志和评估跑通，再把它作为后续实验的稳定基线。当前版本不加入
LangGraph、长期记忆、反思、重规划或微调。

## 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| Plain ReAct 主流程 | 已完成 | 每步只调用一次 LLM，只执行一个动作 |
| Mock 环境与 Mock LLM | 已完成 | 无需 API Key，可离线验证完整流程 |
| ALFWorld 文字环境 | 已完成 | 已在 WSL 中完成真实 `reset()` / `step()` 验证 |
| ALFWorld expert 完整通关测试 | 已完成 | 使用环境内置 expert，不调用 LLM |
| ALFWorld 数据集 | 已完成 | 下载到 WSL 本地缓存，不进入 Git |
| Gemini 免费层 LLM API | 已验证 | `gemini-2.5-flash-lite` 连通测试通过；免费额度可能不足以持续跑完整 episode |
| Ollama 本地 LLM | 已验证 | Windows 运行 `qwen3:4b`，通过 WSL 虚拟网卡端口转发供 Ubuntu 调用 |
| 本地模型真实 episode | 已完成首轮 | 完整运行 30 步并生成评估报告；首轮任务未成功，作为后续改进起点 |

真实环境 smoke test 已验证：

```text
Overall we have 134 games in split=eval_out_of_distribution
Real ALFWorld reset/step smoke test passed.
Action: go to cabinet 1
Reward: 0.0
Done: False
```

## 第一版阶段报告

完整的技术路线、安装过程、问题解决记录、首轮真实实验分析和后续方向见：

```text
docs/FIRST_BASELINE_REPORT.md
```

## Baseline 边界

当前版本只实现 Plain ReAct：

```text
Observation
    -> 构造 Prompt
    -> LLM 输出 Thought 和一条 Action
    -> 解析第一条 Action
    -> env.step(action)
    -> 记录日志
    -> 继续或结束
```

以下能力暂时不加入，以便后续做清晰的对照实验：

- LangGraph
- 长期记忆
- 反思
- 重规划
- Planner / Executor 分层
- 微调

## 快速开始

### 1. 离线 Mock 测试

Mock 测试不依赖 ALFWorld，也不会调用真实 LLM API。

在 Windows PowerShell 中：

```powershell
cd "C:\Users\Mr.Orange\Desktop\agent\git-workspace\agent\ReAct_baseline"
python main.py --mock --clear-results
python evaluator.py
```

在 WSL Ubuntu 中，对应路径是：

```bash
cd "/mnt/c/Users/Mr.Orange/Desktop/agent/git-workspace/agent/ReAct_baseline"
python main.py --mock --clear-results
python evaluator.py
```

预期结果：

```text
task_success_rate: 1.0
average_steps: 2.0
action_valid_rate: 1.0
```

### 2. 安装 ALFWorld 文字环境

ALFWorld 的 TextWorld 依赖需要 Linux。在 Windows 上请使用 WSL Ubuntu。

TextWorld 支持 Python `3.9` 至 `3.12`。不要使用 Python `3.13` 或更新版本。
安装脚本会自动选择兼容解释器。

在 Ubuntu 终端执行：

```bash
cd "/mnt/c/Users/Mr.Orange/Desktop/agent/git-workspace/agent/ReAct_baseline"
bash scripts/setup_text_baseline.sh
source ~/.venvs/react-baseline/bin/activate
```

脚本会：

1. 创建 Linux 虚拟环境 `~/.venvs/react-baseline`。
2. 安装文字版 ALFWorld 与 PyYAML。
3. 执行 `alfworld-download`。
4. 将数据保存到 `~/.cache/alfworld/`。

虚拟环境与数据集均保存在 Git 仓库之外。

### 3. 验证真实 ALFWorld

激活 WSL 虚拟环境后执行：

```bash
python real_env_smoke_test.py
```

这个测试不会调用 LLM。它只验证：

```text
加载官方配置
    -> 初始化 AlfredTWEnv
    -> reset()
    -> 读取 admissible_commands
    -> 执行一个合法动作
    -> step(action)
```

### 4. 使用 ALFWorld Expert 验证完整任务

真实环境单步测试通过后，可以让 ALFWorld 内置的 handcoded expert 离线完成一个
`train` split 任务：

```bash
python expert_env_smoke_test.py
```

这个测试不调用 LLM。它会优先选择较稳定的单物体放置任务，循环读取
`extra.expert_plan`，执行下一条专家动作，直到环境返回成功信号。由于 ALFWorld
内置 handcoded expert 在少数任务上可能超时，脚本最多尝试 5 个任务，并打印每次
失败原因。

用途：

```text
验证 ALFWorld 数据集可用
    -> 验证多步环境状态更新
    -> 验证成功判定 info["won"] -> info["success"]
    -> 验证一个真实任务可以完整结束
```

注意：expert 测试只验证环境，不代表 ReAct Agent 已经学会完成任务。真实 baseline
仍然必须使用 LLM 选择动作。

### 5. 接入真实 LLM API

第一版建议使用 Gemini 免费层：

```bash
gemini-2.5-flash-lite
```

它通过 Google 官方 OpenAI-compatible 接口接入，现有 LLM 客户端无需修改。

在 Google AI Studio 创建 API Key 后，复制示例环境变量文件：

```bash
cp .env.example .env.local
```

编辑 `.env.local`，只在本地填写：

```bash
export LLM_API_KEY="你的 Gemini API Key"
```

不要将 API Key 发到聊天或提交到 Git。

Gemini 免费层有每分钟请求数限制。模板默认设置：

```bash
export LLM_MIN_REQUEST_INTERVAL_SECONDS="4.2"
export LLM_MAX_RETRIES="3"
export LLM_RETRY_BASE_SECONDS="5"
```

Plain ReAct 每一步都会调用一次 LLM。请求节流不会改变 Agent 决策，只会降低调用
速度。遇到 `HTTP 429` 时，客户端会读取服务端错误信息并退避重试。

加载配置并先执行一次最小连通测试：

```bash
source .env.local
python api_smoke_test.py
```

看到 `LLM API smoke test passed.` 后，先运行一个真实 ALFWorld episode：

```bash
python main.py --episodes 1 --max-steps 30 --clear-results
```

检查 `results/step_logs.jsonl` 后，再扩大到 10 个任务：

```bash
python main.py --episodes 10 --max-steps 30 --clear-results
```

当前客户端使用 OpenAI-compatible `/chat/completions` 接口。可接入 OpenAI 或其他
兼容服务。

### 6. 可选：使用 Ollama 本地模型

如果暂时不希望消耗云端 API 额度，可以在 Windows 安装 Ollama，并拉取：

```powershell
ollama pull qwen3:4b
netsh interface portproxy add v4tov4 listenaddress=<WSL网卡地址> listenport=11434 connectaddress=127.0.0.1 connectport=11434
```

`netsh` 命令需要在管理员 PowerShell 中执行。它只在 Windows 的 WSL 虚拟网卡上
暴露转发端口，不应将 Ollama 直接监听到 `0.0.0.0`。

本机测试中，`qwen3:4b` 可以通过 Ollama 的 OpenAI-compatible 接口稳定返回：

```text
Thought: I should inspect the room.
Action: look
```

将本地 `.env.local` 调整为：

```bash
export LLM_PROVIDER="openai_compatible"
export LLM_MODEL="qwen3:4b"
export LLM_API_KEY="ollama"
export LLM_BASE_URL="http://$(ip route show default | awk '{print $3}'):11434/v1"
export LLM_MIN_REQUEST_INTERVAL_SECONDS="0"
```

WSL 默认使用 NAT 网络。通过默认网关读取 Windows 主机地址，比假设
`localhost:11434` 一定会转发更稳妥。

然后在 WSL 中验证：

```bash
source .env.local
python api_smoke_test.py
python main.py --episodes 1 --max-steps 30 --clear-results
```

本地 `4B` 模型适合验证流程并建立能力下限，不应默认替代更强云端模型的正式对照。
本机也测试过 `llama3.2:3b`：模型可以运行，但未能稳定遵循两行 ReAct 输出格式，
因此不作为推荐配置。

## 架构

```text
main.py
  |
  +-- env.reset()
  |     +-- MockALFWorldEnv
  |     +-- ALFWorldEnv -> AlfredTWEnv
  |
  +-- ReActAgent.act()
  |     +-- prompts.build_react_prompt()
  |     +-- llm_client.generate()
  |     +-- parse_thought()
  |     +-- parse_action()
  |
  +-- env.step(action)
  |
  +-- logger
  |     +-- step_logs.jsonl
  |     +-- episode_summary.jsonl
  |
  +-- failure_classifier
  |
  +-- evaluator
        +-- evaluation_report.json
```

## 项目结构

```text
ReAct_baseline/
├── main.py                       # 多 episode 运行入口
├── config.py                     # 环境变量与默认配置
├── alfworld_env.py               # Mock 与真实 ALFWorld 环境适配
├── llm_client.py                 # Mock 与 OpenAI-compatible LLM 客户端
├── react_agent.py                # Prompt 调用、Thought / Action 解析
├── prompts.py                    # Plain ReAct Prompt
├── logger.py                     # JSONL 日志写入
├── failure_classifier.py         # 规则式失败分类
├── evaluator.py                  # 指标汇总
├── api_smoke_test.py             # 单次 API 连通与 Action 格式测试
├── real_env_smoke_test.py        # 真实环境 reset / step 验证
├── expert_env_smoke_test.py      # 内置 expert 离线完整通关验证
├── docs/
│   └── FIRST_BASELINE_REPORT.md   # 第一版路线、问题记录与实验分析
├── configs/
│   └── base_config.yaml          # ALFWorld 官方文字环境配置
├── scripts/
│   └── setup_text_baseline.sh    # WSL 安装与数据下载脚本
├── results/
│   └── .gitkeep                  # 保留空目录，运行结果不提交
├── .env.example                  # API 配置模板，不含真实密钥
├── .gitignore                    # 屏蔽本地产物
└── requirements.txt              # 文字版 ALFWorld 最小依赖
```

## 核心设计约束

### 单步只执行一个动作

Prompt 要求模型输出：

```text
Thought: ...
Action: ...
```

`parse_action()` 只读取第一个 `Action:` 后面的第一行。即使模型额外输出内容，
环境也只接收一条动作。

### 保存原始响应

每一步都会保存 `raw_response`。这样可以区分：

- LLM 是否偏离格式
- 解析器是否提取错误
- 环境是否拒绝动作

### success 不等于 done

`done` 表示 episode 结束，不必然表示任务成功。

成功判定顺序：

```text
优先读取 info["success"]
    -> ALFWorld info["won"] 会在环境适配层规范化为 info["success"]
    -> 如果没有 success，再使用 done and reward > 0 回退判断
```

## 日志

运行后生成：

```text
results/
├── step_logs.jsonl
├── episode_summary.jsonl
└── evaluation_report.json
```

### Step-level 字段

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

### Episode-level 字段

```text
episode_id
goal
success
num_steps
total_reward
failure_reason
final_observation
```

## 评估指标

`python evaluator.py` 会输出：

```text
task success rate
average steps
average steps for successful episodes
average steps for failed episodes
action valid rate
failure reason distribution
```

真实 ALFWorld 环境的 `action_valid` 使用执行动作前的 `admissible_commands`
进行判断。该指标表示动作是否符合当前环境允许的命令集合，不等同于动作是否有助于
完成任务。

失败原因使用简单规则分类：

```text
invalid_action
repeated_action_loop
max_steps_exceeded
unknown_failure
```

## 数据与 Git 隔离

以下内容只保存在本地，不提交到 GitHub：

```text
WSL 虚拟环境: ~/.venvs/react-baseline
ALFWorld 数据集: ~/.cache/alfworld
实验日志: results/*.jsonl
评估报告: results/*.json
API 密钥: .env.local
```

Git 仓库只保存代码、配置模板、官方 YAML 和可复现脚本。

## 常用命令

```bash
# 查看 CLI 参数
python main.py --help
python evaluator.py --help

# 离线 mock 回归测试
python main.py --mock --clear-results

# 真实环境 smoke test
source ~/.venvs/react-baseline/bin/activate
python real_env_smoke_test.py

# 真实环境 expert 完整通关测试，无需 API Key
python expert_env_smoke_test.py

# 真实 API baseline
source .env.local
python api_smoke_test.py
python main.py --episodes 1 --max-steps 30 --clear-results
python main.py --episodes 10 --max-steps 30 --clear-results
```

## 后续实验

当前 Plain ReAct 版本应保持稳定。新增复杂能力时，建议使用独立分支，并与该
baseline 对照：

```text
Plain ReAct
    -> + memory
    -> + reflection
    -> + replanning
    -> + Planner / Executor
```

每次只增加一个变量，才能判断改动是否真正提升 ALFWorld 成功率。
