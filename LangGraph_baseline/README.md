# ALFWorld LangGraph ReAct Baseline

This directory starts the complex-framework track for the ALFWorld project.

It keeps the same task interface, prompt, LLM client, logging format, and evaluator as
`ReAct_baseline`, but rewrites the episode loop as an explicit LangGraph state machine.

## Goal

The first LangGraph version is intentionally conservative:

```text
initialize
  -> act
  -> step
  -> route
  -> finalize
```

It does not add memory, reflection, replanning, or fine-tuning yet. The purpose is to create a
graph-shaped execution surface where those branches can be added later without disturbing the
successful Plain ReAct baseline.

## Why LangGraph Here

Plain ReAct hides control flow inside a Python `for` loop. LangGraph makes the state transitions
explicit:

```text
initialize_node:
    reset env, extract goal, clear agent history

act_node:
    build ReAct prompt, call LLM, parse Thought / Action

step_node:
    execute env.step(action), log result, update feedback and guard counters

route_after_step:
    continue to act or stop at finalize

finalize_node:
    write episode summary
```

This makes it easier to add future nodes such as:

```text
format_repair
invalid_action_retry
reflection
replan
human_review
memory_update
```

## Install

Use the same WSL Python environment as `ReAct_baseline`, then install LangGraph:

```bash
cd "/mnt/c/Users/Mr.Orange/Desktop/agent/git-workspace/agent/LangGraph_baseline"
source ~/.venvs/react-baseline/bin/activate
python -m pip install -r requirements.txt
```

Official LangGraph docs describe the core pattern as defining a `StateGraph`, adding nodes and
edges, then compiling the graph before invocation.

## Run Mock

```bash
python graph_main.py --mock --clear-results
```

Expected shape:

```text
task_success_rate: 1.0
average_steps: 2.0
action_valid_rate: 1.0
```

## Run Real ALFWorld

Use the same local `.env.local` style as `ReAct_baseline`.

```bash
source ../ReAct_baseline/.env.local
python graph_main.py --episodes 1 --max-steps 80 --clear-results
```

Results are written to:

```text
LangGraph_baseline/results/
```

Generated logs and local secrets are ignored by Git.

## Relationship To ReAct Baseline

`ReAct_baseline` remains the stable comparison point.

`LangGraph_baseline` should only change orchestration first. If it later gains reflection,
replanning, or memory, each capability should be added as one separate experiment so the effect is
measurable.
