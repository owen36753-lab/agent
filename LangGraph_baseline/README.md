# ALFWorld LangGraph ReAct Baseline

This directory starts the complex-framework track for the ALFWorld project.

It keeps the same task interface, prompt, LLM client, logging format, and evaluator as
`ReAct_baseline`, but rewrites the episode loop as an explicit LangGraph state machine.

## Goal

The first LangGraph version is intentionally conservative:

```text
initialize
  -> progress_hint, when a direct subgoal action is available
  -> act
  -> format_repair, only when Action parsing fails
  -> invalid_action_retry, only when Action is not available
  -> step
  -> stuck_reflection, only when a valid action loop repeats
  -> route
  -> finalize
```

It does not add memory, reflection, replanning, or fine-tuning yet. The purpose is to create a
graph-shaped execution surface where small control-flow repairs can be added without disturbing the
successful Plain ReAct baseline.

## Why LangGraph Here

Plain ReAct hides control flow inside a Python `for` loop. LangGraph makes the state transitions
explicit:

```text
initialize_node:
    reset env, extract goal, clear agent history

progress_hint_node:
    inspect the goal, previous actions, and admissible_commands; if a direct subgoal action is
    available, inject a short feedback hint before the next ReAct call

act_node:
    build ReAct prompt, call LLM, parse Thought / Action

format_repair_node:
    if the LLM response has no parseable Action line, ask the model to repair only the format before
    touching the environment

invalid_action_retry_node:
    if the Action line is parseable but not in admissible_commands, ask the model to choose exactly
    one available action before touching the environment

step_node:
    execute env.step(action), log result, update feedback and guard counters

stuck_reflection_node:
    if the same valid action or a simple two-action cycle repeats without success, inject a short
    feedback message telling the next act step to choose a different strategy

route_after_step:
    continue to act or stop at finalize

finalize_node:
    write episode summary
```

This makes it easier to add future nodes such as:

```text
invalid_action_retry
reflection
replan
human_review
memory_update
```

The current implementation includes a lightweight task-progress hint, two graph-native repair
branches, and one stuck guard:

```text
initialize / step
  -> progress_hint -> act, when actions like take/heat/move-to-target are directly available

act
  -> step, when Action is parseable
  -> format_repair -> step, when Action parsing fails once
  -> invalid_action_retry -> step, when Action is parseable but unavailable

step
  -> stuck_reflection -> act, when a valid action loop repeats without success
```

If repair still fails, the episode falls back to the existing logged failure path. This keeps the
behavior debuggable: every environment step still records the raw response, and repair text is
appended under `[FORMAT_REPAIR_RAW_RESPONSE]` or `[INVALID_ACTION_RETRY_RAW_RESPONSE]` markers.

Both repair branches are intentionally one-shot guards. They reduce obvious LLM interface failures
without adding memory, reflection, or replanning, so the experiment remains close to the original
ReAct baseline.

The stuck reflection branch is also intentionally small. It does not summarize the whole trajectory
or create a new plan. It only detects a repeated-action loop, including simple two-action cycles,
and passes targeted feedback into the next ReAct prompt.

## Run Isolation

Each run writes to its own directory by default:

```text
results/runs/<timestamp>/
```

This prevents a timed-out or still-shutting-down process from appending to the next run's logs. The
program prints the selected `run_id` and `results_dir` before starting episodes.

Use an explicit id when you want reproducible paths:

```bash
python graph_main.py --mock --run-id mock_debug --clear-results
```

Use a wall-clock guard for long local-model runs:

```bash
python graph_main.py --episodes 1 --max-steps 45 --max-runtime-seconds 1800
```

The runtime guard is checked between environment steps. If the budget is exceeded, the episode is
finalized with `early_stop_reason: runtime_exceeded`.

Relative `ALFWORLD_CONFIG_PATH` values are resolved against `ReAct_baseline` first, so the same
`.env.local` can be reused from this directory.

## Progress Hints

`progress_hint_node` is intentionally narrower than a planner. It does not search the map or create
a full route. It only detects obvious one-step progress from the currently available actions:

```text
take <goal-object> ...
heat/cool/clean <held-object> with ...
go to the required treatment tool
open the goal receptacle
move <ready-object> to <goal-receptacle>
```

The hint is passed through the existing `Previous step feedback` slot, so the LLM still has to output
the final `Thought:` and `Action:` lines. This targets failures where the model sees a correct action
such as `heat apple 3 with microwave 1` but chooses to wander instead.

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
python graph_main.py --episodes 1 --max-steps 80 --max-runtime-seconds 1800
```

To restrict evaluation to one ALFWorld task family, pass `--task-types`:

```bash
python graph_main.py --episodes 1 --task-types 1 --run-id pick_place_smoke
```

Task type ids:

```text
1 Pick & Place
2 Examine in Light
3 Clean & Place
4 Heat & Place
5 Cool & Place
6 Pick Two & Place
```

Results are written to:

```text
LangGraph_baseline/results/runs/<run_id>/
```

Generated logs and local secrets are ignored by Git.

## Relationship To ReAct Baseline

`ReAct_baseline` remains the stable comparison point.

`LangGraph_baseline` should only change orchestration first. If it later gains reflection,
replanning, or memory, each capability should be added as one separate experiment so the effect is
measurable.
