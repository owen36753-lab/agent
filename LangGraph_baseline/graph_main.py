"""Run the ALFWorld ReAct baseline as an explicit LangGraph workflow."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
REACT_BASELINE_DIR = REPO_DIR / "ReAct_baseline"
if str(REACT_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(REACT_BASELINE_DIR))

from alfworld_env import ALFWorldEnv, MockALFWorldEnv  # noqa: E402
from config import (  # noqa: E402
    ALFWORLD_CONFIG_PATH,
    ALFWORLD_SPLIT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MIN_REQUEST_INTERVAL_SECONDS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_RETRY_BASE_SECONDS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONSECUTIVE_FORMAT_ERRORS,
    MAX_STEPS,
    NUM_EPISODES,
)
from evaluator import evaluate  # noqa: E402
from failure_classifier import classify_failure  # noqa: E402
from llm_client import MockLLMClient, OpenAICompatibleLLMClient  # noqa: E402
from logger import ExperimentLogger  # noqa: E402
from react_agent import ReActAgent, parse_action, parse_thought  # noqa: E402
from object_grounding import build_object_grounding, object_instance_from_action  # noqa: E402


RESULTS_DIR = PROJECT_DIR / "results"
STUCK_REPEAT_WINDOW = 3
STUCK_CYCLE_WINDOW = 4
MAX_STUCK_REFLECTIONS_PER_EPISODE = 2


class GraphState(TypedDict, total=False):
    episode_id: int
    env: Any
    agent: ReActAgent
    logger: ExperimentLogger
    max_steps: int
    max_runtime_seconds: float | None
    start_time: float
    goal: str
    observation: str
    current_info: dict[str, Any]
    admissible_commands: list[str]
    thought: str
    action: str
    raw_response: str
    format_repair_attempted: bool
    invalid_action_retry_attempted: bool
    object_grounding: dict[str, Any]
    object_grounding_count: int
    progress_hint: str | None
    progress_hint_count: int
    stuck_reflection_count: int
    stuck_reflection_reason: str | None
    step_number: int
    success: bool
    total_reward: float
    final_observation: str
    step_logs: list[dict[str, Any]]
    feedback: str | None
    consecutive_format_errors: int
    early_stop_reason: str | None
    done: bool
    summary: dict[str, Any]


def _success_from(info: dict[str, Any], done: bool, reward: float) -> bool:
    if "success" in info:
        return bool(info["success"])
    return bool(done and reward > 0)


def _action_valid_from(info: dict[str, Any], action: str) -> bool:
    if not action:
        return False
    return bool(info.get("action_valid", True))


def _goal_from(observation: str, info: dict[str, Any]) -> str:
    return str(info.get("goal") or info.get("task") or observation)


def _admissible_commands_from(info: dict[str, Any]) -> list[str]:
    commands = info.get("admissible_commands") or []
    return [str(command) for command in commands]


def _feedback_from_previous_step(
    action: str,
    action_valid: bool,
    format_error: bool,
    admissible_commands: list[str],
) -> str | None:
    if format_error:
        return (
            "Your previous response did not contain a valid 'Action: ...' line. "
            "Now output exactly two lines: 'Thought: ...' and 'Action: ...'. "
            "Copy the Action exactly from the available actions list."
        )
    if not action_valid:
        preview = "; ".join(admissible_commands[:8])
        return (
            f"Previous action was invalid: {action!r}. "
            f"Choose exactly one available action. Examples now available: {preview}"
        )
    return None


def _parse_task(goal: str) -> dict[str, str | None]:
    goal_text = goal.lower()
    treatment = next(
        (name for name in ("heat", "cool", "clean") if re.search(rf"\b{name}\b", goal_text)),
        None,
    )
    object_match = None
    if treatment:
        object_match = re.search(
            rf"\b{treatment}\s+(?:(?:some|a|an|the|two)\s+)?([a-z]+)",
            goal_text,
        )
    if not object_match:
        object_match = re.search(
            r"\bput\s+(?:(?:some|a|an|the|two)\s+)?([a-z]+)",
            goal_text,
        )
    target_match = re.search(r"\b(?:put|place|move)\b.*\b(?:in|into|on)\s+([a-z]+)", goal_text)
    return {
        "object": object_match.group(1) if object_match else None,
        "treatment": treatment,
        "target": target_match.group(1) if target_match else None,
    }


def _first_action_matching(admissible_commands: list[str], predicate: Any) -> str | None:
    return next((action for action in admissible_commands if predicate(action)), None)


def _progress_hint_for_action(action: str, reason: str) -> str:
    return (
        "Task progress hint: choose this exact available action now if it is still listed: "
        f"{action!r}. {reason}"
    )


def _build_progress_hint(
    goal: str,
    step_logs: list[dict[str, Any]],
    admissible_commands: list[str],
    object_grounding: dict[str, Any] | None = None,
) -> str | None:
    task = _parse_task(goal)
    object_name = task["object"]
    treatment = task["treatment"]
    target = task["target"]
    if not object_name:
        return None

    grounding = object_grounding or build_object_grounding(
        object_name,
        step_logs,
        admissible_commands,
        treatment,
    )
    held_object = grounding.get("held_object")
    treated_objects = set(grounding.get("treated_objects", []))
    object_prefix = held_object or object_name
    is_treated = bool(held_object and held_object in treated_objects)

    take_action = _first_action_matching(
        admissible_commands,
        lambda action: action.startswith("take ")
        and any(
            candidate["action"] == action for candidate in grounding.get("current_candidates", [])
        ),
    )
    if not held_object and take_action:
        take_instance = object_instance_from_action(take_action) or object_name
        return _progress_hint_for_action(
            take_action,
            (
                f"The visible object {take_instance!r} is grounded as the goal object "
                f"{object_name!r} and should be picked up before treatment or placement."
            ),
        )

    if held_object and treatment and not is_treated:
        treatment_action = _first_action_matching(
            admissible_commands,
            lambda action: action.startswith(f"{treatment} {held_object} with "),
        )
        if treatment_action:
            return _progress_hint_for_action(
                treatment_action,
                f"The agent is already holding {held_object!r}; the next required subgoal is to {treatment} it.",
            )

        tool_for_treatment = {"heat": "microwave", "cool": "fridge", "clean": "sinkbasin"}.get(
            treatment
        )
        if tool_for_treatment:
            go_to_tool = _first_action_matching(
                admissible_commands,
                lambda action: action.startswith(f"go to {tool_for_treatment} "),
            )
            if go_to_tool:
                return _progress_hint_for_action(
                    go_to_tool,
                    f"The agent is holding {held_object!r}; go to the {tool_for_treatment} to {treatment} it.",
                )

    ready_to_place = bool(held_object and (not treatment or is_treated))
    if ready_to_place and target:
        move_to_target = _first_action_matching(
            admissible_commands,
            lambda action: action.startswith(f"move {object_prefix} to {target} "),
        )
        if move_to_target:
            return _progress_hint_for_action(
                move_to_target,
                f"The object is ready for the final placement in/on the goal receptacle {target!r}.",
            )

        open_target = _first_action_matching(
            admissible_commands,
            lambda action: action.startswith(f"open {target} "),
        )
        if open_target:
            return _progress_hint_for_action(
                open_target,
                f"The object is ready for placement, but the goal receptacle {target!r} is closed.",
            )

        go_to_target = _first_action_matching(
            admissible_commands,
            lambda action: action.startswith(f"go to {target} "),
        )
        if go_to_target:
            return _progress_hint_for_action(
                go_to_target,
                f"The object is ready for final placement; go to the goal receptacle {target!r}.",
            )

    return None


def _build_format_repair_prompt(
    goal: str,
    observation: str,
    raw_response: str,
    admissible_commands: list[str],
) -> str:
    actions = "\n".join(f"- {action}" for action in admissible_commands) or "(not provided)"
    return (
        "Your previous ALFWorld response did not contain a valid Action line.\n"
        "Repair the response without changing the task intent.\n"
        "Output exactly two lines and nothing else:\n"
        "Thought: <brief reason>\n"
        "Action: <one available action copied exactly>\n\n"
        f"Goal: {goal}\n"
        f"Current observation: {observation}\n"
        f"Previous raw response:\n{raw_response}\n\n"
        f"Available actions:\n{actions}\n"
    )


def _build_invalid_action_retry_prompt(
    goal: str,
    observation: str,
    previous_action: str,
    admissible_commands: list[str],
) -> str:
    actions = "\n".join(f"- {action}" for action in admissible_commands) or "(not provided)"
    return (
        "Your previous ALFWorld action was not in the available actions list.\n"
        "Choose a valid action for the same observation.\n"
        "Output exactly two lines and nothing else:\n"
        "Thought: <brief reason>\n"
        "Action: <one available action copied exactly>\n\n"
        f"Goal: {goal}\n"
        f"Current observation: {observation}\n"
        f"Invalid previous action: {previous_action}\n\n"
        f"Available actions:\n{actions}\n"
    )


def _is_action_available(action: str, admissible_commands: list[str]) -> bool:
    if not action or not admissible_commands:
        return True
    return action in admissible_commands


def _is_repeated_action_loop(step_logs: list[dict[str, Any]]) -> bool:
    if len(step_logs) < STUCK_REPEAT_WINDOW:
        return False
    recent_logs = step_logs[-STUCK_REPEAT_WINDOW:]
    recent_actions = [str(item.get("action", "")) for item in recent_logs]
    if not recent_actions[0] or len(set(recent_actions)) != 1:
        return False
    return all(
        item.get("action_valid", False)
        and not item.get("format_error", False)
        and not item.get("success", False)
        for item in recent_logs
    )


def _is_alternating_action_loop(step_logs: list[dict[str, Any]]) -> bool:
    if len(step_logs) < STUCK_CYCLE_WINDOW:
        return False
    recent_logs = step_logs[-STUCK_CYCLE_WINDOW:]
    recent_actions = [str(item.get("action", "")) for item in recent_logs]
    if not all(recent_actions):
        return False
    if recent_actions[0] == recent_actions[1]:
        return False
    if recent_actions != recent_actions[:2] * 2:
        return False
    return all(
        item.get("action_valid", False)
        and not item.get("format_error", False)
        and not item.get("success", False)
        for item in recent_logs
    )


def _stuck_loop_reason(step_logs: list[dict[str, Any]]) -> str | None:
    if _is_repeated_action_loop(step_logs):
        action = step_logs[-1]["action"]
        return (
            f"Repeated the same valid action {action!r} for "
            f"{STUCK_REPEAT_WINDOW} consecutive steps without success."
        )
    if _is_alternating_action_loop(step_logs):
        actions = [str(item["action"]) for item in step_logs[-STUCK_CYCLE_WINDOW:]]
        pattern = " -> ".join(actions[:2])
        return (
            f"Repeated the valid action cycle {pattern!r} for "
            f"{STUCK_CYCLE_WINDOW} steps without success."
        )
    return None


def _should_trigger_stuck_reflection(state: GraphState) -> bool:
    if state["done"] or state.get("feedback"):
        return False
    if state.get("stuck_reflection_count", 0) >= MAX_STUCK_REFLECTIONS_PER_EPISODE:
        return False
    return _stuck_loop_reason(state["step_logs"]) is not None


def _runtime_exceeded(state: GraphState) -> bool:
    max_runtime_seconds = state.get("max_runtime_seconds")
    if not max_runtime_seconds:
        return False
    return time.monotonic() - state["start_time"] >= max_runtime_seconds


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_alfworld_config_path(config_path: str) -> str:
    candidate = Path(config_path)
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_absolute():
        react_relative = REACT_BASELINE_DIR / candidate
        if react_relative.is_file():
            return str(react_relative)

    parts = list(candidate.parts)
    if "configs" in parts:
        config_suffix = Path(*parts[parts.index("configs") :])
        react_config = REACT_BASELINE_DIR / config_suffix
        if react_config.is_file():
            return str(react_config)

    react_named_config = REACT_BASELINE_DIR / "configs" / candidate.name
    if react_named_config.is_file():
        return str(react_named_config)
    return config_path


def _parse_task_types(raw_task_types: str | None) -> list[int] | None:
    if not raw_task_types:
        return None
    return [int(item.strip()) for item in raw_task_types.split(",") if item.strip()]


def _build_components(use_mock: bool, task_types: list[int] | None = None) -> tuple[Any, ReActAgent]:
    if use_mock:
        return MockALFWorldEnv(), ReActAgent(MockLLMClient())
    if LLM_PROVIDER != "openai_compatible":
        raise ValueError("Set LLM_PROVIDER=openai_compatible for real runs, or use --mock.")

    env = ALFWorldEnv(
        _resolve_alfworld_config_path(ALFWORLD_CONFIG_PATH),
        ALFWORLD_SPLIT,
        task_types=task_types,
    )
    client = OpenAICompatibleLLMClient(
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        min_request_interval_seconds=LLM_MIN_REQUEST_INTERVAL_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        retry_base_seconds=LLM_RETRY_BASE_SECONDS,
    )
    return env, ReActAgent(client)


def initialize_node(state: GraphState) -> dict[str, Any]:
    env = state["env"]
    agent = state["agent"]
    observation, reset_info = env.reset()
    agent.reset()
    return {
        "goal": _goal_from(observation, reset_info),
        "observation": observation,
        "current_info": reset_info,
        "step_number": 0,
        "success": False,
        "total_reward": 0.0,
        "final_observation": observation,
        "step_logs": [],
        "feedback": None,
        "object_grounding": {},
        "object_grounding_count": 0,
        "progress_hint": None,
        "progress_hint_count": 0,
        "consecutive_format_errors": 0,
        "stuck_reflection_count": 0,
        "stuck_reflection_reason": None,
        "early_stop_reason": None,
        "done": False,
    }


def object_grounding_node(state: GraphState) -> dict[str, Any]:
    task = _parse_task(state["goal"])
    object_name = task["object"]
    if not object_name:
        return {"object_grounding": {}}
    admissible_commands = _admissible_commands_from(state["current_info"])
    grounding = build_object_grounding(
        object_name,
        state["step_logs"],
        admissible_commands,
        task["treatment"],
    )
    return {
        "admissible_commands": admissible_commands,
        "object_grounding": grounding,
        "object_grounding_count": state.get("object_grounding_count", 0) + 1,
    }


def progress_hint_node(state: GraphState) -> dict[str, Any]:
    if state.get("feedback"):
        return {"progress_hint": None}

    admissible_commands = state.get("admissible_commands") or _admissible_commands_from(
        state["current_info"]
    )
    progress_hint = _build_progress_hint(
        state["goal"],
        state["step_logs"],
        admissible_commands,
        state.get("object_grounding"),
    )
    if not progress_hint:
        return {"progress_hint": None}
    return {
        "feedback": progress_hint,
        "progress_hint": progress_hint,
        "progress_hint_count": state.get("progress_hint_count", 0) + 1,
    }


def act_node(state: GraphState) -> dict[str, Any]:
    admissible_commands = _admissible_commands_from(state["current_info"])
    thought, action, raw_response = state["agent"].act(
        state["goal"],
        state["observation"],
        admissible_commands,
        state.get("feedback"),
    )
    return {
        "admissible_commands": admissible_commands,
        "thought": thought,
        "action": action,
        "raw_response": raw_response,
        "format_repair_attempted": False,
        "invalid_action_retry_attempted": False,
        "object_grounding": state.get("object_grounding", {}),
    }


def format_repair_node(state: GraphState) -> dict[str, Any]:
    repair_prompt = _build_format_repair_prompt(
        state["goal"],
        state["observation"],
        state["raw_response"],
        state["admissible_commands"],
    )
    repair_response = state["agent"].llm_client.generate(repair_prompt)
    repaired_thought = parse_thought(repair_response)
    repaired_action = parse_action(repair_response)
    return {
        "thought": repaired_thought or state["thought"],
        "action": repaired_action,
        "raw_response": (
            f"{state['raw_response']}\n\n"
            f"[FORMAT_REPAIR_RAW_RESPONSE]\n{repair_response}"
        ),
        "format_repair_attempted": True,
    }


def invalid_action_retry_node(state: GraphState) -> dict[str, Any]:
    retry_prompt = _build_invalid_action_retry_prompt(
        state["goal"],
        state["observation"],
        state["action"],
        state["admissible_commands"],
    )
    retry_response = state["agent"].llm_client.generate(retry_prompt)
    retry_thought = parse_thought(retry_response)
    retry_action = parse_action(retry_response)
    return {
        "thought": retry_thought or state["thought"],
        "action": retry_action,
        "raw_response": (
            f"{state['raw_response']}\n\n"
            f"[INVALID_ACTION_RETRY_RAW_RESPONSE]\n{retry_response}"
        ),
        "invalid_action_retry_attempted": True,
    }


def step_node(state: GraphState) -> dict[str, Any]:
    step_number = state["step_number"] + 1
    action = state.get("action", "")
    format_error = not action
    new_observation, reward, done, info = state["env"].step(action)
    success = _success_from(info, done, reward)
    action_valid = _action_valid_from(info, action)
    consecutive_format_errors = (
        state.get("consecutive_format_errors", 0) + 1 if format_error else 0
    )

    step_log = {
        "episode_id": state["episode_id"],
        "step": step_number,
        "goal": state["goal"],
        "observation": state["observation"],
        "admissible_commands": state["admissible_commands"],
        "thought": state["thought"],
        "action": action,
        "raw_response": state["raw_response"],
        "format_repair_attempted": state.get("format_repair_attempted", False),
        "invalid_action_retry_attempted": state.get("invalid_action_retry_attempted", False),
        "object_grounding": state.get("object_grounding", {}),
        "object_grounding_count": state.get("object_grounding_count", 0),
        "progress_hint": state.get("progress_hint"),
        "progress_hint_count": state.get("progress_hint_count", 0),
        "stuck_reflection_count": state.get("stuck_reflection_count", 0),
        "stuck_reflection_reason": state.get("stuck_reflection_reason"),
        "new_observation": new_observation,
        "reward": reward,
        "done": done,
        "success": success,
        "action_valid": action_valid,
        "format_error": format_error,
        "consecutive_format_errors": consecutive_format_errors,
    }
    state["logger"].log_step(step_log)

    if not format_error:
        state["agent"].record_step(
            state["observation"],
            state["thought"],
            action,
            new_observation,
        )

    early_stop_reason = state.get("early_stop_reason")
    if consecutive_format_errors >= MAX_CONSECUTIVE_FORMAT_ERRORS:
        early_stop_reason = "format_collapse"
    if not early_stop_reason and _runtime_exceeded(state):
        early_stop_reason = "runtime_exceeded"

    reached_max_steps = step_number >= state["max_steps"]
    graph_done = bool(done or success or early_stop_reason or reached_max_steps)
    feedback = _feedback_from_previous_step(
        action,
        action_valid,
        format_error,
        state["admissible_commands"],
    )

    return {
        "step_number": step_number,
        "success": success,
        "total_reward": state["total_reward"] + reward,
        "final_observation": new_observation,
        "current_info": info,
        "observation": state["observation"] if format_error else new_observation,
        "step_logs": state["step_logs"] + [step_log],
        "feedback": feedback,
        "progress_hint": None,
        "consecutive_format_errors": consecutive_format_errors,
        "stuck_reflection_reason": None,
        "early_stop_reason": early_stop_reason,
        "done": graph_done,
    }


def stuck_reflection_node(state: GraphState) -> dict[str, Any]:
    recent_actions = [str(item["action"]) for item in state["step_logs"][-STUCK_CYCLE_WINDOW:]]
    admissible_commands = _admissible_commands_from(state["current_info"])
    avoid_actions = set(recent_actions)
    alternatives = [action for action in admissible_commands if action not in avoid_actions]
    preview = "; ".join(alternatives[:8]) or "(no alternatives provided)"
    reason = _stuck_loop_reason(state["step_logs"]) or "The recent actions are not making progress."
    feedback = (
        f"{reason} Do not repeat that action immediately. "
        f"Pick a different action that changes location, opens a new container, "
        f"takes a visible object, treats a held object, or places it toward the goal. "
        f"Alternative available actions include: {preview}"
    )
    return {
        "feedback": feedback,
        "stuck_reflection_count": state.get("stuck_reflection_count", 0) + 1,
        "stuck_reflection_reason": reason,
    }


def finalize_node(state: GraphState) -> dict[str, Any]:
    failure_reason = None
    if not state["success"]:
        failure_reason = state.get("early_stop_reason") or classify_failure(
            state["step_logs"],
            reached_max_steps=state["step_number"] >= state["max_steps"],
        )

    summary = {
        "episode_id": state["episode_id"],
        "goal": state["goal"],
        "success": state["success"],
        "num_steps": state["step_number"],
        "total_reward": state["total_reward"],
        "failure_reason": failure_reason,
        "early_stop_reason": state.get("early_stop_reason"),
        "final_observation": state["final_observation"],
    }
    state["logger"].log_episode(summary)
    return {"summary": summary}


def route_after_step(
    state: GraphState,
) -> Literal["object_grounding", "stuck_reflection", "finalize"]:
    if state["done"]:
        return "finalize"
    if _should_trigger_stuck_reflection(state):
        return "stuck_reflection"
    return "object_grounding"


def route_after_act(state: GraphState) -> Literal["format_repair", "invalid_action_retry", "step"]:
    action = state.get("action", "")
    if not action:
        return "step" if state.get("format_repair_attempted", False) else "format_repair"
    if not _is_action_available(action, state.get("admissible_commands", [])):
        if not state.get("invalid_action_retry_attempted", False):
            return "invalid_action_retry"
    return "step"


def build_graph() -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "Install LangGraph before running this baseline: "
            "python -m pip install -r requirements.txt"
        ) from exc

    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize_node)
    graph.add_node("object_grounding", object_grounding_node)
    graph.add_node("progress_hint", progress_hint_node)
    graph.add_node("act", act_node)
    graph.add_node("format_repair", format_repair_node)
    graph.add_node("invalid_action_retry", invalid_action_retry_node)
    graph.add_node("stuck_reflection", stuck_reflection_node)
    graph.add_node("step", step_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "object_grounding")
    graph.add_edge("object_grounding", "progress_hint")
    graph.add_edge("progress_hint", "act")
    graph.add_conditional_edges(
        "act",
        route_after_act,
        {
            "format_repair": "format_repair",
            "invalid_action_retry": "invalid_action_retry",
            "step": "step",
        },
    )
    graph.add_conditional_edges(
        "format_repair",
        route_after_act,
        {
            "format_repair": "format_repair",
            "invalid_action_retry": "invalid_action_retry",
            "step": "step",
        },
    )
    graph.add_edge("invalid_action_retry", "step")
    graph.add_conditional_edges(
        "step",
        route_after_step,
        {
            "object_grounding": "object_grounding",
            "stuck_reflection": "stuck_reflection",
            "finalize": "finalize",
        },
    )
    graph.add_edge("stuck_reflection", "object_grounding")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_episode(
    episode_id: int,
    env: Any,
    agent: ReActAgent,
    logger: ExperimentLogger,
    max_steps: int,
    max_runtime_seconds: float | None,
) -> dict[str, Any]:
    graph = build_graph()
    final_state = graph.invoke(
        {
            "episode_id": episode_id,
            "env": env,
            "agent": agent,
            "logger": logger,
            "max_steps": max_steps,
            "max_runtime_seconds": max_runtime_seconds,
            "start_time": time.monotonic(),
        }
    )
    return final_state["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Stop after this wall-clock budget between steps. 0 disables the guard.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to a timestamp, isolating logs per run.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional explicit results directory. Defaults to results/runs/<run-id>.",
    )
    parser.add_argument(
        "--task-types",
        default=None,
        help="Comma-separated ALFWorld task type ids, e.g. 1 or 3,5. Ignored for --mock.",
    )
    parser.add_argument("--mock", action="store_true", help="Run deterministic graph smoke test.")
    parser.add_argument("--clear-results", action="store_true")
    args = parser.parse_args()

    task_types = _parse_task_types(args.task_types)
    env, agent = _build_components(args.mock, task_types)
    run_id = args.run_id or _default_run_id()
    results_dir = args.results_dir or RESULTS_DIR / "runs" / run_id
    logger = ExperimentLogger(results_dir, clear_existing=args.clear_results)
    print(json.dumps({"run_id": run_id, "results_dir": str(results_dir)}, ensure_ascii=False))

    for episode_id in range(args.episodes):
        summary = run_episode(
            episode_id,
            env,
            agent,
            logger,
            args.max_steps,
            args.max_runtime_seconds or None,
        )
        print(json.dumps(summary, ensure_ascii=False))

    print(json.dumps(evaluate(results_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
