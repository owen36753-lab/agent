"""Run the ALFWorld ReAct baseline as an explicit LangGraph workflow."""

from __future__ import annotations

import argparse
import json
import sys
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


RESULTS_DIR = PROJECT_DIR / "results"


class GraphState(TypedDict, total=False):
    episode_id: int
    env: Any
    agent: ReActAgent
    logger: ExperimentLogger
    max_steps: int
    goal: str
    observation: str
    current_info: dict[str, Any]
    admissible_commands: list[str]
    thought: str
    action: str
    raw_response: str
    format_repair_attempted: bool
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


def _build_components(use_mock: bool) -> tuple[Any, ReActAgent]:
    if use_mock:
        return MockALFWorldEnv(), ReActAgent(MockLLMClient())
    if LLM_PROVIDER != "openai_compatible":
        raise ValueError("Set LLM_PROVIDER=openai_compatible for real runs, or use --mock.")

    env = ALFWorldEnv(ALFWORLD_CONFIG_PATH, ALFWORLD_SPLIT)
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
        "consecutive_format_errors": 0,
        "early_stop_reason": None,
        "done": False,
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
        "consecutive_format_errors": consecutive_format_errors,
        "early_stop_reason": early_stop_reason,
        "done": graph_done,
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


def route_after_step(state: GraphState) -> Literal["act", "finalize"]:
    return "finalize" if state["done"] else "act"


def route_after_act(state: GraphState) -> Literal["format_repair", "step"]:
    if state.get("action") or state.get("format_repair_attempted", False):
        return "step"
    return "format_repair"


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
    graph.add_node("act", act_node)
    graph.add_node("format_repair", format_repair_node)
    graph.add_node("step", step_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "act")
    graph.add_conditional_edges(
        "act",
        route_after_act,
        {"format_repair": "format_repair", "step": "step"},
    )
    graph.add_edge("format_repair", "step")
    graph.add_conditional_edges(
        "step",
        route_after_step,
        {"act": "act", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_episode(
    episode_id: int,
    env: Any,
    agent: ReActAgent,
    logger: ExperimentLogger,
    max_steps: int,
) -> dict[str, Any]:
    graph = build_graph()
    final_state = graph.invoke(
        {
            "episode_id": episode_id,
            "env": env,
            "agent": agent,
            "logger": logger,
            "max_steps": max_steps,
        }
    )
    return final_state["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--mock", action="store_true", help="Run deterministic graph smoke test.")
    parser.add_argument("--clear-results", action="store_true")
    args = parser.parse_args()

    env, agent = _build_components(args.mock)
    logger = ExperimentLogger(RESULTS_DIR, clear_existing=args.clear_results)

    for episode_id in range(args.episodes):
        summary = run_episode(episode_id, env, agent, logger, args.max_steps)
        print(json.dumps(summary, ensure_ascii=False))

    print(json.dumps(evaluate(RESULTS_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
