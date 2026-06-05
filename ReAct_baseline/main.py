"""Run plain ReAct episodes in ALFWorld."""

from __future__ import annotations

import argparse
import json
from typing import Any

from alfworld_env import ALFWorldEnv, MockALFWorldEnv
from config import (
    ALFWORLD_CONFIG_PATH,
    ALFWORLD_SPLIT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_MAX_RETRIES,
    LLM_MIN_REQUEST_INTERVAL_SECONDS,
    LLM_PROVIDER,
    LLM_RETRY_BASE_SECONDS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONSECUTIVE_FORMAT_ERRORS,
    MAX_STEPS,
    NUM_EPISODES,
    RESULTS_DIR,
)
from evaluator import evaluate
from failure_classifier import classify_failure
from llm_client import MockLLMClient, OpenAICompatibleLLMClient
from logger import ExperimentLogger
from react_agent import ReActAgent


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


def run_episode(
    episode_id: int,
    env: Any,
    agent: ReActAgent,
    logger: ExperimentLogger,
    max_steps: int,
) -> dict[str, Any]:
    observation, reset_info = env.reset()
    goal = _goal_from(observation, reset_info)
    agent.reset()

    success = False
    total_reward = 0.0
    step_logs: list[dict[str, Any]] = []
    final_observation = observation
    current_info = reset_info
    feedback: str | None = None
    consecutive_format_errors = 0
    early_stop_reason = None

    for step_number in range(1, max_steps + 1):
        admissible_commands = _admissible_commands_from(current_info)
        thought, action, raw_response = agent.act(
            goal,
            observation,
            admissible_commands,
            feedback,
        )
        format_error = not action
        new_observation, reward, done, info = env.step(action)
        success = _success_from(info, done, reward)
        action_valid = _action_valid_from(info, action)
        consecutive_format_errors = (
            consecutive_format_errors + 1 if format_error else 0
        )
        total_reward += reward

        step_log = {
            "episode_id": episode_id,
            "step": step_number,
            "goal": goal,
            "observation": observation,
            "admissible_commands": admissible_commands,
            "thought": thought,
            "action": action,
            "raw_response": raw_response,
            "new_observation": new_observation,
            "reward": reward,
            "done": done,
            "success": success,
            "action_valid": action_valid,
            "format_error": format_error,
            "consecutive_format_errors": consecutive_format_errors,
        }
        logger.log_step(step_log)
        step_logs.append(step_log)
        if not format_error:
            agent.record_step(observation, thought, action, new_observation)
        feedback = _feedback_from_previous_step(
            action,
            action_valid,
            format_error,
            admissible_commands,
        )
        if not format_error:
            observation = new_observation
        final_observation = new_observation
        current_info = info

        if consecutive_format_errors >= MAX_CONSECUTIVE_FORMAT_ERRORS:
            early_stop_reason = "format_collapse"
            break
        if done:
            break

    num_steps = len(step_logs)
    failure_reason = None
    if not success:
        failure_reason = early_stop_reason or classify_failure(
            step_logs,
            reached_max_steps=num_steps >= max_steps,
        )

    summary = {
        "episode_id": episode_id,
        "goal": goal,
        "success": success,
        "num_steps": num_steps,
        "total_reward": total_reward,
        "failure_reason": failure_reason,
        "early_stop_reason": early_stop_reason,
        "final_observation": final_observation,
    }
    logger.log_episode(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--mock", action="store_true", help="Run deterministic local smoke test.")
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
