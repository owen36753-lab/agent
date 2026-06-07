"""Rule-based failure labels for baseline analysis."""

from __future__ import annotations

from typing import Any


def _is_valid_stalled_step(step: dict[str, Any]) -> bool:
    return (
        bool(step.get("action"))
        and step.get("action_valid", True)
        and not step.get("format_error", False)
        and not step.get("success", False)
    )


def _has_tail_consecutive_repeat(step_logs: list[dict[str, Any]], window: int = 3) -> bool:
    if len(step_logs) < window:
        return False
    recent = step_logs[-window:]
    actions = [step.get("action") for step in recent]
    return bool(
        actions[0]
        and len(set(actions)) == 1
        and all(_is_valid_stalled_step(step) for step in recent)
    )


def _has_tail_alternating_cycle(step_logs: list[dict[str, Any]], window: int = 4) -> bool:
    if len(step_logs) < window:
        return False
    recent = step_logs[-window:]
    actions = [step.get("action") for step in recent]
    return bool(
        all(actions)
        and actions[0] != actions[1]
        and actions == actions[:2] * 2
        and all(_is_valid_stalled_step(step) for step in recent)
    )


def classify_failure(step_logs: list[dict[str, Any]], reached_max_steps: bool) -> str | None:
    if not step_logs:
        return "unknown_failure"

    if any(step.get("format_error") for step in step_logs):
        return "format_error"

    if any(not step["action_valid"] for step in step_logs):
        return "invalid_action"

    if _has_tail_consecutive_repeat(step_logs) or _has_tail_alternating_cycle(step_logs):
        return "repeated_action_loop"

    if reached_max_steps:
        return "max_steps_exceeded"

    return "unknown_failure"
