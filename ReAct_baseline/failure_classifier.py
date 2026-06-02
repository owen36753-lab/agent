"""Rule-based failure labels for baseline analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any


def classify_failure(step_logs: list[dict[str, Any]], reached_max_steps: bool) -> str | None:
    if not step_logs:
        return "unknown_failure"

    if any(not step["action_valid"] for step in step_logs):
        return "invalid_action"

    actions = [step["action"] for step in step_logs if step["action"]]
    if actions:
        most_common_count = Counter(actions).most_common(1)[0][1]
        if most_common_count >= 3:
            return "repeated_action_loop"

    if reached_max_steps:
        return "max_steps_exceeded"

    return "unknown_failure"

