"""Prompt templates for the plain ReAct agent."""

from __future__ import annotations

REACT_SYSTEM_PROMPT = """You are an agent acting in the ALFWorld text environment.
Choose exactly one environment action for the current step.
Do not emit multiple actions. Do not add text after the Action line.

Respond exactly in this format:
Thought: <brief reasoning>
Action: <one action>
"""


def build_react_prompt(goal: str, observation: str, history: list[dict[str, str]]) -> str:
    """Build a compact plain ReAct prompt from the current trajectory."""
    trajectory_lines: list[str] = []
    for item in history:
        trajectory_lines.extend(
            [
                f"Observation: {item['observation']}",
                f"Thought: {item['thought']}",
                f"Action: {item['action']}",
                f"New observation: {item['new_observation']}",
            ]
        )

    trajectory = "\n".join(trajectory_lines) if trajectory_lines else "(no previous steps)"
    return (
        f"{REACT_SYSTEM_PROMPT}\n"
        f"Goal: {goal}\n"
        f"Trajectory:\n{trajectory}\n"
        f"Current observation: {observation}\n"
        "Thought:"
    )

