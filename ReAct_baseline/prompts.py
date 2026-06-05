"""Prompt templates for the plain ReAct agent."""

from __future__ import annotations

REACT_SYSTEM_PROMPT = """You are an agent acting in the ALFWorld text environment.
Choose exactly one environment action for the current step.
If available actions are provided, the Action line must be copied exactly from that list.
Do not emit multiple actions. Do not add text after the Action line.
Do not use Markdown, bullets, code fences, backticks, **Action**, or Answer:.

ALFWorld action hints:
- If the current observation shows the needed object and a take action exists, take it.
- If you are at a closed container and an open action exists, open it before leaving.
- If you are holding the needed object, do the required treatment next: heat with microwave, cool with fridge, clean with sinkbasin.
- After the needed treatment is done, move the object to the goal receptacle.
- Search broadly across plausible receptacles and surfaces; do not only enumerate cabinets.
- Prefer actions that make direct progress toward finding, taking, treating, or placing the goal object.

Respond exactly in this format:
Thought: <brief reasoning>
Action: <one action>
"""


def _format_available_actions(available_actions: list[str]) -> str:
    if not available_actions:
        return "(not provided)"
    return "\n".join(f"- {action}" for action in available_actions)


def build_react_prompt(
    goal: str,
    observation: str,
    history: list[dict[str, str]],
    available_actions: list[str] | None = None,
    feedback: str | None = None,
) -> str:
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
    actions = _format_available_actions(available_actions or [])
    feedback_text = feedback or "(none)"
    return (
        f"{REACT_SYSTEM_PROMPT}\n"
        f"Goal: {goal}\n"
        f"Trajectory:\n{trajectory}\n"
        f"Current observation: {observation}\n"
        f"Previous step feedback: {feedback_text}\n"
        f"Available actions for this step:\n{actions}\n"
        "Thought:"
    )
