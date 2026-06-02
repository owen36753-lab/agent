"""Plain ReAct agent without memory, reflection, or replanning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_client import LLMClient
from prompts import build_react_prompt


def parse_thought(raw_response: str) -> str:
    """Extract the first Thought line for logging."""
    match = re.search(r"(?im)^Thought:\s*(.*)$", raw_response)
    return match.group(1).strip() if match else ""


def parse_action(raw_response: str) -> str:
    """Extract only the first line after the first Action: marker."""
    match = re.search(r"(?im)^Action:\s*([^\r\n]*)", raw_response)
    return match.group(1).strip() if match else ""


@dataclass
class ReActAgent:
    llm_client: LLMClient
    history: list[dict[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def act(self, goal: str, observation: str) -> tuple[str, str, str]:
        prompt = build_react_prompt(goal, observation, self.history)
        raw_response = self.llm_client.generate(prompt)
        return parse_thought(raw_response), parse_action(raw_response), raw_response

    def record_step(
        self,
        observation: str,
        thought: str,
        action: str,
        new_observation: str,
    ) -> None:
        self.history.append(
            {
                "observation": observation,
                "thought": thought,
                "action": action,
                "new_observation": new_observation,
            }
        )

