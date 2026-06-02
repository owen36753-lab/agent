"""Replaceable LLM clients with a single generate(prompt) interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import request


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate one ReAct response."""


@dataclass
class OpenAICompatibleLLMClient:
    """Minimal client for OpenAI-compatible chat completion endpoints."""

    api_key: str
    model: str
    base_url: str
    temperature: float = 0.0
    timeout_seconds: int = 60

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY is required for the openai_compatible provider.")
        if not self.model:
            raise ValueError("LLM_MODEL is required for the openai_compatible provider.")

        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]


class MockLLMClient:
    """Deterministic client used only for local smoke tests."""

    def generate(self, prompt: str) -> str:
        current_observation = prompt.rsplit("Current observation: ", maxsplit=1)[-1]
        if current_observation.startswith("You are in a room."):
            return "Thought: I should inspect the room.\nAction: look"
        return "Thought: I found the target and should finish.\nAction: finish"
