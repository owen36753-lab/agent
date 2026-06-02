"""Replaceable LLM clients with a single generate(prompt) interface."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Protocol
from urllib.error import HTTPError
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
    min_request_interval_seconds: float = 0.0
    max_retries: int = 3
    retry_base_seconds: float = 5.0
    _last_request_at: float = field(default=0.0, init=False, repr=False)

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
        for attempt in range(self.max_retries + 1):
            self._wait_for_request_slot()
            try:
                self._last_request_at = time.monotonic()
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code != 429 or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"LLM API request failed with HTTP {exc.code}: {error_body}"
                    ) from exc

                delay = self._retry_delay_seconds(exc, error_body, attempt)
                print(
                    f"LLM API rate limit reached (HTTP 429). "
                    f"Retrying in {delay:.1f}s ({attempt + 1}/{self.max_retries})."
                )
                print(f"LLM API response: {error_body}")
                time.sleep(delay)

        raise RuntimeError("LLM API request failed after retries.")

    def _wait_for_request_slot(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = self.min_request_interval_seconds - elapsed
        if delay > 0:
            time.sleep(delay)

    def _retry_delay_seconds(self, error: HTTPError, body: str, attempt: int) -> float:
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), self.retry_base_seconds)
            except ValueError:
                pass

        match = re.search(
            r"(?:retry in\s*|retryDelay['\"]?\s*:\s*['\"]?)([0-9.]+)s",
            body,
            re.IGNORECASE,
        )
        if match:
            return max(float(match.group(1)), self.retry_base_seconds)
        return self.retry_base_seconds * (2**attempt)


class MockLLMClient:
    """Deterministic client used only for local smoke tests."""

    def generate(self, prompt: str) -> str:
        current_observation = prompt.rsplit("Current observation: ", maxsplit=1)[-1]
        if current_observation.startswith("You are in a room."):
            return "Thought: I should inspect the room.\nAction: look"
        return "Thought: I found the target and should finish.\nAction: finish"
