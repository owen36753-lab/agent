"""Send one minimal request to the configured OpenAI-compatible LLM endpoint."""

from __future__ import annotations

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MIN_REQUEST_INTERVAL_SECONDS,
    LLM_MODEL,
    LLM_RETRY_BASE_SECONDS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)
from llm_client import OpenAICompatibleLLMClient
from react_agent import parse_action, parse_thought


def main() -> None:
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
    response = client.generate(
        "Return exactly two lines and nothing else:\n"
        "Thought: I should inspect the room.\n"
        "Action: look\n"
    )
    thought = parse_thought(response)
    action = parse_action(response)
    if not action:
        raise RuntimeError(f"LLM response did not contain a parseable Action line: {response!r}")

    print("LLM API smoke test passed.")
    print(f"Model: {LLM_MODEL}")
    print(f"Base URL: {LLM_BASE_URL}")
    print(f"Thought: {thought}")
    print(f"Action: {action}")
    print(f"Raw response: {response!r}")


if __name__ == "__main__":
    main()
