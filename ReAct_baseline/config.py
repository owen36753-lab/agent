"""Configuration for the minimal ALFWorld ReAct baseline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "1"))

ALFWORLD_CONFIG_PATH = os.getenv("ALFWORLD_CONFIG_PATH", "")
ALFWORLD_SPLIT = os.getenv("ALFWORLD_SPLIT", "eval_out_of_distribution")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MIN_REQUEST_INTERVAL_SECONDS = float(
    os.getenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "0")
)
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "5"))
