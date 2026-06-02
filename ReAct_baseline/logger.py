"""JSONL logging for reproducible baseline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, results_dir: Path, clear_existing: bool = False) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = results_dir / "step_logs.jsonl"
        self.episode_log_path = results_dir / "episode_summary.jsonl"

        if clear_existing:
            self.step_log_path.unlink(missing_ok=True)
            self.episode_log_path.unlink(missing_ok=True)

    def log_step(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.step_log_path, record)

    def log_episode(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.episode_log_path, record)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

