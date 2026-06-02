"""Compute aggregate metrics from JSONL baseline logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from config import RESULTS_DIR


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _average(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(results_dir: Path) -> dict[str, Any]:
    steps = _read_jsonl(results_dir / "step_logs.jsonl")
    episodes = _read_jsonl(results_dir / "episode_summary.jsonl")

    successful = [episode for episode in episodes if episode["success"]]
    failed = [episode for episode in episodes if not episode["success"]]
    valid_actions = sum(1 for step in steps if step["action_valid"])

    report = {
        "num_episodes": len(episodes),
        "task_success_rate": len(successful) / len(episodes) if episodes else 0.0,
        "average_steps": _average([episode["num_steps"] for episode in episodes]),
        "average_steps_successful": _average([episode["num_steps"] for episode in successful]),
        "average_steps_failed": _average([episode["num_steps"] for episode in failed]),
        "action_valid_rate": valid_actions / len(steps) if steps else 0.0,
        "failure_reason_distribution": dict(
            Counter(episode["failure_reason"] for episode in failed)
        ),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.results_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

