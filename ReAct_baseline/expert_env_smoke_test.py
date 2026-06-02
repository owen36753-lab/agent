"""Solve one real ALFWorld train task with the built-in handcoded expert."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from alfworld_env import ALFWorldEnv


MAX_EXPERT_STEPS = 150
MAX_EPISODE_ATTEMPTS = 5
SMOKE_TEST_TASK_TYPES = [1]


def _next_expert_action(info: dict[str, Any]) -> str:
    plan = info.get("extra.expert_plan") or []
    if isinstance(plan, str):
        return plan
    return str(plan[0]) if plan else ""


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "configs" / "base_config.yaml"
    if not os.getenv("ALFWORLD_DATA"):
        os.environ["ALFWORLD_DATA"] = str(Path.home() / ".cache" / "alfworld")

    env = ALFWorldEnv(
        config_path=str(config_path),
        split="train",
        task_types=SMOKE_TEST_TASK_TYPES,
    )

    for attempt in range(1, MAX_EPISODE_ATTEMPTS + 1):
        observation, info = env.reset()
        print("=" * 70)
        print(f"REAL ALFWORLD EXPERT SMOKE TEST | attempt {attempt}/{MAX_EPISODE_ATTEMPTS}")
        print("=" * 70)
        print(observation)
        print("-" * 70)

        for step_number in range(1, MAX_EXPERT_STEPS + 1):
            action = _next_expert_action(info)
            if not action:
                print("No expert action available; trying another task.")
                break

            observation, reward, done, info = env.step(action)
            success = bool(info.get("success", False))
            print(f"[{step_number:>3}] {action}")

            if success:
                print("-" * 70)
                print(f"SUCCESS: True | steps: {step_number} | reward: {reward}")
                print(f"Final observation: {observation}")
                return
            if done:
                print(
                    "Episode ended before expert success; "
                    f"reward={reward}, info_keys={sorted(info)}. Trying another task."
                )
                break
        else:
            print(f"Expert exceeded {MAX_EXPERT_STEPS} steps; trying another task.")

    raise RuntimeError(
        f"Expert did not solve a task in {MAX_EPISODE_ATTEMPTS} attempts. "
        "ALFWorld loaded correctly, but the handcoded expert needs investigation."
    )


if __name__ == "__main__":
    main()
