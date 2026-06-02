"""Verify one reset() and one valid step() against real text-only ALFWorld."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from alfworld_env import ALFWorldEnv
from config import ALFWORLD_CONFIG_PATH, ALFWORLD_SPLIT


def _select_action(info: dict[str, Any]) -> str:
    commands = info.get("admissible_commands") or []
    if not commands:
        raise RuntimeError("ALFWorld did not return admissible_commands after reset().")
    return str(commands[0])


def main() -> None:
    default_config = Path(__file__).resolve().parent / "configs" / "base_config.yaml"
    config_path = ALFWORLD_CONFIG_PATH or str(default_config)
    if not os.getenv("ALFWORLD_DATA"):
        os.environ["ALFWORLD_DATA"] = str(Path.home() / ".cache" / "alfworld")

    env = ALFWorldEnv(config_path=config_path, split=ALFWORLD_SPLIT)
    observation, info = env.reset()
    action = _select_action(info)
    new_observation, reward, done, new_info = env.step(action)
    if not new_info.get("action_valid"):
        raise RuntimeError(f"Adapter rejected an admissible action: {action}")

    print("Real ALFWorld reset/step smoke test passed.")
    print(f"Observation: {observation[:300]}")
    print(f"Action: {action}")
    print(f"New observation: {new_observation[:300]}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info keys: {sorted(new_info)}")


if __name__ == "__main__":
    main()
