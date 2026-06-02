"""ALFWorld environment adapter with reset() and step(action)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


class ALFWorldEnv:
    """Wrap ALFWorld's batched text environment as a single-episode interface."""

    def __init__(self, config_path: str, split: str) -> None:
        if not config_path:
            raise ValueError("Set ALFWORLD_CONFIG_PATH to an ALFWorld YAML config file.")
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"ALFWorld config does not exist: {config_path}")

        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise RuntimeError(
                "Install ALFWorld and PyYAML before running the real environment."
            ) from exc

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        env_type = config["env"]["type"]
        env_class = get_environment(env_type)
        self._env = env_class(config, train_eval=split).init_env(batch_size=1)

    def reset(self) -> tuple[str, dict[str, Any]]:
        observations, infos = self._env.reset()
        return str(_first(observations)), self._normalize_info(infos)

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        observations, rewards, dones, infos = self._env.step([action])
        return (
            str(_first(observations)),
            float(_first(rewards)),
            bool(_first(dones)),
            self._normalize_info(infos),
        )

    @staticmethod
    def _normalize_info(infos: Any) -> dict[str, Any]:
        if isinstance(infos, list):
            normalized = dict(infos[0]) if infos else {}
        elif isinstance(infos, dict):
            normalized = {key: _first(value) for key, value in infos.items()}
        else:
            normalized = {}

        if "success" not in normalized and "won" in normalized:
            normalized["success"] = bool(normalized["won"])
        return normalized


class MockALFWorldEnv:
    """Small deterministic environment for verifying the baseline wiring."""

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> tuple[str, dict[str, Any]]:
        self._step = 0
        return "You are in a room. Find and finish the mock task.", {"goal": "Finish the mock task."}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._step += 1
        if self._step == 1 and action == "look":
            return "The target is visible.", 0.0, False, {"action_valid": True}
        if self._step == 2 and action == "finish":
            return "Task complete.", 1.0, True, {"success": True, "action_valid": True}
        return "Nothing happens.", 0.0, False, {"action_valid": False}
