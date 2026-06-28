"""Adapter for the public dm-meltingpot 2.4.0 API.

Imports are lazy so the rest of the project remains runnable on machines where
DeepMind Lab2D has no compatible wheel.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pacer_marl.common import MultiAgentEnv, StepResult


class MeltingPotAdapter(MultiAgentEnv):
    def __init__(
        self,
        name: str,
        kind: str = "meltingpot_substrate",
        observation_keys: Iterable[str] = ("RGB", "COLLECTIVE_REWARD"),
        rgb_size: int = 10,
        max_steps: int = 1000,
    ) -> None:
        try:
            from meltingpot import scenario, substrate
        except ImportError as exc:
            raise RuntimeError(
                "Melting Pot is unavailable. On a supported Linux/Python platform, "
                "run: python -m pip install -r requirements-meltingpot.txt"
            ) from exc
        self.name = name
        self.kind = kind
        self.observation_keys = tuple(observation_keys)
        self.rgb_size = int(rgb_size)
        self.max_steps = int(max_steps)
        self.t = 0
        if kind == "meltingpot_scenario":
            self.env = scenario.build(name)
        elif kind == "meltingpot_substrate":
            factory = substrate.get_factory(name)
            self.env = factory.build(factory.default_player_roles())
        else:
            raise ValueError(f"Unsupported Melting Pot kind: {kind}")
        specs = tuple(self.env.action_spec())
        self.num_agents = len(specs)
        action_counts = {int(spec.maximum) - int(spec.minimum) + 1 for spec in specs}
        if len(action_counts) != 1:
            raise ValueError("PACER reference trainer requires equal discrete action spaces")
        self.num_actions = action_counts.pop()
        self.observation_dim = 0

    def _encode_rgb(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 3:
            return image.reshape(-1)
        rows = np.linspace(0, image.shape[0] - 1, self.rgb_size).astype(int)
        cols = np.linspace(0, image.shape[1] - 1, self.rgb_size).astype(int)
        sampled = image[rows][:, cols]
        return (sampled / 255.0).reshape(-1)

    def _encode_one(self, observation: dict[str, Any]) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for key in self.observation_keys:
            if key not in observation:
                continue
            value = np.asarray(observation[key])
            if key.endswith("RGB"):
                pieces.append(self._encode_rgb(value))
            else:
                flat = np.nan_to_num(value.astype(np.float64).reshape(-1))
                pieces.append(np.clip(flat, -20.0, 20.0) / 20.0)
        if not pieces:
            available = ", ".join(sorted(observation))
            raise KeyError(
                f"None of {self.observation_keys!r} found; available: {available}"
            )
        return np.concatenate(pieces)

    def _encode(self, observations: Any) -> np.ndarray:
        matrix = np.stack([self._encode_one(dict(obs)) for obs in observations])
        if self.observation_dim == 0:
            self.observation_dim = int(matrix.shape[1])
        elif matrix.shape[1] != self.observation_dim:
            raise ValueError("Observation shape changed within a Melting Pot run")
        return matrix

    def reset(self) -> np.ndarray:
        self.t = 0
        timestep = self.env.reset()
        return self._encode(timestep.observation)

    def step(self, actions: np.ndarray) -> StepResult:
        action_list = [int(a) for a in np.asarray(actions).reshape(-1)]
        if len(action_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(action_list)}")
        timestep = self.env.step(action_list)
        self.t += 1
        rewards = timestep.reward
        if rewards is None:
            rewards = np.zeros(self.num_agents, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        done = bool(timestep.last()) or self.t >= self.max_steps
        collective = float(rewards.sum())
        return StepResult(
            self._encode(timestep.observation),
            rewards,
            done,
            {"environment": self.name, "collective_reward": collective},
        )

    def close(self) -> None:
        self.env.close()

