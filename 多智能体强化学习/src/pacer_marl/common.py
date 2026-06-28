"""Shared data structures and numerical helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


def softmax(logits: Array) -> Array:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(np.clip(x, -60.0, 60.0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def discounted_returns(rewards: Array, gamma: float) -> Array:
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def clipped_norm(x: Array, maximum: float) -> Array:
    norm = float(np.linalg.norm(x))
    if norm <= maximum or norm == 0.0:
        return x
    return x * (maximum / norm)


@dataclass
class StepResult:
    observations: Array
    rewards: Array
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class MultiAgentEnv:
    """Small interface shared by the toy and Melting Pot adapters."""

    num_agents: int
    num_actions: int
    observation_dim: int

    def reset(self) -> Array:
        raise NotImplementedError

    def step(self, actions: Array) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        return None

