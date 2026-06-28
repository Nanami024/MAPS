"""Decentralized online context state for partner adaptation."""

from __future__ import annotations

import numpy as np


class OnlinePartnerContext:
    """Compresses local history using a stable random projection and EWMA.

    It consumes only local observation, own action and own reward, preserving
    decentralized execution. In a larger implementation this component can be
    replaced by a learned GRU/Transformer without changing the policy API.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        context_dim: int,
        decay: float,
        seed: int,
    ) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.decay = float(decay)
        rng = np.random.default_rng(seed)
        self.obs_projection = rng.normal(
            0.0, 1.0 / np.sqrt(max(1, obs_dim)), size=(context_dim, obs_dim)
        )
        self.aux_projection = rng.normal(
            0.0,
            1.0 / np.sqrt(num_actions + 3),
            size=(context_dim, num_actions + 3),
        )
        self.state = np.zeros(context_dim, dtype=np.float64)
        self.previous_observation = np.zeros(obs_dim, dtype=np.float64)

    def reset(self, observation: np.ndarray) -> np.ndarray:
        self.state.fill(0.0)
        self.previous_observation = np.asarray(observation, dtype=np.float64).copy()
        return self.state.copy()

    def update(self, observation: np.ndarray, action: int, reward: float) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        action_onehot = np.eye(self.num_actions, dtype=np.float64)[int(action)]
        delta = observation - self.previous_observation
        aux = np.concatenate(
            [action_onehot, [float(reward), float(delta.mean()), float(delta.std())]]
        )
        innovation = np.tanh(
            self.obs_projection @ observation + self.aux_projection @ aux
        )
        self.state = np.tanh(self.decay * self.state + (1.0 - self.decay) * innovation)
        self.previous_observation = observation.copy()
        return self.state.copy()

