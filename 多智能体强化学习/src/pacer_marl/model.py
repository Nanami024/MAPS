"""NumPy actor-critic with partner-conditioned role experts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pacer_marl.common import clipped_norm, discounted_returns, softmax


@dataclass
class ActionRecord:
    action: int
    features: np.ndarray
    gate_features: np.ndarray
    probabilities: np.ndarray
    gate: np.ndarray
    expert_probabilities: np.ndarray
    value: float
    reward: float = 0.0


class InfluenceTracker:
    """EWMA estimator of per-role interventional return signatures."""

    def __init__(self, num_roles: int, decay: float = 0.94) -> None:
        self.values = np.zeros(num_roles, dtype=np.float64)
        self.mass = np.full(num_roles, 1e-3, dtype=np.float64)
        self.decay = float(decay)

    def update(self, responsibilities: np.ndarray, episode_return: float) -> None:
        r = np.asarray(responsibilities, dtype=np.float64)
        r = r / max(float(r.sum()), 1e-12)
        self.values = self.decay * self.values + (1.0 - self.decay) * r * episode_return
        self.mass = self.decay * self.mass + (1.0 - self.decay) * r

    def normalized(self) -> np.ndarray:
        estimate = self.values / np.maximum(self.mass, 1e-3)
        centered = estimate - estimate.mean()
        scale = estimate.std() + 1e-6
        return np.clip(centered / scale, -2.5, 2.5)


class RoleMixtureActorCritic:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        context_dim: int = 10,
        num_roles: int = 4,
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.context_dim = int(context_dim)
        self.num_roles = int(num_roles)
        self.feature_dim = self.obs_dim + self.context_dim + 1
        self.gate_dim = self.context_dim + 1
        self.rng = np.random.default_rng(seed)
        scale = 0.08 / np.sqrt(max(1, self.feature_dim))
        self.experts = self.rng.normal(
            0.0, scale, size=(self.num_roles, self.num_actions, self.feature_dim)
        )
        # A small orthogonal bias prevents identical experts at initialization.
        for role in range(self.num_roles):
            self.experts[role, role % self.num_actions, -1] += 0.15
        self.gate_weights = self.rng.normal(
            0.0, 0.05, size=(self.num_roles, self.gate_dim)
        )
        self.gate_bias = np.zeros(self.num_roles, dtype=np.float64)
        self.value_weights = np.zeros(self.feature_dim, dtype=np.float64)
        self.influence = InfluenceTracker(self.num_roles)

    def _inputs(self, observation: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observation = np.clip(np.asarray(observation, dtype=np.float64), -5.0, 5.0)
        context = np.asarray(context, dtype=np.float64)
        features = np.concatenate([observation, context, [1.0]])
        gate_features = np.concatenate([context, [1.0]])
        return features, gate_features

    def distribution(
        self, observation: np.ndarray, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        features, gate_features = self._inputs(observation, context)
        expert_logits = np.einsum("kad,d->ka", self.experts, features)
        expert_probs = softmax(expert_logits)
        gate_logits = self.gate_weights @ gate_features + self.gate_bias
        gate = softmax(gate_logits)
        probabilities = gate @ expert_probs
        probabilities = probabilities / probabilities.sum()
        value = float(self.value_weights @ features)
        return probabilities, gate, expert_probs, features, gate_features, value

    def act(
        self, observation: np.ndarray, context: np.ndarray, deterministic: bool = False
    ) -> ActionRecord:
        p, gate, experts, features, gate_features, value = self.distribution(
            observation, context
        )
        action = int(np.argmax(p)) if deterministic else int(self.rng.choice(self.num_actions, p=p))
        return ActionRecord(action, features, gate_features, p, gate, experts, value)

    def update_episode(
        self,
        records: list[ActionRecord],
        gamma: float,
        actor_lr: float,
        critic_lr: float,
        entropy_coef: float,
        diversity_coef: float,
        influence_coef: float,
        gradient_clip: float,
    ) -> dict[str, float]:
        if not records:
            return {"loss": 0.0, "role_entropy": 0.0}
        returns = discounted_returns(np.array([r.reward for r in records]), gamma)
        values = np.array([r.value for r in records])
        advantages = np.clip(returns - values, -10.0, 10.0)
        grad_experts = np.zeros_like(self.experts)
        grad_gate = np.zeros_like(self.gate_weights)
        grad_gate_bias = np.zeros_like(self.gate_bias)
        grad_value = np.zeros_like(self.value_weights)
        responsibility_sum = np.zeros(self.num_roles)
        entropy_sum = 0.0
        influence_signal = self.influence.normalized()

        for record, advantage, target in zip(records, advantages, returns):
            a = record.action
            p_a = max(float(record.probabilities[a]), 1e-12)
            posterior = record.gate * record.expert_probabilities[:, a] / p_a
            posterior = posterior / max(float(posterior.sum()), 1e-12)
            responsibility_sum += posterior
            onehot = np.eye(self.num_actions)[a]
            for role in range(self.num_roles):
                score = posterior[role] * (onehot - record.expert_probabilities[role])
                # Entropy pressure toward uniformity, plus role separation from the mean expert.
                score += entropy_coef * (1.0 / self.num_actions - record.expert_probabilities[role])
                mean_policy = record.expert_probabilities.mean(axis=0)
                separation = record.expert_probabilities[role] - mean_policy
                score += diversity_coef * separation
                grad_experts[role] += advantage * np.outer(score, record.features)
            gate_score = posterior - record.gate
            gate_score += influence_coef * record.gate * (
                influence_signal - float(record.gate @ influence_signal)
            )
            grad_gate += advantage * np.outer(gate_score, record.gate_features)
            grad_gate_bias += advantage * gate_score
            grad_value += (target - record.value) * record.features
            entropy_sum += -float(np.sum(record.gate * np.log(record.gate + 1e-12)))

        n = float(len(records))
        self.experts += actor_lr * clipped_norm(grad_experts / n, gradient_clip)
        self.gate_weights += actor_lr * clipped_norm(grad_gate / n, gradient_clip)
        self.gate_bias += actor_lr * clipped_norm(grad_gate_bias / n, gradient_clip)
        self.value_weights += critic_lr * clipped_norm(grad_value / n, gradient_clip)
        self.influence.update(responsibility_sum, float(sum(r.reward for r in records)))
        return {
            "loss": float(np.mean((returns - values) ** 2)),
            "role_entropy": entropy_sum / n,
            "advantage_mean": float(advantages.mean()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs_dim=self.obs_dim,
            num_actions=self.num_actions,
            context_dim=self.context_dim,
            num_roles=self.num_roles,
            experts=self.experts,
            gate_weights=self.gate_weights,
            gate_bias=self.gate_bias,
            value_weights=self.value_weights,
            influence_values=self.influence.values,
            influence_mass=self.influence.mass,
        )

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "RoleMixtureActorCritic":
        data = np.load(path)
        model = cls(
            int(data["obs_dim"]),
            int(data["num_actions"]),
            int(data["context_dim"]),
            int(data["num_roles"]),
            seed=seed,
        )
        model.experts = data["experts"]
        model.gate_weights = data["gate_weights"]
        model.gate_bias = data["gate_bias"]
        model.value_weights = data["value_weights"]
        model.influence.values = data["influence_values"]
        model.influence.mass = data["influence_mass"]
        return model

