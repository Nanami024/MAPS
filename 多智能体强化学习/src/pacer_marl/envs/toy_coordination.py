"""Procedural cooperative game with train/test partner separation.

The learner controls one focal player. A scripted background player has a
latent convention. Tasks vary in their payoff relation, cue and observation
noise, which makes the environment useful for checking partner inference,
cross-environment training and zero-shot cross-play without Lab2D.
"""

from __future__ import annotations

import numpy as np

from pacer_marl.common import MultiAgentEnv, StepResult


class ProceduralCoordinationEnv(MultiAgentEnv):
    TRAIN_PARTNERS = ("leader", "follower", "alternator", "reciprocator")
    EVAL_PARTNERS = ("noisy_leader", "delayed_follower", "contrarian")

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 28,
        num_actions: int = 4,
        partner_types: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.horizon = int(horizon)
        self.num_actions = int(num_actions)
        self.num_agents = 1
        self.partner_types = tuple(partner_types or self.TRAIN_PARTNERS)
        self.observation_dim = 3 * self.num_actions + 9
        self.t = 0
        self.task = 0
        self.cue = 0
        self.shift = 0
        self.partner_type = self.partner_types[0]
        self.partner_action = 0
        self.prev_partner_action = 0
        self.prev_learner_action = 0
        self.prev_reward = 0.0
        self.coordination_count = 0

    def reset(self) -> np.ndarray:
        self.t = 0
        self.task = int(self.rng.integers(0, 4))
        self.cue = int(self.rng.integers(0, self.num_actions))
        self.shift = int(self.rng.integers(0, self.num_actions))
        self.partner_type = str(self.rng.choice(self.partner_types))
        self.partner_action = self.cue
        self.prev_partner_action = self.cue
        self.prev_learner_action = int(self.rng.integers(0, self.num_actions))
        self.prev_reward = 0.0
        self.coordination_count = 0
        return self._observation()[None, :]

    def _partner_policy(self) -> int:
        a = self.num_actions
        if self.partner_type == "leader":
            return self.cue
        if self.partner_type == "follower":
            return self.prev_learner_action
        if self.partner_type == "alternator":
            return (self.cue + self.t) % a
        if self.partner_type == "reciprocator":
            return self.prev_learner_action if self.prev_reward > 0 else self.cue
        if self.partner_type == "noisy_leader":
            if self.rng.random() < 0.22:
                return int(self.rng.integers(0, a))
            return self.cue
        if self.partner_type == "delayed_follower":
            return self.prev_partner_action if self.t % 3 else self.prev_learner_action
        if self.partner_type == "contrarian":
            return (self.prev_learner_action + 1 + self.shift) % a
        raise ValueError(f"Unknown partner type: {self.partner_type}")

    def _success(self, learner_action: int, partner_action: int) -> bool:
        if self.task == 0:  # direct convention matching
            return learner_action == partner_action
        if self.task == 1:  # publicly cued joint target
            return learner_action == self.cue and partner_action == self.cue
        if self.task == 2:  # complementary role allocation
            return learner_action == (partner_action + self.shift) % self.num_actions
        # Temporal hand-off: respond to the partner's previous move.
        return learner_action == self.prev_partner_action

    def _observation(self) -> np.ndarray:
        cue = np.eye(self.num_actions, dtype=np.float64)[self.cue]
        partner = np.eye(self.num_actions, dtype=np.float64)[self.partner_action]
        previous = np.eye(self.num_actions, dtype=np.float64)[self.prev_partner_action]
        task = np.eye(4, dtype=np.float64)[self.task]
        time = self.t / max(1, self.horizon - 1)
        scalars = np.array(
            [time, self.prev_reward, self.shift / max(1, self.num_actions - 1),
             self.coordination_count / max(1, self.t), 1.0],
            dtype=np.float64,
        )
        obs = np.concatenate([cue, partner, previous, task, scalars])
        # Observation noise emulates imperfect visual decoding of a partner.
        if self.t > 0 and self.rng.random() < 0.04:
            obs[self.num_actions:2 * self.num_actions] = 0.0
        return obs

    def step(self, actions: np.ndarray) -> StepResult:
        learner_action = int(np.asarray(actions).reshape(-1)[0])
        if not 0 <= learner_action < self.num_actions:
            raise ValueError(f"action {learner_action} outside [0, {self.num_actions})")
        partner_action = self._partner_policy()
        success = self._success(learner_action, partner_action)
        reward = 1.0 if success else -0.15
        # Small cost discourages a degenerate always-switching convention.
        if learner_action != self.prev_learner_action:
            reward -= 0.015
        self.coordination_count += int(success)
        self.prev_partner_action = self.partner_action
        self.partner_action = partner_action
        self.prev_learner_action = learner_action
        self.prev_reward = reward
        self.t += 1
        done = self.t >= self.horizon
        info = {
            "partner_type": self.partner_type,
            "task": self.task,
            "coordination": float(success),
            "coordination_rate": self.coordination_count / self.t,
            "partner_action": partner_action,
        }
        return StepResult(self._observation()[None, :], np.array([reward]), done, info)

