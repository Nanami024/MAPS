"""Training and evaluation loop for PACER."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pacer_marl.context import OnlinePartnerContext
from pacer_marl.envs.meltingpot_adapter import MeltingPotAdapter
from pacer_marl.envs.toy_coordination import ProceduralCoordinationEnv
from pacer_marl.model import ActionRecord, RoleMixtureActorCritic


def make_environment(config: dict[str, Any], seed: int, evaluation: bool = False):
    env_cfg = config["environment"]
    kind = env_cfg.get("kind", "toy")
    if kind == "toy":
        partners = env_cfg.get("eval_partners" if evaluation else "train_partners")
        return ProceduralCoordinationEnv(
            seed=seed,
            horizon=env_cfg.get("horizon", 28),
            num_actions=env_cfg.get("num_actions", 4),
            partner_types=partners,
        )
    names = env_cfg.get("names") or [env_cfg["name"]]
    name = names[seed % len(names)]
    return MeltingPotAdapter(
        name=name,
        kind=kind,
        observation_keys=env_cfg.get("observation_keys", ["RGB", "COLLECTIVE_REWARD"]),
        rgb_size=env_cfg.get("rgb_size", 10),
        max_steps=env_cfg.get("max_steps", 1000),
    )


def _contexts(env, observations: np.ndarray, algo: dict[str, Any], seed: int):
    encoders = [
        OnlinePartnerContext(
            env.observation_dim,
            env.num_actions,
            algo.get("context_dim", 10),
            algo.get("context_decay", 0.8),
            seed + 1009 * i,
        )
        for i in range(env.num_agents)
    ]
    states = np.stack([e.reset(o) for e, o in zip(encoders, observations)])
    return encoders, states


def run_episode(
    env,
    model: RoleMixtureActorCritic,
    algo: dict[str, Any],
    training: bool,
    deterministic: bool = False,
) -> dict[str, Any]:
    observations = env.reset()
    encoders, contexts = _contexts(env, observations, algo, seed=int(model.rng.integers(1 << 30)))
    trajectories: list[list[ActionRecord]] = [[] for _ in range(env.num_agents)]
    total_rewards = np.zeros(env.num_agents, dtype=np.float64)
    coordination = []
    final_info: dict[str, Any] = {}
    while True:
        records = [
            model.act(obs, ctx, deterministic=deterministic)
            for obs, ctx in zip(observations, contexts)
        ]
        actions = np.array([record.action for record in records], dtype=np.int64)
        step = env.step(actions)
        for i, record in enumerate(records):
            record.reward = float(step.rewards[i])
            trajectories[i].append(record)
        total_rewards += step.rewards
        if "coordination" in step.info:
            coordination.append(float(step.info["coordination"]))
        contexts = np.stack(
            [
                encoder.update(obs, int(action), float(reward))
                for encoder, obs, action, reward in zip(
                    encoders, step.observations, actions, step.rewards
                )
            ]
        )
        observations = step.observations
        final_info = step.info
        if step.done:
            break

    update_stats = []
    if training:
        for trajectory in trajectories:
            update_stats.append(
                model.update_episode(
                    trajectory,
                    gamma=algo.get("gamma", 0.99),
                    actor_lr=algo.get("actor_lr", 0.002),
                    critic_lr=algo.get("critic_lr", 0.004),
                    entropy_coef=algo.get("entropy_coef", 0.002),
                    diversity_coef=algo.get("diversity_coef", 0.006),
                    influence_coef=algo.get("influence_coef", 0.01),
                    gradient_clip=algo.get("gradient_clip", 3.0),
                )
            )
    role_entropies = [s["role_entropy"] for s in update_stats] or [0.0]
    return {
        "return_mean": float(total_rewards.mean()),
        "return_sum": float(total_rewards.sum()),
        "coordination_rate": float(np.mean(coordination)) if coordination else float("nan"),
        "role_entropy": float(np.mean(role_entropies)),
        "steps": len(trajectories[0]),
        "partner_type": final_info.get("partner_type", "meltingpot"),
        "environment": final_info.get("environment", getattr(env, "name", "toy")),
    }


def train(config: dict[str, Any], output_dir: str | Path) -> RoleMixtureActorCritic:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    first_env = make_environment(config, seed)
    first_obs = first_env.reset()
    algo = config["algorithm"]
    model = RoleMixtureActorCritic(
        first_env.observation_dim,
        first_env.num_actions,
        context_dim=algo.get("context_dim", 10),
        num_roles=algo.get("num_roles", 4),
        seed=seed,
    )
    first_env.close()
    fieldnames = [
        "episode", "return_mean", "return_sum", "coordination_rate",
        "role_entropy", "steps", "partner_type", "environment",
    ]
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        recent = []
        for episode in range(int(config.get("episodes", 500))):
            env = make_environment(config, seed + episode)
            if env.observation_dim == 0:
                env.reset()
            if env.observation_dim != model.obs_dim or env.num_actions != model.num_actions:
                env.close()
                raise ValueError("All sampled environments must share encoded shape and action count")
            stats = run_episode(env, model, algo, training=True)
            env.close()
            row = {"episode": episode + 1, **stats}
            writer.writerow(row)
            handle.flush()
            recent.append(stats["return_mean"])
            recent = recent[-50:]
            if (episode + 1) % int(config.get("log_every", 25)) == 0:
                print(
                    f"episode={episode + 1:5d} return50={np.mean(recent):8.3f} "
                    f"role_H={stats['role_entropy']:.3f} partner={stats['partner_type']}"
                )
            checkpoint_every = int(config.get("checkpoint_every", 100))
            if checkpoint_every and (episode + 1) % checkpoint_every == 0:
                model.save(output / f"checkpoint_{episode + 1:06d}.npz")
    model.save(output / "checkpoint_final.npz")
    return model


def evaluate(
    config: dict[str, Any],
    model: RoleMixtureActorCritic,
    episodes: int | None = None,
    evaluation_pool: bool = True,
) -> dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    episodes = int(episodes or eval_cfg.get("episodes", 100))
    deterministic = bool(eval_cfg.get("deterministic", True))
    seed = int(config.get("seed", 0)) + 100_000
    rows = []
    for episode in range(episodes):
        env = make_environment(config, seed + episode, evaluation=evaluation_pool)
        if env.observation_dim == 0:
            env.reset()
        rows.append(
            run_episode(env, model, config["algorithm"], training=False, deterministic=deterministic)
        )
        env.close()
    returns = np.array([r["return_mean"] for r in rows])
    by_partner: dict[str, list[float]] = {}
    for row in rows:
        by_partner.setdefault(row["partner_type"], []).append(row["return_mean"])
    return {
        "episodes": episodes,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_p10": float(np.quantile(returns, 0.10)),
        "by_partner": {
            name: {"mean": float(np.mean(values)), "n": len(values)}
            for name, values in sorted(by_partner.items())
        },
    }


def write_evaluation(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

