"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pacer_marl.model import RoleMixtureActorCritic
from pacer_marl.trainer import evaluate, train, write_evaluation


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def smoke_config(episodes: int) -> dict:
    config = load_config(ROOT / "configs" / "pacer_toy.json")
    config["episodes"] = int(episodes)
    config["log_every"] = max(1, int(episodes) // 3)
    config["checkpoint_every"] = 0
    config["evaluation"]["episodes"] = 12
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PACER-MP reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a PACER policy")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--output", required=True)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--episodes", type=int)
    p_eval.add_argument("--output", default="evaluation.json")

    p_smoke = sub.add_parser("smoke", help="short end-to-end dependency-light run")
    p_smoke.add_argument("--episodes", type=int, default=30)
    p_smoke.add_argument("--output", default="runs/smoke")

    args = parser.parse_args(argv)
    if args.command == "train":
        config = load_config(args.config)
        train(config, args.output)
        print(f"checkpoint: {Path(args.output) / 'checkpoint_final.npz'}")
    elif args.command == "evaluate":
        config = load_config(args.config)
        model = RoleMixtureActorCritic.load(args.checkpoint, seed=config.get("seed", 0))
        result = evaluate(config, model, args.episodes)
        write_evaluation(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        config = smoke_config(args.episodes)
        model = train(config, args.output)
        result = evaluate(config, model, config["evaluation"]["episodes"])
        output = Path(args.output) / "evaluation.json"
        write_evaluation(result, output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"smoke run complete: {output}")


if __name__ == "__main__":
    main()

