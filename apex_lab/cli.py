"""CLI entry point.

Usage::

    python -m apex_lab.cli run --config experiments/default.yaml
    python -m apex_lab.cli list-algorithms
    python -m apex_lab.cli compare --algorithms a_star,rrt_star --track oval
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


# Registry of all available algorithms (name -> import path).
_ALGORITHM_REGISTRY: dict[str, str] = {
    # Classical planners
    "a_star": "apex_lab.algorithms.classical.a_star.AStarPlanner",
    "rrt": "apex_lab.algorithms.classical.rrt.RRTPlanner",
    "rrt_star": "apex_lab.algorithms.classical.rrt_star.RRTStarPlanner",
    "genetic_algorithm": "apex_lab.algorithms.classical.genetic.GAPlanner",
    "mpc": "apex_lab.algorithms.classical.mpc.MPCPlanner",
    # Learning agents
    "q_learning": "apex_lab.algorithms.learning.q_learning.QLearningAgent",
    "dqn": "apex_lab.algorithms.learning.dqn.DQNAgent",
    "ppo": "apex_lab.algorithms.learning.ppo.PPOAgent",
}


def cmd_run(args: argparse.Namespace) -> None:
    """Load a YAML config and run the experiment."""
    from apex_lab.configs.loader import load_config

    config = load_config(args.config)
    print(f"[apex-lab] Running experiment: {config.experiment_name}")
    print(f"  track      : {config.track}")
    print(f"  algorithms : {[a.name for a in config.algorithms]}")
    print(f"  episodes   : {config.episodes}")
    print(f"  seed       : {config.seed}")
    raise NotImplementedError("Experiment runner is not implemented yet.")


def cmd_list_algorithms(_args: argparse.Namespace) -> None:
    """Print all registered algorithms."""
    col_w = max(len(k) for k in _ALGORITHM_REGISTRY)
    print(f"{'Name':<{col_w}}  Class")
    print("-" * (col_w + 2 + 60))
    for name, cls_path in sorted(_ALGORITHM_REGISTRY.items()):
        print(f"{name:<{col_w}}  {cls_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run a quick head-to-head comparison on a named track."""
    algorithm_names = [a.strip() for a in args.algorithms.split(",")]
    unknown = [n for n in algorithm_names if n not in _ALGORITHM_REGISTRY]
    if unknown:
        print(f"[apex-lab] Unknown algorithm(s): {unknown}", file=sys.stderr)
        print(f"  Available: {sorted(_ALGORITHM_REGISTRY)}", file=sys.stderr)
        sys.exit(1)

    print(f"[apex-lab] Comparing {algorithm_names} on track '{args.track}'")
    raise NotImplementedError("Compare command is not implemented yet.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m apex_lab.cli",
        description="apex-lab: racing-line optimisation benchmark",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run an experiment from a YAML config.")
    p_run.add_argument("--config", required=True, help="Path to experiment YAML.")
    p_run.set_defaults(func=cmd_run)

    # list-algorithms
    p_list = sub.add_parser("list-algorithms", help="List all available algorithms.")
    p_list.set_defaults(func=cmd_list_algorithms)

    # compare
    p_compare = sub.add_parser("compare", help="Quick head-to-head comparison.")
    p_compare.add_argument(
        "--algorithms",
        required=True,
        help="Comma-separated algorithm names (e.g. a_star,rrt_star).",
    )
    p_compare.add_argument("--track", required=True, help="Track name (e.g. oval).")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
