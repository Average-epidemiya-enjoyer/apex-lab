"""YAML config loading and dataclass schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AlgorithmConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    dt: float = 0.05
    max_steps: int = 10_000


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Mirrors the structure of experiment YAML files.
    """

    experiment_name: str
    track: str
    algorithms: list[AlgorithmConfig]
    episodes: int = 1
    seed: int = 42
    render: bool = False
    output_dir: str = "runs"
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    metrics: list[str] = field(default_factory=lambda: ["lap_time", "path_length"])


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML file into an :class:`ExperimentConfig`.

    Args:
        path: Path to the YAML experiment file.

    Returns:
        Validated :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if "experiment_name" not in raw:
        raise ValueError("Config must contain 'experiment_name'.")
    if "track" not in raw:
        raise ValueError("Config must contain 'track'.")
    if "algorithms" not in raw:
        raise ValueError("Config must contain 'algorithms'.")

    algorithms = [
        AlgorithmConfig(name=a["name"], params=a.get("params", {}))
        for a in raw["algorithms"]
    ]

    sim_raw = raw.get("simulator", {})
    simulator = SimulatorConfig(
        dt=sim_raw.get("dt", 0.05),
        max_steps=sim_raw.get("max_steps", 10_000),
    )

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        track=raw["track"],
        algorithms=algorithms,
        episodes=raw.get("episodes", 1),
        seed=raw.get("seed", 42),
        render=raw.get("render", False),
        output_dir=raw.get("output_dir", "runs"),
        simulator=simulator,
        metrics=raw.get("metrics", ["lap_time", "path_length"]),
    )
