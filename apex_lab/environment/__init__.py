"""Модуль окружения: трасса, физика машины, симулятор, среда gymnasium."""

from apex_lab.environment.track import Track
from apex_lab.environment.car import CarModel, CarParams, CarState
from apex_lab.environment.reward import DefaultReward, RewardStrategy, SparseReward
from apex_lab.environment.race_env import EnvConfig, RaceEnvironment

__all__ = [
    "Track",
    "CarModel",
    "CarParams",
    "CarState",
    "RewardStrategy",
    "DefaultReward",
    "SparseReward",
    "EnvConfig",
    "RaceEnvironment",
]
