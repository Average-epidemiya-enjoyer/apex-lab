"""Стратегии вычисления reward для RaceEnvironment (паттерн Strategy)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from apex_lab.environment.car import CarState
from apex_lab.environment.track import Track


class RewardStrategy(ABC):
    """Абстрактный интерфейс reward-функции.

    Позволяет подменять награду без изменения кода среды — достаточно
    передать другой объект-стратегию в ``RaceEnvironment``.
    """

    @abstractmethod
    def compute(
        self,
        prev_state: CarState,
        next_state: CarState,
        action: tuple[float, float],
        progress_delta: float,
        terminated: bool,
        track: Track,
        dt: float,
    ) -> float:
        """Вычислить скалярную награду за один шаг среды.

        Args:
            prev_state: Состояние до шага.
            next_state: Состояние после шага.
            action: ``(throttle, steer)`` применённые на этом шаге.
            progress_delta: Изменение прогресса по трассе (∈ [0, 1]).
            terminated: True, если эпизод завершился (авария / финиш).
            track: Объект трассы.
            dt: Шаг симуляции, с.

        Returns:
            Скалярная награда.
        """


@dataclass
class DefaultReward(RewardStrategy):
    """Reward по умолчанию: прогресс минус штрафы.

    reward = progress_scale · Δprogress
             - off_track_coeff · max(0, -dist)   # штраф за выезд
             - time_penalty · dt                 # поощряет скорость
             - crash_penalty   (только при terminated)

    Attributes:
        progress_scale: Масштаб награды за прогресс.
        off_track_coeff: Коэффициент штрафа за выезд с трассы (за метр).
        time_penalty: Штраф за время (за секунду симуляции).
        crash_penalty: Единовременный штраф при аварии.
        speed_reward_coeff: Дополнительная награда за скорость (необязательно).
    """

    progress_scale: float = 10.0
    off_track_coeff: float = 2.0
    time_penalty: float = 0.02
    crash_penalty: float = 10.0
    speed_reward_coeff: float = 0.0

    def compute(
        self,
        prev_state: CarState,
        next_state: CarState,
        action: tuple[float, float],
        progress_delta: float,
        terminated: bool,
        track: Track,
        dt: float,
    ) -> float:
        reward = self.progress_scale * progress_delta
        reward -= self.time_penalty * dt

        dist = track.signed_distance_to_boundary(next_state.x, next_state.y)
        if dist < 0.0:
            reward -= self.off_track_coeff * abs(dist)

        if terminated:
            reward -= self.crash_penalty

        if self.speed_reward_coeff > 0.0:
            reward += self.speed_reward_coeff * next_state.speed / track.width

        return float(reward)


@dataclass
class SparseReward(RewardStrategy):
    """Разреженная reward: только +1 за круг, -1 за аварию.

    Используется для отладки алгоритмов, способных работать с редкими
    сигналами вознаграждения.
    """

    lap_reward: float = 100.0
    crash_penalty: float = 50.0

    def compute(
        self,
        prev_state: CarState,
        next_state: CarState,
        action: tuple[float, float],
        progress_delta: float,
        terminated: bool,
        track: Track,
        dt: float,
    ) -> float:
        # Переход через финишную черту (progress оборачивается ~1→0)
        if progress_delta < -0.5:
            return float(self.lap_reward)
        if terminated:
            return -float(self.crash_penalty)
        return 0.0
