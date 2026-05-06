"""Параметры и физическая модель гоночного автомобиля (кинематическая bicycle model)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

# Ускорение свободного падения, м/с²
_G = 9.81


@dataclass(frozen=True)
class CarParams:
    """Физические константы автомобиля.

    Attributes:
        mass: Масса, кг.
        max_speed: Максимальная скорость, м/с.
        max_accel: Максимальное продольное ускорение (газ), м/с².
        max_decel: Максимальное торможение (положительное значение), м/с².
        max_steer_angle: Максимальный угол поворота колёс, рад.
        wheelbase: Колёсная база (расстояние между осями), м.
        drag_coeff: Безразмерный коэффициент аэродинамического сопротивления.
        tire_grip: Максимальное поперечное ускорение в единицах g.
    """

    mass: float = 700.0
    max_speed: float = 80.0
    max_accel: float = 15.0
    max_decel: float = 30.0
    max_steer_angle: float = 0.5
    wheelbase: float = 2.9
    drag_coeff: float = 0.35
    tire_grip: float = 2.5


@dataclass(frozen=True)
class CarState:
    """Иммутабельный снимок состояния автомобиля в момент времени.

    Attributes:
        x: Координата X в мировой СК, м.
        y: Координата Y в мировой СК, м.
        heading: Курс, рад (0 = восток, CCW положительный).
        speed: Продольная скорость, м/с (≥ 0).
        steer_angle: Текущий угол поворота руля, рад.
    """

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    steer_angle: float = 0.0

    def as_array(self) -> NDArray[np.float32]:
        """Вернуть состояние как вещественный вектор (5,)."""
        return np.array(
            [self.x, self.y, self.heading, self.speed, self.steer_angle],
            dtype=np.float32,
        )


class CarModel:
    """Кинематическая bicycle model с ограничением сцепления шин.

    Уравнения движения (упрощённая модель, CoG в середине колёсной базы)::

        dx/dt  = v · cos(θ)
        dy/dt  = v · sin(θ)
        dθ/dt  = v · tan(δ_eff) / L
        dv/dt  = a_long - drag · v²

    Ограничение сцепления: поперечное ускорение a_y = v² · |tan(δ)| / L
    не должно превышать tire_grip · g. При превышении эффективный угол
    руля δ_eff урезается до допустимого значения (эффект недостаточной
    поворачиваемости — «снос»).
    """

    def __init__(self, params: CarParams | None = None) -> None:
        self.params = params or CarParams()

    def step(
        self,
        state: CarState,
        action: NDArray[np.float32] | tuple[float, float],
        dt: float,
    ) -> CarState:
        """Выполнить шаг интегрирования методом Эйлера.

        Args:
            state: Текущее состояние (иммутабельный объект).
            action: ``(throttle, steer)`` — оба в диапазоне [-1, 1].
                    throttle > 0 — газ, throttle < 0 — тормоз.
            dt: Шаг по времени, с.

        Returns:
            Новый ``CarState`` после шага.
        """
        p = self.params
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steer_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Целевой угол руля
        delta_target = steer_cmd * p.max_steer_angle

        # Ограничение сцепления: урезаем δ при высокой скорости
        delta_eff = self._apply_grip_limit(state.speed, delta_target, p)

        # Продольное ускорение
        if throttle >= 0.0:
            accel = throttle * p.max_accel
        else:
            accel = throttle * p.max_decel  # торможение (a < 0)

        # Аэродинамическое сопротивление (квадратичное)
        drag = p.drag_coeff * state.speed ** 2 / p.mass

        # Интегрирование методом Эйлера
        v = state.speed
        theta = state.heading

        new_x = state.x + v * math.cos(theta) * dt
        new_y = state.y + v * math.sin(theta) * dt
        new_theta = theta + v * math.tan(delta_eff) / p.wheelbase * dt
        new_v = v + (accel - drag) * dt

        # Физические ограничения
        new_v = float(np.clip(new_v, 0.0, p.max_speed))
        new_theta = float((new_theta + math.pi) % (2.0 * math.pi) - math.pi)

        return replace(
            state,
            x=new_x,
            y=new_y,
            heading=new_theta,
            speed=new_v,
            steer_angle=delta_eff,
        )

    @staticmethod
    def _apply_grip_limit(speed: float, delta: float, p: CarParams) -> float:
        """Урезать угол поворота руля по ограничению сцепления шин.

        Боковое ускорение: a_y = v² · |tan(δ)| / L ≤ tire_grip · g
        Из этого следует: |δ_max| = arctan(tire_grip · g · L / v²)
        """
        if speed < 1.0:
            # При малой скорости сцепление не ограничивает
            return float(np.clip(delta, -p.max_steer_angle, p.max_steer_angle))

        max_tan = p.tire_grip * _G * p.wheelbase / (speed ** 2)
        delta_max = min(math.atan(max_tan), p.max_steer_angle)
        return float(np.clip(delta, -delta_max, delta_max))

    def lateral_acceleration(self, state: CarState) -> float:
        """Поперечное ускорение, м/с² (при текущих v и δ)."""
        if abs(state.steer_angle) < 1e-9 or state.speed < 1e-3:
            return 0.0
        return state.speed ** 2 * math.tan(state.steer_angle) / self.params.wheelbase
