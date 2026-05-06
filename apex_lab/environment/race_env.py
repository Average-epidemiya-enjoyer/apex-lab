"""RaceEnvironment — gymnasium.Env обёртка над симулятором трассы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.typing import NDArray

from apex_lab.environment.car import CarModel, CarParams, CarState
from apex_lab.environment.reward import DefaultReward, RewardStrategy
from apex_lab.environment.track import Track


@dataclass
class EnvConfig:
    """Конфигурация среды RaceEnvironment.

    Attributes:
        dt: Шаг симуляции, с.
        max_steps: Максимум шагов в эпизоде (лимит по времени).
        n_lookahead: Число lookahead-точек трассы в observation.
        lookahead_spacing: Шаг индекса при выборке lookahead-точек.
        obs_normalize: Нормализовать observation в диапазон [-1, 1].
        crash_on_boundary: True → эпизод завершается при выезде с трассы.
    """

    dt: float = 0.05
    max_steps: int = 5_000
    n_lookahead: int = 10
    lookahead_spacing: int = 5
    obs_normalize: bool = True
    crash_on_boundary: bool = True


# Размерность observation: 6 базовых + n_lookahead * 2
_N_BASE_OBS = 6


class RaceEnvironment(gym.Env):
    """Среда для задачи поиска оптимальной траектории на гоночной трассе.

    Observation space (всё нормализовано в [-1, 1] при obs_normalize=True)::

        [speed/max_speed,                    # 0
         steer/max_steer,                    # 1
         sin(heading), cos(heading),          # 2-3
         progress,                           # 4
         lateral_offset / (width/2),         # 5
         wp_0_fwd, wp_0_left,                # 6-7   lookahead ->
         ...                                 #       | в СК машины
         wp_{N-1}_fwd, wp_{N-1}_left]        #       /

    Action space: Box(-1, 1, shape=(2,))
        action[0] = throttle  (−1 = полный тормоз, +1 = полный газ)
        action[1] = steer     (−1 = максимум влево, +1 = максимум вправо)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        track: Track | str = "oval",
        car_params: CarParams | None = None,
        reward_strategy: RewardStrategy | None = None,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._track = self._resolve_track(track)
        self._car = CarModel(car_params or CarParams())
        self._reward_fn = reward_strategy or DefaultReward()
        self._cfg = config or EnvConfig()
        self.render_mode = render_mode

        obs_dim = _N_BASE_OBS + self._cfg.n_lookahead * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._state: CarState = self._make_initial_state()
        self._step_count: int = 0
        self._prev_progress: float = 0.0
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------
    # gymnasium.Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Сбросить среду в начало эпизода.

        Returns:
            ``(observation, info)``
        """
        super().reset(seed=seed)
        self._state = self._make_initial_state()
        self._step_count = 0
        self._prev_progress = self._track.progress(self._state.x, self._state.y)
        self._total_reward = 0.0
        return self._get_obs(), {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Выполнить один шаг среды.

        Args:
            action: Массив ``[throttle, steer]`` ∈ [-1, 1].

        Returns:
            ``(observation, reward, terminated, truncated, info)``
        """
        prev_state = self._state
        next_state = self._car.step(self._state, action, self._cfg.dt)
        self._state = next_state
        self._step_count += 1

        curr_progress = self._track.progress(next_state.x, next_state.y)
        progress_delta = curr_progress - self._prev_progress

        # Корректировка при пересечении финишной черты (0 → 1 → 0)
        if progress_delta < -0.5:
            progress_delta += 1.0
        self._prev_progress = curr_progress

        # Условия завершения
        off_track = not self._track.is_inside(next_state.x, next_state.y)
        terminated = off_track and self._cfg.crash_on_boundary
        truncated = self._step_count >= self._cfg.max_steps

        reward = self._reward_fn.compute(
            prev_state=prev_state,
            next_state=next_state,
            action=(float(action[0]), float(action[1])),
            progress_delta=progress_delta,
            terminated=terminated,
            track=self._track,
            dt=self._cfg.dt,
        )
        self._total_reward += reward

        info: dict[str, Any] = {
            "progress": curr_progress,
            "speed": next_state.speed,
            "lateral_offset": self._track.lateral_offset(next_state.x, next_state.y),
            "off_track": off_track,
            "step": self._step_count,
            "total_reward": self._total_reward,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        """Отрисовать текущее состояние среды.

        В режиме ``"human"`` открывает/обновляет matplotlib-окно.
        В режиме ``"rgb_array"`` возвращает массив пикселей (H, W, 3).
        """
        if self.render_mode is None:
            return None

        import matplotlib
        import matplotlib.pyplot as plt

        fig = self._get_or_create_figure()
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
        ax.cla()

        # Рисуем трассу и положение машины
        self._track.plot(ax=ax, title=f"{self._track.name} | step={self._step_count}")
        ax.scatter(
            self._state.x, self._state.y,
            color="yellow", s=120, zorder=10, label="Машина"
        )
        # Вектор курса
        import math
        dx = 8.0 * math.cos(self._state.heading)
        dy = 8.0 * math.sin(self._state.heading)
        ax.annotate(
            "", xy=(self._state.x + dx, self._state.y + dy),
            xytext=(self._state.x, self._state.y),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
            zorder=11,
        )

        fig.canvas.draw()

        if self.render_mode == "rgb_array":
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

        plt.pause(0.001)
        return None

    def close(self) -> None:
        """Закрыть окно отрисовки."""
        import matplotlib.pyplot as plt
        if hasattr(self, "_fig") and self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _make_initial_state(self) -> CarState:
        """Поставить машину на стартовую точку трассы."""
        start = self._track.centre_line[0]
        # Начальный курс — по направлению от первой точки ко второй
        nxt = self._track.centre_line[1]
        heading = float(np.arctan2(nxt[1] - start[1], nxt[0] - start[0]))
        return CarState(x=float(start[0]), y=float(start[1]), heading=heading)

    def _get_obs(self) -> NDArray[np.float32]:
        """Составить и (опционально) нормализовать вектор observation."""
        p = self._car.params
        s = self._state
        t = self._track
        cfg = self._cfg

        progress = t.progress(s.x, s.y)
        lat_off = t.lateral_offset(s.x, s.y)
        waypoints = t.get_lookahead_waypoints(
            s.x, s.y, s.heading, n=cfg.n_lookahead, spacing=cfg.lookahead_spacing
        )

        base = np.array([
            s.speed / p.max_speed,
            s.steer_angle / p.max_steer_angle,
            np.sin(s.heading),
            np.cos(s.heading),
            progress,
            lat_off / (t.width / 2.0),
        ], dtype=np.float32)

        # Нормировка waypoints по масштабу трассы
        wp_scale = max(t.total_length / 20.0, 1.0)
        wps_flat = (waypoints / wp_scale).astype(np.float32).flatten()

        return np.concatenate([base, wps_flat])

    def _get_or_create_figure(self) -> Any:
        """Получить или создать matplotlib figure для render."""
        import matplotlib.pyplot as plt
        if not hasattr(self, "_fig") or self._fig is None:
            self._fig = plt.figure(figsize=(8, 6), facecolor="#0d1117")
            self._fig.add_subplot(111)
        return self._fig

    @staticmethod
    def _resolve_track(track: Track | str) -> Track:
        """Разрешить имя трассы в объект Track."""
        if isinstance(track, Track):
            return track
        presets = {
            "oval": Track.oval,
            "figure_eight": Track.figure_eight,
            "monza_like": Track.monza_like,
            "nurburgring_like": Track.nurburgring_like,
            "hairpin_test": Track.hairpin_test,
        }
        if track not in presets:
            raise ValueError(
                f"Неизвестное название трассы {track!r}. "
                f"Доступные: {sorted(presets)}"
            )
        return presets[track]()

    # ------------------------------------------------------------------
    # Свойства для внешнего доступа
    # ------------------------------------------------------------------

    @property
    def track(self) -> Track:
        """Текущая трасса."""
        return self._track

    @property
    def state(self) -> CarState:
        """Текущее состояние автомобиля."""
        return self._state

    @property
    def obs_dim(self) -> int:
        """Размерность вектора наблюдений."""
        return int(self.observation_space.shape[0])
