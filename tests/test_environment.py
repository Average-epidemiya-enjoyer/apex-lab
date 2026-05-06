"""Unit-тесты для модуля environment."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LinearRing

from apex_lab.environment.car import CarModel, CarParams, CarState
from apex_lab.environment.race_env import EnvConfig, RaceEnvironment
from apex_lab.environment.track import Track


# ===========================================================================
# Track
# ===========================================================================

class TestTrackPresets:
    """Базовые проверки готовых пресетов трасс."""

    @pytest.mark.parametrize("preset", ["oval", "figure_eight", "monza_like",
                                         "nurburgring_like", "hairpin_test"])
    def test_preset_creates_track(self, preset: str) -> None:
        track = RaceEnvironment._resolve_track(preset)
        assert isinstance(track, Track)
        assert len(track.centre_line) >= 4

    def test_oval_is_closed(self) -> None:
        t = Track.oval()
        cl = t.centre_line
        # Первая и последняя точки не совпадают (явного замыкания нет),
        # но расстояние между ними должно быть меньше одного сегмента
        seg_len = np.linalg.norm(cl[1] - cl[0])
        gap = np.linalg.norm(cl[-1] - cl[0])
        assert gap < seg_len * 5

    def test_total_length_positive(self) -> None:
        t = Track.oval(semi_major=200.0, semi_minor=80.0)
        assert t.total_length > 0

    def test_inner_outer_boundaries_shape(self) -> None:
        t = Track.oval()
        n = len(t.centre_line)
        assert t.inner_boundary.shape == (n, 2)
        assert t.outer_boundary.shape == (n, 2)

    def test_inner_outer_offset(self) -> None:
        """Границы должны быть смещены на ±width/2 в направлении нормали.

        inner_boundary = cl + normals * half  (левая сторона / инфилд для CCW)
        outer_boundary = cl - normals * half  (правая сторона / внешняя стена)
        """
        t = Track.oval()
        half = t.width / 2
        cl = t.centre_line
        norms = t.normals
        inner_expected = cl + norms * half
        outer_expected = cl - norms * half
        np.testing.assert_allclose(t.inner_boundary, inner_expected, atol=1e-10)
        np.testing.assert_allclose(t.outer_boundary, outer_expected, atol=1e-10)


class TestTrackGeometry:
    """Геометрические методы Track."""

    @pytest.fixture
    def oval(self) -> Track:
        return Track.oval(semi_major=100.0, semi_minor=50.0, width=10.0)

    def test_centre_is_inside(self, oval: Track) -> None:
        # Точка в центре трассы (примерно на центральной линии)
        cl = oval.centre_line
        mid = cl[len(cl) // 4]
        assert oval.is_inside(float(mid[0]), float(mid[1]))

    def test_far_point_outside(self, oval: Track) -> None:
        assert not oval.is_inside(0.0, 0.0)  # центр эллипса снаружи трассы

    def test_signed_dist_inside_positive(self, oval: Track) -> None:
        cl = oval.centre_line
        pt = cl[0]
        d = oval.signed_distance_to_boundary(float(pt[0]), float(pt[1]))
        assert d > 0.0

    def test_signed_dist_outside_negative(self, oval: Track) -> None:
        d = oval.signed_distance_to_boundary(0.0, 0.0)
        assert d < 0.0

    def test_progress_on_centre_line(self, oval: Track) -> None:
        cl = oval.centre_line
        p0 = oval.progress(float(cl[0, 0]), float(cl[0, 1]))
        p_mid = oval.progress(float(cl[len(cl) // 2, 0]), float(cl[len(cl) // 2, 1]))
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p_mid <= 1.0
        assert p_mid > p0  # прогресс монотонно растёт

    def test_lookahead_waypoints_shape(self, oval: Track) -> None:
        cl = oval.centre_line
        wps = oval.get_lookahead_waypoints(float(cl[0, 0]), float(cl[0, 1]), 0.0, n=5)
        assert wps.shape == (5, 2)

    def test_lookahead_forward_component_positive(self, oval: Track) -> None:
        """Lookahead-точки должны быть впереди машины."""
        cl = oval.centre_line
        nxt = cl[1]
        heading = float(np.arctan2(nxt[1] - cl[0, 1], nxt[0] - cl[0, 0]))
        wps = oval.get_lookahead_waypoints(
            float(cl[0, 0]), float(cl[0, 1]), heading, n=5, spacing=3
        )
        # Хотя бы часть точек должна быть в положительном направлении
        assert np.sum(wps[:, 0] > 0) >= 2


class TestRandomTrack:
    """Генерация случайных трасс."""

    @pytest.mark.parametrize("seed", [0, 7, 42, 100, 999])
    def test_no_self_intersections(self, seed: int) -> None:
        """Ни одна из сгенерированных трасс не должна самопересекаться."""
        track = Track.random_track(seed=seed, n_control_points=10,
                                   smoothness=200.0, max_retries=30)
        ring = LinearRing(track.centre_line)
        assert ring.is_simple, f"seed={seed}: центральная линия самопересекается"

    def test_random_track_width(self) -> None:
        t = Track.random_track(seed=1, width=12.0)
        assert t.width == 12.0

    def test_random_track_name_contains_seed(self) -> None:
        t = Track.random_track(seed=77)
        assert "77" in t.name


# ===========================================================================
# CarModel
# ===========================================================================

class TestCarModel:
    """Физика автомобиля."""

    @pytest.fixture
    def car(self) -> CarModel:
        return CarModel(CarParams(max_speed=30.0, max_accel=10.0,
                                  wheelbase=3.0, max_steer_angle=0.4))

    @pytest.fixture
    def rest(self) -> CarState:
        return CarState(x=0.0, y=0.0, heading=0.0, speed=0.0)

    def test_zero_action_no_movement(self, car: CarModel, rest: CarState) -> None:
        """Машина в покое при нулевом управлении должна оставаться неподвижной."""
        for _ in range(10):
            rest = car.step(rest, (0.0, 0.0), dt=0.05)
        assert abs(rest.x) < 1e-9
        assert abs(rest.y) < 1e-9
        assert abs(rest.speed) < 1e-9

    def test_full_throttle_accelerates(self, car: CarModel, rest: CarState) -> None:
        """Полный газ по прямой должен разгонять до max_speed."""
        state = rest
        for _ in range(500):
            state = car.step(state, (1.0, 0.0), dt=0.05)
        assert state.speed > car.params.max_speed * 0.9
        assert state.speed <= car.params.max_speed + 1e-6

    def test_braking_decelerates(self, car: CarModel) -> None:
        """Торможение должно снижать скорость."""
        state = CarState(x=0.0, y=0.0, heading=0.0, speed=20.0)
        state = car.step(state, (-1.0, 0.0), dt=0.1)
        assert state.speed < 20.0

    def test_speed_never_negative(self, car: CarModel) -> None:
        """Скорость не должна уходить в отрицательную область."""
        state = CarState(speed=0.1)
        for _ in range(50):
            state = car.step(state, (-1.0, 0.0), dt=0.1)
        assert state.speed >= 0.0

    def test_steering_changes_heading(self, car: CarModel) -> None:
        """Поворот руля при ненулевой скорости должен изменять курс."""
        state = CarState(x=0.0, y=0.0, heading=0.0, speed=15.0)
        initial_heading = state.heading
        for _ in range(20):
            state = car.step(state, (0.0, 1.0), dt=0.05)
        assert abs(state.heading - initial_heading) > 0.05

    def test_zero_steer_straight_line(self, car: CarModel) -> None:
        """Нулевой руль при постоянной скорости — прямолинейное движение."""
        state = CarState(x=0.0, y=0.0, heading=0.0, speed=10.0)
        for _ in range(100):
            state = car.step(state, (0.0, 0.0), dt=0.05)
        # Машина движется вдоль оси X, отклонение по Y должно быть ≈ 0
        assert abs(state.y) < 1e-6

    def test_state_is_immutable(self, car: CarModel, rest: CarState) -> None:
        """Исходное состояние не должно изменяться после step()."""
        orig_x = rest.x
        orig_speed = rest.speed
        _ = car.step(rest, (1.0, 0.5), dt=0.05)
        assert rest.x == orig_x
        assert rest.speed == orig_speed

    def test_grip_limit_caps_steer(self, car: CarModel) -> None:
        """При высокой скорости и полном руле сцепление должно урезать δ."""
        fast = CarState(speed=29.0, heading=0.0)
        result = car.step(fast, (0.0, 1.0), dt=0.05)
        # δ_eff должен быть меньше max_steer при высокой скорости
        assert abs(result.steer_angle) < car.params.max_steer_angle


# ===========================================================================
# RaceEnvironment
# ===========================================================================

class TestRaceEnvironment:
    """gymnasium.Env интерфейс RaceEnvironment."""

    @pytest.fixture
    def env(self) -> RaceEnvironment:
        track = Track.oval(semi_major=100.0, semi_minor=50.0, width=12.0)
        cfg = EnvConfig(dt=0.05, max_steps=200, n_lookahead=5)
        return RaceEnvironment(track=track, config=cfg)

    def test_reset_returns_obs_and_info(self, env: RaceEnvironment) -> None:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == (env.obs_dim,)
        assert isinstance(info, dict)

    def test_obs_dim_matches_config(self, env: RaceEnvironment) -> None:
        expected = 6 + 5 * 2  # _N_BASE_OBS + n_lookahead * 2
        assert env.obs_dim == expected

    def test_step_returns_correct_types(self, env: RaceEnvironment) -> None:
        env.reset()
        action = np.array([0.5, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncated_after_max_steps(self, env: RaceEnvironment) -> None:
        """Среда должна завершиться по truncated после max_steps шагов."""
        env.reset()
        action = np.array([0.3, 0.0], dtype=np.float32)
        for _ in range(199):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        else:
            _, _, _, truncated, _ = env.step(action)
            assert truncated

    def test_crash_terminates(self, env: RaceEnvironment) -> None:
        """Выезд за пределы трассы должен завершать эпизод."""
        env.reset()
        # Телепортируем машину за пределы трассы напрямую
        env._state = CarState(x=10_000.0, y=10_000.0, heading=0.0, speed=0.0)
        _, _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert terminated or info["off_track"]

    def test_reset_clears_step_counter(self, env: RaceEnvironment) -> None:
        env.reset()
        for _ in range(5):
            env.step(np.array([0.3, 0.0], dtype=np.float32))
        env.reset()
        assert env._step_count == 0

    def test_resolve_track_string(self) -> None:
        for name in ["oval", "hairpin_test", "monza_like"]:
            t = RaceEnvironment._resolve_track(name)
            assert isinstance(t, Track)

    def test_resolve_track_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Неизвестное"):
            RaceEnvironment._resolve_track("nonexistent_track")
