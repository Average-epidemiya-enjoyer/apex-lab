"""Геометрия гоночной трассы: хранение, генерация и аналитика."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep
from shapely.geometry import LinearRing, LineString, Point, Polygon


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------

def _resample_equal_spacing(
    points: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """Пересемплировать кривую в n точек с равномерным шагом по длине дуги."""
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    target = np.linspace(0.0, total, n)
    xi = np.interp(target, cum, points[:, 0])
    yi = np.interp(target, cum, points[:, 1])
    return np.column_stack([xi, yi])


def _compute_normals(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Нормали к замкнутой кривой (90° против часовой стрелки от касательной)."""
    n = len(pts)
    tangents = np.zeros_like(pts)
    for i in range(n):
        delta = pts[(i + 1) % n] - pts[(i - 1) % n]
        norm = float(np.linalg.norm(delta))
        tangents[i] = delta / norm if norm > 1e-9 else np.array([1.0, 0.0])
    # (tx, ty) → (-ty, tx): поворот на 90° CCW
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _spline_closed(
    control_pts: NDArray[np.float64],
    n_samples: int,
    smoothness: float = 0.0,
) -> NDArray[np.float64]:
    """B-сплайн через control_pts с замыканием (per=True)."""
    cx = np.append(control_pts[:, 0], control_pts[0, 0])
    cy = np.append(control_pts[:, 1], control_pts[0, 1])
    tck, _ = splprep([cx, cy], s=smoothness, per=True, k=3)
    t = np.linspace(0, 1, n_samples + 1)[:-1]
    xl, yl = splev(t, tck)
    return np.column_stack([xl, yl])


# ---------------------------------------------------------------------------
# Класс Track
# ---------------------------------------------------------------------------

class Track:
    """Гоночная трасса.

    Хранит дискретизированную центральную линию, ширину и геометрию границ.
    Поддерживает загрузку из YAML/JSON и процедурную генерацию.

    Система координат: мировая 2D (x — восток, y — север, углы CCW).
    Единицы: метры.
    """

    def __init__(
        self,
        centre_line: NDArray[np.float64],
        width: float = 10.0,
        name: str = "unnamed",
        *,
        _skip_simple_check: bool = False,
    ) -> None:
        """
        Args:
            centre_line: Массив (N, 2) с координатами центральной линии.
                         Первая точка ≠ последней (замыкание подразумевается).
            width: Ширина трассы в метрах.
            name: Название трассы.
        """
        if centre_line.ndim != 2 or centre_line.shape[1] != 2:
            raise ValueError("centre_line должен быть массивом формы (N, 2)")
        if len(centre_line) < 4:
            raise ValueError("centre_line должен содержать не менее 4 точек")

        self.name = name
        self.width = float(width)
        self._cl: NDArray[np.float64] = centre_line.astype(np.float64)

        # Lazy-кэш для тяжёлых вычислений
        self._normals: NDArray[np.float64] | None = None
        self._cum: NDArray[np.float64] | None = None
        self._poly: Polygon | None = None
        self._inner: NDArray[np.float64] | None = None
        self._outer: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def centre_line(self) -> NDArray[np.float64]:
        """Центральная линия (N, 2), метры."""
        return self._cl

    @property
    def normals(self) -> NDArray[np.float64]:
        """Нормали в каждой точке центральной линии (CCW = «налево»)."""
        if self._normals is None:
            self._normals = _compute_normals(self._cl)
        return self._normals

    @property
    def _cum_lengths(self) -> NDArray[np.float64]:
        """Накопленные длины дуг, включая замыкающий сегмент."""
        if self._cum is None:
            closed = np.vstack([self._cl, self._cl[:1]])
            segs = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            self._cum = np.concatenate([[0.0], np.cumsum(segs)])
        return self._cum

    @property
    def total_length(self) -> float:
        """Полная длина трассы (длина замкнутого контура), метры."""
        return float(self._cum_lengths[-1])

    @property
    def inner_boundary(self) -> NDArray[np.float64]:
        """Внутренняя граница (сторона инфилда, левее по ходу движения).

        Для CCW-трассы: сдвиг по нормали (которая указывает влево/к центру).
        """
        if self._inner is None:
            self._inner = self._cl + self.normals * self.width / 2
        return self._inner

    @property
    def outer_boundary(self) -> NDArray[np.float64]:
        """Внешняя граница (стена снаружи, правее по ходу движения)."""
        if self._outer is None:
            self._outer = self._cl - self.normals * self.width / 2
        return self._outer

    @property
    def track_polygon(self) -> Polygon:
        """Shapely Polygon, покрывающий ездовую полосу трассы.

        Использует Polygon(exterior_ring, [interior_hole]) для корректного
        представления кольцеобразной области трассы.
        Exterior = outer_boundary (дальняя от инфилда граница).
        Hole     = inner_boundary (ближняя к инфилду граница).
        """
        if self._poly is None:
            ext = np.vstack([self.outer_boundary, self.outer_boundary[:1]])
            hole = np.vstack([self.inner_boundary, self.inner_boundary[:1]])
            poly = Polygon(ext, [hole])
            if not poly.is_valid:
                # Исправляем самопересечения через нулевой буфер
                poly = poly.buffer(0)
            if not poly.is_valid:
                closed_line = np.vstack([self._cl, self._cl[:1]])
                poly = LineString(closed_line).buffer(self.width / 2)
            self._poly = poly
        return self._poly

    # ------------------------------------------------------------------
    # Геометрические методы
    # ------------------------------------------------------------------

    def nearest_centre_index(self, x: float, y: float) -> int:
        """Индекс ближайшей точки центральной линии к (x, y)."""
        dists = np.linalg.norm(self._cl - np.array([x, y]), axis=1)
        return int(np.argmin(dists))

    def progress(self, x: float, y: float) -> float:
        """Прогресс вдоль трассы в диапазоне [0, 1].

        0.0 — стартовая точка, 1.0 — полный круг.
        """
        idx = self.nearest_centre_index(x, y)
        return float(self._cum_lengths[idx] / self.total_length)

    def signed_distance_to_boundary(self, x: float, y: float) -> float:
        """Знаковое расстояние до ближайшей границы трассы, метры.

        Положительное — точка внутри трассы.
        Отрицательное — точка за пределами.
        """
        pt = Point(x, y)
        dist = float(self.track_polygon.boundary.distance(pt))
        # covers() включает граничные точки (contains() — нет)
        return dist if self.track_polygon.covers(pt) else -dist

    def is_inside(self, x: float, y: float) -> bool:
        """True, если (x, y) находится в пределах ездовой полосы."""
        return bool(self.track_polygon.contains(Point(x, y)))

    def lateral_offset(self, x: float, y: float) -> float:
        """Боковое отклонение от центральной линии, метры.

        Положительное — точка левее центра (со стороны внешней границы).
        """
        idx = self.nearest_centre_index(x, y)
        rel = np.array([x, y]) - self._cl[idx]
        return float(np.dot(rel, self.normals[idx]))

    def get_lookahead_waypoints(
        self,
        x: float,
        y: float,
        heading: float,
        n: int = 10,
        spacing: int = 5,
    ) -> NDArray[np.float64]:
        """Возвращает n путевых точек впереди в системе координат машины.

        Система координат машины: ось X — вперёд (по курсу), Y — налево.

        Args:
            x, y: Текущее положение машины в метрах.
            heading: Курс машины в радианах.
            n: Количество lookahead-точек.
            spacing: Шаг через этот индекс в массиве centre_line.

        Returns:
            Массив (n, 2): каждая строка — (вперёд, влево) в метрах.
        """
        idx = self.nearest_centre_index(x, y)
        n_cl = len(self._cl)
        fwd = np.array([np.cos(heading), np.sin(heading)])
        left = np.array([-np.sin(heading), np.cos(heading)])
        car_pos = np.array([x, y])

        result = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            wp = self._cl[(idx + (i + 1) * spacing) % n_cl]
            rel = wp - car_pos
            result[i, 0] = float(np.dot(rel, fwd))
            result[i, 1] = float(np.dot(rel, left))
        return result

    # ------------------------------------------------------------------
    # Загрузка из файла
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Track":
        """Загрузить трассу из YAML-файла."""
        import yaml
        with Path(path).open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Track":
        """Загрузить трассу из JSON-файла."""
        with Path(path).open() as fh:
            data = json.load(fh)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(
            centre_line=np.asarray(data["centre_line"], dtype=float),
            width=float(data.get("width", 10.0)),
            name=str(data.get("name", "loaded")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в JSON-совместимый словарь."""
        return {"name": self.name, "width": self.width, "centre_line": self._cl.tolist()}

    def save_yaml(self, path: str | Path) -> None:
        """Сохранить трассу в YAML-файл."""
        import yaml
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            yaml.dump(self.to_dict(), fh)

    def save_json(self, path: str | Path) -> None:
        """Сохранить трассу в JSON-файл."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            json.dump(self.to_dict(), fh)

    # ------------------------------------------------------------------
    # Процедурная генерация
    # ------------------------------------------------------------------

    @classmethod
    def random_track(
        cls,
        seed: int = 42,
        n_control_points: int = 12,
        smoothness: float = 200.0,
        width: float = 10.0,
        radius: float = 80.0,
        radius_var: float = 30.0,
        n_samples: int = 500,
        max_retries: int = 25,
    ) -> "Track":
        """Случайная трасса на основе B-сплайна по точкам на окружности.

        Генерирует контрольные точки в полярных координатах, строит
        сглаженный B-сплайн (scipy.splprep, per=True), проверяет
        отсутствие самопересечений через shapely.LinearRing.is_simple.

        Args:
            seed: Seed для воспроизводимости.
            n_control_points: Число случайных контрольных точек.
            smoothness: Параметр сглаживания splprep (0 = точная интерполяция).
            width: Ширина трассы, м.
            radius: Средний радиус окружности контрольных точек, м.
            radius_var: Амплитуда случайного отклонения радиуса, м.
            n_samples: Число точек дискретизации итоговой центральной линии.
            max_retries: Максимум попыток генерации.

        Raises:
            RuntimeError: Если не удалось сгенерировать трассу без
                          самопересечений за max_retries попыток.
        """
        rng = np.random.default_rng(seed)
        for attempt in range(max_retries):
            # Равномерно расставленные базовые углы + небольшое случайное возмущение.
            # Это гарантирует отсутствие кластеризации без дополнительной фильтрации.
            base = np.linspace(0.0, 2.0 * np.pi, n_control_points + 1)[:-1]
            max_jitter = np.pi / n_control_points * 0.55
            angles = base + rng.uniform(-max_jitter, max_jitter, n_control_points)
            radii = radius + rng.uniform(-radius_var, radius_var, n_control_points)
            radii = np.clip(radii, radius * 0.25, radius * 1.75)
            ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            try:
                cl = _spline_closed(ctrl, n_samples, smoothness=smoothness)
            except Exception:
                continue
            if LinearRing(cl).is_simple:
                return cls(cl, width=width, name=f"random_s{seed}")
        raise RuntimeError(
            f"Не удалось сгенерировать трассу без самопересечений "
            f"за {max_retries} попыток (seed={seed}). "
            "Попробуйте уменьшить radius_var или увеличить n_control_points."
        )

    # ------------------------------------------------------------------
    # Готовые пресеты
    # ------------------------------------------------------------------

    @classmethod
    def oval(
        cls,
        semi_major: float = 200.0,
        semi_minor: float = 80.0,
        width: float = 12.0,
        n_samples: int = 500,
    ) -> "Track":
        """Овальная трасса (эллипс).

        Args:
            semi_major: Большая полуось, м.
            semi_minor: Малая полуось, м.
            width: Ширина трассы, м.
            n_samples: Число точек центральной линии.
        """
        t = np.linspace(0.0, 2.0 * np.pi, n_samples + 1)[:-1]
        cl = np.column_stack([semi_major * np.cos(t), semi_minor * np.sin(t)])
        return cls(cl, width=width, name="oval")

    @classmethod
    def figure_eight(
        cls,
        radius: float = 80.0,
        width: float = 10.0,
        n_samples: int = 500,
    ) -> "Track":
        """Трасса в форме цифры 8 (лемниската).

        Содержит самопересечение в центре — для тестирования алгоритмов
        на нестандартных топологиях трасс.
        """
        t = np.linspace(0.0, 2.0 * np.pi, n_samples + 1)[:-1]
        # Лемниската Бернулли в параметрической форме
        x = radius * np.sin(t)
        y = radius * np.sin(t) * np.cos(t)
        cl = np.column_stack([x, y])
        return cls(cl, width=width, name="figure_eight", _skip_simple_check=True)

    @classmethod
    def monza_like(
        cls,
        scale: float = 1.0,
        width: float = 12.0,
        n_samples: int = 600,
    ) -> "Track":
        """Высокоскоростная трасса с двумя шиканами в стиле Монцы."""
        # Нормированные контрольные точки, scale=1 ≈ ~5 км
        ctrl = np.array([
            [0.0,  0.0],
            [1.5,  0.1],
            [3.0,  0.0],
            [4.2,  0.3],
            [4.5,  1.0],  # первая шикана
            [4.2,  1.7],
            [3.3,  1.9],
            [2.7,  2.5],  # вторая шикана
            [2.0,  2.7],
            [0.8,  2.6],
            [-0.3, 2.2],
            [-0.6, 1.3],
            [-0.4, 0.4],
        ]) * 55.0 * scale
        return cls(_spline_closed(ctrl, n_samples), width=width, name="monza_like")

    @classmethod
    def nurburgring_like(
        cls,
        scale: float = 1.0,
        width: float = 10.0,
        n_samples: int = 800,
    ) -> "Track":
        """Технически сложная трасса с множеством поворотов в стиле Нюрбургринга."""
        ctrl = np.array([
            [ 0.0,  0.0],
            [ 1.2,  0.2],
            [ 2.2,  0.1],
            [ 3.0,  0.7],
            [ 3.2,  1.6],
            [ 2.8,  2.3],
            [ 2.1,  2.7],
            [ 1.5,  2.5],
            [ 1.1,  3.0],
            [ 0.5,  3.6],
            [-0.1,  3.9],
            [-0.7,  3.7],
            [-1.1,  3.1],
            [-0.9,  2.4],
            [-1.3,  1.8],
            [-1.6,  1.1],
            [-1.4,  0.5],
            [-0.7,  0.1],
        ]) * 42.0 * scale
        return cls(_spline_closed(ctrl, n_samples), width=width, name="nurburgring_like")

    @classmethod
    def hairpin_test(
        cls,
        straight_length: float = 150.0,
        hairpin_radius: float = 12.0,
        width: float = 10.0,
        n_samples: int = 400,
    ) -> "Track":
        """Простая трасса с одной шпилькой для тестирования острых поворотов."""
        r = hairpin_radius
        L = straight_length
        ctrl = np.array([
            [0.0,        0.0],
            [0.0,        L * 0.33],
            [0.0,        L * 0.67],
            [0.0,        L],           # вход в шпильку
            [r,          L + r * 1.1], # апекс шпильки
            [2 * r,      L],           # выход из шпильки
            [2 * r,      L * 0.67],
            [2 * r,      L * 0.33],
            [2 * r,      0.0],         # вход в нижний разворот
            [r,          -r * 1.1],    # апекс нижнего разворота
        ])
        return cls(_spline_closed(ctrl, n_samples), width=width, name="hairpin_test")

    # ------------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: Any | None = None,
        trajectory: NDArray[np.float64] | None = None,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> Any:
        """Нарисовать трассу с помощью matplotlib.

        Args:
            ax: Оси matplotlib (создаются автоматически, если None).
            trajectory: Траектория (N, 2) для отображения поверх трассы.
            title: Заголовок; по умолчанию — имя трассы.
            save_path: Путь для сохранения PNG.

        Returns:
            Объект осей matplotlib.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")

        outer = np.vstack([self.outer_boundary, self.outer_boundary[:1]])
        inner = np.vstack([self.inner_boundary, self.inner_boundary[:1]])

        ax.fill(outer[:, 0], outer[:, 1], color="#2d2d2d", zorder=0)
        ax.fill(inner[:, 0], inner[:, 1], color="#1a6b1a", zorder=1)
        ax.plot(outer[:, 0], outer[:, 1], color="#888", linewidth=1.0, zorder=2)
        ax.plot(inner[:, 0], inner[:, 1], color="#888", linewidth=1.0, zorder=2)

        cl = np.vstack([self._cl, self._cl[:1]])
        ax.plot(cl[:, 0], cl[:, 1], "--", color="white", linewidth=0.8,
                alpha=0.5, zorder=3, label="Центр")
        ax.scatter(self._cl[0, 0], self._cl[0, 1], color="red",
                   s=80, zorder=5, label="Старт")

        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                    color="cyan", linewidth=2.0, zorder=4, label="Траектория")

        ax.set_aspect("equal")
        ax.set_facecolor("#0d1117")
        ax.set_title(title or self.name, color="white", fontsize=11)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(loc="upper right", facecolor="#222", labelcolor="white",
                  fontsize=8, framealpha=0.8)

        info = f"L={self.total_length:.0f} м  w={self.width:.0f} м  N={len(self._cl)}"
        ax.text(0.02, 0.02, info, transform=ax.transAxes,
                color="#aaa", fontsize=7, va="bottom")

        if save_path is not None:
            sp = Path(save_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(sp, bbox_inches="tight", facecolor="#0d1117", dpi=150)

        return ax

    def __repr__(self) -> str:
        return (
            f"Track(name={self.name!r}, width={self.width}m, "
            f"length={self.total_length:.1f}m, n={len(self._cl)})"
        )
