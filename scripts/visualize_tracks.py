"""Генерация PNG-превью всех пресетов трасс + случайных трасс.

Использование:
    python scripts/visualize_tracks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # рендеринг без дисплея

import matplotlib.pyplot as plt
import numpy as np

# Позволяет запускать скрипт из корня проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from apex_lab.environment.track import Track

ASSETS = Path(__file__).parent.parent / "experiments" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

PRESETS: list[tuple[str, Track]] = [
    ("oval",            Track.oval()),
    ("figure_eight",    Track.figure_eight()),
    ("monza_like",      Track.monza_like()),
    ("nurburgring_like",Track.nurburgring_like()),
    ("hairpin_test",    Track.hairpin_test()),
    ("random (seed=42)",Track.random_track(seed=42)),
]


def save_grid() -> None:
    """Сохранить все пресеты на одном листе 3×2."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#0d1117")
    fig.suptitle("apex-lab — пресеты трасс", color="white", fontsize=16, y=0.98)

    for ax, (name, track) in zip(axes.flat, PRESETS):
        track.plot(ax=ax, title=name)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS / "track_presets.png"
    plt.savefig(out, facecolor="#0d1117", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out}")


def save_individual() -> None:
    """Сохранить каждый пресет отдельным файлом."""
    for name, track in PRESETS:
        slug = name.split(" ")[0]
        out = ASSETS / f"track_{slug}.png"
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
        track.plot(ax=ax, title=name, save_path=out)
        plt.close(fig)
        print(f"Сохранено: {out}")


def save_random_variety() -> None:
    """4 случайных трассы с разными seed для демонстрации разнообразия."""
    seeds = [1, 13, 99, 256]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0d1117")
    fig.suptitle("Случайные трассы (random_track)", color="white", fontsize=14)

    for ax, seed in zip(axes.flat, seeds):
        t = Track.random_track(seed=seed)
        t.plot(ax=ax, title=f"seed={seed}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS / "random_tracks.png"
    plt.savefig(out, facecolor="#0d1117", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    print("Генерация визуализаций трасс...")
    save_grid()
    save_individual()
    save_random_variety()
    print("Готово. Файлы сохранены в experiments/assets/")
