"""Generate tracks_overview.png into experiments/assets/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apex_lab.environment.track import Track

presets = [
    ("oval",             Track.oval()),
    ("monza_like",       Track.monza_like()),
    ("nurburgring_like", Track.nurburgring_like()),
    ("hairpin_test",     Track.hairpin_test()),
    ("random_s42",       Track.random_track(seed=42)),
    ("random_s7",        Track.random_track(seed=7)),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="#0d1117")
fig.suptitle("apex-lab  |  Track Presets", color="white", fontsize=15, fontweight="bold", y=1.01)

for ax, (label, track) in zip(axes.flat, presets):
    track.plot(ax=ax, title=f"{label}\nL={track.total_length:.0f} m  w={track.width:.0f} m")

plt.tight_layout()
out = Path("experiments/assets/tracks_overview.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="#0d1117", dpi=150)
print(f"Saved: {out}")
