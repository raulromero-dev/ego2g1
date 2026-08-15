"""Step width restricted to genuine locomotion, applied identically to both datasets.

LAFAN1's ``walk*`` files are not 6 minutes of walking. They contain standing, shuffling,
side-steps, backing up and turning in place, and every one of those inflates step width --
a side-step is nearly *all* step width. That is very likely why the actors read 19.3 cm when
the biomechanics literature puts normal walking at 8-13 cm.

So filter both sides to frames where the subject is actually travelling forward, walking or
turning, and nothing else. The filter is deliberately the same function for my SMPL-X and for
their BVH, because a filter applied to only one side is just a thumb on the scale.

A frame counts as forward locomotion when:
  * the pelvis is moving at a walking pace, and
  * the pelvis is *facing* roughly where it is going (excludes side-steps and walking backwards).

Facing comes from the hip-to-hip vector rotated into the horizontal plane, which both skeletons
provide; nothing here depends on a skeleton-specific joint definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ego2g1.eval import bvh                      # noqa: E402
from ego2g1.eval.gait import step_width_series, _runs   # noqa: E402

RAW = Path("/private/tmp/claude-501/-Users-raulromero/"
           "15ae2624-107f-467a-a2c4-cd44d1c68c78/scratchpad/lafan1_raw")

MIN_SPEED = 0.6          # m/s -- below this is shuffling or standing
MAX_OFF_HEADING = 40.0   # deg between facing and travel; excludes side-steps / backing up
MIN_RUN_S = 1.5          # ignore locomotion bursts too short to contain a step


def locomotion_mask(pelvis: np.ndarray, l_hip: np.ndarray, r_hip: np.ndarray,
                    fps: float) -> np.ndarray:
    """Per-frame: is this person walking forward (possibly turning)?"""
    win = max(3, int(round(0.25 * fps)))
    k = np.ones(win) / win

    vel = np.gradient(pelvis[:, :2], axis=0) * fps
    vel = np.stack([np.convolve(vel[:, i], k, mode="same") for i in range(2)], axis=1)
    speed = np.linalg.norm(vel, axis=1)

    across = r_hip[:, :2] - l_hip[:, :2]                 # points from left hip to right hip
    facing = np.stack([across[:, 1], -across[:, 0]], axis=1)   # rotate -90 deg -> forward
    fn = np.linalg.norm(facing, axis=1, keepdims=True)
    vn = np.linalg.norm(vel, axis=1, keepdims=True)
    cos = np.sum(facing / np.maximum(fn, 1e-9) * vel / np.maximum(vn, 1e-9), axis=1)

    # sign convention differs between skeletons; take whichever makes walking "forward"
    if np.median(cos[speed > MIN_SPEED]) < 0:
        cos = -cos
    return (speed > MIN_SPEED) & (cos > np.cos(np.radians(MAX_OFF_HEADING)))


def spans(mask: np.ndarray, fps: float) -> list[tuple[int, int]]:
    return [(a, b) for a, b in _runs(mask, fps) if (b - a) / fps >= MIN_RUN_S]


def main() -> int:
    root = Path.home() / "ego2g1"
    out: dict[str, dict] = {}

    # ---- actors: LAFAN1 BVH ---------------------------------------------------
    act, act_kept, act_total = [], 0, 0
    for f in sorted(RAW.glob("walk*.bvh")):
        j, m, ft = bvh.parse(f)
        nm = bvh.names(j)
        P = bvh.forward_kinematics(j, m)
        fps = 1 / ft
        LA, RA = nm.index("LeftFoot"), nm.index("RightFoot")
        LU, RU = nm.index("LeftUpLeg"), nm.index("RightUpLeg")
        mask = locomotion_mask(P[:, 0], P[:, LU], P[:, RU], fps)
        act_total += len(P)
        for a, b in spans(mask, fps):
            act_kept += b - a
            w, _ = step_width_series(P[a:b], fps, LA, RA)
            act.extend(w)

    # ---- me: SMPL-X -----------------------------------------------------------
    mine, my_kept, my_total = [], 0, 0
    for p in sorted((root / "data/20_human").glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        J = d["joints_pos_m"].astype(float)
        fps = float(d["fps"])
        if len(J) < 90:
            continue
        mask = locomotion_mask(J[:, 0], J[:, 1], J[:, 2], fps)
        my_total += len(J)
        for a, b in spans(mask, fps):
            my_kept += b - a
            w, _ = step_width_series(J[a:b], fps, 7, 8)
            mine.extend(w)

    mine, act = np.asarray(mine), np.asarray(act)
    print(f"  frames kept   me {my_kept}/{my_total} ({100*my_kept/my_total:.0f}%)   "
          f"actors {act_kept}/{act_total} ({100*act_kept/act_total:.0f}%)")
    print(f"\n  {'':8s} {'steps':>6s} {'median':>8s} {'IQR':>14s}")
    for name, x in (("me", mine), ("actors", act)):
        print(f"  {name:8s} {len(x):6d} {100*np.median(x):7.1f} cm "
              f"{100*np.percentile(x,25):5.1f}-{100*np.percentile(x,75):<5.1f}")
    delta = 100 * (np.median(mine) - np.median(act)) / np.median(act)
    print(f"\n  mine is {delta:+.0f}% vs actors (locomotion only)")
    print(f"  unfiltered was: me 7.1 cm, actors 19.3 cm  (-63%)")

    out = {"mine_cm": [float(100 * x) for x in mine],
           "actors_cm": [float(100 * x) for x in act],
           "summary": {"mine_median_cm": float(100 * np.median(mine)),
                       "actors_median_cm": float(100 * np.median(act)),
                       "delta_pct": float(delta),
                       "frac_kept_mine": my_kept / my_total,
                       "frac_kept_actors": act_kept / act_total}}
    (root / "data/50_eval/width_locomotion.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote data/50_eval/width_locomotion.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
