"""Measure step width straight off the footage, with no SMPL and no robot in the path.

Why this exists. Step width came out three different ways depending on what it was measured on:

    my SMPL-X (monocular -> GVHMR)   7.1 cm
    my motion retargeted to the G1  18.6 cm
    LAFAN1 actors (mocap BVH)       19.3 cm

The G1 number is not a measurement of anyone -- retargeting left the actors' width alone
(19.3 -> 18.8) but inflated mine 2.6x, which is the IK clamping a narrow stance up to the
robot's own minimum. That leaves SMPL-X against mocap, and lateral foot placement is precisely
what a single camera recovers worst. So neither surviving number settles it.

This does. MediaPipe gives 2-D landmarks per frame; contacts come from image-space foot speed
(a planted foot is stationary in a fixed camera, whatever the depth); step width uses the same
local construction as the 3-D code. The result is normalised by the subject's own hip width,
measured in the same frames -- both are lateral quantities in the frontal plane, so the
perspective scaling divides out and no focal length or subject height is needed.

Only clips where the subject walks toward or away from the camera are usable: there, lateral
displacement lies in the image plane (the well-observed direction) rather than along depth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ego2g1.eval.gait import _runs  # noqa: E402

MODEL = Path("/private/tmp/claude-501/-Users-raulromero/"
             "15ae2624-107f-467a-a2c4-cd44d1c68c78/scratchpad/pose_landmarker_heavy.task")
CLIPS = ("s02_p022", "s02_p023", "s02_p024", "s02_p025", "s02_p026")

L_HIP, R_HIP, L_ANK, R_ANK = 23, 24, 27, 28
MIN_VIS = 0.6


def landmarks(video: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-frame pixel landmarks (T,33,2) and visibility (T,33)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (PoseLandmarker, PoseLandmarkerOptions,
                                               RunningMode)

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=0.5)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    pts, vis, t = [], [], 0

    with PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(t * 1000 / fps))
            if res.pose_landmarks:
                p = res.pose_landmarks[0]
                pts.append([[q.x * w, q.y * h] for q in p])
                vis.append([q.visibility for q in p])
            else:
                pts.append(np.full((33, 2), np.nan))
                vis.append(np.zeros(33))
            t += 1
    cap.release()
    return np.array(pts, float), np.array(vis, float), fps


def contacts(ank: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Frames where a foot is planted: image-space speed well below its own swing speed."""
    v = np.full(len(ank), np.nan)
    v[:-1] = np.linalg.norm(np.diff(ank, axis=0), axis=1) * fps
    v[-1] = v[-2]
    ref = np.nanpercentile(v, 75)
    return _runs(np.nan_to_num(v, nan=1e9) < 0.25 * max(ref, 1e-6), fps)


def widths(pts: np.ndarray, vis: np.ndarray, fps: float) -> list[float]:
    """Step widths in units of the subject's own hip width."""
    ok = (vis[:, [L_HIP, R_HIP, L_ANK, R_ANK]] > MIN_VIS).all(axis=1)
    out = []
    per_side = {}
    for side, idx in (("L", L_ANK), ("R", R_ANK)):
        per_side[side] = []
        for a, b in contacts(pts[:, idx], fps):
            if not ok[a:b].any():
                continue
            m = (a + b) // 2
            hip = np.linalg.norm(pts[m, L_HIP] - pts[m, R_HIP])
            if not np.isfinite(hip) or hip < 5:
                continue
            per_side[side].append((m / fps, pts[a:b, idx].mean(axis=0), hip))

    for this, other in (("L", "R"), ("R", "L")):
        for t, p, hip in per_side[this]:
            before = [c for c in per_side[other] if c[0] < t]
            after = [c for c in per_side[other] if c[0] > t]
            if not before or not after:
                continue
            p0, p1 = before[-1][1], after[0][1]
            seg = p1 - p0
            n = np.linalg.norm(seg)
            if n < 10:                                   # pixels; ill-conditioned baseline
                continue
            d = p - p0
            out.append(float(abs(seg[0] * d[1] - seg[1] * d[0]) / n / hip))
    return out


def main() -> int:
    root = Path.home() / "ego2g1"
    result = {}
    for cid in CLIPS:
        vid = next((root / "data/05_clips").glob(f"*/exo/{cid}.mp4"), None)
        if vid is None:
            print(f"  {cid}: not found")
            continue
        pts, vis, fps = landmarks(vid)
        w = widths(pts, vis, fps)
        result[cid] = w
        print(f"  {cid}: {len(pts)} frames, {len(w)} steps, "
              f"median {np.median(w):.3f} hip-widths" if w else f"  {cid}: no steps")

    allw = [x for v in result.values() for x in v]
    out = root / "data/50_eval/width_2d.json"
    out.write_text(json.dumps({"per_clip": result,
                               "median_hipnorm": float(np.median(allw)) if allw else None,
                               "n_steps": len(allw)}, indent=1))
    if allw:
        print(f"\n  ALL: {len(allw)} steps, median {np.median(allw):.3f} hip-widths "
              f"(IQR {np.percentile(allw, 25):.3f}-{np.percentile(allw, 75):.3f})")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
