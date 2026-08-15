"""Export one gait cycle of my recovered SMPL-X body as 2-D points the browser can draw.

Shipping a video would work, but a video cannot be recoloured for dark mode, cannot be
scrubbed by scroll, and cannot be drawn at the page's own device pixel ratio. Shipping the
geometry instead keeps all three, and one gait cycle of decimated vertices is smaller than
the mp4 would have been.

Decisions that keep it small enough to be worth doing:
  * project to the FRONTAL plane here, not in the browser -- (X, Y) in SMPL-X's canonical
    frame. A profile view cannot show lateral foot placement at all: both feet land on the
    same image line however far apart they are. Facing the camera is the only view in which
    the toe-out and the narrow heel base are visible.
  * decimate to ~900 of the 10,475 vertices, chosen by a fixed stride so the sample is
    spread over the whole body rather than clustered on the face
  * quantise to int16 in millimetres; the body is ~1.8 m, so the error is well under a pixel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from width_locomotion_only import locomotion_mask, spans   # noqa: E402

CLIP = "s02_p009"
N_FRAMES = 40
N_POINTS = 900


def main() -> int:
    import smplx

    root = Path.home() / "ego2g1"
    d = np.load(root / f"data/20_human/{CLIP}.npz", allow_pickle=True)
    J = d["joints_pos_m"].astype(float)[:, :22]
    fps = float(d["fps"])
    a, b = max(spans(locomotion_mask(J[:, 0], J[:, 1], J[:, 2], fps), fps),
               key=lambda r: r[1] - r[0])

    pose = d["body_pose_aa"].astype(float)[a:b]
    betas = d["betas"].astype(float)[:10]
    idx = np.linspace(0, len(pose) - 1, N_FRAMES).round().astype(int)

    bm = smplx.create(str(root / "third_party/GMR/assets/body_models"), model_type="smplx",
                      gender="neutral", use_pca=False, batch_size=len(idx), num_betas=10)
    with torch.no_grad():
        # root orientation is left at zero: it is stored Z-up while SMPL-X is Y-up, and the
        # figure only needs to stand and walk in profile. All joint articulation is kept.
        V = bm(body_pose=torch.tensor(pose[idx].reshape(len(idx), 63), dtype=torch.float32),
               global_orient=torch.zeros(len(idx), 3),
               betas=torch.tensor(np.tile(betas, (len(idx), 1)), dtype=torch.float32)
               ).vertices.numpy()

    stride = max(1, len(V[0]) // N_POINTS)
    P = V[:, ::stride, :][:, :N_POINTS, :][:, :, [0, 1]]     # frontal: (X, Y)

    P[:, :, 0] -= P[:, :, 0].mean()                          # centre horizontally
    P[:, :, 1] -= P[:, :, 1].min()                           # feet on zero
    height = float(P[:, :, 1].max())
    P = P / height                                           # unit stature

    q = np.round(P * 10000).astype(np.int16)                 # 0.1 mm at unit stature
    out = root / "data/50_eval/mesh_frames.json"
    out.write_text(json.dumps({
        "clip": CLIP, "n_frames": len(q), "n_points": q.shape[1],
        "quant": 10000, "height_m": height,
        "frames": [f.reshape(-1).tolist() for f in q],
    }, separators=(",", ":")))
    kb = out.stat().st_size / 1024
    print(f"  {len(q)} frames x {q.shape[1]} pts -> {out} ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
