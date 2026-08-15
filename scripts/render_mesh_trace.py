"""Form-trace from the recovered SMPL-X *body*, not the skeleton.

The stick-figure version shows joint positions; this shows the surface GVHMR actually
recovered. Both walkers drive the **same neutral SMPL-X body**, so body shape is removed
from the comparison entirely and only motion differs -- which is what makes it apples to
apples. Mine comes from GVHMR's fitted pose; the actor's comes from mapping LAFAN1's BVH
joint rotations onto the matching SMPL-X joints.

Rendering is a density accumulation rather than a mesh rasteriser: every pose scatters its
10,475 vertices into a float buffer, and the buffer is mapped to ink at the end. Overlapping
poses therefore darken where the body dwells, which is exactly the quality the reference
images have, and it costs one numpy scatter per pose instead of 20,908 filled triangles.

CAVEAT worth keeping in view: LAFAN1's rest pose and SMPL-X's rest pose are both roughly
T-poses but not identical, and this maps local rotations across without a rest-pose
calibration. Legs (straight down in both) transfer well; shoulders carry a few degrees of
error. Good enough to compare gait, not good enough to measure arm angles from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ego2g1.eval import bvh                                    # noqa: E402
from ego2g1.viz.smplx_white import _cycle                      # noqa: E402
from width_locomotion_only import locomotion_mask, spans, RAW   # noqa: E402

MODEL_DIR = Path.home() / "ego2g1/third_party/GMR/assets/body_models"

#: SMPL-X body_pose index (0..20, i.e. joint 1..21) <- LAFAN1 BVH joint name.
BVH_TO_SMPLX = {
    0: "LeftUpLeg", 1: "RightUpLeg", 2: "Spine", 3: "LeftLeg", 4: "RightLeg",
    5: "Spine1", 6: "LeftFoot", 7: "RightFoot", 8: "Spine2", 9: "LeftToe",
    10: "RightToe", 11: "Neck", 12: "LeftShoulder", 13: "RightShoulder", 14: "Head",
    15: "LeftArm", 16: "RightArm", 17: "LeftForeArm", 18: "RightForeArm",
    19: "LeftHand", 20: "RightHand",
}


def body_model():
    import smplx
    return smplx.create(str(MODEL_DIR), model_type="smplx", gender="neutral",
                        use_pca=False, batch_size=1)


def verts_from_pose(bm, body_pose: np.ndarray, global_orient: np.ndarray) -> np.ndarray:
    """(T,21,3) axis-angle + (T,3) root -> (T, 10475, 3) vertices, neutral shape."""
    out = []
    with torch.no_grad():
        for t in range(len(body_pose)):
            o = bm(body_pose=torch.tensor(body_pose[t].reshape(1, 63), dtype=torch.float32),
                   global_orient=torch.tensor(global_orient[t].reshape(1, 3), dtype=torch.float32),
                   betas=torch.zeros(1, 10))
            out.append(o.vertices[0].numpy())
    return np.stack(out)


def bvh_local_rotations(joints, motion) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-joint local rotation vectors from the BVH channels, plus the root's."""
    col, rot = 0, {}
    for j in joints:
        rcols = [col + n for n, c in enumerate(j.channels) if c.endswith("rotation")]
        order = "".join(c[0].upper() for c in j.channels if c.endswith("rotation"))
        if rcols:
            rot[j.name] = Rotation.from_euler(order, motion[:, rcols], degrees=True).as_rotvec()
        col += len(j.channels)
    return rot, rot.get("Hips", np.zeros((len(motion), 3)))


def accumulate(all_verts: list[np.ndarray], size, spacing_frac=0.30, cycles=2.6, n=30):
    """Scatter every pose's vertices into a density buffer; return it normalised 0..1."""
    W, H = size
    buf = np.zeros((H, W), dtype=np.float32)

    # SMPL-X canonical is Y-up with X across the shoulders and Z through the body, so a
    # profile view is the (Z, Y) plane. Projecting (X, Y) -- the obvious guess -- gives a
    # front view, in which a walk is almost invisible.
    V = np.stack(all_verts)[:, :, [2, 1]]                    # (T, nv, 2) profile
    stature = np.percentile(V[:, :, 1], 99) - np.percentile(V[:, :, 1], 1)
    V = V / stature
    V[:, :, 1] -= V[:, :, 1].min()

    step = spacing_frac
    span = (n - 1) * step
    scale = W / (span + 1.5)
    y0 = H - int(0.06 * H)

    for i in range(n):
        t = int(((i / max(n - 1, 1)) * cycles % 1.0) * (len(V) - 1))
        v = V[t]
        x = (v[:, 0] - v[:, 0].mean() + i * step + 0.75) * scale
        y = y0 - v[:, 1] * scale
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        np.add.at(buf, (y[ok].astype(int), x[ok].astype(int)), 1.0 if i < n - 1 else 3.0)

    return buf


def to_image(buf: np.ndarray, rgb, gamma=0.42, cap=6.0) -> np.ndarray:
    a = np.clip(buf / cap, 0, 1) ** gamma
    img = np.ones((*buf.shape, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = 1.0 - a * (1.0 - rgb[c] / 255.0)
    return (img * 255).astype(np.uint8)


def main() -> int:
    root = Path.home() / "ego2g1"
    bm = body_model()
    SS = 2
    W, H = 1500, 300

    # ---- mine: GVHMR's own fit ------------------------------------------------
    d = np.load(root / "data/20_human/s02_p009.npz", allow_pickle=True)
    J = d["joints_pos_m"].astype(float)[:, :22]
    fps = float(d["fps"])
    a, b = max(spans(locomotion_mask(J[:, 0], J[:, 1], J[:, 2], fps), fps),
               key=lambda r: r[1] - r[0])
    pose = d["body_pose_aa"].astype(float)[a:b]
    quat = d["root_quat_wxyz"].astype(float)[a:b]
    orient = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec()
    idx = np.linspace(0, len(pose) - 1, 64).round().astype(int)
    mine = verts_from_pose(bm, pose[idx], np.zeros_like(orient[idx]))
    print(f"  mine: {mine.shape}")

    # ---- actor: LAFAN1 BVH rotations onto the same body -----------------------
    joints, motion, ft = bvh.parse(RAW / "walk3_subject4.bvh")
    nm = bvh.names(joints)
    P = bvh.forward_kinematics(joints, motion)
    f2 = 1 / ft
    a2, b2 = max(spans(locomotion_mask(P[:, 0], P[:, nm.index("LeftUpLeg")],
                                       P[:, nm.index("RightUpLeg")], f2), f2),
                 key=lambda r: r[1] - r[0])
    rot, hips = bvh_local_rotations(joints, motion)
    T = b2 - a2
    ap = np.zeros((T, 21, 3))
    for k, name in BVH_TO_SMPLX.items():
        if name in rot:
            ap[:, k] = rot[name][a2:b2]
    # BVH is Y-up; bring the root into the same Z-up frame the rest of the repo uses
    R_zup = Rotation.from_euler("x", 90, degrees=True)
    ao = (R_zup * Rotation.from_rotvec(hips[a2:b2])).as_rotvec()
    jdx = np.linspace(0, T - 1, 64).round().astype(int)
    act = verts_from_pose(bm, ap[jdx], np.zeros_like(ao[jdx]))
    print(f"  actor: {act.shape}")

    out = Path(root / "data/qa/smplx_white/mesh_trace.jpg")
    lanes = [to_image(accumulate(list(mine), (W, H)), (17, 17, 17)),
             to_image(accumulate(list(act), (W, H)), (138, 138, 138))]
    canvas = np.vstack(lanes)
    Image.fromarray(canvas).resize((2400, 860), Image.LANCZOS).save(out, quality=94)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
