"""Convert GVHMR output into our motion schema.

This is the seam where silent bugs live, because four independent things have to be right and
none of them raise when they are wrong:

1. **Axis convention.** GVHMR's world frame is Y-up; everything downstream (MuJoCo, GMR's IK
   targets, AMASS) is Z-up. Verified empirically against real output: the translation span along
   ``y`` is the smallest of the three axes, which is what "up" looks like for someone walking.
2. **Metric scale.** GVHMR's ``betas`` come from a subject who is a few hundred pixels tall in
   much of this footage. We therefore measure the mesh and rescale to a tape-measured height
   rather than trusting the shape estimate — the scale lands directly on stride length.
3. **Floor height.** GVHMR's world origin is not the ground. Feet must end up near z=0 or the
   retargeter places the robot underground or hovering.
4. **Time.** Every frame carries both a clip-relative timestamp and ``src_time_s``, the offset
   into the original session recording. That is the only link back to the source video, the ego
   footage, and the per-frame subject track.

``body_pose`` is deliberately never rotated. It is parent-relative, so it holds no world-frame
information; rotating it corrupts the pose while looking plausible. GMR's own (commented-out)
attempt at the axis fix makes exactly this mistake, and additionally multiplies axis-angle
vectors as if they were points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ego2g1 import conventions as C

#: SMPL-X body joints whose lowest point approximates ground contact.
FOOT_JOINTS = (7, 8, 10, 11)   # ankles and feet


@dataclass
class ImportReport:
    clip_id: str
    n_frames: int
    smpl_height_m: float
    world_scale: float
    floor_z_m: float
    root_height_med_m: float
    up_axis_span_ratio: float
    warnings: list[str]


def _median_betas(betas: np.ndarray) -> np.ndarray:
    """GVHMR emits per-frame betas; shape should not drift within a clip."""
    return np.median(np.asarray(betas, dtype=np.float64), axis=0).astype(np.float32)


def load_gvhmr(path: Path | str) -> dict:
    """Read ``hmr4d_results.pt`` without needing torch at import time."""
    import torch
    return torch.load(str(path), map_location="cpu")


def to_zup(global_orient_aa: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Y-up -> Z-up for the world-frame quantities only."""
    R_zy = C.R_ZUP_FROM_YUP
    transl_z = (R_zy @ np.asarray(transl, dtype=np.float64).T).T
    rot = (Rotation.from_matrix(R_zy)
           * Rotation.from_rotvec(np.asarray(global_orient_aa, dtype=np.float64)))
    return rot.as_rotvec(), transl_z


def convert(results: dict, *, clip_id: str, subject_height_m: float,
            src_start_s: float, fps: float = 30.0,
            body_model=None) -> tuple[dict, ImportReport]:
    """GVHMR results -> our SMPL schema, Z-up, metric, floor-corrected, asserted."""
    import torch

    g = results["smpl_params_global"]
    to_np = lambda x: (x.detach().cpu().numpy() if isinstance(x, torch.Tensor)
                       else np.asarray(x))

    body_pose = to_np(g["body_pose"]).astype(np.float64)          # (T,63) parent-relative
    global_orient = to_np(g["global_orient"]).astype(np.float64)  # (T,3) world
    transl = to_np(g["transl"]).astype(np.float64)                # (T,3) world, Y-up
    betas = _median_betas(to_np(g["betas"]))
    T = len(transl)
    warnings: list[str] = []

    # Sanity-check the up axis before trusting the conversion: for walking, the up axis should
    # have the smallest travel. If it does not, the input is not what we think it is.
    span = transl.max(0) - transl.min(0)
    up_ratio = float(span[1] / (span.max() + 1e-9))
    if int(span.argmin()) != 1:
        warnings.append(
            f"expected Y (index 1) to have the smallest translation span, got index "
            f"{int(span.argmin())} (spans x={span[0]:.2f} y={span[1]:.2f} z={span[2]:.2f}) — "
            "GVHMR may not be Y-up here, or the subject barely moved")

    global_orient, transl = to_zup(global_orient, transl)

    # Forward the body model to get joints and a mesh we can measure.
    from ego2g1.retarget.gmr_runner import make_body_model
    if body_model is None or getattr(body_model, "batch_size", None) != T:
        body_model = make_body_model(T)
    out = body_model(
        betas=torch.tensor(betas, dtype=torch.float32).reshape(1, -1).expand(T, -1),
        global_orient=torch.tensor(global_orient, dtype=torch.float32),
        body_pose=torch.tensor(body_pose, dtype=torch.float32),
        transl=torch.tensor(transl, dtype=torch.float32),
    )
    verts = out.vertices.detach().numpy()
    joints = out.joints.detach().numpy()

    # Measure the mesh rather than trusting betas, then rescale to the tape measure.
    smpl_height_m = float(np.median(verts[:, :, 2].max(1) - verts[:, :, 2].min(1)))
    world_scale = subject_height_m / smpl_height_m
    transl *= world_scale
    joints *= world_scale
    verts *= world_scale

    # Ground the motion: the 5th percentile of the lowest foot height is a robust floor estimate,
    # tolerant of a few frames where a foot is mis-placed.
    foot_z = joints[:, list(FOOT_JOINTS), 2].min(axis=1)
    floor_z = float(np.percentile(foot_z, 5))
    transl[:, 2] -= floor_z
    joints[:, :, 2] -= floor_z

    root_pos = transl.astype(np.float32)
    root_quat = C.make_quat_continuous(
        C.quat_wxyz_from_rotvec(global_orient)).astype(np.float32)

    C.assert_quat_wxyz(root_quat, f"{clip_id}:root_quat_wxyz")
    C.assert_no_nan({"root_pos_m": root_pos, "joints_pos_m": joints})
    try:
        C.assert_zup_motion(root_pos, f"{clip_id}:root_pos_m")
    except AssertionError as exc:
        # Do not hard-fail a batch on one bad clip; record it and let the caller gate.
        warnings.append(str(exc))

    timestamps = (np.arange(T) / fps).astype(np.float64)
    schema = {
        "schema_version": np.int32(1),
        "clip_id": clip_id,
        "world_up_axis": "Z",
        "quat_convention": "wxyz",
        "units": "m,rad,s",
        "body_model": "smplx_neutral",
        "n_frames": np.int32(T),
        "fps": np.float64(fps),
        "timestamps_s": timestamps,
        "src_time_s": timestamps + float(src_start_s),
        "betas": betas,
        "root_pos_m": root_pos,
        "root_quat_wxyz": root_quat,
        "body_pose_aa": body_pose.reshape(T, 21, 3).astype(np.float32),
        "joints_pos_m": joints.astype(np.float32),
        "subject_height_m": np.float32(subject_height_m),
        "smpl_height_m": np.float32(smpl_height_m),
        "world_scale_applied": np.float32(world_scale),
        "scale_source": "measured",
        "floor_z_m": np.float32(floor_z),
        "K_px": to_np(results["K_fullimg"])[0].astype(np.float32)
                 if "K_fullimg" in results else np.zeros((3, 3), np.float32),
        "static_cam": np.bool_(True),
    }

    report = ImportReport(
        clip_id=clip_id, n_frames=T, smpl_height_m=smpl_height_m, world_scale=world_scale,
        floor_z_m=floor_z, root_height_med_m=float(np.median(root_pos[:, 2])),
        up_axis_span_ratio=up_ratio, warnings=warnings,
    )
    return schema, report


def save(schema: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **schema)
    return path


def to_smplx_motion(schema: dict):
    """Our SMPL schema -> the input type the retargeter takes."""
    from ego2g1.retarget.gmr_runner import SmplxMotion
    return SmplxMotion(
        betas=schema["betas"],
        global_orient=C.rotvec_from_quat_wxyz(schema["root_quat_wxyz"]).astype(np.float32),
        body_pose=schema["body_pose_aa"],
        transl=schema["root_pos_m"],
        fps=float(schema["fps"]),
    )
