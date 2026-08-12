"""Turn retargeted G1 motion into the reference-motion format an RL tracker consumes.

Three jobs, none of them glamorous and all of them easy to get subtly wrong:

1. **Resample 30 -> 50 Hz.** Trackers run at 50 Hz control. Positions interpolate linearly;
   rotations must SLERP, because component-wise interpolation of quaternions does not produce a
   rotation and shows up later as a limb that swings wide through fast turns.
2. **Differentiate.** Joint and body velocities are what the tracking reward actually compares,
   and they must be computed *after* resampling — differentiating at 30 Hz and then resampling
   the derivative smears every contact transient.
3. **Forward kinematics.** The tracker compares body poses in the world, not just joint angles,
   so every frame is pushed through ``mj_forward`` to read all 38 body positions and orientations.

Root pose lives as body 0 of the ``body_*`` arrays rather than as separate fields — that is
mjlab's convention, and duplicating it invites the two copies to disagree.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ego2g1 import conventions as C

TARGET_FPS = 50.0


def resample(qpos: np.ndarray, fps_in: float, fps_out: float = TARGET_FPS) -> np.ndarray:
    """Resample a ``(T, 36)`` qpos track, SLERPing the root rotation."""
    n_in = len(qpos)
    t_in = np.arange(n_in) / fps_in
    duration = t_in[-1]
    # np.arange accumulates rounding, so the final sample can land a hair past `duration`.
    # Slerp rejects anything outside its input range, so clamp rather than hope.
    t_out = np.clip(np.arange(0.0, duration + 1e-9, 1.0 / fps_out), t_in[0], t_in[-1])

    pos = np.stack([np.interp(t_out, t_in, qpos[:, i]) for i in range(3)], axis=1)
    joints = np.stack([np.interp(t_out, t_in, qpos[:, 7 + i]) for i in range(C.G1_NU)], axis=1)

    quat = C.make_quat_continuous(qpos[:, 3:7])
    rots = Rotation.from_quat(quat, scalar_first=True)
    quat_out = Slerp(t_in, rots)(t_out).as_quat(scalar_first=True)

    return np.concatenate([pos, quat_out, joints], axis=1)


def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central differences with one-sided ends — same length in, same length out."""
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def _angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a quaternion track.

    Computed from relative rotations rather than by differentiating components, which would not
    be a physical angular velocity.
    """
    r = Rotation.from_quat(quat_wxyz, scalar_first=True)
    n = len(quat_wxyz)
    omega = np.zeros((n, 3))
    if n > 1:
        rel = (r[1:] * r[:-1].inv()).as_rotvec() / dt
        omega[:-1] = rel
        omega[-1] = rel[-1]
    return omega


def forward_kinematics(qpos: np.ndarray, model: mujoco.MjModel,
                       data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Body positions and WXYZ orientations for every frame, shape ``(T, nbody, ...)``."""
    n = len(qpos)
    pos = np.zeros((n, model.nbody, 3))
    quat = np.zeros((n, model.nbody, 4))
    for t, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos
        quat[t] = data.xquat          # MuJoCo xquat is WXYZ
    return pos, quat


def build_reference(qpos_30: np.ndarray, fps_in: float, *, model=None, data=None,
                    fps_out: float = TARGET_FPS) -> dict:
    """Retargeted qpos -> mjlab-style reference motion."""
    if model is None or data is None:
        from ego2g1.viz.mujoco_playback import load_g1
        model, data = load_g1()

    qpos = resample(qpos_30, fps_in, fps_out)
    dt = 1.0 / fps_out
    n = len(qpos)

    joint_pos = qpos[:, 7:]
    joint_vel = _central_diff(joint_pos, dt)

    body_pos, body_quat = forward_kinematics(qpos, model, data)

    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    for b in range(model.nbody):
        body_lin_vel[:, b] = _central_diff(body_pos[:, b], dt)
        body_ang_vel[:, b] = _angular_velocity(C.make_quat_continuous(body_quat[:, b]), dt)

    return {
        "fps": np.array([fps_out], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
        "n_frames": np.int32(n),
        "duration_s": np.float32(n / fps_out),
    }


def sanity(ref: dict) -> dict:
    """Numbers that reveal a bad reference before a GPU ever sees it."""
    jv = ref["joint_vel"]
    lv = ref["body_lin_vel_w"]
    root_z = ref["body_pos_w"][:, 1, 2]        # body 1 = pelvis (0 is world)
    return {
        "n_frames": int(ref["n_frames"]),
        "duration_s": float(ref["duration_s"]),
        "joint_vel_max": float(np.abs(jv).max()),
        "joint_vel_p99": float(np.percentile(np.abs(jv), 99)),
        "root_speed_max": float(np.linalg.norm(lv[:, 1], axis=1).max()),
        "root_z_med": float(np.median(root_z)),
        "root_z_range": float(root_z.max() - root_z.min()),
    }


def save(ref: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ref)
    return path
