"""Retarget SMPL-X motion onto the Unitree G1, driving GMR as a library.

Deliberately not ``scripts/gvhmr_to_robot.py``. That script:

- hard-codes ``human_height = 1.66 + 0.1 * betas[0]``, which is a guess from a shape estimate,
  where we have a tape measure;
- applies no Y-up -> Z-up correction (the fix is present but commented out in GMR's own source);
- saves the root quaternion as **XYZW** into a field named ``root_rot`` despite the README
  claiming wxyz; and
- constructs a passive MuJoCo viewer unconditionally, which on macOS forces ``mjpython`` and on a
  headless box needs a virtual display.

Driving the library directly avoids all four. What GMR wants per frame is a dict of
``{joint_name: (position_xyz, quaternion_wxyz)}`` holding **global** joint orientations
accumulated down the kinematic chain — not the parent-relative rotations SMPL-X stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ego2g1 import conventions as C

WARMUP_ITERS = 8

G1_XML = Path("third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


@dataclass
class SmplxMotion:
    """SMPL-X motion in *our* convention: Z-up, metres, WXYZ where quaternions appear."""

    betas: np.ndarray          # (10,)
    global_orient: np.ndarray  # (T,3) axis-angle, Z-up world
    body_pose: np.ndarray      # (T,21,3) axis-angle, parent-relative
    transl: np.ndarray         # (T,3) metres, Z-up world
    fps: float = 30.0

    def __len__(self) -> int:
        return len(self.global_orient)


def yup_to_zup(global_orient: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a SMPL-X motion from GVHMR's Y-up world into our Z-up world.

    ``body_pose`` is intentionally untouched: it is parent-relative, so it carries no world-frame
    information and rotating it would corrupt the pose. Only the root orientation and the global
    translation live in the world frame.

    Note this *composes* rotation matrices rather than multiplying the axis-angle vectors —
    GMR's commented-out attempt does the latter, which is not a rotation of a rotation.
    """
    R_zy = C.R_ZUP_FROM_YUP
    transl_z = (R_zy @ np.asarray(transl, dtype=np.float64).T).T
    rot = Rotation.from_matrix(R_zy) * Rotation.from_rotvec(np.asarray(global_orient, np.float64))
    return rot.as_rotvec(), transl_z


def make_body_model(batch_size: int, model_dir: str = "assets/body_models", gender: str = "neutral"):
    """Build SMPL-X sized to the sequence.

    ``smplx`` allocates its default jaw/eye/hand poses at construction time, so a model created
    with ``batch_size=1`` cannot be called with T frames — the unspecified parts stay batch-1 and
    the concat inside ``forward`` fails. Size it up front.
    """
    import smplx
    return smplx.create(model_dir, model_type="smplx", gender=gender,
                        use_pca=False, ext="npz", batch_size=batch_size)


def smplx_frames(motion: SmplxMotion, body_model=None) -> tuple[list[dict], float]:
    """Run SMPL-X forward and build the per-frame dicts GMR consumes."""
    from general_motion_retargeting.utils.smpl import JOINT_NAMES

    T = len(motion)
    if body_model is None or getattr(body_model, "batch_size", None) != T:
        body_model = make_body_model(T)
    out = body_model(
        betas=torch.tensor(motion.betas, dtype=torch.float32).reshape(1, -1).expand(T, -1),
        global_orient=torch.tensor(motion.global_orient, dtype=torch.float32),
        body_pose=torch.tensor(motion.body_pose.reshape(T, -1), dtype=torch.float32),
        transl=torch.tensor(motion.transl, dtype=torch.float32),
        # full_pose is None unless asked for, and it is what carries the per-joint local
        # rotations we accumulate into global orientations below.
        return_full_pose=True,
    )
    joints = out.joints.detach().numpy()
    verts = out.vertices.detach().numpy()
    full_pose = out.full_pose.detach().numpy().reshape(T, -1, 3)

    parents = body_model.parents.numpy()
    names = JOINT_NAMES[: len(parents)]

    # Measure the subject from the mesh in the most upright frame, rather than trusting betas.
    heights = verts[:, :, 2].max(1) - verts[:, :, 2].min(1)
    smpl_height_m = float(np.median(heights))

    frames: list[dict] = []
    for t in range(T):
        rots: list[Rotation] = []
        frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, name in enumerate(names):
            local = Rotation.from_rotvec(full_pose[t, i])
            rot = local if i == 0 else rots[parents[i]] * local
            rots.append(rot)
            frame[name] = (joints[t, i], rot.as_quat(scalar_first=True))  # scalar_first = WXYZ
        frames.append(frame)

    return frames, smpl_height_m


def retarget(motion: SmplxMotion, body_model=None, *, subject_height_m: float,
             robot: str = "unitree_g1", verbose: bool = True) -> dict[str, np.ndarray]:
    """SMPL-X motion -> G1 ``qpos``. Returns arrays in our schema."""
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    frames, smpl_height_m = smplx_frames(motion, body_model)

    # Scale from a real measurement, never from the beta heuristic: betas come from a person who
    # is a few hundred pixels tall in much of the footage, and this scale lands directly on
    # stride length and gait speed.
    world_scale = subject_height_m / smpl_height_m
    if verbose:
        print(f"  SMPL-X mesh height {smpl_height_m:.3f} m -> subject {subject_height_m:.3f} m "
              f"(scale {world_scale:.4f})")

    retargeter = GMR(actual_human_height=subject_height_m, src_human="smplx", tgt_robot=robot)

    # GMR's IK is stateful: each solve warm-starts from the previous one. The very first frame
    # therefore starts from the robot's default pose and only partially converges, which showed
    # up as a 4-5 m root teleport between frames 0 and 1 (156 m/s) while every later frame moved
    # a normal ~3 cm. Solve the first frame repeatedly until the solution settles, then record.
    for _ in range(WARMUP_ITERS):
        retargeter.retarget(frames[0])

    qpos = np.zeros((len(frames), C.G1_NQ), dtype=np.float64)
    for t, frame in enumerate(frames):
        qpos[t] = retargeter.retarget(frame)

    root_pos = qpos[:, :3].astype(np.float32)
    root_quat = C.make_quat_continuous(qpos[:, 3:7]).astype(np.float32)
    joint_pos = qpos[:, 7:].astype(np.float32)

    # GMR's internal qpos is MuJoCo-ordered, so this should already be WXYZ — assert rather
    # than assume, because its save path converts to XYZW and that is easy to inherit.
    C.assert_quat_wxyz(root_quat, "root_quat_wxyz")
    C.assert_zup_motion(root_pos, "root_pos_m")
    C.assert_no_nan({"root_pos_m": root_pos, "joint_pos_rad": joint_pos})

    return {
        "root_pos_m": root_pos,
        "root_quat_wxyz": root_quat,
        "joint_pos_rad": joint_pos,
        "joint_names": np.array(C.G1_JOINT_NAMES),
        "fps": np.float64(motion.fps),
        "timestamps_s": (np.arange(len(frames)) / motion.fps).astype(np.float64),
        "smpl_height_m": np.float32(smpl_height_m),
        "subject_height_m": np.float32(subject_height_m),
        "world_scale_applied": np.float32(world_scale),
        "scale_source": "measured",
        "quat_convention": "wxyz",
        "world_up_axis": "Z",
    }


def save(result: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    return path


def to_qpos(result: dict) -> np.ndarray:
    """Our schema -> the ``(T, 36)`` array MuJoCo wants."""
    return np.concatenate([result["root_pos_m"], result["root_quat_wxyz"],
                           result["joint_pos_rad"]], axis=1).astype(np.float64)
