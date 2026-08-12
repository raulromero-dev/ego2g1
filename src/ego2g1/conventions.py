"""Coordinate frames, units, and joint ordering — asserted at every stage boundary.

This pipeline crosses four tools that disagree with each other in ways that fail *silently*:

- ARKit / Record3D / LAFAN1 CSV / Rerun use **XYZW** quaternions; MuJoCo, mjlab and
  ``scipy``'s ``scalar_first=True`` use **WXYZ**.
- GVHMR's world frame is **Y-up**; MuJoCo, GMR's IK targets and AMASS are **Z-up**.
- GMR's README claims its output is wxyz "to align with mujoco", but
  ``scripts/smplx_to_robot.py`` saves ``qpos[3:7][[1,2,3,0]]`` — that is XYZW.

A swapped ``w`` does not raise; it trains to a mediocre policy, or produces a robot that
falls over for no visible reason. So every array crossing a stage boundary gets checked
here, and every quaternion field is *named* with its convention (``root_quat_wxyz``,
never ``root_rot``).

Verified against ``third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml`` on 2026-08-12:
``nq=36``, ``nv=35``, ``nu=29``, ``njnt=30`` (a ``pelvis`` freejoint plus 29 hinges).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# --- world -------------------------------------------------------------------

WORLD_UP = "Z"
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)
QUAT_CONVENTION = "wxyz"
UNITS = "m,rad,s"

#: GVHMR emits a right-handed **Y-up** world. Everything downstream is Z-up.
#: Maps (x, y, z)_yup -> (x, -z, y)_zup.
R_ZUP_FROM_YUP = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float64)

#: 35 mm full-frame sensor diagonal, for converting a focal length in pixels to "mm equivalent".
FULLFRAME_DIAG_MM = 43.267

# --- Unitree G1, 29 DoF ------------------------------------------------------

#: Joint order as MuJoCo reports it for ``g1_mocap_29dof.xml``: 6+6 legs, 3 waist, 7+7 arms.
#: ``qpos`` layout is ``[root_pos(3), root_quat_wxyz(4), joint_pos_rad(29)]`` = 36.
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
G1_NQ = 36
G1_NU = 29
G1_STAND_HEIGHT_M = 0.793  # qpos0[2]

SMPLX_BODY_JOINTS = 21  # body_pose is (T, 21, 3) axis-angle, parent-relative

# --- quaternions -------------------------------------------------------------


def quat_wxyz_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle -> WXYZ quaternion, shape (..., 3) -> (..., 4)."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat(scalar_first=True)


def rotvec_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """WXYZ quaternion -> axis-angle, shape (..., 4) -> (..., 3)."""
    return Rotation.from_quat(np.asarray(quat, dtype=np.float64), scalar_first=True).as_rotvec()


def make_quat_continuous(quat_wxyz: np.ndarray) -> np.ndarray:
    """Remove sign flips along time.

    ``q`` and ``-q`` are the same rotation, so estimators flip freely between frames. That is
    invisible in a render and catastrophic to anything that differentiates orientation
    (angular velocity, jerk, or a tracking reward).
    """
    q = np.array(quat_wxyz, dtype=np.float64, copy=True)
    flips = np.einsum("ij,ij->i", q[1:], q[:-1]) < 0
    sign = np.concatenate([[1.0], np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)])
    return (q * sign[:, None]).astype(quat_wxyz.dtype)


# --- assertions --------------------------------------------------------------


def assert_quat_wxyz(quat: np.ndarray, name: str = "quat", *, tol: float = 1e-3,
                     sniff_order: bool = False) -> None:
    """Unit norm and time-continuity. Optionally sniff for a swapped component order.

    ``sniff_order`` is **off by default and should stay off for trusted sources.** The heuristic
    compares mean|first| against mean|last|, which is only meaningful when the motion has little
    yaw. A person walking a corridor and turning around has root rotations near 180 degrees about
    the vertical, where a perfectly valid WXYZ quaternion has w near 0 and z near 1 — and the
    sniff test then reports XYZW for correct data. Measured on this project's own clips:
    mean|w| ran 0.66-0.81 against mean|z| 0.55-0.71, close enough that 8 of 26 clips tripped a
    0.25 threshold.

    Use it only when ingesting a source whose convention is genuinely unknown. For MuJoCo qpos
    the order is guaranteed by definition and sniffing can only produce false alarms.
    """
    q = np.asarray(quat)
    if q.ndim != 2 or q.shape[1] != 4:
        raise AssertionError(f"{name}: expected (T,4), got {q.shape}")

    norms = np.linalg.norm(q, axis=1)
    if not np.allclose(norms, 1.0, atol=tol):
        raise AssertionError(
            f"{name}: not unit quaternions (norm range {norms.min():.4f}..{norms.max():.4f})")

    if sniff_order and np.mean(np.abs(q[:, 3])) > np.mean(np.abs(q[:, 0])) + 0.45:
        raise AssertionError(
            f"{name}: may be XYZW, not WXYZ — mean|last|={np.mean(np.abs(q[:, 3])):.3f} "
            f"vs mean|first|={np.mean(np.abs(q[:, 0])):.3f}. "
            "GMR's save path emits XYZW; convert with quat[:, [3, 0, 1, 2]].")

    if len(q) > 1 and np.any(np.einsum("ij,ij->i", q[1:], q[:-1]) < 0):
        raise AssertionError(f"{name}: sign flips along time — call make_quat_continuous first")


def assert_zup_motion(root_pos_m: np.ndarray, name: str = "root_pos_m",
                      *, lo: float = 0.4, hi: float = 2.2) -> None:
    """Root height must be plausible for a standing human or G1 pelvis.

    Catches the Y-up/Z-up mix-up, which otherwise shows up as a body lying on its side —
    and, because GVHMR's whole contribution is a gravity-aligned prior, as confident nonsense.
    """
    p = np.asarray(root_pos_m)
    if p.ndim != 2 or p.shape[1] != 3:
        raise AssertionError(f"{name}: expected (T,3), got {p.shape}")

    med = float(np.median(p[:, 2]))
    if not lo <= med <= hi:
        spread = p.max(0) - p.min(0)
        raise AssertionError(
            f"{name}: median z = {med:.3f} m, outside [{lo}, {hi}]. Axis convention is likely "
            f"wrong — per-axis travel was x={spread[0]:.2f} y={spread[1]:.2f} z={spread[2]:.2f} m "
            "(for walking, the up axis should vary least).")


def assert_float32(arrays: dict[str, np.ndarray]) -> None:
    bad = {k: v.dtype for k, v in arrays.items() if np.asarray(v).dtype != np.float32}
    if bad:
        raise AssertionError(f"expected float32 arrays, got {bad}")


def assert_no_nan(arrays: dict[str, np.ndarray]) -> None:
    bad = [k for k, v in arrays.items() if not np.all(np.isfinite(np.asarray(v)))]
    if bad:
        raise AssertionError(f"non-finite values in {bad}")


def assert_g1_joint_order(names: "list[str] | tuple[str, ...]") -> None:
    """Fail loudly if the model's joint order ever diverges from what we baked in.

    Unitree ships several G1 revisions and DoF variants; a reordered chain silently maps every
    joint target to the wrong actuator.
    """
    got = tuple(names)
    if got != G1_JOINT_NAMES:
        for i, (a, b) in enumerate(zip(got, G1_JOINT_NAMES)):
            if a != b:
                raise AssertionError(
                    f"G1 joint order diverges at index {i}: model has {a!r}, expected {b!r}")
        raise AssertionError(f"G1 joint count is {len(got)}, expected {len(G1_JOINT_NAMES)}")


# --- focal length ------------------------------------------------------------


def focal_px_from_measurement(subject_px_height: float, distance_m: float,
                              subject_height_m: float) -> float:
    """Pinhole focal length in pixels, from a tape measure and one frame.

    GVHMR defaults to ``f = image diagonal`` (~43 mm equivalent, ~53 degrees diagonal FOV).
    A phone's main camera is ~24-27 mm equivalent, so the default assumes a lens roughly 1.7x
    longer than reality — and since GVHMR consumes K, every global translation, and therefore
    stride length and gait speed, inflates by about that factor.
    """
    return subject_px_height * distance_m / subject_height_m


def focal_mm_from_px(focal_px: float, img_w: int, img_h: int) -> float:
    """Pixel focal length -> 35 mm equivalent (GVHMR's ``--f_mm`` takes an int)."""
    return focal_px * FULLFRAME_DIAG_MM / float(np.hypot(img_w, img_h))
