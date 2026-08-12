"""Kinematic playback of a G1 motion in MuJoCo, rendered offscreen to mp4.

**Physics is off.** This calls ``mj_forward``, never ``mj_step`` — it sets ``qpos`` and asks
MuJoCo only to run forward kinematics. That is deliberate: this is the QA rig for retargeting,
so it must show exactly what the retargeter produced, including foot skate and floor
penetration, rather than a physically-plausible correction of it.

Offscreen rendering via ``mujoco.Renderer`` works under plain ``python``. Only the interactive
viewer (``mujoco.viewer.launch_passive``) needs ``mjpython`` on macOS, which is why GMR is
driven as a library elsewhere rather than through its own viewer-constructing scripts.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from ego2g1 import conventions as C

DEFAULT_G1_XML = Path("third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


def load_g1(xml_path: Path | str = DEFAULT_G1_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the G1 and verify it is the model this pipeline was written against."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(
            f"G1 model not found at {xml_path}. Clone GMR: "
            "git clone https://github.com/YanjieZe/GMR third_party/GMR")

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    if model.nq != C.G1_NQ or model.nu != C.G1_NU:
        raise AssertionError(
            f"{xml_path.name}: nq={model.nq}, nu={model.nu}; expected {C.G1_NQ}/{C.G1_NU}. "
            "Wrong G1 variant (23 DoF? with hands?) — joint targets would silently misalign.")

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise AssertionError(f"{xml_path.name}: joint 0 is not the free root joint")
    C.assert_g1_joint_order(names[1:])

    return model, mujoco.MjData(model)


def qpos_from_motion(root_pos_m: np.ndarray, root_quat_wxyz: np.ndarray,
                     joint_pos_rad: np.ndarray) -> np.ndarray:
    """Assemble ``(T, 36)`` qpos from our motion schema, asserting conventions on the way in."""
    root_pos_m = np.asarray(root_pos_m, dtype=np.float64)
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
    joint_pos_rad = np.asarray(joint_pos_rad, dtype=np.float64)

    C.assert_quat_wxyz(root_quat_wxyz, "root_quat_wxyz")
    C.assert_zup_motion(root_pos_m, "root_pos_m")
    if joint_pos_rad.shape[1] != C.G1_NU:
        raise AssertionError(
            f"joint_pos_rad: expected (T,{C.G1_NU}), got {joint_pos_rad.shape}")

    return np.concatenate([root_pos_m, root_quat_wxyz, joint_pos_rad], axis=1)


def report_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, float]:
    """Fraction of frames each joint spends outside its limit.

    A retargeted clip that saturates the G1's narrow ankle-roll (+/-15 deg) or waist
    (+/-30 deg) range is not trainable, and this is far cheaper to check than to discover
    during RL.
    """
    joints = qpos[:, 7:]
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    violation = (joints < lo[None, :]) | (joints > hi[None, :])
    per_joint = violation.mean(axis=0)
    return {
        "violation_frac_overall": float(violation.mean()),
        "worst_joint": C.G1_JOINT_NAMES[int(np.argmax(per_joint))],
        "worst_joint_frac": float(per_joint.max()),
        "n_joints_violating": int((per_joint > 0).sum()),
    }


def render_qpos(qpos: np.ndarray, out_path: Path | str, *,
                model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None,
                fps: float = 30.0, width: int = 640, height: int = 480,
                azimuth: float = 135.0, elevation: float = -15.0, distance: float = 3.2,
                track_root: bool = True) -> Path:
    """Render a ``(T, 36)`` qpos sequence to mp4. Returns the output path."""
    if model is None or data is None:
        model, data = load_g1()

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape[1] != model.nq:
        raise AssertionError(f"qpos: expected (T,{model.nq}), got {qpos.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    # Follow the root so a walking robot does not stroll out of frame.
    cam.lookat[:] = qpos[0, :3] if track_root else qpos[:, :3].mean(axis=0)

    with mujoco.Renderer(model, height=height, width=width) as renderer, \
            imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)          # kinematics only — never mj_step
            if track_root:
                cam.lookat[:] = frame[:3]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

    return out_path


def synthetic_walk(n_frames: int = 150, fps: float = 30.0, *,
                   step_hz: float = 1.8, forward_speed: float = 0.8) -> np.ndarray:
    """A crude sinusoidal gait, used only to prove the render loop before real data exists.

    This is NOT a gait model and makes no biomechanical claim — it exists so step 0 can be
    verified with zero downloads, zero gated assets, and no GPU.
    """
    t = np.arange(n_frames) / fps
    phase = 2 * np.pi * step_hz * t

    joints = np.zeros((n_frames, C.G1_NU), dtype=np.float64)
    idx = {name: i for i, name in enumerate(C.G1_JOINT_NAMES)}

    for side, sign in (("left", 1.0), ("right", -1.0)):
        swing = sign * np.sin(phase)
        joints[:, idx[f"{side}_hip_pitch_joint"]] = 0.45 * swing
        joints[:, idx[f"{side}_knee_joint"]] = 0.55 * np.clip(-swing, 0, None) + 0.10
        joints[:, idx[f"{side}_ankle_pitch_joint"]] = -0.20 * swing
        # Arms counter-swing against the legs — the coupling that reads as "walking".
        joints[:, idx[f"{side}_shoulder_pitch_joint"]] = -0.35 * swing
        joints[:, idx[f"{side}_elbow_joint"]] = 0.30

    root_pos = np.zeros((n_frames, 3), dtype=np.float64)
    root_pos[:, 0] = forward_speed * t
    # Vertical oscillation at twice step frequency — once per step, twice per stride.
    root_pos[:, 2] = C.G1_STAND_HEIGHT_M + 0.012 * np.cos(2 * phase)

    root_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))

    return qpos_from_motion(root_pos, root_quat, joints)
