"""Gait parameters from a G1 motion — the descriptor that defines "walks like Raul".

Everything here is computed identically on a retargeted reference and on a policy rollout, so
the same numbers describe your motion, the baseline's motion, and the trained robot's motion.
That is the whole point: a comparison is only meaningful if both sides go through one function.

**Timing metrics are scale-free and distance metrics are not.** Cadence, step time, duty factor,
symmetry, and phase offsets depend only on *when* contacts happen, so they survive the
uncalibrated focal length in the capture pipeline untouched. Stride length, step width, and gait
speed are all distances and inherit that error directly. They are computed here and reported, but
every one is flagged ``scale_dependent`` so no downstream claim leans on them by accident.

Contacts are detected from foot height rather than from simulator contact forces, so the same
code works on kinematic reference motion (where no contact forces exist) and on physics rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from ego2g1 import conventions as C

#: Ankle-roll links are the lowest bodies in the G1's foot; index into the model's body array.
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
#: Height below which a foot counts as loaded. The G1's ankle-roll link sits a few cm off the
#: ground when the sole is flat, so this is deliberately above zero.
CONTACT_H = 0.08
#: Ignore contacts shorter than this; they are detection chatter, not steps.
MIN_CONTACT_S = 0.10

SCALE_DEPENDENT = ("stride_length_m", "step_length_m", "step_width_m", "gait_speed_mps",
                   "com_vertical_osc_m", "foot_clearance_m")


@dataclass
class GaitSignature:
    """Timing metrics first — those are the trustworthy ones."""

    n_steps: int = 0
    cadence_hz: float = 0.0            # steps per second
    step_time_mean_s: float = 0.0
    step_time_cv: float = 0.0          # variability; a personal trait, not noise
    duty_factor: float = 0.0           # fraction of the cycle each foot is loaded
    stance_time_s: float = 0.0
    swing_time_s: float = 0.0
    symmetry_index: float = 0.0        # 0 = perfectly symmetric left/right step times
    double_support_frac: float = 0.0
    arm_swing_amp_rad: float = 0.0
    arm_leg_phase_rad: float = 0.0     # arm-leg counter-swing coupling
    torso_pitch_mean_rad: float = 0.0
    torso_pitch_std_rad: float = 0.0

    # --- distance metrics: contaminated by focal-length error, kept for completeness ---
    stride_length_m: float = 0.0
    step_length_m: float = 0.0
    step_width_m: float = 0.0
    gait_speed_mps: float = 0.0
    com_vertical_osc_m: float = 0.0
    foot_clearance_m: float = 0.0

    scale_dependent: tuple[str, ...] = field(default=SCALE_DEPENDENT)

    def timing_only(self) -> dict[str, float]:
        """The subset safe to make claims about."""
        return {k: v for k, v in asdict(self).items()
                if k not in SCALE_DEPENDENT and isinstance(v, (int, float))}


def detect_contact(foot_pos: np.ndarray, fps: float, *,
                   speed_frac: float = 0.25, height_margin: float = 0.06) -> np.ndarray:
    """Per-frame contact mask from foot *motion*, not from a height threshold alone.

    Height alone is not usable here and tuning it is worse than useless. The G1's ankle-roll link
    sits 3-5 cm above the floor when the sole is flat and swings up to 0.44 m, with no clean gap
    between stance and swing in the height distribution. Picking a threshold until the duty factor
    reads its textbook 0.60 would be choosing the answer and then reporting it as a measurement.

    What actually defines stance is that a loaded foot does not translate. So: a foot is in
    contact when its horizontal speed is well below the body's forward speed **and** it is in the
    lower part of its own height range. The speed criterion is self-normalising -- it needs no
    absolute scale, which also makes it immune to the focal-length error.
    """
    horiz = np.zeros(len(foot_pos))
    horiz[:-1] = np.linalg.norm(np.diff(foot_pos[:, :2], axis=0), axis=1) * fps
    horiz[-1] = horiz[-2] if len(horiz) > 1 else 0.0

    # Reference speed: the 75th percentile of foot speed approximates swing speed.
    swing_speed = np.percentile(horiz, 75)
    slow = horiz < speed_frac * max(swing_speed, 1e-6)

    z = foot_pos[:, 2]
    low = z < (z.min() + height_margin)
    return slow & low


def _runs(mask: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Contiguous True runs, discarding those shorter than MIN_CONTACT_S."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    min_len = max(1, int(round(MIN_CONTACT_S * fps)))
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def _contact_intervals(foot_z: np.ndarray, fps: float, *, contact_h: float = CONTACT_H
                       ) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where a foot is loaded (height-only; legacy path)."""
    loaded = foot_z < contact_h
    if not loaded.any():
        return []
    edges = np.diff(np.concatenate([[0], loaded.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    min_len = max(1, int(round(MIN_CONTACT_S * fps)))
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def _circular_phase_offset(a: np.ndarray, b: np.ndarray) -> float:
    """Phase lag between two oscillating signals, via cross-spectrum at the dominant frequency."""
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    fa, fb = np.fft.rfft(a), np.fft.rfft(b)
    k = int(np.argmax(np.abs(fa[1:])) + 1)
    return float(np.angle(fa[k] * np.conj(fb[k])))


def compute(body_pos_w: np.ndarray, joint_pos: np.ndarray, fps: float, *,
            foot_idx: tuple[int, int], pelvis_idx: int = 0,
            shoulder_pitch_idx: tuple[int, int] = (15, 22),
            hip_pitch_idx: tuple[int, int] = (0, 6)) -> GaitSignature:
    """Gait signature from body positions and joint angles.

    ``body_pos_w`` is ``(T, nbody, 3)`` and ``joint_pos`` is ``(T, 29)`` in G1 joint order.
    """
    sig = GaitSignature()
    T = len(body_pos_w)
    if T < int(fps):                       # under a second is not a gait
        return sig

    lz = body_pos_w[:, foot_idx[0], 2]
    rz = body_pos_w[:, foot_idx[1], 2]
    l_contact = detect_contact(body_pos_w[:, foot_idx[0]], fps)
    r_contact = detect_contact(body_pos_w[:, foot_idx[1]], fps)
    left = _runs(l_contact, fps)
    right = _runs(r_contact, fps)
    if len(left) < 2 or len(right) < 2:
        return sig

    # --- timing ---------------------------------------------------------------
    strikes = sorted([(a, "L") for a, _ in left] + [(a, "R") for a, _ in right])
    step_times = np.diff([s[0] for s in strikes]) / fps
    step_times = step_times[step_times > MIN_CONTACT_S]
    if step_times.size < 2:
        return sig

    sig.n_steps = int(len(step_times))
    sig.step_time_mean_s = float(step_times.mean())
    sig.cadence_hz = float(1.0 / sig.step_time_mean_s)
    sig.step_time_cv = float(step_times.std() / (step_times.mean() + 1e-9))

    stance = np.array([(b - a) / fps for a, b in left + right])
    sig.stance_time_s = float(stance.mean())
    cycle = 2.0 * sig.step_time_mean_s     # a full stride is two steps
    sig.duty_factor = float(sig.stance_time_s / cycle)
    sig.swing_time_s = float(max(cycle - sig.stance_time_s, 0.0))

    # Symmetry: left steps vs right steps. Genuinely personal — most people are slightly asymmetric.
    l_times = np.array([(b - a) / fps for a, b in left])
    r_times = np.array([(b - a) / fps for a, b in right])
    denom = 0.5 * (l_times.mean() + r_times.mean())
    sig.symmetry_index = float(abs(l_times.mean() - r_times.mean()) / (denom + 1e-9))

    both = l_contact & r_contact
    sig.double_support_frac = float(both.mean())

    # --- upper body -----------------------------------------------------------
    l_sh, r_sh = joint_pos[:, shoulder_pitch_idx[0]], joint_pos[:, shoulder_pitch_idx[1]]
    sig.arm_swing_amp_rad = float(0.5 * ((l_sh.max() - l_sh.min()) + (r_sh.max() - r_sh.min())) / 2)
    sig.arm_leg_phase_rad = _circular_phase_offset(l_sh, joint_pos[:, hip_pitch_idx[0]])

    pelvis_q = body_pos_w[:, pelvis_idx]
    # Torso pitch proxy: waist pitch joint, index 14 in G1 order.
    sig.torso_pitch_mean_rad = float(joint_pos[:, 14].mean())
    sig.torso_pitch_std_rad = float(joint_pos[:, 14].std())

    # --- distances (scale-dependent) -----------------------------------------
    horiz = np.linalg.norm(np.diff(pelvis_q[:, :2], axis=0), axis=1)
    sig.gait_speed_mps = float(horiz.mean() * fps)
    sig.stride_length_m = float(sig.gait_speed_mps * cycle)
    sig.step_length_m = float(sig.gait_speed_mps * sig.step_time_mean_s)
    sig.step_width_m = float(np.abs(
        body_pos_w[:, foot_idx[0], 1] - body_pos_w[:, foot_idx[1], 1]).mean())
    sig.com_vertical_osc_m = float(pelvis_q[:, 2].max() - pelvis_q[:, 2].min())
    sig.foot_clearance_m = float(max(lz.max(), rz.max()))
    return sig


def compare(a: GaitSignature, b: GaitSignature, *, timing_only: bool = True) -> dict[str, dict]:
    """Per-metric comparison. Defaults to the scale-free subset."""
    da = a.timing_only() if timing_only else {k: v for k, v in asdict(a).items()
                                              if isinstance(v, (int, float))}
    db = b.timing_only() if timing_only else {k: v for k, v in asdict(b).items()
                                              if isinstance(v, (int, float))}
    out = {}
    for k in da:
        va, vb = da[k], db.get(k, 0.0)
        out[k] = {"a": va, "b": vb, "abs_diff": va - vb,
                  "pct_diff": 100.0 * (va - vb) / vb if abs(vb) > 1e-9 else float("nan")}
    return out


def step_width_series(body_pos_w: np.ndarray, fps: float, left: int, right: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-step width, by the gait-lab definition. Returns (widths_m, time_s).

    An earlier version of this fitted one straight line to the whole episode and measured
    lateral offset against it. That silently produced garbage on this dataset: the passes are
    out-and-back, so net displacement is ~0.5 m over a ~20 m path and the fitted heading is
    noise. Anything measured against it is noise too.

    The standard definition (Huxham et al. 2006) never needs a global heading. For each foot
    contact, take the two *contralateral* contacts bracketing it in time; the line between them
    is the local line of progression, and the step width is this foot's perpendicular distance
    from it. Purely local, so turning, curved paths and out-and-back passes are all fine.

    Scale-free in practice: both sides are measured by forward kinematics on the same G1 body,
    so the quantity depends on joint angles, not on the camera focal length or on subject size.
    """
    def contacts(idx: int) -> list[tuple[float, np.ndarray]]:
        mask = detect_contact(body_pos_w[:, idx], fps)
        out = []
        for a, b in _runs(mask, fps):
            mid = (a + b) // 2
            out.append((mid / fps, body_pos_w[a:b, idx, :2].mean(axis=0)))
        return out

    lc, rc = contacts(left), contacts(right)
    widths, times = [], []
    for this, other in ((lc, rc), (rc, lc)):
        for t, p in this:
            before = [c for c in other if c[0] < t]
            after = [c for c in other if c[0] > t]
            if not before or not after:
                continue                       # no bracketing pair -> no line of progression
            p0, p1 = before[-1][1], after[0][1]
            seg = p1 - p0
            n = np.linalg.norm(seg)
            if n < 0.05:                       # contralateral foot barely moved; ill-conditioned
                continue
            # perpendicular distance from p to the infinite line through p0,p1.
            # 2-D cross written out: numpy 2 removed the 2-vector form of np.cross.
            d = p - p0
            widths.append(abs(seg[0] * d[1] - seg[1] * d[0]) / n)
            times.append(t)

    order = np.argsort(times)
    return np.asarray(widths)[order], np.asarray(times)[order]


def foot_progression_series(body_pos_w: np.ndarray, body_quat_w: np.ndarray, fps: float,
                            left: int, right: int) -> np.ndarray:
    """Per-step foot progression angle in degrees: how far the foot points off the direction
    of travel. Positive = toe-out ("duck-footed"); ~0 = foot aligned with travel.

    Uses the same local line of progression as :func:`step_width_series`, for the same reason:
    a heading fitted to a whole episode is meaningless when the walk is out-and-back. The
    foot's own axis is the body frame's +x, taken from MuJoCo's ``xquat`` at mid-stance.
    """
    def rot_x(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])  # world x,y of body +x

    def contacts(idx):
        mask = detect_contact(body_pos_w[:, idx], fps)
        out = []
        for a, b in _runs(mask, fps):
            mid = (a + b) // 2
            out.append((mid / fps, body_pos_w[a:b, idx, :2].mean(axis=0), rot_x(body_quat_w[mid, idx])))
        return out

    lc, rc = contacts(left), contacts(right)
    angles = []
    for this, other in ((lc, rc), (rc, lc)):
        for t, _p, axis in this:
            before = [c for c in other if c[0] < t]
            after = [c for c in other if c[0] > t]
            if not before or not after:
                continue
            seg = after[0][1] - before[-1][1]
            if np.linalg.norm(seg) < 0.05:
                continue
            travel = np.arctan2(seg[1], seg[0])
            foot = np.arctan2(axis[1], axis[0])
            angles.append(abs(np.degrees((foot - travel + np.pi) % (2 * np.pi) - np.pi)))
    return np.asarray(angles)


def step_width_facing(joints: np.ndarray, fps: float, left: int, right: int,
                      l_hip: int, r_hip: int) -> np.ndarray:
    """Step width measured against the pelvis's facing direction, not against foot travel.

    :func:`step_width_series` needs the contralateral foot to have moved between contacts to
    define a line of progression. On a treadmill it has not -- successive contacts land in the
    same place -- so that estimator returns nothing at all on datasets like BMLrub.

    Here the sagittal plane is defined by the hip-to-hip axis, and step width is the separation
    of the two feet along that axis at the moment both are loaded. This is the definition
    treadmill gait labs use, and it is valid overground too, which is what lets one estimator
    run on both sides of a comparison.
    """
    lc = detect_contact(joints[:, left], fps)
    rc = detect_contact(joints[:, right], fps)
    both = lc & rc                                   # double support: both feet down
    out = []
    for a, b in _runs(both, fps):
        m = (a + b) // 2
        across = joints[m, r_hip, :2] - joints[m, l_hip, :2]
        n = np.linalg.norm(across)
        if n < 1e-6:
            continue
        across = across / n
        sep = joints[m, left, :2] - joints[m, right, :2]
        out.append(abs(float(sep @ across)))         # component along the hip axis
    return np.asarray(out)
