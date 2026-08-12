"""Batch driver: GVHMR results -> SMPL schema -> G1 motion -> rendered playback.

One entry point so the whole back half of the pipeline is a single command with a single report.
Every stage is independently re-runnable and skips work already on disk, because the expensive
part (GVHMR) is upstream and nothing here should ever force a re-run of it.

Failures are collected rather than raised. One bad clip out of 72 should not abort a batch —
it should be recorded, excluded, and visible in the summary.
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ego2g1.capture.sessions import ClipEntry, load_index, load_sessions

GVHMR_RAW = Path("data/20_human/raw_gvhmr")
HUMAN_DIR = Path("data/20_human")
RETARGET_DIR = Path("data/30_retarget")
PLAYBACK_DIR = Path("data/qa/playback")


@dataclass
class ClipResult:
    clip_id: str
    stage: str = "pending"
    ok: bool = False
    n_frames: int = 0
    smpl_height_m: float = 0.0
    world_scale: float = 0.0
    root_height_med_m: float = 0.0
    joint_limit_violation_frac: float = 0.0
    worst_joint: str = ""
    foot_skate_m_per_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def foot_skate(joints_pos_m: np.ndarray, fps: float,
               foot_idx=(7, 8, 10, 11), contact_h: float = 0.06) -> float:
    """Mean horizontal foot speed while a foot is close to the ground.

    A planted foot should not translate. This is the single most diagnostic number for whether a
    reference motion is physically learnable — an RL tracker cannot reproduce a foot that slides
    while in contact, and no amount of training fixes it.
    """
    feet = joints_pos_m[:, list(foot_idx), :]
    vel = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) * fps
    contact = feet[:-1, :, 2] < contact_h
    return float(vel[contact].mean()) if contact.any() else 0.0


def run_clip(entry: ClipEntry, subject_height_m: float, *,
             render: bool = False, overwrite: bool = False) -> ClipResult:
    from ego2g1.pose import gvhmr_import as gi
    from ego2g1.retarget import gmr_runner as gr
    from ego2g1.viz.mujoco_playback import load_g1, render_qpos, report_joint_limits

    res = ClipResult(clip_id=entry.clip_id)
    raw = GVHMR_RAW / entry.clip_id / "hmr4d_results.pt"
    human_npz = HUMAN_DIR / f"{entry.clip_id}.npz"
    robot_npz = RETARGET_DIR / f"{entry.clip_id}.npz"

    try:
        # --- import -------------------------------------------------------------
        res.stage = "import"
        if human_npz.exists() and not overwrite:
            schema = dict(np.load(human_npz, allow_pickle=True))
        else:
            if not raw.exists():
                res.error = f"missing GVHMR output: {raw}"
                return res
            schema, report = gi.convert(gi.load_gvhmr(raw), clip_id=entry.clip_id,
                                        subject_height_m=subject_height_m,
                                        src_start_s=entry.exo_start_s, fps=entry.fps)
            gi.save(schema, human_npz)
            res.warnings += report.warnings

        res.n_frames = int(schema["n_frames"])
        res.smpl_height_m = float(schema["smpl_height_m"])
        res.world_scale = float(schema["world_scale_applied"])
        res.foot_skate_m_per_s = foot_skate(schema["joints_pos_m"], float(schema["fps"]))

        # --- retarget -----------------------------------------------------------
        res.stage = "retarget"
        if robot_npz.exists() and not overwrite:
            robot = dict(np.load(robot_npz, allow_pickle=True))
        else:
            robot = gr.retarget(gi.to_smplx_motion(schema),
                                subject_height_m=subject_height_m, verbose=False)
            gr.save(robot, robot_npz)

        res.root_height_med_m = float(np.median(robot["root_pos_m"][:, 2]))

        # --- gate ---------------------------------------------------------------
        res.stage = "gate"
        model, data = load_g1()
        qpos = gr.to_qpos(robot)
        lim = report_joint_limits(model, qpos)
        res.joint_limit_violation_frac = lim["violation_frac_overall"]
        res.worst_joint = lim["worst_joint"]

        if render:
            res.stage = "render"
            render_qpos(qpos, PLAYBACK_DIR / f"{entry.clip_id}.mp4", model=model, data=data)

        res.stage = "done"
        res.ok = True
    except Exception as exc:  # noqa: BLE001 - a bad clip must not abort the batch
        res.error = f"{type(exc).__name__}: {exc}"
        res.warnings.append(traceback.format_exc(limit=3))
    return res


def run_all(*, render_top_n: int = 6, overwrite: bool = False,
            limit: int | None = None) -> list[ClipResult]:
    """Process every indexed clip that has GVHMR output."""
    sessions = {s.session_id: s for s in load_sessions()}
    entries = load_index()
    if limit:
        entries = entries[:limit]

    # Render only the largest-subject clips: renders are slow and mostly redundant, and the
    # close ones are where retargeting quality is actually judgeable.
    render_ids = {e.clip_id for e in
                  sorted(entries, key=lambda e: -e.subj_px_height_med)[:render_top_n]}

    results = []
    for i, entry in enumerate(entries, 1):
        height = sessions[entry.session_id].subject_height_m
        r = run_clip(entry, height, render=entry.clip_id in render_ids, overwrite=overwrite)
        results.append(r)
        flag = "ok " if r.ok else "FAIL"
        print(f"[{i:3d}/{len(entries)}] {flag} {r.clip_id}  {r.n_frames:4d}f  "
              f"scale {r.world_scale:.3f}  rootz {r.root_height_med_m:.3f}  "
              f"skate {r.foot_skate_m_per_s:.3f} m/s  lim {r.joint_limit_violation_frac*100:.1f}%"
              + (f"  {r.error}" if r.error else ""), flush=True)
    return results


def summarize(results: list[ClipResult]) -> dict:
    ok = [r for r in results if r.ok]
    if not ok:
        return {"n_total": len(results), "n_ok": 0, "failures": [r.clip_id for r in results]}
    skate = np.array([r.foot_skate_m_per_s for r in ok])
    lim = np.array([r.joint_limit_violation_frac for r in ok])
    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "failures": [f"{r.clip_id}: {r.error}" for r in results if not r.ok],
        "foot_skate_median": float(np.median(skate)),
        "foot_skate_p90": float(np.percentile(skate, 90)),
        "joint_limit_median": float(np.median(lim)),
        "world_scale_median": float(np.median([r.world_scale for r in ok])),
        "root_height_median": float(np.median([r.root_height_med_m for r in ok])),
        "clean_clips": [r.clip_id for r in ok
                        if r.foot_skate_m_per_s < 0.15 and r.joint_limit_violation_frac < 0.05],
    }
