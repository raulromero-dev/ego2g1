"""Render G1 playback beside the source exo video for the same moment."""
import json, subprocess, numpy as np
from pathlib import Path
from ego2g1.retarget import gmr_runner as gr
from ego2g1.viz.mujoco_playback import load_g1, render_qpos
from ego2g1.capture.sessions import load_index, load_sessions

SPANS = ["s01_p004_s0", "s01_p007_s0", "s02_p000_s1", "s01_p014_s0"]
idx = {e.clip_id: e for e in load_index()}
sess = {s.session_id: s for s in load_sessions()}
model, data = load_g1()
out = Path("data/qa/playback"); out.mkdir(parents=True, exist_ok=True)

for sid in SPANS:
    d = dict(np.load(f"data/30_retarget/spans/{sid}.npz", allow_pickle=True))
    q = gr.to_qpos(d)
    fps = float(d["fps"]); dur = len(q) / fps
    src_t0 = float(np.asarray(d["src_time_s"])[0])
    clip_id = str(d["source_clip"])
    e = idx[clip_id]
    robot_mp4 = out / f"{sid}_robot.mp4"
    render_qpos(q, robot_mp4, model=model, data=data, fps=fps, width=480, height=640,
                azimuth=125, elevation=-12, distance=3.4)
    # matching window of the ORIGINAL session video
    src = sess[clip_id[:3]].exo_path
    sbs = out / f"{sid}_sidebyside.mp4"
    subprocess.run([
        "ffmpeg","-v","error","-y",
        "-ss", f"{src_t0:.3f}", "-t", f"{dur:.3f}", "-i", src,
        "-i", str(robot_mp4),
        "-filter_complex",
        "[0:v]scale=-2:640,setpts=PTS-STARTPTS[a];[1:v]scale=-2:640,setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2",
        "-an","-r",str(int(fps)), str(sbs)], check=True)
    print(f"  {sid}: {len(q)} frames, {dur:.1f}s  src t={src_t0:.1f}s -> {sbs.name}", flush=True)
