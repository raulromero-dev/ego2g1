"""Retarget every usable span to the G1 and gate on physical plausibility."""
import json, numpy as np
from pathlib import Path
from ego2g1 import pipeline as P
from ego2g1.retarget import gmr_runner as gr
from ego2g1.pose import gvhmr_import as gi
from ego2g1.viz.mujoco_playback import load_g1, report_joint_limits
from ego2g1.capture.sessions import load_sessions

sessions = {s.session_id: s for s in load_sessions()}
spans = json.load(open("data/30_retarget/spans.json"))
model, data = load_g1()
out_dir = Path("data/30_retarget/spans"); out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for i, sp in enumerate(spans, 1):
    cid, si, a, b = sp["clip_id"], sp["span"], sp["start"], sp["end"]
    span_id = f"{cid}_s{si}"
    dst = out_dir / f"{span_id}.npz"
    H = sessions[cid[:3]].subject_height_m
    try:
        schema = dict(np.load(f"data/20_human/{cid}.npz", allow_pickle=True))
        sub = P.slice_schema(schema, a, b, suffix=f"_s{si}")
        if dst.exists():
            robot = dict(np.load(dst, allow_pickle=True))
        else:
            robot = gr.retarget(gi.to_smplx_motion(sub), subject_height_m=H, verbose=False)
            robot["span_id"] = span_id; robot["source_clip"] = cid
            robot["src_time_s"] = sub["src_time_s"]
            gr.save(robot, dst)
        q = gr.to_qpos(robot)
        lim = report_joint_limits(model, q)
        rows.append(dict(span_id=span_id, n=len(q),
                         root_z=float(np.median(q[:, 2])),
                         skate=P.foot_skate(sub["joints_pos_m"], float(sub["fps"])),
                         lim=lim["violation_frac_overall"], worst=lim["worst_joint"]))
    except Exception as ex:
        rows.append(dict(span_id=span_id, n=0, root_z=0, skate=9.9, lim=9.9, worst=str(ex)[:40]))
    if i % 20 == 0: print(f"  ...{i}/{len(spans)}", flush=True)

ok = [r for r in rows if r["n"] > 0]
print(f"\n=== RETARGET: {len(ok)}/{len(rows)} spans ===")
print(f"  root_z  median {np.median([r['root_z'] for r in ok]):.3f} m   (G1 stance 0.793)")
print(f"  skate   median {np.median([r['skate'] for r in ok]):.3f} m/s")
print(f"  joint-limit violations median {100*np.median([r['lim'] for r in ok]):.2f}%")
clean = [r for r in ok if r["skate"] < 0.15 and r["lim"] < 0.05]
print(f"  clean spans (skate<0.15, lim<5%): {len(clean)}/{len(ok)}"
      f"   = {sum(r['n'] for r in clean)/30/60:.1f} min")
json.dump(rows, open("data/30_retarget/retarget_report.json", "w"), indent=1)
best = sorted(clean, key=lambda r: r["skate"])[:6]
print("\n  best spans:", ", ".join(r["span_id"] for r in best))
