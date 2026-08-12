"""All retargeted spans -> mjlab reference-motion NPZs at 50 Hz."""
import json, numpy as np
from pathlib import Path
from ego2g1.motion import reference as R
from ego2g1.retarget import gmr_runner as gr
from ego2g1.viz.mujoco_playback import load_g1

model, data = load_g1()
src = sorted(Path("data/30_retarget/spans").glob("*.npz"))
out_dir = Path("data/35_reference/mine"); out_dir.mkdir(parents=True, exist_ok=True)
rows, total_s = [], 0.0
for i, p in enumerate(src, 1):
    d = dict(np.load(p, allow_pickle=True))
    ref = R.build_reference(gr.to_qpos(d), float(d["fps"]), model=model, data=data)
    R.save(ref, out_dir / p.name)
    s = R.sanity(ref); s["span_id"] = p.stem
    rows.append(s); total_s += s["duration_s"]
    if i % 25 == 0: print(f"  ...{i}/{len(src)}", flush=True)

jv = np.array([r["joint_vel_max"] for r in rows])
rs = np.array([r["root_speed_max"] for r in rows])
rz = np.array([r["root_z_med"] for r in rows])
print(f"\n=== {len(rows)} reference motions @ 50 Hz, {total_s/60:.1f} min ===")
print(f"  joint_vel max : median {np.median(jv):5.2f}  p90 {np.percentile(jv,90):5.2f}  max {jv.max():5.2f} rad/s")
print(f"  root speed max: median {np.median(rs):5.2f}  p90 {np.percentile(rs,90):5.2f}  max {rs.max():5.2f} m/s")
print(f"  root z median : {np.median(rz):.3f} m   (G1 stance 0.793)")
# implausible motion is unlearnable; flag it now rather than after a training run
bad = [r["span_id"] for r in rows if r["joint_vel_max"] > 25 or r["root_speed_max"] > 4.0]
print(f"  flagged implausible (joint_vel>25 rad/s or root>4 m/s): {len(bad)}")
if bad: print("   ", ", ".join(bad[:8]))
json.dump(rows, open("data/35_reference/mine_report.json","w"), indent=1)
