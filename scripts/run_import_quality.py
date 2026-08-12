import numpy as np, os, json
from pathlib import Path
from ego2g1 import pipeline as P
from ego2g1.pose import gvhmr_import as gi
from ego2g1.retarget import gmr_runner as gr
from ego2g1.capture.sessions import load_index, load_sessions

sessions={s.session_id:s for s in load_sessions()}
entries=[e for e in load_index() if (P.GVHMR_RAW/e.clip_id/"hmr4d_results.pt").exists()]
print(f"{len(entries)} clips with GVHMR output\n", flush=True)

rows=[]; spans_out=[]
for i,e in enumerate(entries,1):
    H=sessions[e.session_id].subject_height_m
    try:
        npz=P.HUMAN_DIR/f"{e.clip_id}.npz"
        if npz.exists():
            schema=dict(np.load(npz, allow_pickle=True))
        else:
            schema,rep=gi.convert(gi.load_gvhmr(P.GVHMR_RAW/e.clip_id/"hmr4d_results.pt"),
                clip_id=e.clip_id, subject_height_m=H, src_start_s=e.exo_start_s, fps=e.fps)
            gi.save(schema,npz)
        fps=float(schema["fps"]); n=int(schema["n_frames"])
        ok=P.frame_quality(schema,H); spans=P.good_spans(ok,fps)
        kept=sum(b-a for a,b in spans)
        rows.append((e.clip_id,n,float(ok.mean()),len(spans),kept/fps))
        for si,(a,b) in enumerate(spans):
            spans_out.append((e.clip_id,si,a,b,(b-a)/fps))
    except Exception as ex:
        rows.append((e.clip_id,0,0.0,0,0.0)); print(f"  FAIL {e.clip_id}: {ex}", flush=True)
    if i%12==0: print(f"  ...{i}/{len(entries)}", flush=True)

tot_f=sum(r[1] for r in rows); tot_k=sum(r[4] for r in rows)
print(f"\n=== IMPORT + QUALITY over {len(rows)} clips ===")
print(f"  input        {tot_f/30/60:.1f} min")
print(f"  usable spans {len(spans_out)} spans, {tot_k/60:.1f} min ({100*tot_k*30/max(tot_f,1):.0f}% of frames)")
print(f"  clips contributing: {sum(1 for r in rows if r[3]>0)}/{len(rows)}")
by={}
for cid,si,a,b,d in spans_out: by.setdefault(cid[:3],[0,0.0]); by[cid[:3]][0]+=1; by[cid[:3]][1]+=d
for k in sorted(by): print(f"    {k}: {by[k][0]:3d} spans, {by[k][1]/60:.1f} min")
json.dump([{"clip_id":c,"span":s,"start":a,"end":b,"dur_s":d} for c,s,a,b,d in spans_out],
          open("data/30_retarget/spans.json","w"), indent=1)
print(f"\n  wrote data/30_retarget/spans.json")
