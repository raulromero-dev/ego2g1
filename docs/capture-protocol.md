# Capture protocol — paired ego + exo

Run this for every session. It takes ~15 minutes of setup and about 20 minutes of recording.

**Why paired:** the tripod sees your legs, the hat sees what you see. The tripod is ground truth
for body motion; the hat is the visual stream plus a cross-check on gait. Recorded together, every
session is permanently *both* a dataset and a validation set. See
[plan.md](plan.md#the-reframe-worth-taking-seriously).

---

## Gear

| Role | Device | Notes |
|---|---|---|
| **Exo** (body motion) | **Two** phones/cameras on tripods | See below — two is the validated number, and one is materially worse |
| **Ego** (vision + head motion) | iPhone 13 on the hat | Plain Camera.app is acceptable now — see below |
| Sync | Your hands and legs | A clap **and** a jump. No hardware needed. |
| Scale | Tape measure | Your height, each tripod's lens height, distance to your walking line |

### Use two exo cameras, not one

This is the highest-value change in this document, and it's verified: OpenCap's validation found
that kinematic estimates **"did not substantially improve when using more than two cameras"** —
but going from two down to one costs you exactly the parameters this project is about.

| Parameter | 1 camera | Multi-camera |
|---|---|---|
| Step length (ICC) | 0.947 | 0.977 |
| Gait speed (ICC) | 0.87 | 0.993 |
| **Stride length (ICC)** | **0.658** | **0.981** |
| **Swing time (ICC)** | **0.722** | **0.922** |
| Ankle sagittal RMSE | 15.23° | 4.36° |

One camera gets you cadence, step length, and gait speed. **Two** gets you stride length, swing
time, and frontal-plane angles — which is where individual walking *style* actually lives. Given
your whole thesis is "the robot moves like me," the second camera is not optional.

**Geometry (OpenCap's validated setup):** two cameras at **±45° apart**, **~3 m** from you,
**hip height**. Joint-angle MAE 4.5° across all DoFs. A second phone on a cheap tripod, or one
phone plus a laptop propped at hip height, is enough.

**On the ego camera:** an ARKit logger app is still strictly better and I'd take it if it's free
to do. But it is no longer load-bearing — the tripods carry body motion now. Don't let the rig
block you from recording.

---

## Setup, once per session

1. **Tripods.** Both static, **±45° apart, ~3 m out, at hip height**. Frame so you fill roughly
   60–80% of frame height with **your feet and ~1 m of floor visible** in both. Feet in frame is
   non-negotiable. Never pan or move a camera mid-session — a locked-down camera lets the
   reconstruction skip visual-odometry entirely (GVHMR's `-s` flag), which removes the single
   largest error source and is roughly a **2× accuracy win** for free.
2. **Measure and record** your height, each tripod's lens height off the floor, and the distance
   from each tripod to your walking line. These give you metric scale for free.
3. **Hat.** Mount as rigidly as you can and **do not change the orientation between clips.** Your
   first five clips came out with three different rotation values, which means the phone sat
   differently each time — that alone will look like a change in your posture.
4. **Lock the ego phone's orientation** (Control Centre → rotation lock) so iOS doesn't reorient
   mid-session.
5. **Lighting.** Bright, even, indoors or open shade. Motion blur is the top image-quality killer
   on a head cam, and short exposure needs light.

---

## Per-clip ritual

Both cameras rolling for all of this. Every clip, no exceptions — it's ~15 seconds.

1. **Clap once, sharply, with arms extended in front of you.** Audible on all recordings, visible
   in both tripod views. Audio sync point.
2. **One clear vertical jump.** This is the *visual* sync point, and it matters more than the clap:
   the standard software sync methods (OpenCap, Pose2Sim) cross-correlate **keypoint vertical
   velocity** across cameras, and they work best with an unambiguous vertical movement. The clap
   can fail — a phone with poor audio, wind, a noisy room. The jump is visible to every camera
   including the hat.
3. **A-pose, hold 3 s.** Arms slightly out, feet shoulder-width, still. Gives the body-shape fit a
   clean frame and a zero-velocity anchor.
4. **Look down at the floor, 2 s.** Gives the ego view a floor plane.
5. **Do the activity.**
6. **Clap and jump again at the end.** Bracketing lets you detect clock drift between cameras
   rather than assuming it away.

---

## Shot list

Record **3 takes of each**. Aim for 30–60 s per take. Prioritize top to bottom — the first three
are worth more than everything below them combined.

| Activity | Why it matters |
|---|---|
| **Walk straight, normal pace**, ~8–10 m, turn, walk back | The core gait signature. Most of your value is here. |
| **Walk slow / walk fast** | Cadence and stride vs. speed — this is what makes your gait *yours* rather than one operating point |
| **Stand still, 20 s** | Baseline. Also the cheapest possible policy comparison. |
| Turn in place, both directions | Yaw behaviour |
| Start and stop, several times | Transitions — where generic policies look most robotic |
| Walk carrying something two-handed | Postural compensation under load |
| Reach and pick up an object from a table | Free-space arm motion |
| Walk while carrying | The locomanipulation case, such as it is in v1 |

**Target: ~25 clips. Aim for 40+ minutes of walking across sessions**, not 20.

I under-called this earlier. There is a 2026 result (LIMMT) showing that **3% of AMASS — under
730 clips, under one hour — matches or exceeds full-AMASS tracking performance on a G1.** Curated
minutes are worth far more than raw hours, which means ~40 minutes of *your* well-shot locomotion
is not a rounding error next to a general corpus. It's a real training signal.

So: still not "record for days," but don't stop at 20 minutes either. Quality and variety of
walking beat volume.

---

## Naming and manifest

```
data/00_raw/2026-08-16_livingroom_01/
  ego.mov
  exo_a.mov
  exo_b.mov
  session.json
```

`session.json`:

```json
{
  "session_id": "2026-08-16_livingroom_01",
  "subject_height_m": 1.78,
  "ego":   {"device": "iPhone 13", "app": "Camera.app", "mount": "hat brim, rigid plate"},
  "exo_a": {"device": "...", "lens_height_m": 1.10, "dist_to_walkline_m": 3.0, "angle_deg": -45},
  "exo_b": {"device": "...", "lens_height_m": 1.10, "dist_to_walkline_m": 3.0, "angle_deg": 45},
  "static_cameras": true,
  "location": "living room",
  "notes": "..."
}
```

`static_cameras: true` is what lets the reconstruction skip visual odometry later. Set it honestly
— if you bumped a tripod, say so.

One directory per session, one JSON beside the video, clip segmentation happens later in code.
Don't hand-trim clips — record continuously per activity and cut programmatically at the claps.

---

## The one check to run immediately after your first session

Before recording 25 clips, record **three** and verify:

1. All three videos cover the same time span, and the clap and jump are findable in each.
2. Your **feet are in frame for the entire walking take** in **both** tripod views — this is the single
   most common way a session becomes worthless.
3. The ego view actually shows your hands during the reach-and-pick take.

Then go record the rest.
