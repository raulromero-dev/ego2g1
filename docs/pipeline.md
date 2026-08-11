# The end-to-end pipeline

Plain-language walkthrough of how a video of me walking becomes a Unitree G1 that walks like me.
Seven stages. Each one takes a file in and writes a file out, so any stage can be re-run
without redoing the ones before it.

> Researched Aug 2026 with web verification + an adversarial fact-check pass. Where a claim was
> corrected by the fact-checker, this doc states the corrected version.

---

## The one-paragraph version

A head-mounted camera never sees your legs, so nothing in this pipeline "sees" you walking.
Instead we recover the **6-DoF trajectory of your head**, and feed that into a model that has
learned, from thousands of hours of motion capture, what a body most likely did to move a head
that way. The images are used only for your **hands** and for finding the **floor**. That gives
a human skeleton, which we then bend onto the G1's very different body, and finally hand to a
reinforcement-learning policy that learns to physically reproduce it under gravity and contact.

**Head motion is the load-bearing signal.** Everything below the waist is an educated guess.

---

## Stage 0 — Capture

| | |
|---|---|
| **In** | You, a hat, a phone |
| **Out** | `data/00_raw/{session}/` — video + per-frame camera pose + IMU + intrinsics + calibration |
| **Tool** | An ARKit logger app (**not** the stock Camera app) |
| **Time** | A weekend to build the habit |

The single highest-leverage decision in the project. Two very different worlds:

**With ARKit** (a logger app), the phone gives you a **metric-scale, gravity-aligned 6-DoF pose
per frame, for free**. Stage 1 becomes a parsing job.

**Without ARKit** (stock Camera app), you have pixels only. Stage 1 becomes offline monocular
SLAM — the hardest, slowest, most failure-prone step in the whole pipeline, and it cannot
recover **absolute scale**. A 1.8 m room and a 1.8 cm dollhouse produce identical video.

The rig details that actually matter:

- **Rigid mount.** A phone is ~200 g on a long lever arm. Every hat slip becomes a fake head
  translation, which the model faithfully converts into a fake step. A crown/GoPro-style strap
  beats a hat brim.
- **Portrait, pitched 25–35° down.** Puts your hands in frame while keeping walls and horizon
  up top for tracking features. Past ~45° you're staring at textureless floor and tracking dies.
- **Lock exposure ≤ 1/120 s.** Head yaw hits 200–400°/s; at 1/30 s that smears 7–13° across the
  frame. Motion blur is the top killer of both tracking and hand estimation.
- **Calibration ritual, every clip.** Stand still 3 s → point at floor 2 s (gives the ground
  plane; the retargeter needs it) → 15–20 s of look-left/right/up/down with feet planted. That
  last one is how you solve the neck-to-camera offset without a mocap studio: during pure head
  rotation the camera traces a sphere around the neck pivot, so you can least-squares fit the
  offset. Expect roughly 8–12 cm forward, 2–5 cm above eye level.

**Why the offset matters:** every model downstream was trained where "head" means a body-model
head *joint*, not a camera hanging off a brim. Get this wrong and you inject rotation-induced
translation into the one signal the whole pipeline trusts.

---

## Stage 1 — Head trajectory

| | |
|---|---|
| **In** | video (+ IMU if you have it) |
| **Out** | `data/10_headpose/{clip}.parquet` — `T×7` pose per frame (quaternion + position), metric, Z-up |
| **Tool** | ARKit passthrough, **or** monocular SLAM (DPV-SLAM / DPVO / MASt3R-SLAM) |
| **Time** | Minutes with ARKit; days of fighting without it |

Turn the recording into "where was my head, in the room, at every frame."

The honest state of the art: on head-worn footage, **commercial visual-inertial tracking beats
open-source SLAM by a wide margin.** On a large benchmark of long egocentric sequences, Aria's
own SLAM scores ~70–91 while the best open systems score in the single digits to low teens
(OKVIS2 3.6, ORB-SLAM3 mono-inertial 14.2, DPVO 1.2). Head motion is fast, rotation-heavy, and
blurry — the exact regime where classical SLAM struggles.

Two things break this stage silently:

- **Scale.** Monocular is scale-free. You must inject scale from somewhere — measured eye height
  at t=0, a known-size object in frame, or a depth sensor.
- **Stabilization.** Stock-camera EIS warps frames non-rigidly, so apparent image motion no
  longer matches real camera motion. Tracking against stabilized footage is tracking against a lie.

---

## Stage 2 — Body reconstruction

| | |
|---|---|
| **In** | head trajectory + video frames + floor height |
| **Out** | `data/20_human/{clip}.npz` — SMPL-X motion: body shape, root pose, joint rotations, 30 fps |
| **Tool** | **EgoAllo** (+ HaMeR for hands) |
| **Time** | Inference only — a few dollars of GPU. Glue code is the real cost. |

Head trajectory in, full human body out. A diffusion model trained on the AMASS motion-capture
corpus samples the body motion most consistent with how your head moved.

**EgoAllo** is the best open implementation. Set expectations honestly:

- Accuracy is **~12–19 cm per-joint error** in world frame. On a ~1.3 m G1 that's about a limb
  segment. Fine as loose reference motion for an RL tracker to imitate. **Useless as manipulation
  ground truth.**
- **Your legs are never observed.** Stance width, foot placement, knee flexion during a crouch —
  all generative samples from a prior, not measurements. Do not make claims about them.
- Its prior is **staged mocap**, which under-represents exactly your target: crouching to pick
  something up, carrying a box, opening a door mid-stride.
- It expects **Project Aria** data. Feeding it an iPhone recording takes ~200–400 lines of glue:
  a loader producing head poses in its expected frame, and a floor-height estimate.

Hands come from **HaMeR** run on the frames. Its finger articulation is excellent; its world
placement is poor. So take **wrist position from EgoAllo, finger articulation from HaMeR.**

---

## Stage 3 — Retargeting

| | |
|---|---|
| **In** | SMPL-X human motion |
| **Out** | `data/30_retarget/{clip}.pkl` → 36-column CSV in Unitree convention |
| **Tool** | **GMR** (General Motion Retargeting) |
| **Time** | CPU, 35–70 fps per clip — hours of footage retargets overnight on the MacBook |

Bend a human onto a robot that has different proportions, fewer joints, and hard limits.

This stage is close to solved — **use GMR, write no retargeting code yourself.** It does
SMPL-X → G1 29-DoF via inverse kinematics and its output converts in one script to the CSV
format every downstream trainer reads.

Verified G1 ground truth (from parsing the actual URDFs):

- **29 DoF** = 6 per leg (12) + 3 waist + 7 per arm (14)
- **33.341 kg**, pelvis→ankle **0.7633 m**
- Joint order is shared across `unitree_ros`, `mujoco_menagerie`, the LAFAN1 dataset, and GMR

Where it hurts:

- **The G1 is stiffer than you.** Ankle roll is only ±15°, waist roll/pitch only ±30°. Human
  lateral ankle motion and torso twist blow past that. Deep squats and side-steps will saturate
  and come out visibly wrong.
- **No retargeter enforces self-collision.** Arms can pass through the torso. Eyeball every clip.
- **Drop hands for v1.** Use the 29-DoF model, not the 43-DoF with-hands one — GMR's hand config
  is a no-op anyway, and it inflates the RL action space by ~48% for zero tracking signal.

---

## Stage 4 — Reference motion prep

| | |
|---|---|
| **In** | 30 fps CSV |
| **Out** | `data/30_retarget/{clip}.npz` — resampled to 50 fps, with velocities and body poses |
| **Tool** | `mjlab`'s `csv_to_npz.py` |
| **Time** | Seconds per clip |

Mechanical, but it's where the trainer's exact expectations get met: `fps`, `joint_pos (T,29)`,
`joint_vel (T,29)`, `body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`, `body_ang_vel_w`.
Root pose is body 0 of the `body_*` arrays, not a separate field.

**Watch the rendered preview this script emits.** That video catches bad retargeting before you
spend a single GPU-hour. Note the tool wants a Weights & Biases account and a registry named
`motions` — or patch it to stop after `np.savez`.

---

## Stage 5 — Policy training

| | |
|---|---|
| **In** | reference motion NPZ |
| **Out** | `data/40_policies/{run}/` — checkpoint + ONNX |
| **Tool** | **mjlab** (`Mjlab-Tracking-Flat-Unitree-G1`) |
| **Time** | Hours on one rented 4090 for a tracker |

The reference motion is kinematics — poses with no physics. This stage learns a controller that
reproduces it *under gravity, with contact, without falling*.

**There is no "fine-tune the G1 model" in the LLM sense.** What exists:

- **Zero-training playback:** ProtoMotions (Apache-2.0, actively maintained, ships G1 tracker
  checkpoints) and TWIST let you *play* motion through a pretrained tracker today. Best first
  milestone.
- **Train your own tracker:** mjlab, on one GPU, hours not days. This is the real path.
- **Don't touch SONIC.** It's a genuine G1 foundation model, but its own docs recommend 64+ GPUs
  to fine-tune, and it cost ~21,000 GPU-hours to train. Out of scope by 2–3 orders of magnitude.

Two hard constraints: **training needs an NVIDIA GPU** — the MacBook is eval-only. And **garbage
reference motion is unlearnable**: foot skating, joint-limit violations, or a drifting root make
the policy fail, and no amount of RL fixes it. Quality gates belong at stage 3, not here.

---

## Stage 6 — Evaluation and visualization

| | |
|---|---|
| **In** | two or more policies |
| **Out** | `data/50_eval/` — metrics tables + rendered comparisons |
| **Tool** | MuJoCo rendering + Rerun + analysis plots |

The deliverable: **baseline vs. mine**, across walk / turn / speed sweep / carry / reach-and-pick /
push recovery.

Choose the baseline carefully — two obvious ones are rigged:

- Unitree's shipped pretrained policy is **legs-only (12 actions, arms pinned)**. Any arm-swing
  comparison against it is meaningless.
- MuJoCo Playground's joystick task has reward terms that **actively penalize moving the arms**.

The honest comparison is **the same code, same environment, same training budget, differing only
in which motions it learned from** — yours vs. a content-matched set of someone else's.

**Build this early, not last:** kinematically replay your retargeted motion with physics off,
and compute gait metrics on both that *and* the pre-retargeting human motion. The gap between
them is **style destroyed by retargeting** — a hard ceiling on the entire project. If retargeting
flattens your gait signature, no policy recovers it, and you want to know in week 2.

---

## Conventions that will bite you

Write these down, assert them in code at every stage boundary.

| | |
|---|---|
| **World frame** | Right-handed, **Z-up**, gravity `(0,0,-9.81)`. ARKit is **Y-up** — convert once, at load. |
| **Quaternions** | Canonical **WXYZ**. Name every field explicitly: `root_quat_wxyz`, never `root_rot`. |
| **Units** | metres, radians, seconds. No degrees anywhere. |
| **Time** | Carry both `fps` *and* a per-frame timestamp array. Never trust fps alone after resampling. |
| **dtype** | float32 for arrays, float64 for absolute timestamps. |

**The quaternion trap, specifically.** Four conventions collide here. ARKit, Record3D, the LAFAN1
CSV, and Rerun are **XYZW**. MuJoCo, mjlab, and scipy's `scalar_first=True` are **WXYZ**. Worse:
GMR's README claims it outputs wxyz "to align with MuJoCo", but the source saves
`qpos[3:7][[1,2,3,0]]` — which is **XYZW**. This fails silently. Your robot just falls over.

**LeRobot is not the spine here.** You know it from the SO-ARM101, and its G1 support looks
relevant, but its dataset format has no floating-base state — no root position, no root
quaternion — and assumes a single fps for the whole dataset, while this pipeline runs 30 fps
capture, 50 fps reference, and 200 Hz sim. Use it for policy rollout data later if you want;
don't force the motion data into it.

---

## What breaks first, ranked

1. **Head trajectory quality.** Without ARKit, monocular SLAM on head-worn footage is the
   likeliest total blocker. Symptom: reconstructed motion drifts or teleports.
2. **The neck-to-camera offset.** Silently converts head rotation into fake translation.
   Symptom: plausible-looking motion with persistent foot skate.
3. **Frame/quaternion conventions.** Symptom: robot inverted, twisted, or instantly falling.
4. **Not enough data.** Minutes of footage is a plumbing test, not a training corpus.
