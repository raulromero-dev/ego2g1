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
| **Exo** (body motion) | Second phone / camera on a tripod | Must see you **head to feet**, including the floor around your feet |
| **Ego** (vision + head motion) | iPhone 13 on the hat | Plain Camera.app is acceptable now — see below |
| Sync | Your hands | A sharp clap. No hardware needed. |
| Scale | Tape measure | Measure your height once; measure the tripod height once |

**On the ego camera:** an ARKit logger app is still strictly better and I'd take it if it's free
to do. But it is no longer load-bearing — the tripod carries body motion now. Don't let the rig
block you from recording this weekend.

---

## Setup, once per session

1. **Tripod.** Place it so you fill roughly 60–80% of the frame height, with **your feet and ~1 m
   of floor visible at the bottom**. Feet in frame is non-negotiable — that's the whole point.
   Landscape. Static: never pan or move it mid-session.
2. **Measure and record** your height, the tripod lens height off the floor, and the distance from
   tripod to your walking line. These give you metric scale for free.
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

1. **Clap once, sharply, with arms extended in front of you.** Audible on both recordings, visible
   in the tripod view. This is your frame-accurate sync point.
2. **A-pose, hold 3 s.** Arms slightly out, feet shoulder-width, still. Gives the body-shape fit a
   clean frame and a zero-velocity anchor.
3. **Look down at the floor, 2 s.** Gives the ego view a floor plane.
4. **Do the activity.**
5. **Clap again at the end.** Two sync points bracket the clip and let you detect drift between the
   two clocks.

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

**Total: ~25 clips, ~20 minutes of footage.** That is a *style set*, not a training corpus — and
that's the correct target. Physical validity comes from the free retargeted corpora; personality
comes from this.

---

## Naming and manifest

```
data/00_raw/2026-08-16_livingroom_01/
  ego.mov
  exo.mov
  session.json
```

`session.json`:

```json
{
  "session_id": "2026-08-16_livingroom_01",
  "subject_height_m": 1.78,
  "tripod_lens_height_m": 1.10,
  "tripod_to_walkline_m": 4.0,
  "ego_device": "iPhone 13",
  "ego_app": "Camera.app",
  "ego_mount": "hat brim, rigid plate",
  "exo_device": "...",
  "location": "living room",
  "notes": "..."
}
```

One directory per session, one JSON beside the video, clip segmentation happens later in code.
Don't hand-trim clips — record continuously per activity and cut programmatically at the claps.

---

## The one check to run immediately after your first session

Before recording 25 clips, record **three** and verify:

1. Both videos cover the same time span and the claps are findable.
2. Your **feet are in frame for the entire walking take** in the tripod view — this is the single
   most common way a session becomes worthless.
3. The ego view actually shows your hands during the reach-and-pick take.

Then go record the rest.
