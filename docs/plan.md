# Build plan and risks

Output of an adversarial completeness review over the stage-by-stage research in
[pipeline.md](pipeline.md). Where this document disagrees with that one, this one wins —
it was written to attack the plan, not to describe it.

---

## Build right-to-left, not left-to-right

The natural instinct is to start at stage 0 (capture) and work forward. Don't. Every stage's
research recommends starting at that stage; follow all of them and you get five half-built
stages that have never touched each other.

Build from **the end you can verify toward the end you can't.** One substitution per step, so
every failure is attributable to exactly one component:

| Step | What you swap in | What's proven when it works |
|---|---|---|
| 1 | Download a LAFAN1 G1 motion → play through a pretrained tracker in MuJoCo | The whole skeleton. Nothing of yours in the loop. **Days, not weeks.** |
| 2 | An AMASS SMPL-X clip → GMR → same tracker | You own the retargeting boundary |
| 3 | Tripod video of you → GVHMR → GMR → same tracker | **A robot doing your motion, from your video** |
| 4 | Hat cam → SMPL → same tracker, filmed simultaneously with step 3 | The research contribution |

**Step 3 is ~80% of the value and is nearly turnkey** — GMR ships `scripts/gvhmr_to_robot.py`.
Step 4 is the interesting part and should be built *on top of* a working step 3, never instead
of it.

Why this ordering matters concretely: the seams between stages are where the research was
weakest. Verification found at least one wrong command at *every* stage boundary — a fabricated
NPZ schema, a `--robot` flag that doesn't exist, an undisclosed WandB dependency, GMR needing
`mjpython` on macOS. Nobody researched the seams; they researched the boxes. Right-to-left
forces you through one seam at a time.

---

## The three fatal risks

### 1. The legs are hallucinated — so the capture rig may contribute almost nothing

Below the waist, everything is a generative sample from a mocap-trained prior conditioned on
head motion. Your step length, stance width, knee flexion, foot placement: **unobservable from
a hat brim.** That's 12 of 29 DoF, and all the balance behaviour the RL tracker cares about.

**Symptom:** your clips are plausible but interchangeable — near-constant cadence and stride
regardless of what you actually did. Swap your head trajectory for a stranger's and the legs
come out nearly identical.

**Cheapest test:** record ego + tripod *simultaneously* on day one, diff ego-derived legs against
tripod-derived legs. One afternoon. Answers the existential question before you invest months.

### 2. The one piece of code that doesn't exist is the piece everything depends on

There is **no turnkey iPhone-video → SMPL-X egocentric pipeline.** EgoAllo is bound to Project
Aria (VRS + closed-loop trajectories + semidense points), outputs SMPL-H rather than SMPL-X, and
its "run on raw video" issue has been open and unanswered for a year. This is not a weekend
shim: it's a loader rewrite, a floor-plane estimator, a lever-arm calibration, a coordinate
conversion, and a body-model hop — against a model that has never seen a ~70° rectilinear camera.

**Symptom:** week five, and you still have not produced a single physically-simulated robot
motion of any kind. That means you started at the hard end.

### 3. Locomanipulation without objects in sim is miming

"Picking things up" and "opening doors" need a mesh, mass, friction, and a pose trajectory per
object, plus contact rewards. Your reconstruction contains no object at all and no postural
compensation for load — so "carry a box" is indistinguishable from waving your arms. Also, a
29-DoF G1 has no fingers.

**Symptom:** the sim looks convincing, then you add a 2 kg box and the policy walks through it.

### Honourable mention: there is no success criterion

Sim-only, no hardware, no benchmark, no held-out evaluation. *"The G1 did something that looks
like me"* is unfalsifiable, and projects like that drift indefinitely. Define the metric before
the capture protocol.

---

## The error budget — and why it changes the priorities

| Source | Magnitude |
|---|---|
| ARKit drift over a short clip | 1–5 cm |
| Uncalibrated neck-to-camera lever arm | ~10 cm (fixable) |
| **Ego-pose reconstruction error** | **12–19 cm** |
| Retargeting scale error | few cm |
| RL tracking error | few cm |

**The ego-pose term swamps everything else by 3×.** Which means the capture craft — locked
exposure, 60 fps, per-frame intrinsics, rolling-shutter characterization — is careful
optimization of the *smallest* term in the budget. For v1: cut all of it, use a free
off-the-shelf logger.

---

## The reframe worth taking seriously

> An egocentric camera is an excellent record of **what you saw** and a poor record of
> **what your body did** — and this plan uses it exclusively for the latter.

Invert it. Get body motion from a **tripod** (which observes your legs), and let the hat cam be
what it's genuinely good for: the visual observation stream a policy conditions on. That keeps
the hat, kills the highest-risk stage, and matches what credible 2026 work in this space actually
converged on — exocentric for motion, egocentric for vision.

The tripod is not a nice-to-have parallel recommendation. It is **ground truth, and it costs
~$25.** Record ego and exo simultaneously from the first session and every capture is permanently
both a dataset and a validation set.

---

## What's missing from the plan entirely

- **A conventions module.** Ten silent-failure seams: quaternion order disagrees at four
  boundaries, up-axis at one, joint order at three, DoF count at one, SMPL-H vs SMPL-X at one.
  *Silent* is the operative word — a swapped `w` trains to a mediocre policy instead of crashing.
  One constants file, suffixed field names, assertions at every boundary, one golden clip that
  round-trips the whole chain in CI.
- **A persistent Linux GPU box** as dev infrastructure, not just training rentals. GMR is
  untested on macOS and needs `mjpython`; HaMeR and EgoAllo hardcode CUDA; mjlab is eval-only on
  Mac; Isaac Lab won't install at all. Five stages pinning different CUDA/Python versions on one
  MacBook is a standing tax.
- **Day-one registration walls.** SMPL-X, SMPL-H, MANO, AMASS, gated HF repos, WandB registry.
  Days of wall-clock latency. Start them immediately; they run in parallel with everything.
- **A question about the target robot.** A 29-DoF biped makes everything a balance problem, and
  balance is driven by the legs you cannot observe. An upper-body-only or fixed-base humanoid
  torso would make arm motion — the part ego *does* see — the entire task. No tool defaults to
  that, so nobody considered it. Worth ten minutes before committing months.

---

## Cut list for v1

Hands. Object interaction and contact. Locomanipulation as a category (do locomotion plus
free-space arm motion). Real hardware. A custom Swift capture app. Per-clip RL training — use
pretrained checkpoints first. The lever-arm calibration ritual, until you've proven ego-pose is
your bottleneck rather than your ceiling. Any ambition to collect hundreds of clips.

**Keep:** the tripod, one motion source, one tracker, one MuJoCo replay, and a split-screen video
as the deliverable.

---

## Realistic effort, one person

| Stage | Weeks | Notes |
|---|---|---|
| Known-good motion → pretrained tracker → MuJoCo | 0.5–1 | Mostly environment setup |
| AMASS SMPL-X → GMR → tracker | 1–2 | Registration walls dominate |
| Tripod → GVHMR → GMR → tracker | 2–3 | Near-turnkey. **The real v1.** |
| Ego rig + ARKit logging + calibration | 2–3 | +1 week if writing Swift |
| Ego video → SMPL | **4–8, high variance** | May never work well. This is the research. |
| Per-clip RL tracking in mjlab | 2–3 | Compute is hours; plumbing is weeks |
| Object locomanipulation | 8+ | Don't |

Satisfying v1 (rows 1–3 plus 6): **6–10 weeks.** The plan as originally shaped: 6+ months with a
real chance of no working artifact.

**GPU cost is nearly irrelevant at every row.** The currency here is integration time. The
research's advice to "spend the budget on the tracking policy" is backwards.

---

## On your own data specifically

Free, already-retargeted G1 corpora exist right now: ~17,771 AMASS clips, 40 LAFAN1 clips,
PHUMA's physics-curated hours, BONES-SEED's ~71K motions. **The marginal training value of
hundreds of hand-captured clips over that is approximately zero.**

Your data's unique value is that it is *yours*. That is a **demo property, not a training
property** — and demos need 3 good clips, not 400.
