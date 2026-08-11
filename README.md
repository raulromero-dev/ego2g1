# ego2g1

Turn egocentric video of *me* — shot on a phone strapped under a hat brim — into whole-body
motion for a **Unitree G1**, train a control policy on it in **MuJoCo**, and show, side by side,
how a policy trained on my movement walks and manipulates differently from a stock one.

> Status: **scaffolding.** Nothing works yet. See `docs/plan.md` for the staged build plan
> and `docs/risks.md` for the things most likely to kill this.

## The pipeline

```
   phone on hat            SLAM / VIO           ego-pose model         retargeting
 ┌──────────────┐      ┌───────────────┐      ┌──────────────┐      ┌──────────────┐
 │ RGB video    │      │ 6-DoF head    │      │ SMPL-X body  │      │ G1 joint     │
 │ + IMU        │─────▶│ trajectory    │─────▶│ + hands      │─────▶│ trajectories │
 │ + intrinsics │      │ (metric,      │      │ (global,     │      │ (23/29 DoF,  │
 └──────────────┘      │  gravity-up)  │      │  contacts)   │      │  contacts)   │
    00_raw             └───────────────┘      └──────────────┘      └──────────────┘
                            10_headpose            20_human              30_retarget
                                                                              │
                                          ┌───────────────────────────────────┘
                                          ▼
                              ┌───────────────────────┐      ┌──────────────────────┐
                              │ RL motion-tracking    │─────▶│ baseline vs. mine    │
                              │ policy in MuJoCo/MJX  │      │ metrics + animation  │
                              └───────────────────────┘      └──────────────────────┘
                                     40_policies                     50_eval
```

Each arrow is a stage that can be built, tested, and *looked at* independently. The rule of the
repo: **no stage ships without a visualization that proves it worked.**

## Layout

```
src/ego2g1/
  capture/    recording protocol, phone export parsing, calibration
  pose/       head trajectory + SMPL-X body reconstruction
  retarget/   SMPL-X → G1 joint trajectories
  motion/     motion file I/O, resampling, contact detection, quality filters
  train/      MuJoCo/MJX environments + RL training
  eval/       task suite, gait & style metrics, rollout harness
  viz/        Rerun logging, MuJoCo rendering, comparison plots
configs/      hydra/yaml configs per stage
scripts/      CLI entrypoints, third-party bootstrap
docs/         plan, data schema, risks, decisions
data/         all artifacts, gitignored — see data/README.md
third_party/  upstream repos cloned, not vendored
```

## The deliverable

A comparison across a fixed task suite (walk, turn, speed sweep, carry, reach-and-pick,
walk-while-carrying, push recovery) between:

- **baseline** — a stock G1 policy, and
- **mine** — the same architecture trained on motion retargeted from my own recordings,

reported as gait/style metrics *and* as animation: side-by-side render, ghost overlay,
gait diagrams, joint-angle-vs-phase curves, and a motion embedding showing whether my policy's
gait actually lands near my own.

## License

TBD. Note that SMPL/SMPL-X body models and several upstream repos carry
non-commercial research licenses — check before redistributing anything derived from them.
