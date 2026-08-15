# ego2g1

Turn egocentric video of *me* — shot on a phone strapped under a hat brim — into whole-body
motion for a **Unitree G1**, train a control policy on it in **MuJoCo**, and show, side by side,
how a policy trained on my movement walks and manipulates differently from a stock one.

> Status: **working end to end.** 72 clips of footage became 93 retargeted G1 motions, two
> policies were trained on them, and both were measured. `site/` is the write-up.

## What it found

Measured on human skeletons (my SMPL-X against LAFAN1's BVH), with the same code on both
sides and only steady forward walking kept:

| | me | mocap actors |
|---|---|---|
| step width, at the **ankle** | **6.6 cm** | 15.4 cm |
| step width, at the **forefoot** | 18.4 cm | 17.7 cm |
| foot progression (toe-out) | **23.9°** | 8.0° |
| swing time | 0.43 s | — |

I walk with my heels close together and my toes turned well out. Measuring at the ankle alone
says I plant *narrow*; measuring at the forefoot says I plant the same as anyone. Both are true,
and which landmark you pick decides the answer.

Of that, the trained policy inherited the **timing** and not the **geometry**: 0.42 s of swing
against my 0.43 s, but 25–28 cm between the heels, which is roughly where the G1 stands on its
own. Retargeting preserves joint angles; it does not preserve where the feet go.

**Caveats that matter.** The comparison group is five LAFAN1 performers, not a population, and
their 15.4 cm is wide against the biomechanics literature — so "narrower than these actors" is
supportable and "narrower than most people" is not. My side is estimated from a single camera
and theirs is measured by a mocap rig; the 2-D check in `scripts/measure_width_2d.py` exists to
show the estimate is not distorted (0.499 hip-widths off raw pixels against 0.562 from SMPL-X).

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

## The showcase

`site/` is a static page telling the story, with every number in it read from
`data/50_eval/`. Serve it locally with:

```bash
cd site/public && python3 -m http.server 8899
```

`site/public/index.html` is the write-up. `site/public/film.html` is the same material as a
self-advancing film, one scene per section.

## Licence, and what is *not* in this repo

The **code** is MIT (see `LICENSE`). The **data is not mine to redistribute**, and none of it
is committed:

| | licence | how to get it |
|---|---|---|
| SMPL-X body model | non-commercial research | register at `smpl-x.is.tue.mpg.de` |
| LAFAN1 | Ubisoft licence | `github.com/ubisoft/ubisoft-laforge-animation-dataset` |
| AMASS | non-commercial research | register at `amass.is.tue.mpg.de` |
| GVHMR / GMR / mjlab | see each upstream repo | `scripts/bootstrap_third_party.sh` |

`.gitignore` excludes `data/**`, `third_party/*`, and every `*.npz` / `*.pkl` / `*.pt`, so
body models and motion capture stay out by construction rather than by care.

The footage of me, and the photographs in `site/`, are mine and are shared as part of the
write-up.
