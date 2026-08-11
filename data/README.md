# data/

Numbered by pipeline stage. Everything here is gitignored — see `docs/data-schema.md`
for the field-level schema of each stage, and `docs/versioning.md` for how these
directories are backed up / versioned.

| Dir | Stage | Produced by | Contains |
|---|---|---|---|
| `00_raw/` | Capture | phone | one dir per recording session: video, ARKit/IMU stream, intrinsics, calibration |
| `10_headpose/` | Head tracking | SLAM / VIO | metric-scale 6-DoF camera→world trajectory per clip, gravity-aligned |
| `20_human/` | Body reconstruction | ego-pose model | SMPL-X motion (betas, root, poses) + hands + contacts |
| `30_retarget/` | Retargeting | IK / retargeter | G1 joint trajectories, root pose, contact flags, residuals |
| `40_policies/` | Training | RL | checkpoints, configs, training curves |
| `50_eval/` | Evaluation | eval harness | rollouts, metrics tables, rendered comparison video |

Naming convention for a session: `YYYY-MM-DD_<location>_<activity>_<NN>/`
e.g. `2026-08-14_kitchen_carry-mug_03/`
