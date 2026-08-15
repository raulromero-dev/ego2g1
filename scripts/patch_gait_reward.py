"""Make the style reward actually do something, and stop over-rewarding velocity.

Run 1 produced two policies that were the same experiment twice: A finished at 69.57 mean
reward and B at 69.63, and `Episode_Reward/air_time` was 0.0003. Two separate faults caused
that, and fixing only one of them would not have helped.

FAULT 1 -- the term was switched off.
    config/g1/env_cfgs.py:  cfg.rewards["air_time"].weight = 0.0
Every other reward carried a real weight. The one term encoding this subject's cadence was
explicitly zeroed in the G1 preset.

FAULT 2 -- the reward has the wrong SHAPE, so raising the weight alone would not help.
`mdp.feet_air_time` is a flat band counter:
    in_range = (air_time > 0.05) & (air_time < 0.5);  reward = sum(in_range)
The robot settled at an air time of 0.097 s, which is already inside that band, so it was
collecting full marks for the very gait we wanted to change. Turning the weight up would have
rewarded the fast shuffle harder. The band has to become a target.

The replacement is a Gaussian peaked at the measured swing time. At the current 0.097 s it
returns exp(-(0.097-0.43)^2 / (2*0.18^2)) = 0.18 -- small but non-zero, which is the point:
there is a gradient to climb from where the policy actually is. A narrow band would have been
flat at zero out there and taught it nothing.

TARGET: 0.43 s, derived from cadence rather than measured swing directly. Measuring swing on
the retargeted motion gives 0.567 s, but with a duty factor of 0.469 -- below 0.5, which is
impossible for walking and means the contact detector is merging adjacent swings. Cadence is
the timing metric that has held up throughout: 1.846 steps/s -> 0.542 s per step -> 1.083 s
stride, and a walking swing fraction of 0.40 gives 0.43 s.

Idempotent: re-running is safe, and each file is backed up once.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "mjlab/src/mjlab/tasks/velocity"

#: Measured swing time to aim for, and how forgiving the reward is around it.
TARGET_AIR_S = 0.43
TARGET_STD_S = 0.18
AIR_TIME_WEIGHT = 1.5

#: Velocity tracking dominated the objective at 2.0 + 2.0 against everything else. Lowering
#: it leaves room for the style term to matter without letting the robot stop walking.
TRACK_LIN_WEIGHT = 1.0
TRACK_ANG_WEIGHT = 0.75

NEW_REWARD = '''

def feet_air_time_target(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target: float = 0.43,
  std: float = 0.18,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward a swing time close to a measured target, rather than merely inside a band.

  `feet_air_time` returns 1 per foot for any air time in [0.05, 0.5]. That is flat: a 0.1 s
  shuffle scores exactly as well as a 0.45 s human stride, so it cannot move the gait. This
  peaks at `target` and falls off smoothly, so there is a gradient pointing at the target
  from wherever the policy currently is.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None

  in_air = current_air_time > 0
  closeness = torch.exp(-((current_air_time - target) ** 2) / (2.0 * std**2))
  reward = torch.sum(closeness * in_air.float(), dim=1)

  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  env.extras["log"]["Metrics/air_time_target"] = torch.tensor(target)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward *= (total_command > command_threshold).float()
  return reward
'''


def backup(p: Path) -> None:
    b = p.with_suffix(p.suffix + ".prestyle")
    if not b.exists():
        shutil.copy(p, b)


def main() -> int:
    rewards = ROOT / "mdp/rewards.py"
    envcfg = ROOT / "velocity_env_cfg.py"
    g1cfg = ROOT / "config/g1/env_cfgs.py"
    for p in (rewards, envcfg, g1cfg):
        if not p.exists():
            print(f"  MISSING {p}")
            return 1
        backup(p)

    # 1. add the shaped reward next to the flat one
    txt = rewards.read_text()
    if "def feet_air_time_target(" not in txt:
        anchor = "def feet_air_time("
        i = txt.index(anchor)
        j = txt.index("\ndef ", i + 1)
        txt = txt[:j] + NEW_REWARD + txt[j:]
        rewards.write_text(txt)
        print("  rewards.py: added feet_air_time_target")
    else:
        print("  rewards.py: feet_air_time_target already present")

    # 2. point the term at it, with the measured target, and calm the velocity terms
    txt = envcfg.read_text()
    txt = txt.replace(
        '''    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),''',
        f'''    "air_time": RewardTermCfg(
      func=mdp.feet_air_time_target,
      weight=0.0,  # Override per-robot.
      params={{
        "sensor_name": "feet_ground_contact",
        "target": {TARGET_AIR_S},
        "std": {TARGET_STD_S},
        "command_name": "twist",
        "command_threshold": 0.5,
      }},
    ),''')
    txt = txt.replace(
        '''      func=mdp.track_linear_velocity,
      weight=2.0,''',
        f'''      func=mdp.track_linear_velocity,
      weight={TRACK_LIN_WEIGHT},''')
    txt = txt.replace(
        '''      func=mdp.track_angular_velocity,
      weight=2.0,''',
        f'''      func=mdp.track_angular_velocity,
      weight={TRACK_ANG_WEIGHT},''')
    envcfg.write_text(txt)
    print(f"  velocity_env_cfg.py: air_time -> target reward; "
          f"track_lin {TRACK_LIN_WEIGHT}, track_ang {TRACK_ANG_WEIGHT}")

    # 3. the actual bug: the G1 preset zeroed it
    txt = g1cfg.read_text()
    if 'cfg.rewards["air_time"].weight = 0.0' in txt:
        txt = txt.replace('cfg.rewards["air_time"].weight = 0.0',
                          f'cfg.rewards["air_time"].weight = {AIR_TIME_WEIGHT}')
        g1cfg.write_text(txt)
        print(f"  config/g1/env_cfgs.py: air_time weight 0.0 -> {AIR_TIME_WEIGHT}")
    else:
        print("  config/g1/env_cfgs.py: weight already non-zero")

    # 4. verify
    print("\n  resulting config:")
    for line in envcfg.read_text().splitlines():
        if "feet_air_time_target" in line or '"target"' in line or '"std":' in line:
            print(f"    {line.strip()[:74]}")
    for line in g1cfg.read_text().splitlines():
        if 'rewards["air_time"].weight' in line:
            print(f"    {line.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
