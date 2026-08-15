"""Add the two reward terms that carry this subject's foot GEOMETRY, not just their timing.

A2 fixed the cadence and still did not look like him, and the reason is structural: the only
style term was `air_time`, which is purely temporal. Nothing in the reward mentioned where the
feet go, so there was never any pressure to reproduce the thing that actually makes his walk
recognisable -- heels close together with the toes turned out.

Two terms are added here, both driven by his measured medians:

  step_width  ankle separation perpendicular to travel, target 6.6 cm (his), against a
              default G1 stance of 23.7 cm
  toe_out     each foot's yaw away from the direction of travel, target 23.9 deg (his),
              against roughly 8 deg for the mocap actors

Both are feasible on this robot, which was worth checking before rewarding them: hip roll can
bring the ankles to 0.5 cm apart and hip yaw has +/-90 deg of range. The 18.6 cm that came out
of retargeting was GMR's IK preferring a neutral pose, not a kinematic limit, so a policy can
learn what the retargeter would not produce.

Both use a Gaussian rather than a band, for the reason established in patch_gait_reward.py: a
band is flat, so it gives no gradient from wherever the policy currently sits.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "mjlab/src/mjlab/tasks/velocity"

#: His measured medians. See data/50_eval/ -- forefoot.json and width_locomotion.json.
STEP_WIDTH_TARGET_M = 0.066
STEP_WIDTH_STD_M = 0.055
TOE_OUT_TARGET_RAD = 0.417        # 23.9 degrees
TOE_OUT_STD_RAD = 0.30

STEP_WIDTH_WEIGHT = 1.5
TOE_OUT_WEIGHT = 1.0

NEW = '''

def step_width(
  env: ManagerBasedRlEnv,
  target: float = 0.066,
  std: float = 0.055,
  command_name: str | None = None,
  command_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward planting the feet a measured distance apart, across the direction of travel.

  Separation is taken perpendicular to the base's own heading rather than to a world axis, so
  the term means the same thing when the robot is turning. The G1's neutral stance is 23.7 cm
  and the target here is 6.6 cm, so this pulls hard against the default pose -- which is the
  intent, since that gap is the whole signature.
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  sep = feet[:, 0, :] - feet[:, 1, :]  # [B, 2]

  # heading of the base, as a unit vector in the ground plane
  fwd = quat_apply(asset.data.root_link_quat_w, torch.tensor(
    [1.0, 0.0, 0.0], device=sep.device).expand(sep.shape[0], 3))[:, :2]
  fwd = fwd / torch.clamp(torch.norm(fwd, dim=1, keepdim=True), min=1e-6)
  lateral = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=1)  # perpendicular, in-plane

  width = torch.abs(torch.sum(sep * lateral, dim=1))  # [B]
  reward = torch.exp(-((width - target) ** 2) / (2.0 * std**2))
  env.extras["log"]["Metrics/step_width_mean"] = width.mean()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward = reward * (total > command_threshold).float()
  return reward


def toe_out(
  env: ManagerBasedRlEnv,
  target: float = 0.417,
  std: float = 0.30,
  command_name: str | None = None,
  command_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward each foot pointing outward from the line of travel by a measured angle.

  Signed per foot: the left foot should turn left of travel and the right foot right of it, so
  the two are rewarded toward opposite signs. Rewarding |angle| instead would be satisfied by
  both feet splaying the same way, which is a limp rather than a stance.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quats = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, 2, 4]
  B = quats.shape[0]
  x_axis = torch.tensor([1.0, 0.0, 0.0], device=quats.device)

  base_fwd = quat_apply(asset.data.root_link_quat_w, x_axis.expand(B, 3))[:, :2]
  base_yaw = torch.atan2(base_fwd[:, 1], base_fwd[:, 0])

  total = torch.zeros(B, device=quats.device)
  for f in range(quats.shape[1]):
    foot_fwd = quat_apply(quats[:, f, :], x_axis.expand(B, 3))[:, :2]
    foot_yaw = torch.atan2(foot_fwd[:, 1], foot_fwd[:, 0])
    rel = torch.atan2(torch.sin(foot_yaw - base_yaw), torch.cos(foot_yaw - base_yaw))
    want = target if f == 0 else -target      # site order is (left, right)
    total = total + torch.exp(-((rel - want) ** 2) / (2.0 * std**2))
    if f == 0:
      env.extras["log"]["Metrics/toe_out_left_deg"] = torch.rad2deg(rel).mean()

  reward = total / quats.shape[1]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      c = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward = reward * (c > command_threshold).float()
  return reward
'''


def backup(p: Path) -> None:
    b = p.with_suffix(p.suffix + ".pregeom")
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

    txt = rewards.read_text()
    if "quat_apply" not in txt.split("\n\n")[0] and "import" in txt:
        # make sure quat_apply is available; mjlab exposes it alongside quat_apply_inverse
        if "quat_apply," not in txt and "quat_apply " not in txt:
            txt = txt.replace("quat_apply_inverse", "quat_apply, quat_apply_inverse", 1)
    if "def step_width(" not in txt:
        txt = txt.rstrip() + "\n" + NEW
        rewards.write_text(txt)
        print("  rewards.py: added step_width and toe_out")
    else:
        print("  rewards.py: terms already present")

    txt = envcfg.read_text()
    if '"step_width"' not in txt:
        anchor = '    "foot_clearance": RewardTermCfg('
        block = f'''    "step_width": RewardTermCfg(
      func=mdp.step_width,
      weight=0.0,  # Override per-robot.
      params={{
        "target": {STEP_WIDTH_TARGET_M},
        "std": {STEP_WIDTH_STD_M},
        "command_name": "twist",
        "command_threshold": 0.5,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      }},
    ),
    "toe_out": RewardTermCfg(
      func=mdp.toe_out,
      weight=0.0,  # Override per-robot.
      params={{
        "target": {TOE_OUT_TARGET_RAD},
        "std": {TOE_OUT_STD_RAD},
        "command_name": "twist",
        "command_threshold": 0.5,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      }},
    ),
'''
        txt = txt.replace(anchor, block + anchor, 1)
        envcfg.write_text(txt)
        print("  velocity_env_cfg.py: registered both terms")
    else:
        print("  velocity_env_cfg.py: terms already registered")

    txt = g1cfg.read_text()
    if 'rewards["step_width"]' not in txt:
        anchor = '  cfg.rewards["air_time"].weight'
        block = (
            '  cfg.rewards["step_width"].params["asset_cfg"].site_names = site_names\n'
            '  cfg.rewards["toe_out"].params["asset_cfg"].body_names = (\n'
            '    "left_ankle_roll_link", "right_ankle_roll_link",\n'
            '  )\n'
            f'  cfg.rewards["step_width"].weight = {STEP_WIDTH_WEIGHT}\n'
            f'  cfg.rewards["toe_out"].weight = {TOE_OUT_WEIGHT}\n')
        i = txt.index(anchor)
        txt = txt[:i] + block + txt[i:]
        g1cfg.write_text(txt)
        print(f"  config/g1/env_cfgs.py: step_width={STEP_WIDTH_WEIGHT}, "
              f"toe_out={TOE_OUT_WEIGHT}")
    else:
        print("  config/g1/env_cfgs.py: already wired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
