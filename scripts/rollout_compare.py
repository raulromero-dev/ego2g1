"""Roll out two trained policies under an IDENTICAL velocity command and measure both.

`mjlab.scripts.play` is a demo script: it samples velocity commands at random, so two runs
get asked to walk in different directions. That is fine for a look, useless for a comparison —
you cannot attribute a gait difference to the reward when the two robots were given different
tasks. Comparing them requires pinning the command, which play.py does not expose.

So this harness does three things play.py cannot:

1. **Fixes the twist command** to the same constant for both policies, so the only difference
   between the rollouts is the network.
2. **Fixes the seed** so initial states match.
3. **Records qpos**, so the same `eval/gait.py` that measured the original footage can measure
   the robots. The video is a by-product; the numbers are the point.

Run from the mjlab checkout so the package resolves:
    cd ~/mjlab && uv run python ~/ego2g1/scripts/rollout_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

POLICIES = {
    "A_style": Path.home() / "ego2g1/data/40_policies/policyA_style_model_4999.pt",
    "B_base": Path.home() / "ego2g1/data/40_policies/policyB_base_model_4999.pt",
}
TASK = "Mjlab-Velocity-Flat-Unitree-G1"
OUT = Path.home() / "ego2g1/data/50_eval/rollouts"

#: The command both policies are held to. Straight ahead at a walking pace, no turn — the
#: condition our own footage was recorded under, so the comparison is like-for-like.
COMMAND = (0.9, 0.0, 0.0)     # lin_vel_x m/s, lin_vel_y m/s, ang_vel_yaw rad/s
STEPS = 600                   # 12 s at 50 Hz
SEED = 7


def build(task_id: str, device: str, num_envs: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = num_envs
    cfg.seed = SEED
    if hasattr(cfg, "sim"):
        cfg.sim.device = device
    return ManagerBasedRlEnv(cfg, device=device)


def force_command(env, cmd) -> bool:
    """Overwrite the sampled twist with a constant. Returns False if the term is not found."""
    for name in ("twist", "base_velocity", "velocity"):
        try:
            term = env.command_manager.get_term(name)
        except Exception:
            continue
        vec = torch.tensor(cmd, dtype=torch.float32, device=env.device)
        term.command[:] = vec.unsqueeze(0).expand_as(term.command)
        return True
    return False


def rollout(name: str, ckpt: Path, device: str = "cpu") -> dict:
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    env = build(TASK, device, 1)
    wrapped = RslRlVecEnvWrapper(env)
    rl_cfg = load_rl_cfg(TASK)
    runner = load_runner_cls(TASK)(wrapped, rl_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=device)

    obs, _ = wrapped.get_observations()
    pinned = force_command(env, COMMAND)
    qpos = []
    with torch.inference_mode():
        for _ in range(STEPS):
            act = policy(obs)
            obs, _, _, _ = wrapped.step(act)
            force_command(env, COMMAND)          # re-pin: the manager resamples on a timer
            qpos.append(env.scene["robot"].data.joint_pos[0].cpu().numpy().copy())
    env.close()
    return {"name": name, "pinned": pinned, "qpos": np.array(qpos)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, ckpt in POLICIES.items():
        if not ckpt.exists():
            print(f"  missing checkpoint: {ckpt}")
            return 1
        print(f"  rolling out {name} ...", flush=True)
        r = rollout(name, ckpt)
        np.save(OUT / f"{name}_qpos.npy", r["qpos"])
        results[name] = {"pinned_command": bool(r["pinned"]), "frames": int(len(r["qpos"]))}
        print(f"    {len(r['qpos'])} frames, command pinned: {r['pinned']}")
    (OUT / "rollout_meta.json").write_text(json.dumps(
        {"command": COMMAND, "steps": STEPS, "seed": SEED, "task": TASK, "runs": results}, indent=1))
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
