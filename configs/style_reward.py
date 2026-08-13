"""Style-reward overrides for mjlab's G1 velocity task.

The experiment is a one-parameter ablation. mjlab's stock G1 velocity config sets

    cfg.rewards["air_time"].weight = 0.0

so cadence is *unconstrained* in the baseline — whatever step frequency emerges from velocity
tracking, uprightness, posture and the foot penalties is what the robot adopts. Policy A turns
that single term on and points it at the step timing measured from Raul's own walking. Policy B
changes nothing.

That means any cadence difference between the two policies is attributable to one weight and one
target, both derived from data, rather than to a different corpus, reward mix, or budget.

Measured signature (60 spans, see data/50_eval/gait_target.json):

    cadence           1.846 Hz  = 111 steps/min
    step period       0.542 s   -> stride period 1.084 s
    arm swing         0.259 rad
    arm-leg phase    -2.895 rad (~pi, near-perfect counter-swing)
    torso pitch      +0.114 rad, sd 0.032

**On the swing-time window.** `air_time` rewards time with a foot off the ground, i.e. swing
duration. Swing = (1 - duty_factor) x stride period. Our duty-factor measurement is NOT
trustworthy (see eval/gait.py — it needs sole geometry, not the ankle joint), so rather than
assert a single swing time we bracket the physiologically plausible range implied by the cadence
we *do* trust:

    duty 0.55 -> swing 0.49 s
    duty 0.60 -> swing 0.43 s
    duty 0.65 -> swing 0.38 s

Hence a window of [0.38, 0.49] s. Narrower than mjlab's default [0.05, 0.50], which is wide
enough to permit almost any gait.
"""

from __future__ import annotations

# --- measured, from data/50_eval/gait_target.json ---------------------------------
CADENCE_HZ = 1.846
STEP_PERIOD_S = 0.542
STRIDE_PERIOD_S = 2 * STEP_PERIOD_S
ARM_SWING_RAD = 0.259
TORSO_PITCH_RAD = 0.114

#: Bracketed from the trusted cadence across a plausible duty-factor range (see module docstring).
SWING_MIN_S = 0.38
SWING_MAX_S = 0.49

#: Deliberately modest. Style terms compete with velocity tracking (weight 2.0); weighted too
#: heavily, a policy can hit a target cadence by shuffling in place rather than by walking well.
#: Start low, confirm the policy still tracks commands, and raise only if the style effect is
#: invisible.
AIR_TIME_WEIGHT = 0.25


def apply_style(cfg) -> None:
    """Policy A: enable the cadence term and aim it at the measured signature."""
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["air_time"].params["threshold_min"] = SWING_MIN_S
    cfg.rewards["air_time"].params["threshold_max"] = SWING_MAX_S


def apply_baseline(cfg) -> None:
    """Policy B: stock config. Explicit rather than implicit, so the ablation is visible."""
    cfg.rewards["air_time"].weight = 0.0


SUMMARY = f"""
Policy A (style)     air_time weight {AIR_TIME_WEIGHT}, window [{SWING_MIN_S}, {SWING_MAX_S}] s
Policy B (baseline)  air_time weight 0.0  (mjlab stock)
Everything else identical: same task, architecture, seeds, and step budget.
"""
