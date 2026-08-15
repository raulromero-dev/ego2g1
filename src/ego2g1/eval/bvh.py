"""Minimal BVH reader — enough to get world joint positions out of LAFAN1.

Exists for one reason: step width measured on the retargeted G1 is not a measurement of the
human. The G1 has a fixed pelvis width, and GMR solves joint angles to hit body-frame targets,
so every subject retargeted onto that chassis is pulled toward the robot's own stance. The
same motions read 7.1 cm as SMPL-X and 18.6 cm as G1. Joint *angles* survive retargeting;
a Cartesian width does not.

LAFAN1 ships as BVH on a human skeleton, so parsing it directly puts both sides of the
comparison on real bodies and takes the robot out of the loop entirely.

Output is converted to the repo's conventions: metres, Z-up (BVH is centimetres, Y-up).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class Joint:
    __slots__ = ("name", "offset", "channels", "children", "parent")

    def __init__(self, name: str, parent: "Joint | None"):
        self.name, self.parent = name, parent
        self.offset = np.zeros(3)
        self.channels: list[str] = []
        self.children: list[Joint] = []


def parse(path: Path) -> tuple[list[Joint], np.ndarray, float]:
    """Return (joints in depth-first order, motion frames, frame_time_s)."""
    tokens = Path(path).read_text().split("\n")
    joints: list[Joint] = []
    stack: list[Joint] = []
    i = 0

    while i < len(tokens):
        parts = tokens[i].split()
        i += 1
        if not parts:
            continue
        head = parts[0]
        if head in ("ROOT", "JOINT"):
            j = Joint(parts[1], stack[-1] if stack else None)
            if j.parent:
                j.parent.children.append(j)
            joints.append(j)
            stack.append(j)
        elif head == "End":                      # End Site: no channels, not a real joint
            stack.append(Joint("__end__", stack[-1] if stack else None))
        elif head == "OFFSET":
            stack[-1].offset = np.array([float(x) for x in parts[1:4]])
        elif head == "CHANNELS":
            stack[-1].channels = parts[2:]
        elif head == "}":
            stack.pop()
        elif head == "MOTION":
            n = int(tokens[i].split()[-1]); i += 1
            frame_time = float(tokens[i].split()[-1]); i += 1
            rows = [np.fromstring(tokens[i + k], sep=" ") for k in range(n)]
            return joints, np.array(rows), frame_time

    raise ValueError(f"no MOTION section in {path}")


def forward_kinematics(joints: list[Joint], motion: np.ndarray) -> np.ndarray:
    """World positions, shape (T, n_joints, 3), in metres and Z-up."""
    T = len(motion)
    pos = np.zeros((T, len(joints), 3))
    index = {j: k for k, j in enumerate(joints)}

    col = 0
    plan = []                                    # (joint, position_cols, rotation_cols, order)
    for j in joints:
        pcols = [col + n for n, c in enumerate(j.channels) if c.endswith("position")]
        rcols = [col + n for n, c in enumerate(j.channels) if c.endswith("rotation")]
        order = "".join(c[0].lower() for c in j.channels if c.endswith("rotation"))
        plan.append((j, pcols, rcols, order))
        col += len(j.channels)

    world_R = {}
    for j, pcols, rcols, order in plan:
        k = index[j]
        # BVH rotation channels are listed outermost-first; scipy's intrinsic convention
        # (uppercase) applied in that same order reproduces the standard interpretation.
        R = (Rotation.from_euler(order.upper(), motion[:, rcols], degrees=True).as_matrix()
             if rcols else np.tile(np.eye(3), (T, 1, 1)))
        if j.parent is None:
            world_R[k] = R
            pos[:, k] = motion[:, pcols] if pcols else j.offset
        else:
            p = index[j.parent]
            world_R[k] = world_R[p] @ R
            pos[:, k] = pos[:, p] + np.einsum("tij,j->ti", world_R[p], j.offset)

    pos /= 100.0                                 # centimetres -> metres
    return pos[:, :, [0, 2, 1]]                  # Y-up -> Z-up


def names(joints: list[Joint]) -> list[str]:
    return [j.name for j in joints]
