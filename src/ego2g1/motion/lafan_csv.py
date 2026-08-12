"""Export retargeted G1 motion as LAFAN1-format CSV, so mjlab's own tooling does the rest.

Rather than hand-build the NPZ that mjlab trains from, we emit the same 36-column CSV that the
LAFAN1-retargeted-to-G1 corpus uses and run it through mjlab's ``csv_to_npz.py``. That inherits
their conventions instead of reproducing them from documentation, and it means our clips and the
general corpus pass through identical preprocessing — which matters when the whole experiment is
a controlled comparison between the two.

**The CSV is XYZW.** Verified directly against ``walk1_subject1.csv``: the dominant component of
columns 3-6 is index 3 (mean |c6| = 0.634 vs |c3| = 0.045), i.e. the scalar part is last. Our
internal convention is WXYZ everywhere, so this module is the single boundary where the swap
happens — and it is the only place in the codebase permitted to write XYZW.

Column layout, matching LAFAN1 exactly::

    0..2    root position xyz, metres
    3..6    root quaternion XYZW
    7..35   29 joint angles, radians, in G1 joint order
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ego2g1 import conventions as C

N_COLS = 36
#: LAFAN1 is distributed at 30 fps; mjlab's csv_to_npz resamples to the control rate.
LAFAN_FPS = 30.0


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """The one sanctioned convention swap in this codebase."""
    q = np.asarray(quat_wxyz)
    C.assert_quat_wxyz(q, "quat before XYZW export")
    return q[:, [1, 2, 3, 0]]


def motion_to_rows(root_pos_m: np.ndarray, root_quat_wxyz: np.ndarray,
                   joint_pos_rad: np.ndarray) -> np.ndarray:
    """Assemble the (T, 36) array LAFAN1 CSVs contain."""
    root_pos_m = np.asarray(root_pos_m, dtype=np.float64)
    joint_pos_rad = np.asarray(joint_pos_rad, dtype=np.float64)
    if joint_pos_rad.shape[1] != C.G1_NU:
        raise AssertionError(f"expected {C.G1_NU} joints, got {joint_pos_rad.shape[1]}")

    rows = np.concatenate([root_pos_m, wxyz_to_xyzw(root_quat_wxyz), joint_pos_rad], axis=1)
    if rows.shape[1] != N_COLS:
        raise AssertionError(f"expected {N_COLS} columns, got {rows.shape[1]}")
    return rows


def write_csv(rows: np.ndarray, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow([f"{v:.6f}" for v in r])
    return path


def verify_against_lafan(our_csv: Path | str, lafan_csv: Path | str) -> dict:
    """Compare our export to a reference LAFAN1 file on the properties that must match."""
    ours = np.loadtxt(our_csv, delimiter=",")
    theirs = np.loadtxt(lafan_csv, delimiter=",", max_rows=2000)
    out = {
        "our_shape": ours.shape,
        "cols_match": ours.shape[1] == theirs.shape[1] == N_COLS,
        "our_quat_norm": (float(np.linalg.norm(ours[:, 3:7], axis=1).min()),
                          float(np.linalg.norm(ours[:, 3:7], axis=1).max())),
        # Both must place the scalar component last; if ours lands elsewhere the swap is wrong.
        "our_dominant_quat_idx": int(np.argmax(np.abs(ours[:, 3:7]).mean(0))),
        "their_dominant_quat_idx": int(np.argmax(np.abs(theirs[:, 3:7]).mean(0))),
        "our_root_z_med": float(np.median(ours[:, 2])),
        "their_root_z_med": float(np.median(theirs[:, 2])),
    }
    out["quat_order_matches"] = out["our_dominant_quat_idx"] == out["their_dominant_quat_idx"]
    return out
