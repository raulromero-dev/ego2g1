"""Render the SMPL-X body model so the intermediate representation is visible, not abstract.

SMPL-X is what sits between "video of a person" and "joint angles for a robot". It is not a
point cloud and not a skeleton — it is a *parametric* body:

    shape (betas, 10 numbers)  +  pose (per-joint rotations)  ->  mesh (10475 vertices)
                                                               -> joints (55 positions)

That factorisation is the whole reason this pipeline works. Shape is who you are and is constant
across a clip; pose is what you did and changes every frame. Retargeting consumes the *pose*
(rotations, not positions) and uses shape only to get your proportions right.

Rendering here is deliberately dependency-free: orthographic projection plus depth shading,
drawn with PIL. No OpenGL, no offscreen context, no extra install.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

BODY_MODEL_DIR = Path("assets/body_models")
CANVAS = (560, 760)


def _project(verts: np.ndarray, azim_deg: float, canvas=CANVAS, margin: float = 0.90):
    """Orthographic projection about the vertical axis. SMPL-X canonical space is Y-up."""
    a = np.deg2rad(azim_deg)
    rot = np.array([[np.cos(a), 0.0, np.sin(a)],
                    [0.0,      1.0, 0.0],
                    [-np.sin(a), 0.0, np.cos(a)]])
    p = verts @ rot.T

    w, h = canvas
    span = max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min())
    scale = margin * min(w, h) / span
    cx, cy = (p[:, 0].max() + p[:, 0].min()) / 2, (p[:, 1].max() + p[:, 1].min()) / 2

    x = (p[:, 0] - cx) * scale + w / 2
    y = h / 2 - (p[:, 1] - cy) * scale          # image y grows downward
    return np.stack([x, y], 1), p[:, 2]


def _shade(depth: np.ndarray, lo=(38, 44, 54), hi=(196, 214, 232)) -> np.ndarray:
    """Nearer vertices brighter — enough cue to read the body as 3D."""
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    return (np.array(lo)[None, :] + d[:, None] * (np.array(hi) - np.array(lo))[None, :]).astype(np.uint8)


def render_body(verts: np.ndarray, joints: np.ndarray, parents: np.ndarray, azim: float,
                *, canvas=CANVAS, show_skeleton: bool = True, caption: str = "") -> Image.Image:
    img = Image.new("RGB", canvas, (14, 16, 20))
    draw = ImageDraw.Draw(img)

    xy, z = _project(verts, azim, canvas)
    colors = _shade(z)
    order = np.argsort(z)                      # painter's algorithm: far first
    for i in order[::3]:                       # every 3rd vertex is plenty at this size
        x, y = xy[i]
        draw.point((x, y), fill=tuple(int(c) for c in colors[i]))

    if show_skeleton:
        jxy, _ = _project(joints, azim, canvas)
        # Only the 22 body joints; SMPL-X's remaining 33 are hands/face and clutter the view.
        for child in range(1, min(22, len(parents))):
            p = int(parents[child])
            if p < 0:
                continue
            draw.line([tuple(jxy[p]), tuple(jxy[child])], fill=(255, 176, 59), width=3)
        for j in range(min(22, len(jxy))):
            x, y = jxy[j]
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(43, 179, 201), outline=(14, 16, 20))

    if caption:
        draw.text((16, canvas[1] - 26), caption, fill=(150, 160, 175))
    return img


def load_model(gender: str = "neutral"):
    import smplx
    return smplx.create(str(BODY_MODEL_DIR), model_type="smplx", gender=gender,
                        use_pca=False, ext="npz", batch_size=1)


def forward(model, betas: np.ndarray | None = None, body_pose: np.ndarray | None = None):
    kw = {}
    if betas is not None:
        kw["betas"] = torch.tensor(betas, dtype=torch.float32).reshape(1, -1)
    if body_pose is not None:
        kw["body_pose"] = torch.tensor(body_pose, dtype=torch.float32).reshape(1, -1)
    out = model(**kw)
    return (out.vertices[0].detach().numpy(), out.joints[0].detach().numpy())


def write_video(frames: list[Image.Image], out: Path | str, fps: int = 25) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as w:
        for f in frames:
            w.append_data(np.asarray(f))
    return out
