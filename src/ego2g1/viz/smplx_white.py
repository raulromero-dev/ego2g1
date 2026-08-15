"""Render the recovered SMPL-X body on white — no video behind it.

The QA overlays draw the mesh *on the footage*, which is the right check for "did pose
recovery work" and the wrong image for "what does this walk look like". The footage carries
a corridor, a doorway, a camera angle; all of it competes with the body, and none of it is
the measurement. Stripping the background leaves only what the pipeline actually recovered.

Two outputs, because they answer different questions:

  walk   an animated skeleton, one figure, white ground plus a fading trail of past ankles
  chrono chronophotography -- the same walk sampled at footfalls and superimposed in place,
         so a whole pass reads as one still image

Both project orthographically. A perspective camera would make the far leg smaller than the
near leg, and since the thing being measured is *how far apart the feet are*, any projection
that scales with depth would put a camera artefact straight into the quantity of interest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# SMPL-X body joints. Indices are fixed by the model, not by us.
PELVIS, HEAD = 0, 15
L_ANKLE, R_ANKLE = 7, 8
L_FOOT, R_FOOT = 10, 11

#: (parent, child, side) -- side picks the colour so left/right stay tellable apart.
BONES: tuple[tuple[int, int, str], ...] = (
    (0, 3, "c"), (3, 6, "c"), (6, 9, "c"), (9, 12, "c"), (12, 15, "c"),
    (0, 1, "l"), (1, 4, "l"), (4, 7, "l"), (7, 10, "l"),
    (0, 2, "r"), (2, 5, "r"), (5, 8, "r"), (8, 11, "r"),
    (9, 13, "l"), (13, 16, "l"), (16, 18, "l"), (18, 20, "l"),
    (9, 14, "r"), (14, 17, "r"), (17, 19, "r"), (19, 21, "r"),
)

SS = 3  # supersample factor; PIL has no antialiased line, so draw big and downsample


def _travel_axis(joints: np.ndarray) -> int:
    """Which world axis the walk runs along -- 0 or 1, whichever the pelvis covers more of."""
    path = np.abs(np.diff(joints[:, PELVIS, :2], axis=0)).sum(axis=0)
    return int(np.argmax(path))


def _project(joints: np.ndarray, axis: int) -> np.ndarray:
    """World (T,J,3) -> side-view (T,J,2) in metres, x along travel, y up."""
    return np.stack([joints[..., axis], joints[..., 2]], axis=-1)


class _Frame:
    """Metres -> pixels, with the aspect ratio locked so a stride cannot be stretched."""

    def __init__(self, pts: np.ndarray, w: int, h: int, pad: float = 0.35):
        lo = pts.reshape(-1, 2).min(axis=0) - pad
        hi = pts.reshape(-1, 2).max(axis=0) + pad
        self.scale = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
        self.lo, self.w, self.h = lo, w, h
        span = (hi - lo) * self.scale
        self.off = np.array([(w - span[0]) / 2, (h - span[1]) / 2])

    def __call__(self, p: np.ndarray) -> np.ndarray:
        xy = (p - self.lo) * self.scale + self.off
        return np.stack([xy[..., 0], self.h - xy[..., 1]], axis=-1)


def _draw_body(d: ImageDraw.ImageDraw, px: np.ndarray, colors: dict, width: int) -> None:
    for a, b, side in BONES:
        d.line([tuple(px[a]), tuple(px[b])], fill=colors[side], width=width, joint="curve")
    r = width * 0.9
    hx, hy = px[HEAD]
    d.ellipse([hx - r * 2.1, hy - r * 2.1, hx + r * 2.1, hy + r * 2.1], fill=colors["c"])


def _rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (*(int(h[i:i + 2], 16) for i in (0, 2, 4)), int(round(255 * alpha)))


def render_walk(
    npz: Path, out: Path, *, size=(1280, 720), accent="#7E9338", secondary="#2E6FB7",
    ink="#141414", fps_out=30, trail_s=1.6,
) -> Path:
    """Animated skeleton on white, with the ankles leaving a fading trail."""
    import imageio.v3 as iio

    d = np.load(npz, allow_pickle=True)
    J = d["joints_pos_m"].astype(float)
    fps = float(d["fps"])
    axis = _travel_axis(J)
    pts = _project(J[:, :22], axis)

    W, H = size[0] * SS, size[1] * SS
    # Frame on one stride's worth of body, not on the whole pass. A 13 m walk fitted to a
    # locked-off frame renders the subject a few percent of frame height; the camera tracks
    # instead, so scale comes from the body and only the trail reveals the distance covered.
    body_h = np.ptp(pts[:, :, 1])
    fr = _Frame(np.array([[0.0, 0.0], [body_h * 1.9, body_h * 1.22]]), W, H, pad=0.0)
    px_all = fr(pts - pts[:, PELVIS:PELVIS + 1, :] * np.array([1.0, 0.0])
                - np.array([-body_h * 0.95, 0.0]))
    # NOT floor_z_m -- that records the floor in GVHMR's original frame for provenance, and
    # the import already subtracted it. The contact height is where the feet actually rest.
    ground = fr(np.array([[0.0, np.percentile(J[:, [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT], 2], 2)]]))[0, 1]
    lw = max(2, int(round(W / 210)))
    trail_n = int(trail_s * fps)
    colors = {"c": ink, "l": accent, "r": secondary}

    frames = []
    for t in range(len(px_all)):
        img = Image.new("RGB", (W, H), "#FFFFFF")
        dr = ImageDraw.Draw(img, "RGBA")
        dr.line([(0, ground), (W, ground)], fill=_rgba(ink, 0.16), width=max(1, lw // 3))

        # ankle trail: where the feet have been, fading out behind
        for j, col in ((L_ANKLE, accent), (R_ANKLE, secondary)):
            for k in range(max(0, t - trail_n), t):
                a = 0.30 * (1 - (t - k) / trail_n) ** 1.6
                x, y = px_all[k, j]
                s = lw * 0.7
                dr.ellipse([x - s, y - s, x + s, y + s], fill=_rgba(col, a))

        _draw_body(dr, px_all[t], colors, lw)
        frames.append(np.asarray(img.resize(size, Image.LANCZOS)))

    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, np.stack(frames), fps=fps_out, codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    return out


def render_chrono(
    npz: Path, out: Path, *, size=(2000, 900), accent="#7E9338", secondary="#2E6FB7",
    ink="#141414", n=9, start=0.0, end=1.0,
) -> Path:
    """Chronophotography: n poses from one pass, superimposed where they happened."""
    d = np.load(npz, allow_pickle=True)
    J = d["joints_pos_m"].astype(float)
    axis = _travel_axis(J)
    pts = _project(J[:, :22], axis)

    lo, hi = int(start * len(pts)), int(end * len(pts))
    # Sample at equal *distance*, not equal time. Sampling in time bunches the figures up
    # wherever the walker slowed down -- at the end of a pass, that stacks them into a blur.
    x = pts[lo:hi, PELVIS, 0]
    travel = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(x)))])
    idx = lo + np.searchsorted(travel, np.linspace(0, travel[-1], n))
    idx = np.clip(idx, lo, hi - 1)

    W, H = size[0] * SS, size[1] * SS
    fr = _Frame(pts[idx], W, H, pad=0.28)
    lw = max(2, int(round(W / 300)))

    img = Image.new("RGB", (W, H), "#FFFFFF")
    dr = ImageDraw.Draw(img, "RGBA")
    ground = fr(np.array([[0.0, 0.0]]))[0, 1]
    dr.line([(0, ground), (W, ground)], fill=_rgba(ink, 0.18), width=max(1, lw // 2))

    for i, t in enumerate(idx):
        # oldest faintest, newest solid -- reads as direction of travel without an arrow
        a = 0.22 + 0.78 * (i / (len(idx) - 1)) ** 1.5
        colors = {"c": _rgba(ink, a), "l": _rgba(accent, a), "r": _rgba(secondary, a)}
        _draw_body(dr, fr(pts[t]), colors, lw)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize(size, Image.LANCZOS).save(out, quality=94)
    return out


#: LAFAN1's BVH skeleton, in the joint order :func:`ego2g1.eval.bvh.parse` returns.
BVH_BONES: tuple[tuple[int, int, str], ...] = (
    (0, 9, "c"), (9, 10, "c"), (10, 11, "c"), (11, 12, "c"), (12, 13, "c"),
    (0, 1, "l"), (1, 2, "l"), (2, 3, "l"), (3, 4, "l"),
    (0, 5, "r"), (5, 6, "r"), (6, 7, "r"), (7, 8, "r"),
    (11, 14, "l"), (14, 15, "l"), (15, 16, "l"), (16, 17, "l"),
    (11, 18, "r"), (18, 19, "r"), (19, 20, "r"), (20, 21, "r"),
)
BVH_HEAD = 13


def _cycle(joints: np.ndarray, fps: float, ankle: int, n: int) -> np.ndarray:
    """One gait cycle, resampled to n frames, starting at a foot contact.

    Phase alignment is what makes two different walkers comparable. Sampling both by distance
    or by time would drift them out of step within a stride and the overlay would compare a
    mid-swing pose against a mid-stance one.
    """
    from ego2g1.eval.gait import detect_contact, _runs

    runs = _runs(detect_contact(joints[:, ankle], fps), fps)
    if len(runs) < 2:
        raise ValueError("no gait cycle found")
    # the longest stride available, so the sample is a steady one
    a, b = max(zip(runs, runs[1:]), key=lambda p: p[1][0] - p[0][0])
    src = np.linspace(a[0], b[0], n)
    lo = np.floor(src).astype(int)
    w = (src - lo)[:, None, None]
    hi = np.minimum(lo + 1, len(joints) - 1)
    return joints[lo] * (1 - w) + joints[hi] * w


def render_overlay(
    mine: np.ndarray, theirs: np.ndarray, out: Path, *,
    mine_bones=BONES, their_bones=BVH_BONES, mine_head=HEAD, their_head=BVH_HEAD,
    size=(2200, 820), n=7, ink="#111111", ghost="#B4B4B4", view="side",
) -> Path:
    """Two walkers at matched gait phase, superimposed on white.

    Both are scaled to a common stature and stood on a common floor, so what remains between
    the black figure and the grey one is posture -- how the limbs are arranged at the same
    instant of the stride -- rather than how tall either person is.

    ``view="side"`` is the sagittal plane: leg swing, trunk lean, arm carriage.
    ``view="top"`` looks straight down. Toe-out and foot separation are transverse-plane
    facts, and a side view cannot show them at all -- both feet sit on the same image line
    no matter how far apart they are. This section's whole claim needs the top view.
    """
    def norm(J: np.ndarray) -> np.ndarray:
        J = J - J[:, :, :].reshape(-1, 3).mean(axis=0) * np.array([1, 1, 0])
        h = np.percentile(J[:, :, 2], 99) - np.percentile(J[:, :, 2], 1)
        J = J / h                                     # unit stature
        J[:, :, 2] -= J[:, :, 2].min()                # feet on the floor
        return J

    def align(J: np.ndarray) -> np.ndarray:
        """Rotate about vertical so travel runs along +x.

        Picking the dominant world axis is not enough: a walk at 30 deg to it projects
        sheared, which in the top view turns a straight line of footfalls into a diagonal
        and makes toe-out unreadable.
        """
        d = J[-1, 0, :2] - J[0, 0, :2]
        if np.linalg.norm(d) < 1e-6:
            return J
        th = -np.arctan2(d[1], d[0])
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        out = J.copy()
        out[..., :2] = J[..., :2] @ R.T
        return out

    A, B = align(norm(mine)), align(norm(theirs))
    if view == "top":
        # lateral is magnified: a 6 cm gap next to a 1.7 m body is otherwise invisible.
        # The caption has to say so -- this axis is not 1:1 with the other.
        pa = np.stack([A[..., 0], A[..., 1] * 3.0], axis=-1)
        pb = np.stack([B[..., 0], B[..., 1] * 3.0], axis=-1)
    else:
        pa = np.stack([A[..., 0], A[..., 2]], axis=-1)
        pb = np.stack([B[..., 0], B[..., 2]], axis=-1)
    # travel left-to-right for both, whatever direction each was recorded in
    for p in (pa, pb):
        if p[-1, 0, 0] < p[0, 0, 0]:
            p[..., 0] *= -1
    # each pose is drawn at its own slot, pelvis-centred, so the pair overlays exactly
    pa = pa - pa[:, :1, :] * np.array([1.0, 0.0])
    pb = pb - pb[:, :1, :] * np.array([1.0, 0.0])
    if view == "top":
        # centre each walker's own line of travel on zero -- but over the whole cycle, not
        # per frame, which would subtract out the lateral sway we are trying to show
        pa[..., 1] -= pa[:, 0, 1].mean()
        pb[..., 1] -= pb[:, 0, 1].mean()

    W, H = size[0] * SS, size[1] * SS
    slots = (np.arange(n) * 1.0)[:, None, None] * np.array([1.0, 0.0])   # (n,1,2)
    idx = np.linspace(0, len(pa) - 1, n).round().astype(int)
    jdx = np.linspace(0, len(pb) - 1, n).round().astype(int)
    allpts = np.concatenate([pa[idx] + slots, pb[jdx] + slots])
    fr = _Frame(allpts, W, H, pad=0.22)
    lw = max(2, int(round(W / 330)))

    img = Image.new("RGB", (W, H), "#FFFFFF")
    dr = ImageDraw.Draw(img, "RGBA")
    gy = fr(np.array([[0.0, 0.0]]))[0, 1]
    dr.line([(0, gy), (W, gy)], fill=_rgba(ink, 0.15), width=max(1, lw // 2))
    for k in range(n):
        off = slots[k]                                # (1,2), broadcasts over joints
        # grey first: the baseline sits behind, the subject reads on top
        for pts, i, bones, head, col, a in (
            (pb, jdx[k], their_bones, their_head, ghost, 0.95),
            (pa, idx[k], mine_bones, mine_head, ink, 0.95),
        ):
            colors = {s: _rgba(col, a) for s in "clr"}
            px = fr(pts[i] + off)
            for u, v, _s in bones:
                dr.line([tuple(px[u]), tuple(px[v])], fill=colors["c"], width=lw, joint="curve")
            r = lw * 1.9
            hx, hy = px[head]
            dr.ellipse([hx - r, hy - r, hx + r, hy + r], fill=colors["c"])

    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize(size, Image.LANCZOS).save(out, quality=95)
    return out


def render_lanes(
    subjects, out: Path, *, size=(2400, 1000), n=22, cycles=2.6, spacing=0.30,
    pad=0.30,
) -> Path:
    """Two walkers as dense motion trails, one lane each, leading figure solid.

    ``subjects`` is a sequence of ``(joints, bones, head, colour, label)``. Each lane gets
    ``n`` poses advancing across the frame while cycling through the stride, so the trail
    reads as one body moving rather than a row of separate figures -- the overlap is the
    point, and ``spacing`` well under a body width is what produces it.

    Both lanes are normalised to a common stature and share one scale, so the lanes can be
    read against each other; only posture differs.
    """
    lanes = []
    for J, bones, head, col, label in subjects:
        K = J - J.reshape(-1, 3).mean(axis=0) * np.array([1, 1, 0])
        h = np.percentile(K[:, :, 2], 99) - np.percentile(K[:, :, 2], 1)
        K = K / h
        K[:, :, 2] -= K[:, :, 2].min()
        d = K[-1, 0, :2] - K[0, 0, :2]
        if np.linalg.norm(d) > 1e-6:
            th = -np.arctan2(d[1], d[0])
            c, s = np.cos(th), np.sin(th)
            K[..., :2] = K[..., :2] @ np.array([[c, -s], [s, c]]).T
        p = np.stack([K[..., 0], K[..., 2]], axis=-1)
        if p[-1, 0, 0] < p[0, 0, 0]:
            p[..., 0] *= -1
        p = p - p[:, :1, :] * np.array([1.0, 0.0])          # pelvis-centred per frame
        lanes.append((p, bones, head, col, label))

    W, H = size[0] * SS, size[1] * SS
    span = (n - 1) * spacing
    body = max(np.ptp(p[:, :, 1]) for p, *_ in lanes)
    frame_pts = np.array([[-pad, 0.0], [span + pad, body * 1.06]])
    fr = _Frame(frame_pts, W, H // len(lanes), pad=0.0)
    lw = max(2, int(round(W / 620)))

    img = Image.new("RGB", (W, H), "#FFFFFF")
    dr = ImageDraw.Draw(img, "RGBA")

    for li, (p, bones, head, col, label) in enumerate(lanes):
        dy = li * (H // len(lanes))
        for i in range(n):
            t = int(((i / max(n - 1, 1)) * cycles % 1.0) * (len(p) - 1))
            # the trail is faint and even; only the leading body is solid
            a = 0.95 if i == n - 1 else 0.20 + 0.16 * (i / max(n - 1, 1))
            px = fr(p[t] + np.array([i * spacing, 0.0]))
            px[:, 1] += dy
            rgba = _rgba(col, a)
            for u, v, _s in bones:
                dr.line([tuple(px[u]), tuple(px[v])], fill=rgba, width=lw, joint="curve")
            r = lw * (2.4 if i == n - 1 else 1.7)
            hx, hy = px[head]
            dr.ellipse([hx - r, hy - r, hx + r, hy + r], fill=rgba)


    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize(size, Image.LANCZOS).save(out, quality=95)
    return out
