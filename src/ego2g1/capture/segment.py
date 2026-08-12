"""Find the subject in a static-camera exo recording, and split it into passes.

The exo tripods never move, which makes this far easier than tracking: a per-pixel median over
the whole session *is* the empty room, and anything that differs from it is the subject. No
detector, no model, no GPU.

Two outputs:

- a **per-frame track** (bbox, foreground area, blur) that the manifest turns into quality bands
- **passes**, i.e. contiguous stretches where the subject is present, which are the clips

Frames are decoded through ffmpeg, which applies the display-matrix rotation, so every pixel
coordinate here is in *display* space -- the same space a person sees and the same space the cut
clips end up in. Subject height is therefore a fraction of 1920 for these portrait recordings.
This is deliberately not pyav: pyav's un-rotated view is only correct for verifying the cutter.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ego2g1.capture.probe import probe

#: Analysis resolution. Small enough to decode a 10-minute session in seconds, large enough that
#: Laplacian variance still means something.
ANALYSIS_W = 320
#: Foreground threshold in 8-bit grey levels; well above sensor noise, below real subject contrast.
FG_THRESHOLD = 28


@dataclass
class SubjectTrack:
    """Per-sampled-frame subject measurements, in display-space pixels of the source video."""

    t_s: np.ndarray            # (N,) seconds into the source file
    bbox_xywh_px: np.ndarray   # (N,4) float32, source-resolution pixels; zeros when absent
    fg_area_frac: np.ndarray   # (N,) fraction of frame that is foreground
    present: np.ndarray        # (N,) bool
    blur_lapvar: np.ndarray    # (N,) Laplacian variance inside bbox; low = motion blur
    frame_hw: tuple[int, int]  # source display (height, width)
    sample_fps: float

    @property
    def subj_px_height(self) -> np.ndarray:
        return self.bbox_xywh_px[:, 3]

    @property
    def subj_height_frac(self) -> np.ndarray:
        return self.bbox_xywh_px[:, 3] / float(self.frame_hw[0])


@dataclass
class Pass:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _decode_grey(path: Path, sample_fps: float, width: int) -> tuple[np.ndarray, int, int]:
    """Decode the whole video to a small greyscale array, rotation applied."""
    p = probe(path)
    disp_w, disp_h = p.display_wh
    height = int(round(width * disp_h / disp_w))
    height -= height % 2

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={sample_fps},scale={width}:{height},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout

    n = len(raw) // (width * height)
    frames = np.frombuffer(raw, dtype=np.uint8)[: n * width * height]
    return frames.reshape(n, height, width), disp_h, disp_w


def _laplacian_var(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap focus/blur measure."""
    # Needs at least a 3x3 interior, else the stencil below yields an empty array.
    if img.ndim != 2 or img.shape[0] < 3 or img.shape[1] < 3:
        return 0.0
    lap = (-4.0 * img[1:-1, 1:-1]
           + img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:])
    return float(lap.var())


def track_subject(path: Path | str, *, sample_fps: float = 5.0,
                  width: int = ANALYSIS_W, threshold: int = FG_THRESHOLD) -> SubjectTrack:
    """Background-subtract a static-camera recording into a per-frame subject track."""
    path = Path(path)
    frames, disp_h, disp_w = _decode_grey(path, sample_fps, width)
    n, h, w = frames.shape
    scale = disp_h / float(h)

    # The median over the session is the empty room: the subject is somewhere else most of the
    # time, so they never dominate any pixel's distribution.
    background = np.median(frames, axis=0)

    bbox = np.zeros((n, 4), dtype=np.float32)
    area = np.zeros(n, dtype=np.float32)
    blur = np.zeros(n, dtype=np.float32)

    for i, frame in enumerate(frames):
        mask = np.abs(frame.astype(np.float32) - background) > threshold
        area[i] = mask.mean()
        if not mask.any():
            continue
        rows, cols = np.where(mask)
        # Trim 2% off each axis so a stray reflection or a door opening across the room does not
        # inflate the box; the subject is a dense blob, outliers are not.
        y0, y1 = np.percentile(rows, [1, 99])
        x0, x1 = np.percentile(cols, [1, 99])
        bbox[i] = [x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale]
        blur[i] = _laplacian_var(frame[int(y0):int(y1) + 1, int(x0):int(x1) + 1].astype(np.float32))

    present = (area > 0.005) & (bbox[:, 3] > 0)
    t = np.arange(n, dtype=np.float64) / sample_fps

    return SubjectTrack(t_s=t, bbox_xywh_px=bbox, fg_area_frac=area, present=present,
                        blur_lapvar=blur, frame_hw=(disp_h, disp_w), sample_fps=sample_fps)


def merge_passes(track: SubjectTrack, *, enter_s: float = 0.6, exit_s: float = 0.4,
                 min_duration_s: float = 3.0, pad_s: float = 0.3) -> list[Pass]:
    """Turn per-frame presence into passes, with hysteresis.

    Hysteresis rather than a bare threshold because a single dropped detection would otherwise
    split one walk into two clips. ``min_duration_s`` is 3 s because GVHMR needs temporal context
    — its motion prior operates over a window, and two-second stubs come back unreliable.
    """
    fps = track.sample_fps
    enter_n, exit_n = max(1, int(round(enter_s * fps))), max(1, int(round(exit_s * fps)))
    present = track.present

    passes: list[Pass] = []
    inside = False
    start_i = 0
    run = 0

    for i, p in enumerate(present):
        if not inside:
            run = run + 1 if p else 0
            if run >= enter_n:
                inside, start_i, run = True, i - run + 1, 0
        else:
            run = run + 1 if not p else 0
            if run >= exit_n:
                passes.append((start_i, i - run + 1))
                inside, run = False, 0
    if inside:
        passes.append((start_i, len(present) - 1))

    out: list[Pass] = []
    duration = track.t_s[-1] if len(track.t_s) else 0.0
    for a, b in passes:
        start = max(0.0, float(track.t_s[a]) - pad_s)
        end = min(duration, float(track.t_s[b]) + pad_s)
        if end - start >= min_duration_s:
            out.append(Pass(index=len(out), start_s=start, end_s=end))
    return out


def summarize(track: SubjectTrack, passes: list[Pass]) -> dict[str, float | int]:
    p = track.present
    dt = 1.0 / track.sample_fps
    heights = track.subj_px_height[p]
    return {
        "duration_s": float(track.t_s[-1]) if len(track.t_s) else 0.0,
        "in_frame_s": float(p.sum() * dt),
        "in_frame_frac": float(p.mean()),
        "n_passes": len(passes),
        "pass_total_s": float(sum(x.duration_s for x in passes)),
        "median_pass_s": float(np.median([x.duration_s for x in passes])) if passes else 0.0,
        "subj_px_height_med": float(np.median(heights)) if heights.size else 0.0,
        "frac_ge_864px": float((heights >= 864).mean()) if heights.size else 0.0,
        "frac_lt_576px": float((heights < 576).mean()) if heights.size else 0.0,
    }
