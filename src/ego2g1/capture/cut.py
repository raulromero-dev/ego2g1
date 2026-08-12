"""Cut passes out of the long session recordings, baking rotation into the pixels.

Re-encoding rather than stream-copying is deliberate and does four jobs at once:

1. **Bakes the display-matrix rotation into the pixels.** ``-c copy`` preserves the side data,
   which leaves the bug live: ffmpeg and every player apply it, pyav does not, and GVHMR reads
   with pyav. Measured on this project's own files, all three exo recordings are stored
   1920x1080 with ``rotation=270``.
2. **Normalises frame rate.** Session 2's exo is 30000/1001, so "30 fps" downstream is a lie
   worth ~0.57 s of drift over its 571 s.
3. **Drops audio**, which GVHMR ignores and which is ~10% of the bytes we upload.
4. **Produces a faststart file** that streams while it uploads.

Every output is verified with ``imageio``/pyav -- GVHMR's exact reader -- rather than ffprobe,
because ffprobe is precisely the tool that would tell us the reassuring answer.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ego2g1.capture.probe import VideoProbe, assert_pyav_shape, probe

#: Visually lossless for h264 at this resolution; the upload is ~450 MB for the whole corpus.
EXO_CRF = 20
#: Ego clips are for the gait descriptor and side-by-side QA, never for pose estimation.
EGO_CRF = 26
TARGET_FPS = 30


@dataclass(frozen=True)
class CutSpec:
    clip_id: str
    src: Path
    dst: Path
    start_s: float
    duration_s: float
    crf: int = EXO_CRF


def cut_clip(spec: CutSpec, *, expect_hw: tuple[int, int] | None = None,
             overwrite: bool = False) -> Path:
    """Cut and re-encode one clip, then verify the rotation actually landed."""
    if spec.dst.exists() and not overwrite:
        return spec.dst
    spec.dst.parent.mkdir(parents=True, exist_ok=True)

    # -ss before -i seeks fast; ffmpeg re-encodes from the nearest keyframe and trims
    # accurately because we are not stream-copying.
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{spec.start_s:.3f}",
        "-i", str(spec.src),
        "-t", f"{spec.duration_s:.3f}",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(spec.crf),
        "-pix_fmt", "yuv420p",
        "-r", str(TARGET_FPS),
        "-an",
        "-movflags", "+faststart",
        str(spec.dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if expect_hw is not None:
        # The assert that pays for this whole module. Note it verifies *dimensions*, so it
        # catches 90/270 rotations but NOT 180 (which flips pixels without changing shape) --
        # Video-1's ego is exactly that case.
        assert_pyav_shape(spec.dst, expect_hw)

    return spec.dst


def expected_hw(src_probe: VideoProbe) -> tuple[int, int]:
    """(height, width) the cut clip should have — i.e. the display orientation."""
    w, h = src_probe.display_wh
    return (h, w)


def cut_session(specs: list[CutSpec], src_probe: VideoProbe, *,
                overwrite: bool = False, verbose: bool = True) -> list[Path]:
    """Cut many clips from one source, verifying each."""
    want = expected_hw(src_probe)
    out: list[Path] = []
    for i, spec in enumerate(specs, 1):
        path = cut_clip(spec, expect_hw=want, overwrite=overwrite)
        out.append(path)
        if verbose:
            mb = path.stat().st_size / 1e6
            print(f"  [{i:3d}/{len(specs)}] {spec.clip_id}  "
                  f"{spec.duration_s:5.1f}s  {mb:5.1f} MB  -> {path.name}")
    return out


def verify_cuts(paths: list[Path], expect_hw: tuple[int, int]) -> dict[str, object]:
    """Re-verify a set of clips through pyav. Cheap insurance before a paid GPU session."""
    bad: list[str] = []
    total_bytes = 0
    for p in paths:
        total_bytes += p.stat().st_size
        try:
            assert_pyav_shape(p, expect_hw)
        except AssertionError as exc:
            bad.append(f"{p.name}: {exc}")
    return {
        "n_clips": len(paths),
        "n_bad": len(bad),
        "total_mb": round(total_bytes / 1e6, 1),
        "failures": bad,
    }


def probe_and_report(path: Path | str) -> str:
    """One-line before/after description, for logs and QA sheets."""
    p = probe(path)
    from ego2g1.capture.probe import pyav_shape
    ph, pw = pyav_shape(path)
    agree = "consistent" if (pw, ph) == p.display_wh else "MISMATCH"
    return (f"{Path(path).name}: display {p.display_wh[0]}x{p.display_wh[1]}, "
            f"pyav {pw}x{ph} -> {agree}")
