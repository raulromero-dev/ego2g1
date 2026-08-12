"""ffprobe truth about a video file — especially the parts that lie.

Two traps this module exists to surface:

**Rotation.** iPhone and Pixel files are stored landscape with a rotation entry in the display
matrix; players apply it, and the file is portrait. ffmpeg applies it on decode. **pyav does
not** — and pyav is what GVHMR reads with. So "how big is this video" has two different correct
answers, and picking the wrong one feeds a sideways human into a gravity-aligned pose prior.
``coded_wh`` is what pyav sees; ``display_wh`` is what a person sees.

**Frame rate.** 30000/1001 is not 30. Stored as a float it becomes 29.97002997..., and indexing
frames by ``round(t * fps)`` accumulates drift that looks exactly like bad retargeting. We keep
it as an exact ``Fraction`` until something deliberately normalises it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    coded_wh: tuple[int, int]      # as stored — what pyav yields
    display_wh: tuple[int, int]    # after rotation — what ffmpeg/players yield
    rotation_deg: int
    fps_exact: Fraction
    n_frames: int
    duration_s: float
    codec: str
    has_audio: bool

    @property
    def is_rotated(self) -> bool:
        return self.rotation_deg % 180 != 0

    @property
    def is_portrait(self) -> bool:
        return self.display_wh[1] > self.display_wh[0]

    @property
    def fps_is_exact_integer(self) -> bool:
        return self.fps_exact.denominator == 1

    def describe(self) -> str:
        rot = f", rotation={self.rotation_deg}" if self.rotation_deg else ""
        fps = (f"{self.fps_exact} = {float(self.fps_exact):.5f}"
               if not self.fps_is_exact_integer else str(self.fps_exact))
        return (f"{self.path.name}: coded {self.coded_wh[0]}x{self.coded_wh[1]}"
                f" -> display {self.display_wh[0]}x{self.display_wh[1]}{rot}, "
                f"{fps} fps, {self.n_frames} frames, {self.duration_s:.2f}s, {self.codec}")


def probe(path: Path | str) -> VideoProbe:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)

    streams = info["streams"]
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    has_audio = any(s["codec_type"] == "audio" for s in streams)

    coded_w, coded_h = int(video["width"]), int(video["height"])

    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if rotation == 0 and "rotate" in (video.get("tags") or {}):
        rotation = int(video["tags"]["rotate"])
    rotation %= 360

    display_wh = (coded_h, coded_w) if rotation % 180 else (coded_w, coded_h)

    fps_exact = Fraction(video.get("r_frame_rate", "0/1"))

    n_frames = int(video.get("nb_frames") or 0)
    duration = float(info["format"].get("duration") or 0.0)
    if n_frames == 0 and duration and fps_exact:
        n_frames = int(round(duration * float(fps_exact)))

    return VideoProbe(
        path=path, coded_wh=(coded_w, coded_h), display_wh=display_wh,
        rotation_deg=rotation, fps_exact=fps_exact, n_frames=n_frames,
        duration_s=duration, codec=video.get("codec_name", "?"), has_audio=has_audio,
    )


def pyav_shape(path: Path | str) -> tuple[int, int]:
    """(height, width) as ``imageio``/pyav reports it — GVHMR's exact reader.

    Use this, not ffprobe, to verify a clip is genuinely portrait on disk. The whole point of
    re-encoding is to make this agree with what a human sees.
    """
    import imageio.v3 as iio
    props = iio.improps(str(path), plugin="pyav")
    return int(props.shape[1]), int(props.shape[2])


def assert_pyav_shape(path: Path | str, expected_hw: tuple[int, int]) -> None:
    got = pyav_shape(path)
    if got != tuple(expected_hw):
        raise AssertionError(
            f"{Path(path).name}: pyav sees {got[1]}x{got[0]} (WxH), expected "
            f"{expected_hw[1]}x{expected_hw[0]}. The rotation is still in the display matrix "
            "rather than baked into the pixels — re-encode instead of stream-copying.")
