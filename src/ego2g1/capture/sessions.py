"""Session registry and clip index — the minimum needed to trace any artifact back to source.

Deliberately not the full manifest. Quality bands, Gemini labels and training gates are all
*derived* from `data/qa/segmentation/{sid}_track.npz`, which is already on disk, so they can be
computed later without re-running anything. What cannot be recovered later is the mapping from a
clip back to the exact seconds of the exact source file — so that is what this records.

``src_time_s`` is the canonical link across every downstream stage: it is what lets a SMPL frame
be lined up with the original exo video, with the ego recording (via ``sync_offset_s``), and with
the per-frame subject track.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ego2g1 import conventions as C

CLIPS_DIR = Path("data/05_clips")
TRACK_DIR = Path("data/qa/segmentation")
INDEX_PATH = CLIPS_DIR / "clips.json"
SESSIONS_PATH = Path("data/00_raw/sessions.json")


@dataclass
class ExoRig:
    device: str
    dist_to_walkline_m: float | None = None   # tape measure; enables focal calibration
    lens_height_m: float | None = None
    angle_deg: float | None = None


@dataclass
class Session:
    session_id: str
    location: str
    exo_path: str
    ego_path: str
    #: ego relative to exo; negative means the ego recording started later.
    sync_offset_s: float
    sync_confidence: float
    subject_height_m: float
    exo: ExoRig
    static_cameras: bool = True
    ego_notes: str = ""
    #: Filled in once a frame with known subject distance is measured. Without it GVHMR assumes
    #: f = image diagonal (~43 mm equivalent), which inflates every distance by roughly 1.7x.
    f_px_calibrated: float | None = None
    f_mm_calibrated: int | None = None

    def calibrate_focal(self, subject_px_height: float, img_w: int, img_h: int) -> None:
        if self.exo.dist_to_walkline_m is None:
            return
        f_px = C.focal_px_from_measurement(subject_px_height, self.exo.dist_to_walkline_m,
                                           self.subject_height_m)
        self.f_px_calibrated = float(f_px)
        self.f_mm_calibrated = int(round(C.focal_mm_from_px(f_px, img_w, img_h)))


@dataclass
class ClipEntry:
    clip_id: str
    session_id: str
    exo_clip_path: str
    src_exo_path: str
    exo_start_s: float
    exo_end_s: float
    duration_s: float
    n_frames: int
    fps: float
    #: Same window in the ego recording's timeline.
    ego_start_s: float
    ego_end_s: float
    subj_px_height_med: float
    subj_px_height_min: float
    subj_px_height_max: float
    frame_h: int
    frame_w: int
    in_frame_frac: float
    blur_lapvar_med: float
    #: Ordered severity: how large the subject is, which drives pose-estimation reliability.
    quality_band: str
    #: Orthogonal to the band — a clip can be close AND truncated, or close AND blurred.
    quality_flags: list[str] = field(default_factory=list)

    @property
    def subj_height_frac_med(self) -> float:
        return self.subj_px_height_med / float(self.frame_h)


#: Thresholds as fractions of the 1920 px portrait frame: 0.45 and 0.30.
GOOD_PX, MARGINAL_PX = 864, 576
#: Laplacian variance below this reads as motion blur at the analysis resolution.
BLUR_MIN = 12.0


def _band_and_flags(h_med: float, bbox: np.ndarray, frame_h: int, frame_w: int,
                    blur_med: float, in_frame: float, duration_s: float) -> tuple[str, list[str]]:
    band = "good" if h_med >= GOOD_PX else ("marginal" if h_med >= MARGINAL_PX else "far")

    flags: list[str] = []
    if bbox.size:
        # Truncation breaks the bbox-centred crop that pose estimators assume.
        touch_top = bbox[:, 1] <= 2
        touch_bot = (bbox[:, 1] + bbox[:, 3]) >= frame_h - 2
        if float((touch_top | touch_bot).mean()) >= 0.20:
            flags.append("cropped")
        touch_side = (bbox[:, 0] <= 2) | ((bbox[:, 0] + bbox[:, 2]) >= frame_w - 2)
        if float(touch_side.mean()) >= 0.20:
            flags.append("occluded")
    if blur_med < BLUR_MIN:
        flags.append("blurred")
    if in_frame < 0.90:
        flags.append("intermittent")
    if duration_s < 4.0:
        flags.append("short")
    return band, flags


def build_index(sessions: list[Session], *, fps: float = 30.0) -> list[ClipEntry]:
    """Join cut clips to their source window and per-frame subject measurements."""
    entries: list[ClipEntry] = []

    for s in sessions:
        track_path = TRACK_DIR / f"{s.session_id}_track.npz"
        if not track_path.exists():
            raise FileNotFoundError(f"missing subject track for {s.session_id}: {track_path}")
        tr = np.load(track_path)
        t_s = tr["t_s"]
        bboxes = tr["bbox_xywh_px"]
        heights = bboxes[:, 3]
        # `present` is the authoritative mask: it requires real foreground area, not just a
        # non-zero bbox. Filtering on `heights > 0` admits noise frames and drags the median
        # down by ~3x -- which is exactly the discrepancy this replaced.
        present = tr["present"].astype(bool)
        blur = tr["blur_lapvar"]
        frame_h, frame_w = (int(x) for x in tr["frame_hw"])
        starts, ends = tr["pass_start_s"], tr["pass_end_s"]

        for i, (a, b) in enumerate(zip(starts, ends)):
            clip_path = CLIPS_DIR / s.session_id / "exo" / f"{s.session_id}_p{i:03d}.mp4"
            if not clip_path.exists():
                continue
            window = (t_s >= a) & (t_s <= b)
            sel = window & present
            h = heights[sel]
            duration = float(b - a)
            in_frame = float(sel.sum() / max(1, window.sum()))
            blur_med = float(np.median(blur[sel])) if sel.any() else 0.0
            band, flags = _band_and_flags(
                float(np.median(h)) if h.size else 0.0, bboxes[sel], frame_h, frame_w,
                blur_med, in_frame, duration)
            entries.append(ClipEntry(
                clip_id=f"{s.session_id}_p{i:03d}",
                session_id=s.session_id,
                exo_clip_path=str(clip_path),
                src_exo_path=s.exo_path,
                exo_start_s=float(a), exo_end_s=float(b),
                duration_s=duration,
                n_frames=int(round(duration * fps)),
                fps=fps,
                ego_start_s=float(a) + s.sync_offset_s,
                ego_end_s=float(b) + s.sync_offset_s,
                subj_px_height_med=float(np.median(h)) if h.size else 0.0,
                subj_px_height_min=float(h.min()) if h.size else 0.0,
                subj_px_height_max=float(h.max()) if h.size else 0.0,
                frame_h=frame_h, frame_w=frame_w,
                in_frame_frac=in_frame, blur_lapvar_med=blur_med,
                quality_band=band, quality_flags=flags,
            ))
    return entries


def write_index(sessions: list[Session], entries: list[ClipEntry]) -> tuple[Path, Path]:
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_PATH.write_text(json.dumps([asdict(s) for s in sessions], indent=2))
    INDEX_PATH.write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return SESSIONS_PATH, INDEX_PATH


def load_index() -> list[ClipEntry]:
    return [ClipEntry(**d) for d in json.loads(INDEX_PATH.read_text())]


def load_sessions() -> list[Session]:
    out = []
    for d in json.loads(SESSIONS_PATH.read_text()):
        d = dict(d)
        d["exo"] = ExoRig(**d["exo"])
        out.append(Session(**d))
    return out
