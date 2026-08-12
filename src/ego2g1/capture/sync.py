"""Audio-based synchronization of paired ego/exo recordings.

No clapperboard required. Both cameras record the same acoustic scene, so we align
them by cross-correlating *onset envelopes* (spectral flux) rather than raw energy.

Why spectral flux and not energy: the two microphones hear very different things --
the ego mic is strapped to the subject's head, the exo mic is several metres away --
so absolute level and EQ do not correspond. Sharp transients (footfalls, door clicks,
object contacts) do. Empirically on the SEAS sessions this moved the peak/noise ratio
from ~1.3-2.5x (energy) to ~5.5-11.8x (flux).

The windowed estimate is not optional: a single global correlation cannot tell you
whether the two clocks drift apart. Agreement across windows is the evidence that a
single constant offset is valid for the whole session.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, stft

SAMPLE_RATE = 16_000
ENV_RATE = 100.0  # onset-envelope frames per second


@dataclass
class SyncResult:
    """Offset to apply to align an ego clip with its exo clip.

    ``offset_s`` is ego-relative-to-exo: negative means the ego recording started
    *later*, so exo time ``t`` corresponds to ego time ``t + offset_s``.
    """

    offset_s: float
    confidence: float
    windows: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def drift_s(self) -> float:
        """Spread of per-window offsets. Near zero means one constant offset is valid."""
        if len(self.windows) < 2:
            return 0.0
        offs = [w[1] for w in self.windows]
        return max(offs) - min(offs)

    @property
    def is_trustworthy(self) -> bool:
        return self.confidence >= 3.0 and self.drift_s <= 0.1


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a video's audio track to mono float32 via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32)


def onset_envelope(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Half-wave-rectified spectral flux over the footfall/transient band."""
    freqs, _, spec = stft(x, fs=sample_rate, nperseg=512,
                          noverlap=512 - sample_rate // int(ENV_RATE))
    band = (freqs >= 80) & (freqs <= 2000)
    mag = np.log1p(np.abs(spec[band]) * 20)
    flux = np.diff(mag, axis=1)
    flux[flux < 0] = 0.0
    env = flux.sum(0)
    return env - np.median(env)


def _correlate(a: np.ndarray, b: np.ndarray, max_lag_s: float) -> tuple[float, float]:
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    corr = fftconvolve(a, b[::-1], mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    keep = np.abs(lags) <= int(max_lag_s * ENV_RATE)
    corr, lags = corr[keep], lags[keep]

    peak_idx = int(np.argmax(corr))
    guard = int(ENV_RATE * 0.4)
    background = np.delete(corr, slice(max(0, peak_idx - guard), peak_idx + guard))
    confidence = corr[peak_idx] / (np.percentile(np.abs(background), 99.5) + 1e-12)
    return lags[peak_idx] / ENV_RATE, float(confidence)


def sync_pair(ego_path: Path, exo_path: Path, *,
              max_lag_s: float = 40.0, window_s: float = 60.0) -> SyncResult:
    """Estimate the constant time offset between an ego/exo recording pair."""
    ego_env = onset_envelope(load_audio(ego_path))
    exo_env = onset_envelope(load_audio(exo_path))

    offset, confidence = _correlate(ego_env, exo_env, max_lag_s)

    windows: list[tuple[float, float, float]] = []
    span = int(window_s * ENV_RATE)
    for start in range(0, min(len(ego_env), len(exo_env)) - span, span):
        win_off, win_conf = _correlate(ego_env[start:start + span],
                                       exo_env[start:start + span], max_lag_s=8.0)
        windows.append((start / ENV_RATE, win_off, win_conf))

    # Trust the consensus of confident windows over the global peak when they disagree.
    confident = [w[1] for w in windows if w[2] >= 3.0]
    if confident:
        offset = float(np.median(confident))

    return SyncResult(offset_s=offset, confidence=confidence, windows=windows)
