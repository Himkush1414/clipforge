"""Audio energy, silences and peaks from the 16 kHz analysis WAV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

HOP_SECONDS = 0.05


@dataclass
class AudioAnalysis:
    hop: float
    rms_db: np.ndarray
    flux: np.ndarray
    silences: list[tuple[float, float]]
    peaks: list[float]
    noise_floor_db: float

    def rms_at(self, start: float, end: float) -> float:
        i, j = int(start / self.hop), max(int(start / self.hop) + 1, int(end / self.hop))
        chunk = self.rms_db[i:j]
        return float(chunk.mean()) if chunk.size else -90.0


def find_silences(
    rms_db: np.ndarray, hop: float, threshold_db: float, min_length: float = 0.35
) -> list[tuple[float, float]]:
    quiet = rms_db < threshold_db
    silences = []
    start = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            if (i - start) * hop >= min_length:
                silences.append((round(start * hop, 3), round(i * hop, 3)))
            start = None
    if start is not None and (len(quiet) - start) * hop >= min_length:
        silences.append((round(start * hop, 3), round(len(quiet) * hop, 3)))
    return silences


QUIET_HOP = 0.01
QUIET_MIN_SECONDS = 0.35
LOUD_BLIP_SECONDS = 0.03
BLIP_QUIET_SIDES = 0.1
# Quiet and loud must sit this far apart, or the audio has no real silences, only quieter moments.
MIN_SEPARATION_DB = 15.0


def frame_levels(samples: np.ndarray, hop: int, chunk_frames: int = 6000) -> np.ndarray:
    """RMS level in dB per hop, read in chunks so an hour of memory-mapped audio never loads at once."""
    frames = len(samples) // hop
    out = np.empty(frames, dtype=np.float32)
    for first in range(0, frames, chunk_frames):
        last = min(frames, first + chunk_frames)
        block = np.asarray(samples[first * hop : last * hop], dtype=np.float32).reshape(-1, hop) / 32768.0
        out[first:last] = 20.0 * np.log10(np.maximum(np.sqrt((block * block).mean(axis=1)), 1e-5))
    return out


def quiet_threshold(levels: np.ndarray) -> float | None:
    """Where quiet and loud separate best (Otsu's method), or None when they do not really separate.

    A fixed share of the level range, as before, lands inside steady noise, where ordinary dips then
    count as silence. Looking for the split between the two groups avoids that.
    """
    if levels.size == 0:
        return None
    counts, edges = np.histogram(levels, bins=np.arange(-100.0, 1.0, 1.0))
    centres = (edges[:-1] + edges[1:]) / 2
    weight = counts.astype(np.float64)
    total = weight.sum()
    if total == 0:
        return None
    cumulative = np.cumsum(weight)
    cumulative_mean = np.cumsum(weight * centres)
    best, best_index = -1.0, None
    for index in range(len(centres) - 1):
        low = cumulative[index]
        high = total - low
        if low == 0 or high == 0:
            continue
        mean_low = cumulative_mean[index] / low
        mean_high = (cumulative_mean[-1] - cumulative_mean[index]) / high
        between = low * high * (mean_high - mean_low) ** 2
        if between > best:
            best, best_index = between, index
    if best_index is None:
        return None
    low = cumulative[best_index]
    mean_low = cumulative_mean[best_index] / low
    mean_high = (cumulative_mean[-1] - cumulative_mean[best_index]) / (total - low)
    if mean_high - mean_low < MIN_SEPARATION_DB:
        return None
    return float(edges[best_index + 1])


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, end) index pairs of each run of True."""
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2].tolist(), changes[1::2].tolist(), strict=True))


def find_quiet(wav: Path) -> list[tuple[float, float]]:
    """Silences in an audio file, to the nearest 10 ms."""
    rate, samples = wavfile.read(wav, mmap=True)
    if samples.ndim > 1:
        samples = samples[:, 0]
    return quiet_spans(frame_levels(samples, int(rate * QUIET_HOP)), QUIET_HOP)


def quiet_spans(levels: np.ndarray, hop: float) -> list[tuple[float, float]]:
    threshold = quiet_threshold(levels)
    if threshold is None:
        return []
    quiet = levels < threshold
    # A click or a breath inside a pause is not the end of the pause. A word's first consonant is also
    # short and loud, but speech follows it at once, so a blip only counts when quiet surrounds it.
    side = int(round(BLIP_QUIET_SIDES / hop))
    filled = quiet.copy()
    for first, end in runs(~quiet):
        if (
            (end - first) * hop < LOUD_BLIP_SECONDS
            and first >= side
            and end + side <= len(quiet)
            and quiet[first - side : first].all()
            and quiet[end : end + side].all()
        ):
            filled[first:end] = True
    quiet = filled
    return [
        (round(first * hop, 3), round(end * hop, 3))
        for first, end in runs(quiet)
        if (end - first) * hop >= QUIET_MIN_SECONDS
    ]


def analyze_audio(wav: Path) -> AudioAnalysis:
    rate, samples = wavfile.read(wav, mmap=True)
    if samples.ndim > 1:
        samples = samples[:, 0]
    hop = int(rate * HOP_SECONDS)
    frames = max(1, len(samples) // hop)
    rms_db = np.zeros(frames, dtype=np.float32)
    flux = np.zeros(frames, dtype=np.float32)
    window = np.hanning(hop)
    prev_mag: np.ndarray | None = None
    # Frame by frame so a long video does not load fully into RAM (samples are memory mapped).
    for i in range(frames):
        chunk = np.asarray(samples[i * hop : (i + 1) * hop], dtype=np.float32) / 32768.0
        if chunk.size < hop:
            chunk = np.pad(chunk, (0, hop - chunk.size))
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        rms_db[i] = 20.0 * np.log10(max(rms, 1e-5))
        mag = np.abs(np.fft.rfft(chunk * window))
        if prev_mag is not None:
            flux[i] = float(np.maximum(mag - prev_mag, 0.0).sum())
        prev_mag = mag

    audible = rms_db[rms_db > -80.0]
    noise_floor = float(np.percentile(audible, 10)) if audible.size else -90.0
    silences = quiet_spans(frame_levels(samples, int(rate * QUIET_HOP)), QUIET_HOP)

    distance = max(1, int(0.5 / HOP_SECONDS))
    peak_ids, _ = find_peaks(rms_db, distance=distance, prominence=6.0)
    return AudioAnalysis(
        hop=HOP_SECONDS,
        rms_db=rms_db,
        flux=flux,
        silences=silences,
        peaks=[round(float(i) * HOP_SECONDS, 3) for i in peak_ids],
        noise_floor_db=noise_floor,
    )
