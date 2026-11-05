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
    loud = float(np.percentile(audible, 90)) if audible.size else -30.0
    # Halfway between floor and loud speech, but never above -25 dB and never below -60 dB.
    threshold = float(np.clip(noise_floor + (loud - noise_floor) * 0.3, -60.0, -25.0))
    silences = find_silences(rms_db, HOP_SECONDS, threshold)

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
