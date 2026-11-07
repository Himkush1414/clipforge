#!/usr/bin/env python3
"""Word timing on real speech, where no exact reference exists.

After a real pause, the voice's start is plainly visible as a rise in energy, so words that follow a
pause can be scored against it. Also reports how many segments the confidence gate sends back to
Whisper's times. Audio stays in the work folder, never the repo.

Usage: real_onsets.py speech-16k-mono.wav out.json   (runs in the app's environment)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from clipforge.analysis import align, transcribe
from clipforge.hardware import detect

RATE = 16000
HOP = 160
MIN_PAUSE_FRAMES = 30  # 300 ms


def voice_onsets(samples: np.ndarray) -> list[float]:
    """Times where energy stayed low for a pause, then rose and stayed up for 100 ms."""
    frames = samples[: samples.size // HOP * HOP].reshape(-1, HOP)
    level = np.sqrt((frames**2).mean(axis=1))
    quiet = level < 0.05 * np.percentile(level, 95)
    onsets, run = [], 0
    for index, is_quiet in enumerate(quiet):
        if is_quiet:
            run += 1
            continue
        if run >= MIN_PAUSE_FRAMES and quiet[index : index + 10].sum() == 0:
            onsets.append(index * HOP / RATE)
        run = 0
    return onsets


def score(starts: np.ndarray, onsets: list[float]) -> dict:
    nearest = [starts[np.argmin(np.abs(starts - onset))] - onset for onset in onsets]
    errors = np.array([e * 1000 for e in nearest if abs(e) < 1.0])
    return {
        "onsets_matched": int(errors.size),
        "median_abs_ms": round(float(np.median(np.abs(errors))), 1),
        "p90_abs_ms": round(float(np.percentile(np.abs(errors), 90)), 1),
        "max_abs_ms": round(float(np.abs(errors).max()), 1),
        "median_signed_ms": round(float(np.median(errors)), 1),
        "within_50ms": round(float((np.abs(errors) <= 50).mean()), 3),
        "within_100ms": round(float((np.abs(errors) <= 100).mean()), 3),
    }


def main() -> int:
    wav, out = Path(sys.argv[1]), Path(sys.argv[2])
    seen: dict = {}
    real_align = align.align_segments

    def spy(samples, segments, language, *args, **kwargs):
        seen.update(samples=samples, whisper=[w.start for s in segments for w in s.words])
        seen["segments"] = segments
        return real_align(samples, segments, language, *args, **kwargs)

    align.align_segments = spy
    duration = wavfile.read(wav)[1].size / RATE
    result = transcribe._run(wav, duration, detect().whisper_device, None, None, None)
    onsets = voice_onsets(seen["samples"])
    report = {
        "minutes": round(duration / 60, 1),
        "language": result["language"],
        "timing": result["timing"],
        "words": len(result["words"]),
        "segments": len(seen["segments"]),
        "pause_onsets": len(onsets),
        "whisper": score(np.array(seen["whisper"]), onsets),
        "aligned": score(np.array([w["start"] for w in result["words"]]), onsets),
    }
    print(json.dumps(report, indent=1))
    out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
