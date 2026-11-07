#!/usr/bin/env python3
"""Time and peak memory for aligning about an hour of real speech, by repeating a shorter recording.

Whisper runs once on the recording and is released first, as in the app, so this measures alignment.

Usage: align_memory.py speech-16k-mono.wav out.json   (runs in the app's environment)
"""

from __future__ import annotations

import gc
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.io import wavfile

from clipforge.analysis import align, transcribe
from clipforge.analysis.transcribe import Word
from clipforge.hardware import detect

RATE = 16000


def main() -> int:
    wav, out = Path(sys.argv[1]), Path(sys.argv[2])
    seen: dict = {}
    real_align = align.align_segments

    def capture(samples, segments, language, *args, **kwargs):
        seen.update(samples=samples, segments=segments, language=language)
        return [], "skipped"

    align.align_segments = capture
    duration = wavfile.read(wav)[1].size / RATE
    transcribe._run(wav, duration, detect().whisper_device, None, None, None)
    gc.collect()
    align.align_segments = real_align

    repeats = int(3600 / duration) + 1
    audio = np.tile(seen["samples"], repeats)
    segments = [
        align.Segment(
            s.start + r * duration,
            s.end + r * duration,
            [Word(w.text, w.start + r * duration, w.end + r * duration, w.probability) for w in s.words],
        )
        for r in range(repeats)
        for s in seen["segments"]
    ]

    process = psutil.Process()
    before = process.memory_info().rss
    peak = [before]
    done = threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.05)

    thread = threading.Thread(target=watch)
    thread.start()
    start = time.time()
    words, method = align.align_segments(audio, segments, seen["language"])
    took = time.time() - start
    done.set()
    thread.join()
    report = {
        "minutes_of_speech": round(audio.size / RATE / 60, 1),
        "method": method,
        "segments": len(segments),
        "longest_segment_seconds": round(max(s.end - s.start for s in segments), 1),
        "words": len(words),
        "seconds": round(took, 1),
        "times_realtime": round(audio.size / RATE / took, 1),
        "rss_before_gb": round(before / 1e9, 2),
        "rss_peak_gb": round(peak[0] / 1e9, 2),
    }
    print(json.dumps(report, indent=1))
    out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
