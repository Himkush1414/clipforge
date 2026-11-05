"""Scene cuts and motion curve from one fast low-resolution pass.

Reads small gray frames from an FFmpeg pipe. Memory stays flat on long videos and
every cut has an exact frame number.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clipforge.media.ffmpeg import Cancelled, FFmpegError, find_tool

WIDTH, HEIGHT = 96, 54


@dataclass
class SceneAnalysis:
    fps: float
    frame_count: int
    cut_frames: list[int]
    motion: np.ndarray  # mean absolute frame difference per frame, 0..255

    @property
    def cut_times(self) -> list[float]:
        return [f / self.fps for f in self.cut_frames]


def detect_cuts(diffs: np.ndarray, fps: float, min_gap_seconds: float = 0.25) -> list[int]:
    """A cut is a frame whose difference towers over its neighbourhood (adaptive threshold)."""
    cuts: list[int] = []
    n = len(diffs)
    window = max(3, int(fps))
    min_gap = max(1, int(min_gap_seconds * fps))
    for i in range(1, n):
        value = diffs[i]
        if value < 8.0:
            continue
        lo, hi = max(1, i - window), min(n, i + window + 1)
        around = np.concatenate([diffs[lo:i], diffs[i + 1 : hi]])
        baseline = float(np.median(around)) if around.size else 0.0
        if value > max(18.0, baseline * 3.5 + 6.0) and (not cuts or i - cuts[-1] >= min_gap):
            cuts.append(i)
    return cuts


def analyze_scenes(
    video: Path,
    fps: float,
    duration: float,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SceneAnalysis:
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:flags=area,format=gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout
    frame_size = WIDTH * HEIGHT
    diffs: list[float] = []
    prev: np.ndarray | None = None
    expected = max(1, int(duration * fps))
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
            diffs.append(0.0 if prev is None else float(np.abs(frame - prev).mean()))
            prev = frame
            if len(diffs) % 500 == 0:
                if should_cancel and should_cancel():
                    raise Cancelled("Cancelled")
                if on_progress:
                    on_progress(min(1.0, len(diffs) / expected))
    finally:
        proc.stdout.close()
        code = proc.wait()
    if not diffs:
        raise FFmpegError(f"Could not decode video for scene analysis (exit {code}).")
    motion = np.array(diffs, dtype=np.float32)
    return SceneAnalysis(fps, len(diffs), detect_cuts(motion, fps), motion)
