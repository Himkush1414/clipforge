"""Scene cuts and motion curve from one fast low-resolution pass.

Reads small gray frames from an FFmpeg pipe. Memory stays flat on long videos and
every cut has an exact frame number.
"""

from __future__ import annotations

import math
import subprocess
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clipforge.media.ffmpeg import Cancelled, FFmpegError, find_tool

WIDTH, HEIGHT = 96, 54
MIN_CUT_DIFF = 8.0
COLOR_WEIGHT = 0.75
FLASH_FRAMES = 3  # A picture that returns within this many frames was a flash, not a new scene.


@dataclass
class SceneAnalysis:
    fps: float
    frame_count: int
    cut_frames: list[int]
    motion: np.ndarray  # mean absolute frame difference per frame, 0..255

    @property
    def cut_times(self) -> list[float]:
        return [f / self.fps for f in self.cut_frames]


def picture_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Brightness change, or colour change when larger: flat art can cut between equally bright colours."""
    delta = np.abs(a - b).mean(axis=(1, 2))
    return float(max(delta[0], (delta[1] + delta[2]) * COLOR_WEIGHT))


def detect_cuts(
    diffs: np.ndarray, fps: float, min_gap_seconds: float = 0.25, returns: np.ndarray | None = None
) -> list[int]:
    """A cut is a frame whose difference towers over its neighbourhood (adaptive threshold).

    returns[i] is how far the picture just after frame i differs from the one just before it.
    A small value means an earlier picture is back, so frame i was part of a flash, not a new scene.
    """
    cuts: list[int] = []
    n = len(diffs)
    window = max(3, int(fps))
    min_gap = max(1, int(min_gap_seconds * fps))
    for i in range(1, n):
        value = diffs[i]
        if value < MIN_CUT_DIFF:
            continue
        lo, hi = max(1, i - window), min(n, i + window + 1)
        around = np.concatenate([diffs[lo:i], diffs[i + 1 : hi]])
        baseline = float(np.median(around)) if around.size else 0.0
        if value <= max(18.0, baseline * 3.5 + 6.0) or (cuts and i - cuts[-1] < min_gap):
            continue
        if returns is not None and i < len(returns) and returns[i] < value * 0.35:
            continue
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
        f"scale={WIDTH}:{HEIGHT}:flags=area,format=yuv444p",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout
    frame_size = WIDTH * HEIGHT * 3
    diffs: list[float] = []
    returns: list[float] = []
    recent: deque[np.ndarray] = deque(maxlen=FLASH_FRAMES + 1)
    prev: np.ndarray | None = None
    expected = max(1, int(duration * fps))
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(3, HEIGHT, WIDTH).astype(np.int16)
            index = len(diffs)
            diffs.append(0.0 if prev is None else picture_difference(frame, prev))
            prev = frame
            returns.append(math.inf)
            if diffs[index] >= MIN_CUT_DIFF and len(recent) >= 2:
                # The flash's second edge: did this jump restore a picture from just before it?
                earlier = list(recent)[:-1]
                returns[index] = min(picture_difference(frame, old) for old in earlier)
            candidate = index - FLASH_FRAMES
            if len(recent) == recent.maxlen and diffs[candidate] >= MIN_CUT_DIFF:
                # The flash's first edge: is the picture from before that jump back by now?
                returns[candidate] = min(returns[candidate], picture_difference(frame, recent[0]))
            recent.append(frame)
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
    return SceneAnalysis(
        fps, len(diffs), detect_cuts(motion, fps, returns=np.array(returns, dtype=np.float32)), motion
    )
