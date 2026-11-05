"""Find where the subject is, so a vertical crop can follow it.

Faces come first (OpenCV Haar cascade, bundled with OpenCV). Without a face, the moving area is
used. The virtual camera holds still until the subject leaves a dead zone, then eases across.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from clipforge.media.ffmpeg import find_tool

SAMPLE_WIDTH, SAMPLE_HEIGHT = 320, 180
SAMPLE_FPS = 2
DEAD_ZONE = 0.10
EASE_SECONDS = 0.5
MAX_KEYFRAMES = 40


@dataclass
class CameraKey:
    time: float
    center: float  # 0..1 of source width


@lru_cache(maxsize=1)
def _face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore[attr-defined]


def find_subject(frame: np.ndarray, previous: np.ndarray | None) -> tuple[float | None, float]:
    """Return (center x 0..1, confidence) for one gray frame."""
    faces = _face_detector().detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
    if len(faces):
        x, _, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x + w / 2) / frame.shape[1], 1.0
    if previous is None:
        return None, 0.0
    diff = cv2.absdiff(frame, previous).astype(np.float32)
    diff[diff < 25] = 0
    total = float(diff.sum())
    if total < frame.size * 0.5:
        return None, 0.0
    columns = diff.sum(axis=0)
    return float((columns * np.arange(frame.shape[1])).sum() / total) / frame.shape[1], 0.5


def sample_centers(video: Path, start: float, duration: float) -> list[tuple[float, float | None]]:
    cmd = [
        find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video), "-map", "0:v:0",
        "-vf", f"fps={SAMPLE_FPS},scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT},format=gray",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout
    size = SAMPLE_WIDTH * SAMPLE_HEIGHT
    samples: list[tuple[float, float | None]] = []
    previous = None
    index = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(SAMPLE_HEIGHT, SAMPLE_WIDTH)
        center, _ = find_subject(frame, previous)
        samples.append((index / SAMPLE_FPS, center))
        previous = frame
        index += 1
    proc.stdout.close()
    proc.wait()
    return samples


def plan_camera(samples: list[tuple[float, float | None]], cuts: list[float]) -> list[CameraKey]:
    """Dead-zone camera. At a scene cut it jumps at once, since easing across a cut looks wrong."""
    known = [c for _, c in samples if c is not None]
    position = float(np.median(known)) if known else 0.5
    keys = [CameraKey(0.0, position)]
    window: list[float] = []
    for time, center in samples:
        if center is None:
            continue
        if any(keys[-1].time < cut <= time for cut in cuts):
            position = center
            keys.append(CameraKey(time, position))
            window = []
            continue
        window = [*window[-2:], center]
        target = float(np.median(window))
        if abs(target - position) > DEAD_ZONE:
            keys.append(CameraKey(time, position))
            position = target
            keys.append(CameraKey(time + EASE_SECONDS, position))
    return _thin(keys)


def _thin(keys: list[CameraKey]) -> list[CameraKey]:
    if len(keys) <= MAX_KEYFRAMES:
        return keys
    step = len(keys) / MAX_KEYFRAMES
    return [keys[int(i * step)] for i in range(MAX_KEYFRAMES)]


def crop_x_expression(keys: list[CameraKey], source_width: int, crop_width: int) -> str:
    """FFmpeg expression for the crop's left edge over time, clamped to the frame."""

    def left(center: float) -> float:
        return max(0.0, min(source_width - crop_width, center * source_width - crop_width / 2))

    if len(keys) <= 1:
        return f"{left(keys[0].center if keys else 0.5):.1f}"
    expression = f"{left(keys[-1].center):.1f}"
    for a, b in reversed(list(zip(keys, keys[1:], strict=False))):
        la, lb = left(a.center), left(b.center)
        span = max(b.time - a.time, 1e-3)
        expression = (
            f"if(lt(t,{b.time:.3f}),{la:.1f}+({lb - la:.1f})*(t-{a.time:.3f})/{span:.3f},{expression})"
        )
    return expression
