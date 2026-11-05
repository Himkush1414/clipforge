"""Turn a moment into exact cut points that never touch a spoken word."""

from __future__ import annotations

import math
from bisect import bisect_left

from clipforge.analysis.transcribe import Word

LEAD_IN = 0.12
TAIL_OUT = 0.25
SNAP_WINDOW = 0.3


def frame_floor(t: float, fps: float) -> float:
    return math.floor(t * fps + 1e-6) / fps


def frame_ceil(t: float, fps: float) -> float:
    return math.ceil(t * fps - 1e-6) / fps


def _nearest_cut(cuts: list[float], t: float, low: float, high: float) -> float | None:
    """Closest cut inside [low, high] and within the snap window of t."""
    i = bisect_left(cuts, t)
    options = [c for c in cuts[max(0, i - 2) : i + 2] if low <= c <= high and abs(c - t) <= SNAP_WINDOW]
    return min(options, key=lambda c: abs(c - t)) if options else None


def refine(
    start: float, end: float, words: list[Word], cuts: list[float], fps: float, duration: float
) -> tuple[float, float]:
    """Pad into silence, prefer a scene cut when one is close, then align to frame boundaries."""
    before = [w for w in words if w.end <= start + 1e-3]
    after = [w for w in words if w.start >= end - 1e-3]
    floor = before[-1].end if before else 0.0
    ceiling = after[0].start if after else duration

    new_start = max(floor, start - LEAD_IN)
    new_end = min(ceiling, end + TAIL_OUT)
    snapped_start = _nearest_cut(cuts, new_start, floor, start)
    snapped_end = _nearest_cut(cuts, new_end, end, ceiling)
    new_start = snapped_start if snapped_start is not None else new_start
    new_end = snapped_end if snapped_end is not None else new_end

    # Floor the start and ceil the end so rounding only ever adds, never clips, a word.
    new_start = max(0.0, frame_floor(new_start, fps))
    new_end = min(frame_ceil(duration, fps), frame_ceil(new_end, fps))
    return new_start, new_end


def remove_pauses(
    start: float, end: float, words: list[Word], min_pause: float = 0.8, keep: float = 0.2
) -> list[tuple[float, float]]:
    """Pacing: drop long pauses between words, keeping a short breath on both sides."""
    inside = [w for w in words if w.start >= start and w.end <= end]
    segments: list[tuple[float, float]] = []
    seg_start = start
    for a, b in zip(inside, inside[1:], strict=False):
        if b.start - a.end >= min_pause:
            segments.append((seg_start, a.end + keep))
            seg_start = b.start - keep
    segments.append((seg_start, end))
    return segments
