"""Output sizes and the mapping from source time to output time."""

from __future__ import annotations

from dataclasses import dataclass, field

from clipforge.analysis.transcribe import Word
from clipforge.hardware import RESOLUTIONS

RATIOS = ("9:16", "1:1", "16:9")


def output_size(ratio: str, resolution: str) -> tuple[int, int]:
    """1080p means a 1080-pixel short side at every aspect ratio."""
    short = RESOLUTIONS[resolution]
    if ratio == "9:16":
        return short, short * 16 // 9
    if ratio == "1:1":
        return short, short
    if ratio == "16:9":
        return short * 16 // 9, short
    raise ValueError(f"Unknown ratio: {ratio}")


@dataclass
class Segment:
    start_frame: int
    end_frame: int  # exclusive

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def to_frame_segments(segments: list[tuple[float, float]], fps: float, max_frame: int) -> list[Segment]:
    """Times to frames. Always at least one frame, never past the end, overlaps merged."""
    out: list[Segment] = []
    for start, end in segments:
        a = max(0, round(start * fps))
        b = min(max_frame, round(end * fps))
        if b <= a:
            b = min(max_frame, a + 1)
        if b <= a:
            continue
        if out and a <= out[-1].end_frame:
            out[-1].end_frame = max(out[-1].end_frame, b)
        else:
            out.append(Segment(a, b))
    return out


@dataclass
class Timeline:
    """Where each kept segment lands in the output, accounting for crossfade overlap."""

    segments: list[Segment]
    fps: float
    transition_frames: int = 0
    output_starts: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        position = 0
        self.output_starts = []
        for segment in self.segments:
            self.output_starts.append(position / self.fps)
            position += segment.frames - self.transition_frames
        self._total_frames = position + self.transition_frames

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps

    def remap_words(self, words: list[Word]) -> list[Word]:
        """Keep words that are in the output and move them to output time. Others are dropped."""
        mapped: list[Word] = []
        for word in words:
            for segment, offset in zip(self.segments, self.output_starts, strict=True):
                seg_start, seg_end = segment.start_frame / self.fps, segment.end_frame / self.fps
                middle = (word.start + word.end) / 2
                if seg_start <= middle < seg_end:
                    start = max(word.start, seg_start) - seg_start + offset
                    end = min(word.end, seg_end) - seg_start + offset
                    mapped.append(Word(word.text, round(start, 3), round(end, 3), word.probability))
                    break
        return mapped
