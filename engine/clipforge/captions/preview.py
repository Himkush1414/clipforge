"""Render a real libass preview (one frame or a short clip) for a preset."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from clipforge.analysis.transcribe import Word
from clipforge.captions.ass import write_ass
from clipforge.captions.presets import FONTS_DIR, CaptionPreset
from clipforge.config import get_settings
from clipforge.media.ffmpeg import escape_filter_path, run_ffmpeg

PREVIEW_SIZES = {"9:16": (540, 960), "1:1": (720, 720), "16:9": (960, 540)}
SAMPLE_LINE = "Nobody expected this to work but it was absolutely incredible!"


def sample_words(text: str = SAMPLE_LINE, seconds_per_word: float = 0.32) -> list[Word]:
    words = []
    t = 0.15
    for token in text.split():
        words.append(Word(token, round(t, 3), round(t + seconds_per_word - 0.03, 3)))
        t += seconds_per_word
    return words


def ass_filter(ass_path: Path) -> str:
    return f"ass=filename={escape_filter_path(ass_path)}:fontsdir={escape_filter_path(FONTS_DIR)}"


def render_preview(
    preset: CaptionPreset,
    out: Path,
    ratio: str = "9:16",
    at: float | None = 0.5,
    words: list[Word] | None = None,
    size: tuple[int, int] | None = None,
    background: str | None = None,
) -> Path:
    """Write a PNG of the frame at time `at`, or an MP4 of the whole sample when `at` is None."""
    width, height = size or PREVIEW_SIZES[ratio]
    words = words or sample_words()
    duration = words[-1].end + 0.5
    settings = get_settings()
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="preview-", dir=settings.tmp_dir))
    try:
        ass_path = write_ass(tmp / "p.ass", words, preset, width, height, ratio)
        bg = (
            background
            or f"gradients=s={width}x{height}:c0=0x1f2937:c1=0x4c1d95:x0=0:y0=0:x1={width}:y1={height}:speed=0"
        )
        graph = f"{ass_filter(ass_path)}"
        args: list = [
            "-f",
            "lavfi",
            "-t",
            f"{duration:.2f}",
            "-i",
            f"{bg},fps=30,format=yuv420p",
            "-vf",
            graph,
        ]
        if at is not None:
            args += ["-ss", f"{at:.3f}", "-frames:v", "1", "-update", "1", out]
        else:
            args += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                out,
            ]
        run_ffmpeg(args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
