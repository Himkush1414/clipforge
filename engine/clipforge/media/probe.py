"""Media probing with ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from clipforge.media.ffmpeg import FFmpegError, find_tool


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    fps_num: int
    fps_den: int
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    sample_rate: int | None
    frame_count: int
    size_bytes: int
    bit_rate: int
    rotation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def display_size(self) -> tuple[int, int]:
        return (self.height, self.width) if self.rotation in (90, 270) else (self.width, self.height)


def _rate(value: str | None) -> Fraction | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def probe(path: Path | str) -> MediaInfo:
    """Read stream info. Raises FFmpegError when the file is not playable video."""
    cmd = [
        find_tool("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=120)
    if result.returncode != 0:
        raise FFmpegError(f"Could not read media file: {result.stderr.strip()[:300]}", result.stderr)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1
        ),
        None,
    )
    if video is None:
        raise FFmpegError("No video stream found in this file.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    rate = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")) or Fraction(30)
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    if duration <= 0:
        raise FFmpegError("This file has no measurable duration.")
    nb_frames = video.get("nb_frames")
    frame_count = int(nb_frames) if str(nb_frames or "").isdigit() else round(duration * float(rate))

    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = abs(int(side["rotation"])) % 360

    return MediaInfo(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(rate),
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        video_codec=video.get("codec_name", "unknown"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        frame_count=frame_count,
        size_bytes=int(fmt.get("size") or Path(path).stat().st_size),
        bit_rate=int(fmt.get("bit_rate") or 0),
        rotation=rotation,
    )
