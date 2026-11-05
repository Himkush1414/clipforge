"""Auto mode: choose genre, format, length, count, ratio and resolution from the content.

These are transparent rules, not a trained model. Every choice carries a reason shown in the UI,
so a wrong call is easy to see and redo in manual mode.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clipforge.analysis.audio import AudioAnalysis
from clipforge.analysis.scenes import SceneAnalysis
from clipforge.analysis.scoring import split_sentences
from clipforge.analysis.transcribe import Word
from clipforge.hardware import RESOLUTIONS, detect
from clipforge.media.ffmpeg import find_tool
from clipforge.media.probe import MediaInfo

LONG_FORMAT_MIN_SECONDS = 40 * 60
SAMPLE_FRAMES = 12
SAMPLE_W, SAMPLE_H = 160, 90

GENRE_PRESETS = {
    "anime": "anime-impact",
    "gameplay": "manga-punch",
    "podcast": "podcast-clean",
    "movie": "cinema-subtle",
    "series": "cinema-subtle",
    "other": "karaoke-box",
}


@dataclass
class Signals:
    duration: float
    speech_share: float
    words_per_minute: float
    cuts_per_minute: float
    motion: float
    animation: float
    sentence_seconds: float


@dataclass
class Decision:
    genre: str
    format: str
    ratio: str
    resolution: str
    clip_duration: float
    clip_count: int
    part_numbering: bool
    preset_id: str
    signals: Signals
    reasons: dict[str, str] = field(default_factory=dict)

    def options(self) -> dict:
        return {
            "genre": self.genre,
            "format": self.format,
            "ratio": self.ratio,
            "resolution": self.resolution,
            "clip_duration": self.clip_duration,
            "clip_count": self.clip_count,
            "part_numbering": self.part_numbering,
            "min_score_ratio": 0.6,
            "auto_preset": self.preset_id,
        }

    def report(self) -> dict:
        return {
            "genre": self.genre,
            "format": self.format,
            "ratio": self.ratio,
            "resolution": self.resolution,
            "clip_duration": round(self.clip_duration),
            "clip_count": self.clip_count,
            "preset": self.preset_id,
            "reasons": self.reasons,
            "signals": {k: round(float(v), 3) for k, v in vars(self.signals).items()},
        }


def animation_score(video: Path, duration: float) -> float:
    """Drawn pictures have large flat areas and hard edges; camera footage has fine texture."""
    # One seek per frame: decoding the whole file took 73 s on a one-hour video.
    size = SAMPLE_W * SAMPLE_H
    samples = []
    for index in range(SAMPLE_FRAMES):
        at = duration * (index + 0.5) / SAMPLE_FRAMES
        cmd = [
            find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "4",
            "-ss", f"{at:.3f}", "-i", str(video), "-map", "0:v:0", "-frames:v", "1",
            "-vf", f"scale={SAMPLE_W}:{SAMPLE_H},format=gray", "-f", "rawvideo", "-",
        ]  # fmt: skip
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        if len(out.stdout) >= size:
            samples.append(
                np.frombuffer(out.stdout[:size], np.uint8).reshape(SAMPLE_H, SAMPLE_W).astype(np.int16)
            )
    frames = samples
    scores = []
    for frame in frames:
        if frame.std() < 6:  # A blank or fading frame says nothing about the art style.
            continue
        dx = np.abs(np.diff(frame, axis=1))
        flat = float((dx <= 2).mean())
        edges = float((dx >= 40).mean())
        texture = float(((dx > 2) & (dx < 40)).mean())
        scores.append(flat * 0.6 + min(edges * 8, 1.0) * 0.3 + (1 - min(texture * 3, 1.0)) * 0.1)
    return float(np.median(scores)) if scores else 0.0


def measure(
    words: list[Word], audio: AudioAnalysis, scenes: SceneAnalysis, info: MediaInfo, video: Path | None
) -> Signals:
    _ = audio  # Kept in the signature: loudness signals can join without touching callers.
    minutes = max(info.duration / 60, 1e-3)
    spoken = sum(w.end - w.start for w in words)
    sentences = split_sentences(words)
    return Signals(
        duration=info.duration,
        speech_share=min(1.0, spoken / max(info.duration, 1e-3)),
        words_per_minute=len(words) / minutes,
        cuts_per_minute=len(scenes.cut_frames) / minutes,
        motion=float(scenes.motion.mean()) if scenes.motion.size else 0.0,
        animation=animation_score(video, info.duration) if video else 0.0,
        sentence_seconds=float(np.median([s.end - s.start for s in sentences])) if sentences else 0.0,
    )


def choose_genre(s: Signals) -> tuple[str, str]:
    if s.speech_share >= 0.45 and s.cuts_per_minute < 3:
        return "podcast", (
            f"Speech covers {s.speech_share:.0%} of the video, with {s.cuts_per_minute:.1f} cuts a minute."
        )
    if s.animation >= 0.62:
        return "anime", f"The picture looks drawn: flat color areas and hard edges (score {s.animation:.2f})."
    if s.speech_share < 0.3 and s.motion >= 6:
        return "gameplay", (
            f"Little speech ({s.speech_share:.0%}) and constant on-screen motion (level {s.motion:.1f})."
        )
    if s.cuts_per_minute >= 6:
        if s.duration >= 70 * 60:
            return "movie", f"Fast editing ({s.cuts_per_minute:.1f} cuts a minute) and feature length."
        return "series", f"Fast editing ({s.cuts_per_minute:.1f} cuts a minute) at episode length."
    return "other", "No strong pattern matched, so all signals are weighted evenly."


def choose_format(s: Signals, genre: str) -> tuple[str, str]:
    if (
        s.duration >= LONG_FORMAT_MIN_SECONDS
        and genre in ("podcast", "series", "movie")
        and s.speech_share >= 0.35
    ):
        minutes = s.duration / 60
        return (
            "long",
            f"A {minutes:.0f}-minute, speech-led video has enough content for 10 to 15 minute parts.",
        )
    return "short", "Short vertical clips suit this length and pacing best."


def choose_duration(s: Signals, genre: str, fmt: str) -> tuple[float, str]:
    if fmt == "long":
        # Aim for 10 to 15 minutes, preferring whatever divides the video evenly.
        parts = max(1, round(s.duration / 750))
        length = float(np.clip(s.duration / parts, 600, 900))
        return length, f"{length / 60:.0f} minutes keeps each part inside the 10 to 15 minute range."
    if genre == "podcast" or s.sentence_seconds >= 6:
        length = 50.0
        why = "Long spoken thoughts need room to land."
    elif s.cuts_per_minute >= 15 or genre == "gameplay":
        length = 25.0
        why = "Fast pacing works best in tight clips."
    else:
        length = 35.0
        why = "A middle length suits this pacing."
    if length > s.duration:
        length = s.duration
        why = "The video is shorter than a normal clip, so the clip spans all of it."
    return length, why


def choose_count(s: Signals, length: float, fmt: str) -> tuple[int, str]:
    if fmt == "long":
        count = max(1, min(12, int(s.duration // length)))
        return count, f"{count} full parts fit in this video."
    per_three_minutes = int(s.duration // 180)
    room = int(s.duration // (length * 1.5))
    count = int(np.clip(per_three_minutes, 1, 12))
    count = max(1, min(count, room))
    return (
        count,
        f"Up to {count}, one per three minutes of video. Weak moments are dropped, so it may be fewer.",
    )


def choose_ratio(info: MediaInfo, fmt: str) -> tuple[str, str]:
    width, height = info.display_size
    if height > width:
        return "9:16", "The source is already vertical."
    if fmt == "long":
        return "16:9", "Long videos are watched in landscape."
    return "9:16", "Shorts and Reels are vertical."


def choose_resolution(info: MediaInfo) -> tuple[str, str]:
    """Match the source, never above 4K in auto: 8K takes far longer and rarely pays off in clips."""
    short_side = min(info.display_size)
    allowed = detect().allowed_resolutions
    choice = "1080p"
    for name in ("2K", "4K"):
        if short_side >= RESOLUTIONS[name] and name in allowed:
            choice = name
    if short_side < RESOLUTIONS["1080p"]:
        return (
            choice,
            f"The source is {short_side}p, so 1080p is the lowest export size. It will be upscaled.",
        )
    return choice, f"Matches the {short_side}p source without upscaling."


def decide(
    words: list[Word],
    audio: AudioAnalysis,
    scenes: SceneAnalysis,
    info: MediaInfo,
    video: Path | None,
    channel_settings: dict | None = None,
) -> Decision:
    signals = measure(words, audio, scenes, info, video)
    genre, genre_why = choose_genre(signals)
    fmt, fmt_why = choose_format(signals, genre)
    ratio, ratio_why = choose_ratio(info, fmt)
    if channel_settings:
        fmt, fmt_why = channel_settings.get("format", fmt), "This channel's default format."
        ratio, ratio_why = channel_settings.get("ratio", ratio), "This channel's default ratio."
    length, length_why = choose_duration(signals, genre, fmt)
    count, count_why = choose_count(signals, length, fmt)
    resolution, resolution_why = choose_resolution(info)
    preset = (channel_settings or {}).get("preset_id") or GENRE_PRESETS[genre]
    return Decision(
        genre=genre,
        format=fmt,
        ratio=ratio,
        resolution=resolution,
        clip_duration=length,
        clip_count=count,
        part_numbering=count > 1,
        preset_id=preset,
        signals=signals,
        reasons={
            "genre": genre_why,
            "format": fmt_why,
            "ratio": ratio_why,
            "resolution": resolution_why,
            "clip_duration": length_why,
            "clip_count": count_why,
            "preset": f"The default style for {genre}.",
        },
    )
