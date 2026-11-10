"""Quality checks that run on every exported file. The file on disk is what gets measured."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from clipforge.media.ffmpeg import FFmpegError, run_ffmpeg
from clipforge.media.probe import probe

LOUDNESS_TOLERANCE = 2.0


@dataclass
class Check:
    name: str
    status: str  # ok, warn or fail
    detail: str


def _status(checks: list[Check]) -> str:
    states = {c.status for c in checks}
    return "fail" if "fail" in states else "warn" if "warn" in states else "ok"


def run_qc(
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_duration: float,
    fps: float,
    expect_audio: bool,
    target_lufs: float = -14.0,
    target_peak: float = -1.0,  # the delivery ceiling; renders limit at -1.5 to stay under it
    source_height: int | None = None,
) -> dict:
    checks: list[Check] = []
    try:
        info = probe(path)
    except FFmpegError as exc:
        return {"status": "fail", "checks": [asdict(Check("playable", "fail", str(exc)[:200]))]}
    checks.append(Check("playable", "ok", f"{info.video_codec}, {info.fps:.3f} fps"))

    size_ok = (info.width, info.height) == (expected_width, expected_height)
    checks.append(
        Check(
            "resolution",
            "ok" if size_ok else "fail",
            f"{info.width}x{info.height}, wanted {expected_width}x{expected_height}",
        )  # fmt: skip
    )
    if source_height and min(expected_width, expected_height) > source_height:
        checks.append(
            Check(
                "upscale", "warn", f"Source is {source_height}p, so this export is upscaled and not sharper."
            )
        )

    delta = abs(info.duration - expected_duration)
    checks.append(
        Check(
            "duration",
            "ok" if delta <= 2.5 / fps else "fail",
            f"{info.duration:.3f}s, wanted {expected_duration:.3f}s",
        )
    )
    if expect_audio != info.has_audio:
        checks.append(
            Check("audio", "fail", "Audio track missing" if expect_audio else "Unexpected audio track")
        )
    elif not info.has_audio:
        checks.append(Check("audio", "warn", "This clip has no audio"))

    audio_filter = ["-af", "ebur128=peak=true"] if info.has_audio else []
    video_filter = "scale=480:-2,blackdetect=d=0.5:pix_th=0.08,freezedetect=n=0.002:d=4"
    stderr = run_ffmpeg(
        ["-loglevel", "info", "-i", path, "-vf", video_filter, *audio_filter, "-f", "null", "-"]
    )

    black = sum(float(m) for m in re.findall(r"black_duration:([\d.]+)", stderr))
    if black >= info.duration * 0.5:
        checks.append(Check("black_frames", "fail", f"{black:.1f}s of black"))
    elif black > 0:
        checks.append(Check("black_frames", "warn", f"{black:.1f}s of black"))
    else:
        checks.append(Check("black_frames", "ok", "none"))

    frozen = sum(float(m) for m in re.findall(r"freeze_duration: ?([\d.]+)", stderr))
    checks.append(
        Check("frozen_video", "warn" if frozen else "ok", f"{frozen:.1f}s frozen" if frozen else "none")
    )

    if info.has_audio:
        summary = stderr[stderr.rfind("Summary:") :] if "Summary:" in stderr else ""
        lufs = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
        peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
        if lufs and float(lufs.group(1)) > -70 and info.duration >= 3:
            value = float(lufs.group(1))
            good = abs(value - target_lufs) <= LOUDNESS_TOLERANCE
            checks.append(
                Check("loudness", "ok" if good else "warn", f"{value:.1f} LUFS, target {target_lufs}")
            )
        elif lufs and float(lufs.group(1)) <= -70:
            checks.append(Check("loudness", "warn", "Audio is silent"))
        if peak:
            value = float(peak.group(1))
            checks.append(Check("true_peak", "ok" if value <= target_peak else "warn", f"{value:.1f} dBFS"))
    return {"status": _status(checks), "checks": [asdict(c) for c in checks]}
