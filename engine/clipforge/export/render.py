"""Render a clip: exact frames, reframing, joins, captions, two-pass loudness, encoding."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.analysis.transcribe import Word
from clipforge.captions.ass import write_ass
from clipforge.captions.presets import CaptionPreset
from clipforge.captions.preview import ass_filter
from clipforge.config import get_settings
from clipforge.export.plan import Segment, Timeline, output_size
from clipforge.export.reframe import crop_x_expression, plan_camera, sample_centers
from clipforge.hardware import detect
from clipforge.media.ffmpeg import Cancelled, FFmpegError, run_ffmpeg
from clipforge.media.probe import MediaInfo

log = logging.getLogger("clipforge.render")

TARGET_LUFS = -14.0
TARGET_PEAK = -1.0
TARGET_LRA = 11.0
SEEK_MARGIN = 2.0
CLICK_FADE = 0.012
MIN_LOUDNORM_SECONDS = 1.0
DISK_RESERVE_BYTES = 1024**3


@dataclass
class RenderSpec:
    source: Path
    info: MediaInfo
    segments: list[Segment]
    ratio: str
    resolution: str
    output: Path
    words: list[Word] = field(default_factory=list)
    preset: CaptionPreset | None = None
    fit: str = "smart"  # smart, crop or blur
    transition: str = "cut"  # cut or crossfade
    transition_seconds: float = 0.3
    cuts: list[float] = field(default_factory=list)


@dataclass
class RenderResult:
    path: Path
    frames: int
    duration: float
    encoder: str
    width: int
    height: int


def pick_encoder(width: int, height: int) -> str:
    """H.264 where it fits, HEVC above that. 8K on NVENC needs HEVC (checked by a test encode)."""
    hw = detect()
    long_side = max(width, height)
    h264_limit = hw.h264_max_height * 16 // 9
    return hw.h264_encoder if long_side <= h264_limit else hw.hevc_encoder


def encoder_args(encoder: str) -> list[str]:
    if encoder.endswith("_nvenc"):
        args = ["-c:v", encoder, "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
        args += ["-spatial-aq", "1"]
    elif encoder == "libx265":
        args = ["-c:v", "libx265", "-preset", "medium", "-crf", "20"]
    else:
        args = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    if "hevc" in encoder or "265" in encoder:
        args += ["-tag:v", "hvc1"]
    return args


def _cover_crop(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """Largest even crop of the source with the output's aspect ratio."""
    if src_w * out_h > src_h * out_w:
        crop_h = src_h
        crop_w = round(src_h * out_w / out_h)
    else:
        crop_w = src_w
        crop_h = round(src_w * out_h / out_w)
    return crop_w - crop_w % 2, crop_h - crop_h % 2


def _reframe_filter(spec: RenderSpec, segment: Segment, out_w: int, out_h: int) -> str:
    src_w, src_h = spec.info.display_size
    fit_scale = f"scale={out_w}:{out_h}:flags=lanczos"
    same_shape = abs(src_w * out_h - src_h * out_w) <= max(src_w, src_h)
    if same_shape:
        return fit_scale
    if spec.fit == "blur":
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
            f"crop={out_w}:{out_h},boxblur=luma_radius=40:luma_power=2[bg2];"
            f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
        )
    crop_w, crop_h = _cover_crop(src_w, src_h, out_w, out_h)
    x_expr = f"{(src_w - crop_w) // 2}"
    if spec.fit == "smart" and crop_w < src_w:
        fps = spec.info.fps
        start, length = segment.start_frame / fps, segment.frames / fps
        cuts = [c - start for c in spec.cuts if start < c < start + length]
        keys = plan_camera(sample_centers(spec.source, start, length), cuts)
        x_expr = crop_x_expression(keys, src_w, crop_w)
    y = (src_h - crop_h) // 2
    return f"crop={crop_w}:{crop_h}:x='{x_expr}':y={y},{fit_scale}"


def _input_args(spec: RenderSpec) -> tuple[list[str], list[float]]:
    """One input per segment, each seeking just before the segment so long videos stay fast."""
    args: list[str] = []
    seeks: list[float] = []
    fps = spec.info.fps
    for segment in spec.segments:
        seek_frame = max(0, segment.start_frame - round(SEEK_MARGIN * fps))
        # Half a frame early: the wanted frame's timestamp is safely past the seek point.
        seek = max(0.0, (seek_frame - 0.5) / fps) if seek_frame > 0 else 0.0
        seeks.append(seek)
        if seek > 0:
            args += ["-ss", f"{seek:.6f}"]
        args += ["-i", str(spec.source)]
    return args, seeks


def build_video_graph(spec: RenderSpec, seeks: list[float], out_w: int, out_h: int, ass: Path | None) -> str:
    fps = spec.info.fps
    rate = f"{spec.info.fps_num}/{spec.info.fps_den}"
    parts: list[str] = []
    labels: list[str] = []
    for i, (segment, seek) in enumerate(zip(spec.segments, seeks, strict=True)):
        first_decoded = 0 if seek == 0 else round(seek * fps + 0.5)
        a = segment.start_frame - first_decoded
        b = a + segment.frames
        reframe = _reframe_filter(spec, segment, out_w, out_h)
        chain = (
            f"[{i}:v:0]fps={rate},trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,"
            f"{reframe},setsar=1,format=yuv420p[v{i}]"
        )
        parts.append(chain)
        labels.append(f"[v{i}]")
    last = _join_video(spec, labels, parts, fps)
    if ass:
        parts.append(f"[{last}]{ass_filter(ass)}[vout]")
    else:
        parts.append(f"[{last}]null[vout]")
    return ";".join(parts)


def _join_video(spec: RenderSpec, labels: list[str], parts: list[str], fps: float) -> str:
    if len(labels) == 1:
        return labels[0].strip("[]")
    if spec.transition == "crossfade":
        duration = spec.transition_seconds
        current = labels[0]
        offset = 0.0
        for i, label in enumerate(labels[1:], start=1):
            offset += spec.segments[i - 1].frames / fps - duration
            parts.append(
                f"{current}{label}xfade=transition=fade:duration={duration:.4f}:offset={offset:.4f}[x{i}]"
            )
            current = f"[x{i}]"
        return current.strip("[]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[joined]")
    return "joined"


def build_audio_graph(spec: RenderSpec, seeks: list[float], final_filter: str | None) -> str:
    fps = spec.info.fps
    parts: list[str] = []
    labels: list[str] = []
    for i, (segment, seek) in enumerate(zip(spec.segments, seeks, strict=True)):
        start = segment.start_frame / fps - seek
        length = segment.frames / fps
        fade = min(CLICK_FADE, length / 4)
        parts.append(
            f"[{i}:a:0]atrim=start={start:.6f}:duration={length:.6f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:d={fade:.4f},afade=t=out:st={max(0.0, length - fade):.4f}:d={fade:.4f}[a{i}]"
        )
        labels.append(f"[a{i}]")
    if len(labels) == 1:
        last = labels[0]
    elif spec.transition == "crossfade":
        last = labels[0]
        for i, label in enumerate(labels[1:], start=1):
            parts.append(f"{last}{label}acrossfade=d={spec.transition_seconds:.4f}[ax{i}]")
            last = f"[ax{i}]"
    else:
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[ajoined]")
        last = "[ajoined]"
    tail = f"{final_filter}," if final_filter else ""
    parts.append(f"{last}{tail}aresample=48000,aformat=channel_layouts=stereo[aout]")
    return ";".join(parts)


def measure_loudness(
    spec: RenderSpec,
    input_args: list[str],
    seeks: list[float],
    should_cancel: Callable[[], bool] | None = None,
) -> dict | None:
    """First loudnorm pass. Returns None when the audio is silent or too short to measure."""
    if not spec.info.has_audio:
        return None
    total = sum(s.frames for s in spec.segments) / spec.info.fps
    if total < MIN_LOUDNORM_SECONDS:
        return None
    measure = f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK}:LRA={TARGET_LRA}:print_format=json"
    graph = build_audio_graph(spec, seeks, measure)
    stderr = run_ffmpeg(
        [*input_args, "-filter_complex", graph, "-map", "[aout]", "-f", "null", "-"],
        should_cancel=should_cancel,
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    if not match:
        return None
    stats = json.loads(match.group(0))
    if stats["input_i"] in ("-inf", "inf", "nan") or float(stats["input_i"]) < -70:
        return None
    return stats


def loudnorm_filter(stats: dict | None) -> str | None:
    if stats is None:
        return None
    return (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK}:LRA={TARGET_LRA}"
        f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}:linear=true"
    )


def estimated_bytes(width: int, height: int, seconds: float) -> int:
    """Generous size guess (about 12 Mbit/s at 1080p), used only to warn before a disk fills."""
    return int(width * height * 5.8 * seconds / 8) + 50_000_000


def check_disk_space(width: int, height: int, seconds: float) -> None:
    free = shutil.disk_usage(get_settings().work_dir).free
    needed = estimated_bytes(width, height, seconds) + DISK_RESERVE_BYTES
    if free < needed:
        raise FFmpegError(
            f"Not enough disk space: this clip needs about {needed / 1024**3:.1f} GB "
            f"and {free / 1024**3:.1f} GB is free."
        )


def render_clip(
    spec: RenderSpec,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RenderResult:
    if not spec.segments:
        raise FFmpegError("A clip needs at least one segment.")
    settings = get_settings()
    out_w, out_h = output_size(spec.ratio, spec.resolution)
    fps = spec.info.fps
    transition_frames = 0
    if spec.transition == "crossfade" and len(spec.segments) > 1:
        shortest = min(s.frames for s in spec.segments)
        transition_frames = min(round(spec.transition_seconds * fps), max(1, shortest // 2))
        spec.transition_seconds = transition_frames / fps
    timeline = Timeline(spec.segments, fps, transition_frames)
    check_disk_space(out_w, out_h, timeline.duration)

    spec.output.parent.mkdir(parents=True, exist_ok=True)
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="render-", dir=settings.tmp_dir))
    partial = spec.output.with_name(f".{spec.output.stem}.part{spec.output.suffix}")
    try:
        ass_path = None
        if spec.preset is not None:
            words = timeline.remap_words(spec.words)
            if words:
                ass_path = write_ass(tmp / "captions.ass", words, spec.preset, out_w, out_h, spec.ratio)

        input_args, seeks = _input_args(spec)
        video_graph = build_video_graph(spec, seeks, out_w, out_h, ass_path)
        graph = video_graph
        maps = ["-map", "[vout]"]
        if spec.info.has_audio:
            stats = measure_loudness(spec, input_args, seeks, should_cancel)
            graph += ";" + build_audio_graph(spec, seeks, loudnorm_filter(stats))
            maps += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]

        def encode(encoder: str) -> None:
            run_ffmpeg(
                [
                    *input_args,
                    "-filter_complex",
                    graph,
                    *maps,
                    *encoder_args(encoder),
                    "-frames:v",
                    str(timeline.total_frames),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-t",
                    f"{timeline.duration:.6f}",
                    partial,
                ],  # fmt: skip
                duration=timeline.duration,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )

        encoder = pick_encoder(out_w, out_h)
        try:
            encode(encoder)
        except Cancelled:
            raise
        except FFmpegError as exc:
            if not encoder.endswith("_nvenc"):
                raise
            # The GPU can be busy or out of sessions. A slower render beats a failed job.
            fallback = "libx265" if "hevc" in encoder else "libx264"
            log.warning("%s failed, falling back to %s: %s", encoder, fallback, str(exc)[:200])
            encoder = fallback
            encode(encoder)
        partial.replace(spec.output)
        return RenderResult(spec.output, timeline.total_frames, timeline.duration, encoder, out_w, out_h)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        partial.unlink(missing_ok=True)


def replace_audio(video: Path, audio: Path, out: Path, duration: float) -> Path:
    """Offered only after a claim, with a track the user holds a licence for. The picture is copied."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.stem}.part{out.suffix}")
    # The new track is padded or cut to the clip's length, so picture and duration stay as they were.
    fit = f"apad,atrim=duration={duration:.6f},asetpts=PTS-STARTPTS"
    args: list[str | Path] = ["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    measure = f"{fit},loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK}:LRA={TARGET_LRA}:print_format=json"
    stderr = run_ffmpeg(["-i", audio, "-af", measure, "-f", "null", "-"])
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    stats = json.loads(match.group(0)) if match else None
    silent = stats is None or stats["input_i"] in ("-inf", "inf", "nan") or float(stats["input_i"]) < -70
    final = fit if silent else f"{fit},{loudnorm_filter(stats)}"
    try:
        run_ffmpeg(
            [*args, "-af", f"{final},aresample=48000,aformat=channel_layouts=stereo",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", partial]
        )  # fmt: skip
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)
    return out


def make_thumbnail(video: Path, out: Path, at: float) -> Path:
    run_ffmpeg(["-ss", f"{at:.3f}", "-i", video, "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", out])
    return out
