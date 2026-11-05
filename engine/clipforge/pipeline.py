"""One job from start to finish: analyze, choose moments, render, check, record clips."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

from clipforge.analysis.audio import HOP_SECONDS, AudioAnalysis, analyze_audio
from clipforge.analysis.boundaries import refine, remove_pauses
from clipforge.analysis.scenes import SceneAnalysis, analyze_scenes
from clipforge.analysis.scoring import Moment, find_moments
from clipforge.analysis.transcribe import Word, extract_audio, transcribe, words_from_dicts
from clipforge.autopilot import decide
from clipforge.captions.layout import undrawable_words
from clipforge.captions.presets import BUILTIN_PRESETS, CaptionPreset
from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.export.plan import Segment, to_frame_segments
from clipforge.export.render import RenderSpec, make_thumbnail, render_clip
from clipforge.jobs import JobContext
from clipforge.media.probe import MediaInfo, probe
from clipforge.metadata import build_metadata
from clipforge.models import Channel, Clip, Job, Preset, RightsType, Source, utcnow
from clipforge.qc import run_qc

log = logging.getLogger("clipforge.pipeline")

ANALYSIS_SHARE = 0.4


@dataclass
class Analysis:
    words: list[Word]
    language: str | None
    audio: AudioAnalysis
    scenes: SceneAnalysis


def silent_audio(duration: float) -> AudioAnalysis:
    frames = max(1, int(duration / HOP_SECONDS))
    return AudioAnalysis(
        HOP_SECONDS,
        np.full(frames, -90.0, dtype=np.float32),
        np.zeros(frames, dtype=np.float32),
        [(0.0, round(duration, 3))],
        [],
        -90.0,
    )


def load_or_analyze(source: Source, info: MediaInfo, language: str | None, ctx: JobContext) -> Analysis:
    """Results are cached per source: re-running a video skips straight to rendering."""
    settings = get_settings()
    folder = settings.analysis_dir / source.id
    folder.mkdir(parents=True, exist_ok=True)
    transcript_path = folder / f"transcript-{language or 'auto'}.json"
    scenes_path, audio_path = folder / "scenes.npz", folder / "audio.npz"
    video = Path(source.path)

    tmp = Path(tempfile.mkdtemp(prefix="analysis-", dir=settings.tmp_dir))
    try:
        wav = tmp / "audio.wav"
        need_audio = info.has_audio and not (transcript_path.exists() and audio_path.exists())
        if need_audio:
            ctx.update(stage="extracting audio", progress=0.01)
            extract_audio(video, wav, ctx.should_cancel)

        if info.has_audio:
            ctx.update(stage="transcribing", progress=0.03)
            result = transcribe(
                wav,
                info.duration,
                cache=transcript_path,
                language=language,
                on_progress=lambda p: ctx.update(progress=0.03 + p * 0.22),
                should_cancel=ctx.should_cancel,
            )
        else:
            result = {"language": None, "words": []}
        ctx.check_cancel()

        ctx.update(stage="analyzing audio", progress=0.26)
        if not info.has_audio:
            audio = silent_audio(info.duration)
        elif audio_path.exists():
            audio = _load_audio(audio_path)
        else:
            audio = analyze_audio(wav)
            _save_audio(audio_path, audio)
        ctx.check_cancel()

        ctx.update(stage="detecting scenes", progress=0.28)
        if scenes_path.exists():
            scenes = _load_scenes(scenes_path)
        else:
            scenes = analyze_scenes(
                video,
                info.fps,
                info.duration,
                on_progress=lambda p: ctx.update(progress=0.28 + p * (ANALYSIS_SHARE - 0.28)),
                should_cancel=ctx.should_cancel,
            )
            np.savez_compressed(
                scenes_path,
                fps=scenes.fps,
                frame_count=scenes.frame_count,
                cut_frames=np.array(scenes.cut_frames, dtype=np.int64),
                motion=scenes.motion,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return Analysis(words_from_dicts(result["words"]), result.get("language"), audio, scenes)


def _save_audio(path: Path, audio: AudioAnalysis) -> None:
    np.savez_compressed(
        path,
        hop=audio.hop,
        rms_db=audio.rms_db,
        flux=audio.flux,
        silences=np.array(audio.silences, dtype=np.float32).reshape(-1, 2),
        peaks=np.array(audio.peaks, dtype=np.float32),
        noise_floor_db=audio.noise_floor_db,
    )


def _load_audio(path: Path) -> AudioAnalysis:
    with np.load(path) as data:
        return AudioAnalysis(
            float(data["hop"]),
            data["rms_db"],
            data["flux"],
            [(float(a), float(b)) for a, b in data["silences"]],
            [float(p) for p in data["peaks"]],
            float(data["noise_floor_db"]),
        )


def _load_scenes(path: Path) -> SceneAnalysis:
    with np.load(path) as data:
        return SceneAnalysis(
            float(data["fps"]), int(data["frame_count"]), [int(c) for c in data["cut_frames"]], data["motion"]
        )


def load_preset(preset_id: str | None) -> CaptionPreset:
    key = preset_id or "anime-impact"
    with session_scope() as session:
        row = session.get(Preset, key)
    if row is not None:
        return CaptionPreset(**row.data)
    if key in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[key]
    raise ValueError(f"Caption preset not found: {key}")


def resolve_options(
    job: Job, analysis: Analysis, info: MediaInfo, video: Path, channel: Channel | None
) -> tuple[dict, dict]:
    """Return (options, decisions). Manual mode uses what the user chose."""
    if job.mode == "manual":
        return dict(job.settings["manual"]), {}
    decision = decide(
        analysis.words, analysis.audio, analysis.scenes, info, video, channel.settings if channel else None
    )
    return decision.options(), decision.report()


def _plan_clips(
    job: Job, options: dict, analysis: Analysis, info: MediaInfo
) -> list[tuple[Moment, list[Segment]]]:
    edit = job.settings.get("edit", {})
    cuts = job.settings.get("frame_cuts") or []
    if cuts:
        planned = []
        for cut in cuts:
            segment = Segment(cut["start_frame"], min(cut["end_frame"], info.frame_count))
            if segment.frames <= 0:
                raise ValueError(f"Cut starts past the end of the video ({info.frame_count} frames).")
            start, end = segment.start_frame / info.fps, segment.end_frame / info.fps
            text = " ".join(w.text for w in analysis.words if start <= w.start and w.end <= end)
            planned.append((Moment(start, end, 0.0, {}, text), [segment]))
        return planned

    moments = find_moments(
        analysis.words,
        analysis.audio,
        analysis.scenes,
        info.duration,
        options["genre"],
        options["clip_duration"],
        options["clip_count"],
    )
    ratio = options.get("min_score_ratio")
    if ratio and moments:
        # Auto mode keeps quality over quantity: drop moments far weaker than the best one.
        best = max(m.score for m in moments)
        moments = [m for m in moments if m.score >= best * ratio]
    planned = []
    cut_times = analysis.scenes.cut_times
    for moment in moments:
        start, end = refine(moment.start, moment.end, analysis.words, cut_times, info.fps, info.duration)
        moment.start, moment.end = start, end
        if not moment.text:  # Sparse speech uses non-verbal units, which carry no text of their own.
            moment.text = " ".join(w.text for w in analysis.words if start <= w.start and w.end <= end)
        spans = [(start, end)]
        if edit.get("remove_pauses"):
            spans = remove_pauses(start, end, analysis.words)
        planned.append((moment, to_frame_segments(spans, info.fps, info.frame_count)))
    return planned


def _clip_progress(ctx: JobContext, base: float, step: float) -> Callable[[float], None]:
    return lambda p: ctx.update(progress=base + p * step * 0.9)


def run_job(ctx: JobContext) -> None:
    settings = get_settings()
    with session_scope() as session:
        job = session.get(Job, ctx.job_id)
        assert job is not None
        source = session.get(Source, job.source_id)
        channel = session.get(Channel, job.channel_id) if job.channel_id else None

    # Checked again here so no code path can ever process a video without a rights record.
    if job.rights_type not in {r.value for r in RightsType}:
        raise ValueError("This job has no source rights record, so it cannot run.")
    if source is None or not Path(source.path).exists():
        raise ValueError("The source video is missing.")

    ctx.update(stage="reading video", progress=0.0)
    info = probe(source.path)
    edit = job.settings.get("edit", {})
    analysis = load_or_analyze(source, info, edit.get("language"), ctx)

    ctx.update(stage="choosing moments", progress=ANALYSIS_SHARE)
    options, decisions = resolve_options(job, analysis, info, Path(source.path), channel)
    planned = _plan_clips(job, options, analysis, info)

    notes = []
    wanted = options["clip_count"]
    if job.mode == "manual" and not job.settings.get("frame_cuts") and len(planned) < wanted:
        notes.append(
            f"Asked for {wanted} clips, but this video only has {len(planned)} good non-overlapping moments."
        )
    if not planned:
        raise ValueError(
            f"No clip fits these settings. Try another clip length (the video lasts {info.duration:.0f} s)."
        )
    decisions = {
        **decisions,
        "language": analysis.language,
        "notes": list(notes),
        "words": len(analysis.words),
    }
    with session_scope() as session:
        row = session.get(Job, job.id)
        assert row is not None
        row.decisions = decisions
        session.add(row)
        session.commit()

    preset = None
    if edit.get("captions", True):
        chosen = edit.get("preset_id") or (channel.settings.get("preset_id") if channel else None)
        preset = load_preset(chosen or options.get("auto_preset"))
        missing = undrawable_words(analysis.words, preset, analysis.language)
        if missing:
            notes.append(
                f"{len(missing)} caption words use letters no installed font can draw, such as "
                f'"{missing[0]}", so they may appear as empty boxes. Japanese, Korean and Chinese need '
                "the font from scripts/setup.sh. Right-to-left languages such as Arabic are not supported."
            )
    templates = channel.settings if channel else {}
    corpus = " ".join(w.text for w in analysis.words)
    out_dir = settings.renders_dir / job.id
    numbered = options.get("part_numbering", False) and len(planned) > 1

    for index, (moment, segments) in enumerate(planned):
        ctx.check_cancel()
        base = ANALYSIS_SHARE + (1 - ANALYSIS_SHARE) * index / len(planned)
        step = (1 - ANALYSIS_SHARE) / len(planned)
        ctx.update(stage=f"rendering clip {index + 1} of {len(planned)}", progress=base)
        clip_id = f"{job.id[:8]}{index:02d}"
        with session_scope() as session:
            if session.get(Clip, clip_id) is not None:  # Restarted job: this clip is already done.
                continue
        output = out_dir / f"{clip_id}.mp4"
        spec = RenderSpec(
            source=Path(source.path),
            info=info,
            segments=segments,
            ratio=options["ratio"],
            resolution=options["resolution"],
            output=output,
            words=analysis.words,
            preset=preset,
            fit=edit.get("fit", "smart"),
            transition=edit.get("transition", "cut"),
            cuts=analysis.scenes.cut_times,
            language=analysis.language,
        )
        result = render_clip(
            spec,
            on_progress=_clip_progress(ctx, base, step),
            should_cancel=ctx.should_cancel,
        )
        notes.extend(n for n in result.notes if n not in notes)
        ctx.update(stage=f"checking clip {index + 1} of {len(planned)}")
        qc = run_qc(
            result.path,
            result.width,
            result.height,
            result.duration,
            info.fps,
            info.has_audio,
            source_height=min(info.display_size),
        )
        thumb = make_thumbnail(result.path, output.with_suffix(".jpg"), min(result.duration / 3, 2.0))
        part = index + 1 if numbered else None
        title, description = build_metadata(
            moment.text,
            corpus,
            options["genre"],
            part,
            options["format"],
            job.rights_type,
            job.rights_detail,
            templates.get("title_template", "{title}"),
            templates.get("description_template", "{summary}\n\n{hashtags}"),
        )
        now = utcnow()
        with session_scope() as session:
            session.add(
                Clip(
                    id=clip_id,
                    job_id=job.id,
                    part=part,
                    start=moment.start,
                    end=moment.end,
                    segments=[[s.start_frame, s.end_frame] for s in segments],
                    score=moment.score,
                    reasons=moment.reasons,
                    title=title,
                    description=description,
                    format=options["format"],
                    ratio=options["ratio"],
                    resolution=options["resolution"],
                    width=result.width,
                    height=result.height,
                    duration=result.duration,
                    frames=result.frames,
                    size_bytes=result.path.stat().st_size,
                    path=str(result.path),
                    thumb_path=str(thumb),
                    preset_id=edit.get("preset_id") or options.get("auto_preset"),
                    qc={**qc, "encoder": result.encoder},
                    created_at=now,
                    expires_at=now + timedelta(hours=settings.clip_ttl_hours),
                )
            )
            session.commit()
    if notes != decisions["notes"]:
        with session_scope() as session:
            row = session.get(Job, job.id)
            assert row is not None
            row.decisions = {**row.decisions, "notes": notes}
            session.add(row)
            session.commit()
    log.info("Job %s finished: %s", job.id, json.dumps({"clips": len(planned)}))
