"""Word-level transcription with faster-whisper, cached per source."""

from __future__ import annotations

import gc
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from clipforge.config import get_settings
from clipforge.hardware import detect, load_gpu_runtime
from clipforge.media.ffmpeg import Cancelled, run_ffmpeg

log = logging.getLogger("clipforge.transcribe")

ALIGN_PROGRESS_START = 0.9


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0


def extract_audio(video: Path, out: Path, should_cancel: Callable[[], bool] | None = None) -> Path:
    """Decode to 16 kHz mono WAV: what Whisper and the audio analysis both use.

    Always the first audio track, the one renders play. Left to itself FFmpeg picks the track flagged as
    default, so a file flagging its second language would get captions that do not match its sound.
    """
    run_ffmpeg(
        ["-i", video, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        should_cancel=should_cancel,
    )
    return out


def sanitize_words(words: list[Word], min_length: float = 0.04) -> list[Word]:
    """Whisper can emit overlapping or zero-length words. Force increasing, non-overlapping times."""
    clean: list[Word] = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        start = max(word.start, clean[-1].end if clean else 0.0)
        end = max(word.end, start + min_length)
        clean.append(Word(text, round(start, 3), round(end, 3), round(word.probability, 3)))
    return clean


def _load_model(device: str):
    from faster_whisper import WhisperModel

    settings = get_settings()
    # Keeps every download on the work drive, whatever the machine's default cache location is.
    os.environ.setdefault("HF_HOME", str(settings.models_dir / "hf"))
    if device == "cuda":
        load_gpu_runtime()
        name, compute = settings.whisper_model_gpu, "int8_float16"
    else:
        name, compute = settings.whisper_model_cpu, "int8"
    log.info("Loading Whisper %s on %s (%s)", name, device, compute)
    return WhisperModel(name, device=device, compute_type=compute, download_root=str(settings.models_dir))


def transcribe(
    audio: Path,
    duration: float,
    cache: Path | None = None,
    language: str | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Return {language, words}. Falls back to CPU when the GPU fails (out of memory, driver)."""
    if cache and cache.exists():
        return json.loads(cache.read_text())

    device = detect().whisper_device
    try:
        result = _run(audio, duration, device, language, on_progress, should_cancel)
    except Cancelled:
        raise
    except RuntimeError as exc:
        if device != "cuda":
            raise
        log.warning("GPU transcription failed (%s); retrying on CPU", exc)
        result = _run(audio, duration, "cpu", language, on_progress, should_cancel)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
    return result


def _run(audio, duration, device, language, on_progress, should_cancel) -> dict:
    # Decoded here so faster-whisper never needs its PyAV path. Our WAV is already 16 kHz mono.
    _, pcm = wavfile.read(audio, mmap=True)
    samples = np.asarray(pcm, dtype=np.float32) / 32768.0
    from clipforge.analysis.align import Segment, align_segments, model_folder

    model = _load_model(device)
    found: list[Segment] = []
    try:
        segments, info = model.transcribe(
            samples,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
            condition_on_previous_text=False,
        )
        # Whisper's share of the progress bar, when alignment will run after it.
        share = ALIGN_PROGRESS_START if model_folder(language or info.language) else 1.0
        for segment in segments:
            if should_cancel and should_cancel():
                raise Cancelled("Cancelled")
            found.append(
                Segment(
                    float(segment.start),
                    float(segment.end),
                    [
                        Word(w.word, float(w.start), float(w.end), float(w.probability))
                        for w in segment.words or []
                    ],
                )
            )
            if on_progress and duration:
                on_progress(share * min(1.0, segment.end / duration))
    finally:
        # shortcut: RAM is 7.6 GB here, so the model is reloaded per video instead of kept alive.
        del model
        gc.collect()
    words, timing = align_segments(
        samples,
        found,
        info.language,
        should_cancel,
        (lambda done: on_progress(share + (1.0 - share) * done)) if on_progress else None,
    )
    return {
        "language": info.language,
        "timing": timing,
        "words": [asdict(w) for w in sanitize_words(words)],
    }


def words_from_dicts(rows: list[dict]) -> list[Word]:
    return [Word(r["text"], r["start"], r["end"], r.get("probability", 1.0)) for r in rows]
