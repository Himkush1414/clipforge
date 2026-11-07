#!/usr/bin/env python3
"""Word timing and transcription accuracy against known answers, using generated voices.

Word-by-word samples: each word is synthesized alone, then joined with known gaps, so every word's
true start and end are exact. Natural samples: Kokoro reports its own word timings and sentences flow.
Accuracy samples: the script is the reference. Runs in the benchmark environment, not the app's.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import subprocess
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

RATE = 16000
SCRIPTS = {
    "en": "Welcome back to the workshop. Today we are building a tiny wooden boat from scrap pine. Most people think you need expensive tools for this, but you really do not. Now here is the secret nobody tells you. Always cut away from the centre line, never towards it. If you remember only one thing today, remember that. The light you see tonight left those stars thousands of years ago. Good bread needs only four ingredients, and the biggest mistake is rushing the dough.",
    "es": "Bienvenidos de nuevo al taller. Hoy vamos a construir un pequeño barco de madera. Mucha gente piensa que necesita herramientas caras, pero no es verdad. Aquí está el secreto que nadie te cuenta. Corta siempre hacia fuera y nunca hacia ti. El buen pan solo necesita cuatro ingredientes y un poco de paciencia.",
    "hi": "नमस्ते दोस्तों आज हम लकड़ी की एक छोटी नाव बनाएंगे। बहुत से लोग सोचते हैं कि इसके लिए महंगे औजार चाहिए लेकिन ऐसा नहीं है। अब मैं आपको एक राज़ बताता हूँ। अच्छी रोटी के लिए आटा पानी और थोड़ा धैर्य चाहिए। आज रात आसमान में बहुत सारे तारे दिखाई देंगे।",
    "ar": "مرحبا بكم مرة أخرى في الورشة. اليوم سنبني قاربا خشبيا صغيرا. يعتقد كثير من الناس أنهم بحاجة إلى أدوات غالية لكن هذا غير صحيح. الآن سأخبركم بالسر الذي لا يخبركم به أحد. الخبز الجيد يحتاج إلى الدقيق والماء والصبر.",
    "ja": "皆さん、こんにちは。今日は小さな木の船を作ります。高い道具が必要だと思う人が多いですが、そんなことはありません。ここで誰も教えてくれない秘密をお話しします。おいしいパンに必要なのは、小麦粉と水と少しの忍耐です。今夜はたくさんの星が見えるでしょう。",
}
PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "es": "es_ES-davefx-medium",
    "hi": "hi_IN-pratham-medium",
    "ar": "ar_JO-kareem-medium",
}
WHISPERX_ALIGN = {
    "en": None,
    "es": None,
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-japanese",
}


class VramWatch:
    def __enter__(self):
        self.peak, self._stop = 0, False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop:
            out = subprocess.run(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if out.isdigit():
                self.peak = max(self.peak, int(out))
            time.sleep(0.2)

    def __exit__(self, *_):
        self._stop = True
        self._thread.join(timeout=2)


def to_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    return (
        resample_poly(audio.astype(np.float64), RATE, rate).astype(np.float32)
        if rate != RATE
        else audio.astype(np.float32)
    )


def trim(audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    loud = np.where(np.abs(audio) > threshold * max(np.abs(audio).max(), 1e-6))[0]
    return audio[loud[0] : loud[-1] + 1] if loud.size else audio


def piper_say(voices: Path, language: str):
    from piper import PiperVoice

    voice = PiperVoice.load(str(voices / f"{PIPER_VOICES[language]}.onnx"))

    def say(text: str) -> np.ndarray:
        chunks = list(voice.synthesize(text))
        return to_rate(np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate)

    return say


def word_by_word(say, language: str, seed: int = 3) -> tuple[np.ndarray, list[dict]]:
    """Exact truth: every word's position in the file is known to the sample."""
    rng = random.Random(seed)
    pieces, truth = [np.zeros(int(0.5 * RATE), np.float32)], []
    position = pieces[0].size
    for sentence in re.split(r"(?<=[.。!?؟।])\s*", SCRIPTS[language]):
        tokens = sentence.split()
        for index, token in enumerate(tokens):
            voice = trim(say(token))
            truth.append({"text": token, "start": position / RATE, "end": (position + voice.size) / RATE})
            pieces.append(voice)
            position += voice.size
            gap = np.zeros(
                int(rng.uniform(0.04, 0.14) * RATE) if index + 1 < len(tokens) else int(0.5 * RATE),
                np.float32,
            )
            pieces.append(gap)
            position += gap.size
    return np.concatenate(pieces), truth


def kokoro_natural(language: str = "en") -> tuple[np.ndarray, list[dict]]:
    """Natural flowing sentences. Truth is the word timing Kokoro used to generate the audio."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    pieces, truth = [], []
    offset = 0.0
    for result in pipeline(SCRIPTS[language], voice="af_heart", split_pattern=r"(?<=[.!?])\s+"):
        audio = result.audio.numpy() if hasattr(result.audio, "numpy") else np.asarray(result.audio)
        for token in result.tokens or []:
            if token.start_ts is not None and token.end_ts is not None and re.search(r"\w", token.text):
                truth.append(
                    {"text": token.text, "start": offset + token.start_ts, "end": offset + token.end_ts}
                )
        resampled = to_rate(audio, 24000)
        pieces.append(resampled)
        offset += resampled.size / RATE
    return np.concatenate(pieces), truth


def kokoro_japanese() -> np.ndarray:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")
    pieces = []
    for result in pipeline(SCRIPTS["ja"], voice="jf_alpha", split_pattern=r"(?<=。)"):
        pieces.append(to_rate(np.asarray(result.audio), 24000))
        pieces.append(np.zeros(int(0.3 * RATE), np.float32))
    return np.concatenate(pieces)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNM" or ch.isspace())
    return re.sub(r"\s+", " ", text).strip()


def timing_errors(found: list[dict], truth: list[dict]) -> dict:
    a, b = [normalize(w["text"]) for w in found], [normalize(w["text"]) for w in truth]
    starts, ends = [], []
    for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            f, t = found[block.a + i], truth[block.b + i]
            starts.append(abs(f["start"] - t["start"]) * 1000)
            ends.append(abs(f["end"] - t["end"]) * 1000)
    if not starts:
        return {"words_matched": 0, "words_total": len(truth)}
    return {
        "words_matched": len(starts),
        "words_total": len(truth),
        "start_median_ms": round(float(np.median(starts)), 1),
        "start_p90_ms": round(float(np.percentile(starts, 90)), 1),
        "start_max_ms": round(float(np.max(starts)), 1),
        "end_median_ms": round(float(np.median(ends)), 1),
        "end_p90_ms": round(float(np.percentile(ends, 90)), 1),
    }


def accuracy(found: str, language: str) -> dict:
    from jiwer import cer, wer

    reference, hypothesis = normalize(SCRIPTS[language]), normalize(found)
    if language in ("ja", "zh"):
        reference, hypothesis = reference.replace(" ", ""), hypothesis.replace(" ", "")
    return {
        "wer": None if language in ("ja", "zh") else round(wer(reference, hypothesis), 3),
        "cer": round(cer(reference, hypothesis), 3),
    }


class Whisper:
    def __init__(self, name: str, compute: str, models: Path):
        from faster_whisper import WhisperModel

        from clipforge.hardware import load_gpu_runtime

        load_gpu_runtime()
        self.model = WhisperModel(name, device="cuda", compute_type=compute, download_root=str(models))

    def run(self, audio: np.ndarray, language: str | None) -> tuple[list[dict], list[dict], str]:
        """clipforge's own settings. Returns (words, segments, detected language)."""
        segments, info = self.model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
            condition_on_previous_text=False,
        )
        words, spans = [], []
        for segment in segments:
            found = [{"text": w.word.strip(), "start": w.start, "end": w.end} for w in segment.words or []]
            spans.append({"text": segment.text, "start": segment.start, "end": segment.end, "words": found})
            words += found
        return words, spans, info.language


def whisperx_align(spans: list[dict], audio: np.ndarray, language: str) -> list[dict]:
    import whisperx.alignment as alignment

    model, meta = alignment.load_align_model(language, "cuda", model_name=WHISPERX_ALIGN.get(language))
    result = alignment.align(spans, model, meta, audio, "cuda", return_char_alignments=False)
    words = [
        {"text": w["word"], "start": w["start"], "end": w["end"]}
        for w in result["word_segments"]
        if "start" in w
    ]
    del model
    return words


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path.home() / "clipforge-work")
    parser.add_argument("--models", default="large-v3-turbo:int8_float16,large-v3:int8_float16")
    parser.add_argument("--languages", default="en,es,hi,ar,ja")
    args = parser.parse_args()
    voices = args.work / "bench" / "voices"
    models_dir = args.work / "models"
    audio_sets: dict[str, dict] = {}
    for language in args.languages.split(","):
        entry: dict = {"language": language}
        if language in PIPER_VOICES:
            audio, truth = word_by_word(piper_say(voices, language), language)
            entry["word_by_word"] = {"audio": audio, "truth": truth}
            say = piper_say(voices, language)
            parts = []
            for sentence in re.split(r"(?<=[.。!?؟।])\s*", SCRIPTS[language]):
                if sentence.strip():
                    parts += [say(sentence), np.zeros(int(0.4 * RATE), np.float32)]
            entry["sentences"] = np.concatenate(parts)
        if language == "ja":
            entry["sentences"] = kokoro_japanese()
        if language == "en":
            audio, truth = kokoro_natural()
            entry["natural"] = {"audio": audio, "truth": truth}
        audio_sets[language] = entry
        print(
            "samples ready:",
            language,
            {
                k: (v["audio"].size / RATE if isinstance(v, dict) else v.size / RATE)
                for k, v in entry.items()
                if k != "language"
            },
            flush=True,
        )

    results: dict = {"note": "Generated voices. Reference text and timings are known exactly."}
    import torch

    for spec in args.models.split(","):
        name, compute = spec.split(":")
        with VramWatch() as vram:
            loaded = time.time()
            whisper = Whisper(name, compute, models_dir)
            load_seconds = time.time() - loaded
            model_results: dict = {"load_seconds": round(load_seconds, 1)}
            for language, entry in audio_sets.items():
                row: dict = {}
                clip = entry["sentences"]
                start = time.time()
                words, spans, detected = whisper.run(clip, None)
                row["speed_x_realtime"] = round(clip.size / RATE / (time.time() - start), 1)
                row["detected_language"] = detected
                row["detected_correct"] = detected == language
                row["accuracy_auto_language"] = accuracy(" ".join(w["text"] for w in words), language)
                words, spans, _ = whisper.run(clip, language)
                row["accuracy_language_given"] = accuracy(" ".join(w["text"] for w in words), language)
                for sample in ("word_by_word", "natural"):
                    if sample in entry:
                        words, spans, _ = whisper.run(entry[sample]["audio"], language)
                        row[f"timing_{sample}"] = timing_errors(words, entry[sample]["truth"])
                        if name == "large-v3-turbo":
                            torch.cuda.empty_cache()
                            aligned_start = time.time()
                            try:
                                aligned = whisperx_align(spans, entry[sample]["audio"], language)
                                row[f"timing_{sample}_whisperx_align"] = {
                                    **timing_errors(aligned, entry[sample]["truth"]),
                                    "seconds": round(time.time() - aligned_start, 1),
                                }
                            except Exception as exc:
                                row[f"timing_{sample}_whisperx_align"] = {"error": repr(exc)[:200]}
                            torch.cuda.empty_cache()
                model_results[language] = row
                print(name, language, json.dumps(row, ensure_ascii=False), flush=True)
        model_results["peak_vram_mb"] = vram.peak
        results[name] = model_results
        del whisper
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
