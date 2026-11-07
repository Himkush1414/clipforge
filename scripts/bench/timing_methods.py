#!/usr/bin/env python3
"""Ways to tighten Whisper's word timing, scored on the same exact references as speech_timing.py.

Signed errors matter here: a fixed offset can be corrected, while scatter cannot.
Runs in the benchmark environment, which has the voices and WhisperX. WhisperX is only a reference
point here: clipforge does not use it, and its default Spanish model is licensed CC BY-NC.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import speech_timing as st

from clipforge.analysis import align
from clipforge.analysis.transcribe import Word


def signed(found: list[dict], truth: list[dict]) -> dict:
    a, b = [st.normalize(w["text"]) for w in found], [st.normalize(w["text"]) for w in truth]
    starts, ends = [], []
    for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            f, t = found[block.a + i], truth[block.b + i]
            starts.append((f["start"] - t["start"]) * 1000)
            ends.append((f["end"] - t["end"]) * 1000)
    if not starts:
        return {"words_matched": 0}
    s, e = np.array(starts), np.array(ends)
    return {
        "words_matched": len(starts),
        "start_median_abs_ms": round(float(np.median(np.abs(s))), 1),
        "start_p90_abs_ms": round(float(np.percentile(np.abs(s), 90)), 1),
        "start_max_abs_ms": round(float(np.abs(s).max()), 1),
        "start_median_signed_ms": round(float(np.median(s)), 1),
        "start_within_50ms": round(float((np.abs(s) <= 50).mean()), 3),
        "start_within_100ms": round(float((np.abs(s) <= 100).mean()), 3),
        "end_median_abs_ms": round(float(np.median(np.abs(e))), 1),
        "end_p90_abs_ms": round(float(np.percentile(np.abs(e), 90)), 1),
        "end_max_abs_ms": round(float(np.abs(e).max()), 1),
        "end_median_signed_ms": round(float(np.median(e)), 1),
    }


def clipforge_align(spans: list[dict], audio: np.ndarray, language: str) -> list[dict]:
    segments = [
        align.Segment(s["start"], s["end"], [Word(w["text"], w["start"], w["end"]) for w in s["words"]])
        for s in spans
    ]
    words, method = align.align_segments(audio, segments, language)
    if method != "aligned":
        raise RuntimeError(f"no alignment model installed for {language}")
    return [{"text": w.text, "start": w.start, "end": w.end} for w in words]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path.home() / "clipforge-work")
    parser.add_argument("--languages", default="en,es,hi,ar")
    parser.add_argument("--whisperx", action="store_true", help="also score WhisperX, for reference")
    args = parser.parse_args()
    voices = args.work / "bench" / "voices"
    samples: dict[str, tuple[np.ndarray, list[dict], str]] = {}
    for language in args.languages.split(","):
        samples[f"{language} word by word"] = (
            *st.word_by_word(st.piper_say(voices, language), language),
            language,
        )
    if "en" in args.languages:
        samples["en natural"] = (*st.kokoro_natural(), "en")
    whisper = st.Whisper("large-v3-turbo", "int8_float16", args.work / "models")
    results: dict = {}
    for name, (audio, truth, language) in samples.items():
        start = time.time()
        words, spans, _ = whisper.run(audio, language)
        row = {"whisper": {**signed(words, truth), "seconds": round(time.time() - start, 1)}}
        start = time.time()
        try:
            row["clipforge align"] = {
                **signed(clipforge_align(spans, audio, language), truth),
                "seconds": round(time.time() - start, 2),
            }
        except Exception as exc:
            row["clipforge align"] = {"error": repr(exc)[:160]}
        if args.whisperx:
            import torch

            torch.cuda.empty_cache()
            start = time.time()
            try:
                plain = [{k: s[k] for k in ("text", "start", "end")} for s in spans]
                row["whisperx align"] = {
                    **signed(st.whisperx_align(plain, audio, language), truth),
                    "seconds": round(time.time() - start, 1),
                }
            except Exception as exc:
                row["whisperx align"] = {"error": repr(exc)[:160]}
            gc.collect()
            torch.cuda.empty_cache()
        results[name] = row
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
