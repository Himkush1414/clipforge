#!/usr/bin/env python3
"""Compare ways of finding silences, on material with known gaps.

Sets: seeded pink noise and steady noise with digital and room-tone gaps (the question clipforge asks
of videos with no speech: where is it quiet?), generated speech with exact pauses, and a real podcast,
whose reference pauses come from the aligned transcript and are approximate.

Methods: clipforge's current threshold, the proposed one, Silero VAD v6 (bundled with faster-whisper),
WebRTC VAD and FFmpeg silencedetect. A speech detector answers "is someone talking?", not "is it quiet?",
so on noise it is expected to fail. It is scored anyway, so the table shows where each one applies.

Usage: silence_eval.py OUT.json [--podcast WAV --transcript JSON]   (runs in the benchmark environment)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from clipforge.analysis.audio import find_quiet, find_silences, frame_levels
from clipforge.media.ffmpeg import find_tool

RATE = 16000
MIN_GAP = 0.35


def write(path: Path, samples: np.ndarray) -> Path:
    wavfile.write(path, RATE, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
    return path


def pink(seconds: float, seed: int, level: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(seconds * RATE))
    spectrum = np.fft.rfft(white)
    spectrum /= np.sqrt(np.maximum(np.arange(spectrum.size), 1))
    out = np.fft.irfft(spectrum, white.size)
    return (out / np.abs(out).max() * level).astype(np.float32)


def with_gaps(signal: np.ndarray, gaps: list[tuple[float, float]], floor: float = 0.0, seed: int = 3) -> np.ndarray:
    out = signal.copy()
    rng = np.random.default_rng(seed)
    for a, b in gaps:
        i, j = int(a * RATE), int(b * RATE)
        out[i:j] = rng.standard_normal(j - i).astype(np.float32) * floor
    return out


NOISE_GAPS = [(3.0, 4.0), (7.5, 8.0), (11.0, 13.0), (16.0, 16.4)]
STEADY_GAPS = [(2.0, 2.5), (5.0, 6.2), (9.0, 9.4), (12.5, 14.0), (17.0, 17.6)]


def noise_sets(tmp: Path) -> dict[str, tuple[Path, list]]:
    """Quiet gaps in loud signals."""
    rng = np.random.default_rng(5)
    white = (rng.standard_normal(20 * RATE) * 0.1).astype(np.float32)
    brown = np.cumsum(rng.standard_normal(20 * RATE)).astype(np.float32)
    brown = (brown - np.convolve(brown, np.ones(4000) / 4000, "same")) / 60
    return {
        "pink noise, seeded, digital gaps": (write(tmp / "pink.wav", with_gaps(pink(20, 1), NOISE_GAPS)), NOISE_GAPS),
        "pink noise, room tone in gaps": (
            write(tmp / "pink-room.wav", with_gaps(pink(20, 2), NOISE_GAPS, floor=0.003)),
            NOISE_GAPS,
        ),
        "white noise, steady": (write(tmp / "white.wav", with_gaps(white, STEADY_GAPS)), STEADY_GAPS),
        "brown noise, room tone in gaps": (
            write(tmp / "brown.wav", with_gaps(np.clip(brown, -0.6, 0.6), STEADY_GAPS, floor=0.002)),
            STEADY_GAPS,
        ),
        "pink noise, no gaps at all": (write(tmp / "pink-solid.wav", pink(20, 4)), []),
    }


def speech_set(tmp: Path) -> tuple[Path, list] | None:
    """Generated speech with exact pauses, over quiet room tone."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import speech_timing as st
    except ImportError:
        return None
    say = st.piper_say(Path.home() / "clipforge-work" / "bench" / "voices", "en")
    rng = np.random.default_rng(9)
    pieces, gaps, position = [np.zeros(RATE // 2, np.float32)], [], RATE // 2
    gaps.append((0.0, 0.5))
    for sentence in re.split(r"(?<=[.!?])\s+", st.SCRIPTS["en"]):
        voice = st.trim(say(sentence))
        pause = float(rng.choice([0.2, 0.3, 0.45, 0.6, 0.9, 1.4]))
        pieces += [voice, np.zeros(int(pause * RATE), np.float32)]
        start = (position + voice.size) / RATE
        position += voice.size + int(pause * RATE)
        if pause >= MIN_GAP:
            gaps.append((start, position / RATE))
    audio = np.concatenate(pieces)
    audio += (np.random.default_rng(1).standard_normal(audio.size) * 0.001).astype(np.float32)
    return write(tmp / "speech.wav", audio), gaps


def podcast_set(wav: Path, transcript: Path, seconds: float) -> tuple[Path, list]:
    words = json.loads(transcript.read_text())["words"]
    words = [w for w in words if w["end"] <= seconds]
    gaps = [(a["end"], b["start"]) for a, b in zip(words, words[1:], strict=False) if b["start"] - a["end"] >= MIN_GAP]
    return wav, gaps


# ---------- methods ----------
def current(wav: Path) -> list[tuple[float, float]]:
    """The method before this change: 50 ms frames, a threshold 30% of the way from floor to loud."""
    _, samples = wavfile.read(wav, mmap=True)
    levels = frame_levels(samples, int(RATE * 0.05))
    audible = levels[levels > -80.0]
    floor = float(np.percentile(audible, 10)) if audible.size else -90.0
    loud = float(np.percentile(audible, 90)) if audible.size else -30.0
    threshold = float(np.clip(floor + (loud - floor) * 0.3, -60.0, -25.0))
    return find_silences(levels, 0.05, threshold)


def proposed(wav: Path) -> list[tuple[float, float]]:
    return find_quiet(wav)


def silero(wav: Path) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    _, pcm = wavfile.read(wav)
    audio = pcm.astype(np.float32) / 32768.0
    speech = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=int(MIN_GAP * 1000), speech_pad_ms=0, min_speech_duration_ms=100)
    )
    return complement([(s["start"] / RATE, s["end"] / RATE) for s in speech], audio.size / RATE)


def webrtc(wav: Path, mode: int = 3) -> list[tuple[float, float]]:
    import webrtcvad

    _, pcm = wavfile.read(wav)
    vad = webrtcvad.Vad(mode)
    step = int(0.03 * RATE)
    voiced = []
    for i in range(0, pcm.size - step, step):
        if vad.is_speech(pcm[i : i + step].astype(np.int16).tobytes(), RATE):
            voiced.append((i / RATE, (i + step) / RATE))
    return complement(merge(voiced, 0.1), pcm.size / RATE)


def ffmpeg_silencedetect(wav: Path) -> list[tuple[float, float]]:
    out = subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-i", str(wav), "-af", f"silencedetect=noise=-35dB:d={MIN_GAP}", "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
    duration = len(wavfile.read(wav)[1]) / RATE
    return [(a, ends[i] if i < len(ends) else duration) for i, a in enumerate(starts)]


def merge(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in spans:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def complement(spans: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out, cursor = [], 0.0
    for a, b in spans:
        if a - cursor >= MIN_GAP:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if duration - cursor >= MIN_GAP:
        out.append((cursor, duration))
    return out


METHODS = {
    "current": current,
    "proposed": proposed,
    "Silero VAD v6": silero,
    "WebRTC VAD": webrtc,
    "FFmpeg silencedetect": ffmpeg_silencedetect,
}


def score(found: list[tuple[float, float]], truth: list[tuple[float, float]]) -> dict:
    """Each true gap matches at most one found gap, by overlap. Edge errors are over matched pairs."""
    used: set[int] = set()
    edges: list[float] = []
    matched = 0
    for a, b in truth:
        best, best_overlap = None, 0.0
        for index, (c, d) in enumerate(found):
            share = max(0.0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if index not in used and share > best_overlap:
                best, best_overlap = index, share
        if best is not None and best_overlap >= 0.3:
            used.add(best)
            matched += 1
            edges += [abs(found[best][0] - a) * 1000, abs(found[best][1] - b) * 1000]
    return {
        "gaps": len(truth),
        "found": matched,
        "false_gaps": len(found) - len(used),
        "edge_mean_ms": round(float(np.mean(edges)), 1) if edges else None,
        "edge_worst_ms": round(float(np.max(edges)), 1) if edges else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--podcast", type=Path)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--podcast-seconds", type=float, default=600)
    args = parser.parse_args()
    results: dict = {}
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        sets = noise_sets(tmp)
        speech = speech_set(tmp)
        if speech:
            sets["generated speech, exact pauses"] = speech
        if args.podcast and args.transcript:
            sets["real podcast, approximate pauses"] = podcast_set(args.podcast, args.transcript, args.podcast_seconds)
        for name, (wav, truth) in sets.items():
            row = {}
            for method, find in METHODS.items():
                start = time.time()
                try:
                    row[method] = {**score(find(wav), truth), "seconds": round(time.time() - start, 2)}
                except Exception as exc:
                    row[method] = {"error": repr(exc)[:160]}
            results[name] = row
            print(name)
            for method, cell in row.items():
                print(f"  {method:22} {cell}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
