#!/usr/bin/env python3
"""How much do the scorer's clips overlap the clips a person chose?

Reads a picks file (see my_picks.template.json), runs the scorer on each video, and reports:
  pick recall      share of the person's picks that a scorer clip landed on
  clip precision   share of the scorer's clips that landed on a pick
  mean IoU         for each pick, overlap with its best clip, divided by their combined span
A clip "lands on" a pick when they share at least half of the shorter one. Random windows of the same
length and number are scored too, since a score only means something next to what chance gets.

Analysis is cached per video, so scorer changes can be compared quickly. Audio-only files work too.

Usage:
  pick_eval.py picks.json [--scorer NAME] [--count N] [--out results.json]
  pick_eval.py picks.json --transcript     print each video's timed sentences, to help choose picks
Runs in the app's environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from clipforge.analysis.audio import analyze_audio
from clipforge.analysis.scenes import SceneAnalysis, analyze_scenes
from clipforge.analysis.scoring import find_moments, split_sentences
from clipforge.analysis.transcribe import extract_audio, transcribe, words_from_dicts
from clipforge.config import get_settings
from clipforge.media.ffmpeg import FFmpegError, find_tool
from clipforge.media.probe import MediaInfo, probe

CACHE = Path.home() / "clipforge-work" / "bench" / "pick-eval" / "cache"
RANDOM_TRIALS = 200


def seconds(value: float | int | str) -> float:
    """12.5, "754.5", "12:34.5" or "1:02:03" as seconds."""
    if isinstance(value, int | float):
        return float(value)
    total = 0.0
    for part in value.strip().split(":"):
        total = total * 60 + float(part)
    return total


def probe_or_none(path: Path) -> MediaInfo | None:
    """Audio-only files have no video details, and are scored with no scene information."""
    try:
        return probe(path)
    except FFmpegError:
        return None


def audio_seconds(path: Path) -> float:
    out = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(out.strip())


def load_analysis(path: Path, limit: float | None) -> dict:
    """Words, audio and scenes for a video, cached by file and limit."""
    stat = path.stat()
    key = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{limit}".encode()).hexdigest()[:16]
    folder = CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    info = probe_or_none(path)
    length = info.duration if info else audio_seconds(path)
    duration = min(length, limit) if limit else length
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        transcript = folder / "transcript.json"
        audio_cache = folder / "audio.pkl"
        if not (transcript.exists() and audio_cache.exists()):
            if limit:
                subprocess.run(
                    [
                        find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-t", str(limit),
                        "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
                    ],
                    check=True,
                )  # fmt: skip
            else:
                extract_audio(path, wav)
        result = transcribe(wav, duration, cache=transcript)
        if audio_cache.exists():
            import pickle

            audio = pickle.loads(audio_cache.read_bytes())  # noqa: S301 - our own cache file
        else:
            import pickle

            audio = analyze_audio(wav)
            audio_cache.write_bytes(pickle.dumps(audio))
    scenes_cache = folder / "scenes.npz"
    if scenes_cache.exists():
        with np.load(scenes_cache) as data:
            scenes = SceneAnalysis(
                float(data["fps"]), int(data["frame_count"]), [int(c) for c in data["cut_frames"]], data["motion"]
            )
    elif info:
        scenes = analyze_scenes(path, info.fps, info.duration)
        np.savez_compressed(
            scenes_cache,
            fps=scenes.fps,
            frame_count=scenes.frame_count,
            cut_frames=np.array(scenes.cut_frames, dtype=np.int64),
            motion=scenes.motion,
        )
    else:
        scenes = SceneAnalysis(30.0, int(duration * 30), [], np.zeros(int(duration * 30), dtype=np.float32))
    return {
        "words": words_from_dicts(result["words"]),
        "language": result.get("language"),
        "audio": audio,
        "scenes": scenes,
        "duration": duration,
    }


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def lands_on(clip: tuple[float, float], pick: tuple[float, float]) -> bool:
    shorter = min(clip[1] - clip[0], pick[1] - pick[0])
    return shorter > 0 and overlap(clip, pick) >= 0.5 * shorter


def compare(clips: list[tuple[float, float]], picks: list[tuple[float, float]]) -> dict:
    """The three scores for one set of clips against one set of picks."""
    if not clips or not picks:
        return {"pick_recall": 0.0, "clip_precision": 0.0, "mean_iou": 0.0}
    ious = []
    for pick in picks:
        best = max(overlap(clip, pick) / (max(clip[1], pick[1]) - min(clip[0], pick[0])) for clip in clips)
        ious.append(best)
    return {
        "pick_recall": sum(any(lands_on(c, p) for c in clips) for p in picks) / len(picks),
        "clip_precision": sum(any(lands_on(c, p) for p in picks) for c in clips) / len(clips),
        "mean_iou": statistics.mean(ious),
    }


def random_windows(duration: float, length: float, count: int, rng: random.Random) -> list[tuple[float, float]]:
    """Non-overlapping windows at random places."""
    chosen: list[tuple[float, float]] = []
    for _ in range(count * 40):
        if len(chosen) >= count or duration <= length:
            break
        start = rng.uniform(0, duration - length)
        window = (start, start + length)
        if all(overlap(window, c) == 0 for c in chosen):
            chosen.append(window)
    return chosen


def chance(duration: float, clips: list[tuple[float, float]], picks: list[tuple[float, float]]) -> dict:
    """Mean scores of many random sets, with the same clip length and number as the scorer's."""
    if not clips:
        return {"pick_recall": 0.0, "clip_precision": 0.0, "mean_iou": 0.0}
    length = statistics.mean(c[1] - c[0] for c in clips)
    rng = random.Random(7)
    trials = [compare(random_windows(duration, length, len(clips), rng), picks) for _ in range(RANDOM_TRIALS)]
    return {key: statistics.mean(t[key] for t in trials) for key in trials[0]}


def score_heuristic(analysis: dict, genre: str, clip_seconds: float, count: int) -> list[dict]:
    moments = find_moments(
        analysis["words"], analysis["audio"], analysis["scenes"], analysis["duration"], genre, clip_seconds, count
    )
    return [{"start": m.start, "end": m.end, "score": m.score, "text": m.text} for m in moments]


SCORERS = {"heuristic": score_heuristic}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("picks", type=Path)
    parser.add_argument("--scorer", default="heuristic", choices=sorted(SCORERS))
    parser.add_argument("--count", type=int, help="clips per video. Defaults to the number of picks.")
    parser.add_argument("--transcript", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--show-clips", action="store_true", help="print the text of each chosen clip")
    args = parser.parse_args()
    get_settings()
    videos = json.loads(args.picks.read_text(encoding="utf-8"))["videos"]
    report: dict = {"scorer": args.scorer, "videos": []}
    for video in videos:
        path = Path(video["path"]).expanduser()
        analysis = load_analysis(path, video.get("limit_seconds"))
        if args.transcript:
            print(f"\n# {path.name} ({analysis['duration'] / 60:.1f} min, {analysis['language']})")
            for sentence in split_sentences(analysis["words"]):
                print(f"[{sentence.start:7.1f}-{sentence.end:7.1f}] {sentence.text}")
            continue
        picks = [(seconds(p["start"]), seconds(p["end"])) for p in video["picks"]]
        count = args.count or len(picks)
        chosen = SCORERS[args.scorer](analysis, video.get("genre", "other"), video["clip_seconds"], count)
        clips = [(c["start"], c["end"]) for c in chosen]
        scores = compare(clips, picks)
        baseline = chance(analysis["duration"], clips, picks)
        report["videos"].append(
            {
                "video": path.name,
                "picks": len(picks),
                "clips": len(clips),
                **{k: round(v, 3) for k, v in scores.items()},
                "random": {k: round(v, 3) for k, v in baseline.items()},
                "chosen": [{"start": round(a, 2), "end": round(b, 2)} for a, b in clips],
            }
        )
        print(
            f"{path.name[:44]:44} recall {scores['pick_recall']:.2f} (random {baseline['pick_recall']:.2f})  "
            f"precision {scores['clip_precision']:.2f} (random {baseline['clip_precision']:.2f})  "
            f"IoU {scores['mean_iou']:.2f} (random {baseline['mean_iou']:.2f})"
        )
        if args.show_clips:
            for clip in chosen:
                hit = "HIT " if any(lands_on((clip["start"], clip["end"]), p) for p in picks) else "miss"
                print(f"  {hit} [{clip['start']:7.1f}-{clip['end']:7.1f}] {clip['text'][:170]}")
    if report["videos"]:
        for key in ("pick_recall", "clip_precision", "mean_iou"):
            report[f"mean_{key}"] = round(statistics.mean(v[key] for v in report["videos"]), 3)
            report[f"mean_random_{key}"] = round(statistics.mean(v["random"][key] for v in report["videos"]), 3)
        print(
            f"\nAll videos: recall {report['mean_pick_recall']:.2f} (random {report['mean_random_pick_recall']:.2f})  "
            f"precision {report['mean_clip_precision']:.2f} (random {report['mean_random_clip_precision']:.2f})  "
            f"IoU {report['mean_mean_iou']:.2f} (random {report['mean_random_mean_iou']:.2f})"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
