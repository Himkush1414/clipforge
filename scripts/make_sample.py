#!/usr/bin/env python3
"""Create a sample video we own: original script, synthetic voice, FFmpeg-generated picture.

Usage: make_sample.py OUTPUT.mp4 [--voice PATH_TO_PIPER_VOICE.onnx]
Needs piper-tts for speech. The voice model is downloaded outside the repo by the caller.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = [
    "Welcome back to the workshop. Today we are building a tiny wooden boat from scrap pine.",
    "Most people think you need expensive tools for this. You really do not.",
    "A sharp knife, some sandpaper and a little patience will carry you a very long way.",
    "First, mark the outline of the hull on the block. Take your time with this step.",
    "Now here is the secret nobody tells you. Cut away from the centre line, never towards it!",
    "If you remember only one thing today, remember that. It will save your fingers and your boat.",
    "Sand the hull until it feels smooth. Start with coarse paper, then move to the fine paper.",
    "Let us change the subject for a moment and talk about the night sky.",
    "Have you ever looked up and really wondered how far away those stars are?",
    "The light you see tonight left some of those stars thousands of years ago.",
    "That is incredible when you think about it. You are looking straight into the past!",
    "Astronomers measure these huge distances in light years, and the numbers are enormous.",
    "Our nearest neighbour star is more than four light years away from us.",
    "Even so, a small telescope in your back garden can show you the rings of Saturn.",
    "Now let us talk about bread, because everybody loves fresh bread.",
    "Good bread needs only four ingredients: flour, water, salt and yeast.",
    "The biggest mistake beginners make is rushing the dough. Do not rush it!",
    "Give the dough time to rise slowly, and the flavour becomes deeper and richer.",
    "Bake it in a very hot oven, and listen for that hollow sound when you tap the crust.",
    "So that is everything for today: a boat, the stars and a loaf of bread.",
    "Thank you so much for watching. Go and make something with your own two hands.",
]

SCENES = [
    "gradients=s=1920x1080:c0=0x1b2a49:c1=0x7f5539:speed=0.02",
    "testsrc2=s=1920x1080",
    "mandelbrot=s=1920x1080",
    "gradients=s=1920x1080:c0=0x254d32:c1=0xd4a373:speed=0.05",
    "life=s=1920x1080:mold=10:ratio=0.1:death_color=#102030:life_color=#ffd166,scale=1920:1080",
    "rgbtestsrc=s=1920x1080",
]


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        parts = []
        for i, line in enumerate(SCRIPT):
            wav = tmp / f"{i}.wav"
            run(
                [sys.executable, "-m", "piper", "-m", args.voice, "-f", wav, "--", line],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            parts.append(wav)
        silence = tmp / "gap.wav"
        run([args.ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "anullsrc=r=22050:cl=mono", "-t", "0.6", silence])
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\nfile '{silence}'\n" for p in parts))
        speech = tmp / "speech.wav"
        run([args.ffmpeg, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
             "-c:a", "pcm_s16le", speech])
        probe = subprocess.run(
            [args.ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(speech)], capture_output=True, text=True, check=True)
        duration = float(probe.stdout)
        scene_len = duration / len(SCENES)
        inputs: list = []
        labels = ""
        filters = []
        for i, scene in enumerate(SCENES):
            inputs += ["-f", "lavfi", "-t", f"{scene_len:.3f}", "-i", f"{scene},fps=30,format=yuv420p"]
            filters.append(f"[{i}:v]setsar=1[v{i}]")
            labels += f"[v{i}]"
        graph = ";".join(filters) + f";{labels}concat=n={len(SCENES)}:v=1:a=0[v]"
        run([args.ffmpeg, "-loglevel", "error", "-y", *inputs, "-i", speech, "-filter_complex", graph,
             "-map", "[v]", "-map", f"{len(SCENES)}:a", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", args.output])
    print(args.output, f"{duration:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
