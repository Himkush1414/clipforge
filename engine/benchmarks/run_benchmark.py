#!/usr/bin/env python3
"""Measure the engine against known answers, using generated video and audio only.

Results go to a JSON file. docs/ENGINE.md explains them. No section needs a network or a model.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile

from clipforge.analysis.audio import analyze_audio
from clipforge.analysis.scenes import analyze_scenes
from clipforge.analysis.scoring import find_moments
from clipforge.analysis.transcribe import Word
from clipforge.captions.layout import SAFE_MARGINS
from clipforge.captions.presets import BUILTIN_PRESETS
from clipforge.captions.preview import render_preview
from clipforge.config import get_settings
from clipforge.export.plan import Segment
from clipforge.export.render import RenderSpec, render_clip
from clipforge.hardware import detect
from clipforge.media.ffmpeg import find_tool
from clipforge.media.probe import probe
from clipforge.qc import run_qc

FF = find_tool("ffmpeg")
FPROBE = find_tool("ffprobe")


def ff(*args) -> None:
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def peak_memory() -> "MemoryWatch":
    return MemoryWatch()


class MemoryWatch:
    """Peak RAM of ffmpeg plus this process, and peak GPU memory, sampled while a block runs."""

    def __enter__(self) -> "MemoryWatch":
        self.ram_kb = 0
        self.vram_mb = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop:
            out = subprocess.run(
                "ps -eo rss,comm | awk '/ffmpeg/ {s+=$1} END {print s+0}'",
                shell=True,
                capture_output=True,
                text=True,
            ).stdout
            own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.ram_kb = max(self.ram_kb, int(out.strip() or 0) + own)
            gpu = subprocess.run(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            ).stdout
            if gpu.strip().isdigit():
                self.vram_mb = max(self.vram_mb, int(gpu.strip()))
            time.sleep(0.25)

    def __exit__(self, *_exc) -> None:
        self._stop = True
        self._thread.join(timeout=2)


def frames_of(path: Path, size: int = 16) -> np.ndarray:
    raw = subprocess.run(
        [
            FF,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={size}:{size}:flags=area,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size).astype(np.int16)


def coded_video(path: Path, frames: int, rate: str, gop: int) -> None:
    """Each frame shows its own number as 12 black or white blocks, so any wrong frame is identified."""
    expr = "if(mod(floor(N/pow(2,floor(X/(W/12)))),2),235,16)"
    graph = f"nullsrc=s=384x216:r={rate},format=gray,geq=lum='{expr}'"
    ff(
        "-f",
        "lavfi",
        "-i",
        graph,
        "-frames:v",
        frames,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-qp",
        "4",
        "-g",
        gop,
        "-bf",
        2,
        "-pix_fmt",
        "yuv420p",
        path,
    )


def read_codes(path: Path) -> list[int]:
    width, height = 384, 216
    raw = subprocess.run(
        [
            FF,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={width}:{height}:flags=neighbor,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width)
    block = width // 12
    codes = []
    for frame in frames:
        bits = [frame[60:150, b * block + 8 : (b + 1) * block - 8].mean() > 128 for b in range(12)]
        codes.append(sum(1 << b for b, on in enumerate(bits) if on))
    return codes


def bench_cuts(tmp: Path) -> dict:
    """Random cuts decoded back to frame numbers. Offset 0 means every frame is exactly right."""
    results = {}
    rng = random.Random(7)
    for name, rate, fps, gop in [
        ("30 fps, long GOP", "30", 30.0, 250),
        ("29.97 fps", "30000/1001", 30000 / 1001, 48),
        ("60 fps", "60", 60.0, 120),
        ("23.976 fps", "24000/1001", 24000 / 1001, 96),
    ]:
        source = tmp / f"code-{fps:.2f}.mp4"
        total = 3000
        coded_video(source, total, rate, gop)
        info = probe(source)
        wrong, count = 0, 0
        worst = 0
        for trial in range(10):
            length = rng.choice([1, 2, 3, 5, 12, 40])
            start = rng.randrange(0, total - length)
            out = tmp / f"cut-{fps:.2f}-{trial}.mp4"
            render_clip(
                RenderSpec(source, info, [Segment(start, start + length)], "16:9", "1080p", out, fit="crop")
            )
            got = read_codes(out)
            expected = list(range(start, start + length))
            count += 1
            if got != expected:
                wrong += 1
                worst = max(
                    worst,
                    max((abs(a - b) for a, b in zip(got, expected, strict=False)), default=length),
                    abs(len(got) - length),
                )
        out = tmp / f"many-{fps:.2f}.mp4"
        cuts = sorted(rng.sample(range(0, total - 30, 30), 9))
        segments = [Segment(a, a + rng.choice([5, 10, 20])) for a in cuts]
        render_clip(RenderSpec(source, info, segments, "16:9", "1080p", out, fit="crop"))
        expected = [f for s in segments for f in range(s.start_frame, s.end_frame)]
        got = read_codes(out)
        count += 1
        if got != expected:
            wrong += 1
            worst = max(
                worst,
                max((abs(a - b) for a, b in zip(got, expected, strict=False)), default=0),
                abs(len(got) - len(expected)),
            )
        results[name] = {"cuts_checked": count, "cuts_wrong": wrong, "worst_frame_offset": worst}
    return results


def _sync_source(path: Path, seconds: int) -> list[float]:
    """A white flash and a tone start together once per 4 seconds, at known times."""
    events = [t + 2.0 for t in range(0, seconds - 3, 4)]
    white = "+".join(f"between(t,{t},{t + 0.5})" for t in events)
    tone = "+".join(f"between(t,{t},{t + 0.5})" for t in events)
    graph = (
        f"color=c=black:s=320x180:r=30:d={seconds},drawbox=c=white:t=fill:enable='{white}'[v];"
        f"sine=f=1000:r=48000:d={seconds},volume='if({tone},1,0)':eval=frame[a]"
    )
    ff(
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        30,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        path,
    )
    return events


def _onsets(video: Path, tmp: Path) -> tuple[list[float], list[float]]:
    wav = tmp / "sync.wav"
    ff("-i", video, "-ac", 1, "-ar", 8000, wav)
    rate, samples = wavfile.read(wav)
    window = rate // 200
    rms = np.array([np.abs(samples[i : i + window]).mean() for i in range(0, len(samples) - window, window)])
    on = rms > rms.max() * 0.25
    audio = [i * window / rate for i in range(1, len(on)) if on[i] and not on[i - 1]]
    bright = frames_of(video, 8).mean(axis=(1, 2)) > 128
    fps = probe(video).fps
    video_on = [i / fps for i in range(1, len(bright)) if bright[i] and not bright[i - 1]]
    return video_on, audio


def bench_sync(tmp: Path) -> dict:
    """Offset between flash and tone at the start, middle and end of a clip with many cuts."""
    results = {}
    source = tmp / "sync.mp4"
    events = _sync_source(source, 240)
    info = probe(source)
    fps = info.fps
    for name, count in [
        ("3 segments (direct)", 3),
        ("12 segments (one by one)", 12),
        ("40 segments (one by one)", 40),
    ]:
        segments = []
        for event in events[:count]:
            start = round((event - 1.0) * fps)
            segments.append(Segment(start, start + round(2.0 * fps)))
        out = tmp / f"sync-{count}.mp4"
        render_clip(RenderSpec(source, info, segments, "16:9", "1080p", out, fit="crop"))
        video_on, audio_on = _onsets(out, tmp)
        offsets = [(a - v) * 1000 for v, a in zip(video_on, audio_on, strict=False)]
        spread = len(offsets)
        output = probe(out)
        results[name] = {
            "flashes_found": spread,
            "worst_offset_ms": round(max(abs(o) for o in offsets), 1),
            "drift_first_to_last_ms": round(offsets[-1] - offsets[0], 1),
            "duration_error_ms": round(abs(output.duration - count * 2.0) * 1000, 1),
        }
    return results


def bench_loudness(tmp: Path) -> dict:
    """Target -14 LUFS, -1 dBTP. Sources range from very quiet to clipped."""
    results = {}
    cases = {
        "quiet speech-like": "anoisesrc=d=20:c=pink:r=48000:a=0.02,tremolo=f=3:d=0.9",
        "normal music-like": "anoisesrc=d=20:c=pink:r=48000:a=0.2,tremolo=f=2:d=0.5",
        "loud clipped": "anoisesrc=d=20:c=white:r=48000:a=1.0,volume=12dB,alimiter=limit=1:level=false",
        "wide range": "anoisesrc=d=20:c=pink:r=48000:a=0.3,volume='if(lt(t,10),0.02,1)':eval=frame",
    }
    for index, (name, audio) in enumerate(cases.items()):
        source = tmp / f"loud-{index}.mp4"
        ff(
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x180:r=30:d=20",
            "-f",
            "lavfi",
            "-i",
            audio,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-shortest",
            source,
        )
        info = probe(source)
        out = tmp / f"loudout-{index}.mp4"
        result = render_clip(RenderSpec(source, info, [Segment(0, 600)], "16:9", "1080p", out, fit="crop"))
        checks = {
            c["name"]: c
            for c in run_qc(out, result.width, result.height, result.duration, info.fps, True)["checks"]
        }
        lufs = float(checks["loudness"]["detail"].split()[0])
        peak = float(checks["true_peak"]["detail"].split()[0])
        results[name] = {"lufs": lufs, "lufs_error": round(abs(lufs + 14), 2), "true_peak_dbfs": peak}
    return results


def _ink_box(path: Path, threshold: int = 60) -> tuple[int, int, int, int]:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(int)
    ys, xs = np.where(arr.max(axis=2) > threshold)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bench_captions(tmp: Path) -> dict:
    """Rendered ink measured against the safe margins, for every preset at every ratio."""
    sizes = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}
    words = [
        Word("EXTRAORDINARILY", 0.0, 1.0),
        Word("uncomfortable", 1.0, 2.0),
        Word("CONVERSATIONS", 2.0, 3.0),
    ]
    results = {}
    for ratio, (width, height) in sizes.items():
        margin = SAFE_MARGINS[ratio]
        worst = 10**9
        checked = violations = 0
        for preset in BUILTIN_PRESETS.values():
            clean = preset.model_copy(update={"glow": 0, "glow_color": None})
            for at in (0.1, 0.6, 1.5, 2.5):
                out = render_preview(
                    clean,
                    tmp / "cap.png",
                    ratio,
                    at=at,
                    words=words,
                    size=(width, height),
                    background=f"color=c=black:s={width}x{height}",
                )
                x0, y0, x1, y1 = _ink_box(out)
                room = min(
                    x0 - width * margin["side"],
                    width * (1 - margin["side"]) - x1,
                    y0 - height * margin["top"],
                    height * (1 - margin["bottom"]) - y1,
                )
                worst = min(worst, room)
                checked += 1
                violations += room < 0
        results[ratio] = {
            "frames_checked": checked,
            "outside_safe_area": violations,
            "tightest_margin_px": round(worst, 1),
            "frame": f"{width}x{height}",
        }
    return results


def bench_render(tmp: Path) -> dict:
    """Render speed and peak memory for a 30 second clip with captions, at each resolution."""
    source = tmp / "render-src.mp4"
    ff(
        "-f",
        "lavfi",
        "-i",
        "mandelbrot=s=1920x1080:r=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=220",
        "-t",
        30,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        source,
    )
    info = probe(source)
    words = [Word(f"word{i}", i * 0.4, i * 0.4 + 0.35) for i in range(70)]
    results = {}
    for ratio, resolution in [("9:16", "1080p"), ("16:9", "4K"), ("9:16", "8K")]:
        out = tmp / f"speed-{resolution}.mp4"
        with peak_memory() as memory:
            start = time.time()
            rendered = render_clip(
                RenderSpec(
                    source,
                    info,
                    [Segment(0, 900)],
                    ratio,
                    resolution,
                    out,
                    words=words,
                    preset=BUILTIN_PRESETS["anime-impact"],
                    fit="crop",
                )
            )
            took = time.time() - start
        results[f"{ratio} {resolution}"] = {
            "encoder": rendered.encoder,
            "seconds_for_30s_clip": round(took, 1),
            "speed_x_realtime": round(30 / took, 1),
            "peak_ram_gb": round(memory.ram_kb / 1024**2, 2),
            "peak_gpu_mb": memory.vram_mb,
        }
    return results


def scene_test_video(path: Path) -> dict:
    """Known structure: hard cuts between moving scenes, fades, a flash and fast motion that must not count."""
    parts = [
        ("mandelbrot=s=640x360:r=30:start_scale=3", 60),
        ("testsrc2=s=640x360:r=30", 45),
        ("gradients=s=640x360:r=30:speed=0.05:c0=0x202840:c1=0xd08030", 90),
        ("life=s=640x360:r=30:mold=10:ratio=0.2:death_color=#101820:life_color=#60c0ff,scale=640:360", 50),
        ("testsrc=s=640x360:r=30", 8),  # A very short shot.
        ("rgbtestsrc=s=640x360:r=30", 70),
        ("mandelbrot=s=640x360:r=30:start_scale=1.5:end_scale=0.1", 75),
    ]
    inputs: list = []
    labels = ""
    filters = []
    for i, (src, frames) in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", f"{src},format=yuv420p,setsar=1,fps=30"]
        filters.append(f"[{i}:v]trim=end_frame={frames},setpts=PTS-STARTPTS[p{i}]")
        labels += f"[p{i}]"
    graph = ";".join(filters) + f";{labels}concat=n={len(parts)}:v=1:a=0[v]"
    ff(
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-qp",
        "12",
        "-g",
        300,
        path,
    )
    cuts, position = [], 0
    for _, frames in parts[:-1]:
        position += frames
        cuts.append(position)
    return {"cuts": cuts, "frames": position + parts[-1][1]}


def fast_motion_video(path: Path) -> None:
    """One continuous shot with a large object moving fast and a brief bright flash: no cuts at all."""
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=0x303030:s=640x360:r=30:d=12",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=280x280:r=30:d=12",
        "-filter_complex",
        "[0][1]overlay=x='180+170*sin(t*9)':y='40+30*cos(t*7)',drawbox=c=white:t=fill:enable='between(n,150,151)'",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-qp",
        "12",
        path,
    )


def fades_video(path: Path) -> list[int]:
    """Three scenes joined by 1 second dissolves. Truth is the dissolve midpoint."""
    ff(
        "-f",
        "lavfi",
        "-i",
        "mandelbrot=s=640x360:r=30",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=640x360:r=30",
        "-f",
        "lavfi",
        "-i",
        "gradients=s=640x360:r=30:speed=0.02",
        "-filter_complex",
        "[0]format=yuv420p,trim=duration=4,setpts=PTS-STARTPTS[a];[1]format=yuv420p,trim=duration=4,setpts=PTS-STARTPTS[b];[2]format=yuv420p,trim=duration=4,setpts=PTS-STARTPTS[c];"
        "[a][b]xfade=transition=fade:duration=1:offset=3[x];[x][c]xfade=transition=fade:duration=1:offset=6",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-qp",
        "12",
        path,
    )
    return [105, 195]


def score_cuts(found: list[int], truth: list[int], tolerance: int) -> dict:
    remaining = list(truth)
    hits = 0
    offsets = []
    for frame in found:
        close = [t for t in remaining if abs(t - frame) <= tolerance]
        if close:
            best = min(close, key=lambda t: abs(t - frame))
            remaining.remove(best)
            offsets.append(abs(best - frame))
            hits += 1
    return {
        "found": len(found),
        "true_positives": hits,
        "missed": len(truth) - hits,
        "false_alarms": len(found) - hits,
        "worst_offset_frames": max(offsets, default=0),
    }


def own_cuts(path: Path) -> list[int]:
    info = probe(path)
    return analyze_scenes(path, info.fps, info.duration).cut_frames


def bench_scenes(tmp: Path) -> dict:
    hard = tmp / "scenes-hard.mp4"
    truth = scene_test_video(hard)
    motion = tmp / "scenes-motion.mp4"
    fast_motion_video(motion)
    fades = tmp / "scenes-fades.mp4"
    fade_truth = fades_video(fades)
    start = time.time()
    found = own_cuts(hard)
    took = time.time() - start
    return {
        "hard_cuts": {
            **score_cuts(found, truth["cuts"], 0),
            "truth": len(truth["cuts"]),
            "fps_processed": round(truth["frames"] / took),
        },
        "fast_motion_and_flash": {"false_alarms": len(own_cuts(motion))},
        "dissolves": {**score_cuts(own_cuts(fades), fade_truth, 20), "truth": len(fade_truth)},
    }


def bench_silence(tmp: Path) -> dict:
    """Pink noise with silent gaps at known times."""
    gaps = [(3.0, 4.0), (7.5, 8.0), (11.0, 13.0), (16.0, 16.4)]
    expr = "+".join(f"between(t,{a},{b})" for a, b in gaps)
    wav = tmp / "silence.wav"
    ff(
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=d=20:c=pink:r=16000:a=0.3,volume='if({expr},0,1)':eval=frame",
        "-ac",
        1,
        "-c:a",
        "pcm_s16le",
        wav,
    )
    found = analyze_audio(wav).silences
    errors = []
    for a, b in gaps:
        best = min(found, key=lambda s: abs(s[0] - a) + abs(s[1] - b)) if found else (0, 0)
        errors += [abs(best[0] - a) * 1000, abs(best[1] - b) * 1000]
    return {
        "gaps_truth": len(gaps),
        "gaps_found": len(found),
        "worst_edge_error_ms": round(max(errors), 1),
        "mean_edge_error_ms": round(statistics.mean(errors), 1),
    }


def bench_scoring() -> dict:
    """Plant exciting sections in plain talk. A good scorer picks clips overlapping them."""

    class Audio:
        hop = 0.05

    hits = total = 0
    overlap: list[float] = []
    planted_share = 0.0
    plain = (
        "the plan for the day was simple and we walked along the road to look at the old town near the river "
    )
    exciting = (
        "this is absolutely incredible nobody expected it and the whole crowd went crazy with total shock "
    )
    rng = random.Random(11)
    for _trial in range(10):
        words: list[Word] = []
        t = 0.0
        planted = []
        hot = sorted(rng.sample(range(2, 18), 3))
        for section in range(20):
            is_hot = section in hot
            base = (exciting if is_hot else plain).split()
            begin = t
            for _sentence in range(3):
                for i, token in enumerate(base):
                    end_mark = "!" if is_hot and i == len(base) - 1 else "." if i == len(base) - 1 else ""
                    length = 0.26 if is_hot else 0.34
                    words.append(Word(token + end_mark, round(t, 3), round(t + length, 3)))
                    t += length + 0.02
                t += 0.4
            if is_hot:
                planted.append((begin, t))
        duration = t + 1
        planted_share = sum(b - a for a, b in planted) / duration
        audio = Audio()
        frames = int(duration / audio.hop)
        audio.rms_db = np.full(frames, -30.0, dtype=np.float32)
        audio.flux = np.zeros(frames, dtype=np.float32)
        audio.silences, audio.peaks = [], []
        for begin, end in planted:
            audio.rms_db[int(begin / audio.hop) : int(end / audio.hop)] = -14.0
            audio.flux[int(begin / audio.hop) : int(end / audio.hop)] = 5.0
        scenes = type(
            "S",
            (),
            {
                "fps": 30.0,
                "motion": np.zeros(int(duration * 30), dtype=np.float32),
                "cut_times": [],
                "cut_frames": [],
            },
        )()
        chosen = find_moments(words, audio, scenes, duration, "other", 30, 3)
        for moment in chosen:
            total += 1
            length = moment.end - moment.start
            inside = [max(0.0, min(moment.end, b) - max(moment.start, a)) for a, b in planted]
            overlap.append(sum(inside) / length)
            # Captured: at least 80% of one planted moment is inside this clip.
            hits += any(share >= 0.8 * (b - a) for share, (a, b) in zip(inside, planted, strict=True))
    return {
        "clips_chosen": total,
        "clips_capturing_a_planted_moment": hits,
        "capture_rate": round(hits / max(total, 1), 2),
        "mean_share_of_clip_on_planted": round(statistics.mean(overlap), 2),
        "planted_share_of_video": round(planted_share, 2),
    }


SECTIONS = {
    "cuts": bench_cuts,
    "sync": bench_sync,
    "loudness": bench_loudness,
    "captions": bench_captions,
    "render": bench_render,
    "scenes": bench_scenes,
    "silence": bench_silence,
    "scoring": lambda _tmp: bench_scoring(),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=",".join(SECTIONS))
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    hw = detect()
    version = subprocess.run([FF, "-hide_banner", "-version"], capture_output=True, text=True).stdout.split()[
        2
    ]
    results: dict = {
        "label": args.label,
        "machine": {"gpu": hw.gpu_name, "cpu_threads": hw.cpu_threads, "ram_gb": hw.ram_gb},
        "ffmpeg": version,
    }
    # On the work drive: Windows drives under /mnt are far slower and would distort the speed numbers.
    settings = get_settings()
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=settings.tmp_dir) as name:
        tmp = Path(name)
        for section in args.only.split(","):
            start = time.time()
            print(f"== {section}", flush=True)
            results[section] = SECTIONS[section](tmp)
            print(
                json.dumps(results[section], ensure_ascii=False), f"({time.time() - start:.0f}s)", flush=True
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
