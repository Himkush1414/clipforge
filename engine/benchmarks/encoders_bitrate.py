#!/usr/bin/env python3
"""Encoders at equal bitrate: VMAF at each bitrate, and the bitrate each needs for the same VMAF.

For every source and output size, a lossless reference is made the way a render frames it (centre crop
to the ratio, then scale), so the encoders all start from the same frames. Each encoder then targets the
same bitrates, and VMAF is measured against that reference.

The summary per encoder is the Bjøntegaard rate difference against the first encoder (the current
settings): the average change in bitrate needed for the same VMAF, over the VMAF range both reach.
Negative means fewer bits for the same quality.

Usage:
  encoders_bitrate.py WORKDIR SOURCE[@START] ... [--seconds 20] [--sizes 9:16,16:9] [--out results.json]
  encoders_bitrate.py WORKDIR SOURCE[@START] ... --repeat-check     same input twice: identical results?
  encoders_bitrate.py WORKDIR SOURCE[@START] ... --timing 3         encode time only, 3 runs each
  encoders_bitrate.py --summarize results.json                     rate differences from saved points
START is where the excerpt begins, in seconds. Needs the FFmpeg build in ~/clipforge-work/tools/bin
(with NVENC and libvmaf).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

FF = str(Path.home() / "clipforge-work" / "tools" / "bin" / "ffmpeg")
FFPROBE = str(Path.home() / "clipforge-work" / "tools" / "bin" / "ffprobe")
SIZES = {"9:16": (1080, 1920), "16:9": (1920, 1080), "16:9 4K": (3840, 2160)}
# Bitrates for a 1080p frame; other sizes scale them by pixel count.
RATES_KBPS = [1000, 2000, 4000, 7000, 12000]


def rates_for(size: tuple[int, int]) -> list[int]:
    scale = size[0] * size[1] / (1920 * 1080)
    return [round(kbps * scale) for kbps in RATES_KBPS]


def nvenc_current(kbps: int) -> list[list[str]]:
    """The render's NVENC settings, with a bitrate target in place of constant quality."""
    return [
        ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-spatial-aq", "1"] + _vbv(kbps),
    ]


def nvenc_p7_multipass(kbps: int) -> list[list[str]]:
    return [
        [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-multipass",
            "fullres",
            "-rc",
            "vbr",
            "-spatial-aq",
            "1",
            "-rc-lookahead",
            "32",
        ]  # fmt: skip
        + _vbv(kbps)
    ]


def x264_two_pass(preset: str):
    def args(kbps: int) -> list[list[str]]:
        base = ["-c:v", "libx264", "-preset", preset] + _vbv(kbps)
        return [base + ["-pass", "1"], base + ["-pass", "2"]]

    return args


def _vbv(kbps: int) -> list[str]:
    return ["-b:v", f"{kbps}k", "-maxrate", f"{kbps * 3 // 2}k", "-bufsize", f"{kbps * 2}k"]


ENCODERS = {
    "h264_nvenc p5 (current)": nvenc_current,
    "h264_nvenc p7 multipass": nvenc_p7_multipass,
    "libx264 slow 2-pass": x264_two_pass("slow"),
    "libx264 medium 2-pass": x264_two_pass("medium"),
}


def ffmpeg(*args: str | Path) -> None:
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def make_reference(source: Path, start: float, seconds: float, size: tuple[int, int], out: Path) -> Path:
    width, height = size
    crop = f"crop='min(iw,ih*{width}/{height})':'min(ih,iw*{height}/{width})'"
    ffmpeg(
        "-ss", f"{start}", "-t", f"{seconds}", "-i", source, "-an",
        "-vf", f"{crop},scale={width}:{height}:flags=lanczos,fps=30,format=yuv420p",
        "-c:v", "ffv1", out,
    )  # fmt: skip
    return out


def encode(reference: Path, passes: list[list[str]], out: Path, workdir: Path) -> float:
    started = time.perf_counter()
    if len(passes) == 1:
        ffmpeg("-i", reference, *passes[0], "-pix_fmt", "yuv420p", out)
    else:
        log = workdir / "x264pass"
        # The first pass goes to a throwaway file: the null muxer warns about B-frame timestamps.
        ffmpeg("-i", reference, *passes[0], "-passlogfile", log, "-pix_fmt", "yuv420p", workdir / "pass1.mp4")
        ffmpeg("-i", reference, *passes[1], "-passlogfile", log, "-pix_fmt", "yuv420p", out)
        for leftover in [*workdir.glob("x264pass*"), workdir / "pass1.mp4"]:
            leftover.unlink(missing_ok=True)
    return time.perf_counter() - started


def frame_count(path: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


def vmaf(encoded: Path, reference: Path) -> tuple[float, float]:
    """Frames are paired by number, not timestamp. The lossless reference is Matroska, which rounds
    timestamps to milliseconds, and pairing by timestamp then compares a third of the frames with the
    previous reference frame (measured: 85.2 instead of 97.2 for the same encode)."""
    frames = frame_count(reference)
    if frame_count(encoded) != frames:
        raise RuntimeError(f"{encoded.name} has a different number of frames from the reference")
    log = encoded.with_suffix(".vmaf.json")
    pair = "setpts=N/(30*TB)"
    graph = f"[0:v]{pair}[a];[1:v]{pair}[b];[a][b]libvmaf=log_fmt=json:log_path={log}:n_threads=8"
    ffmpeg("-i", encoded, "-i", reference, "-lavfi", graph, "-f", "null", "-")
    data = json.loads(log.read_text())
    log.unlink()
    if len(data["frames"]) != frames:
        raise RuntimeError(f"VMAF scored {len(data['frames'])} of {frames} frames")
    pooled = data["pooled_metrics"]["vmaf"]
    return round(pooled["mean"], 2), round(pooled["harmonic_mean"], 2)


def actual_kbps(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=bit_rate,duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fmt = json.loads(out)["format"]
    return round(float(fmt["bit_rate"]) / 1000, 1)


HIGH_QUALITY = 93.0  # renders use constant quality 19, which lands at about VMAF 95-97 on this footage


def bd_rate(base: list[dict], other: list[dict], floor: float = 0.0) -> float | None:
    """Average bitrate change (%) of other against base at equal VMAF, over the VMAF range both cover
    (above floor, if given).

    Uses log bitrate as a monotone piecewise-cubic function of VMAF, as in the current Bjøntegaard method.
    """

    def curve(points: list[dict]):
        points = sorted(points, key=lambda p: p["vmaf"])
        q = np.array([p["vmaf"] for p in points])
        r = np.log([p["kbps"] for p in points])
        if np.any(np.diff(q) <= 0):
            return None, q
        return PchipInterpolator(q, r), q

    f1, q1 = curve(base)
    f2, q2 = curve(other)
    if f1 is None or f2 is None:
        return None
    low, high = max(q1.min(), q2.min(), floor), min(q1.max(), q2.max())
    if high - low < 0.5:
        return None
    grid = np.linspace(low, high, 200)
    return round((math.exp(float(np.mean(f2(grid) - f1(grid)))) - 1) * 100, 1)


def run_source(source: Path, start: float, seconds: float, workdir: Path, sizes: list[str]) -> dict:
    results: dict = {}
    for ratio in sizes:
        size = SIZES[ratio]
        reference = make_reference(source, start, seconds, size, workdir / "reference.mkv")
        points: dict[str, list[dict]] = {}
        for name, settings in ENCODERS.items():
            points[name] = []
            for kbps in rates_for(size):
                out = workdir / "encoded.mp4"
                took = encode(reference, settings(kbps), out, workdir)
                mean, harmonic = vmaf(out, reference)
                point = {"target_kbps": kbps, "kbps": actual_kbps(out), "vmaf": mean, "vmaf_harmonic": harmonic,
                         "seconds": round(took, 1)}  # fmt: skip
                points[name].append(point)
                print(source.name, ratio, name, point, flush=True)
                out.unlink()
        base = next(iter(ENCODERS))
        results[ratio] = {
            "points": points,
            "bd_rate_vs_current": {
                name: bd_rate(points[base], points[name]) for name in ENCODERS if name != base
            },
        }
        reference.unlink()
    return results


def repeat_check(source: Path, start: float, seconds: float, workdir: Path) -> dict:
    """Encode the same input twice per encoder: are VMAF and size identical between runs?"""
    reference = make_reference(source, start, seconds, SIZES["16:9"], workdir / "reference.mkv")
    found = {}
    for name, settings in ENCODERS.items():
        runs = []
        for _ in range(2):
            out = workdir / "encoded.mp4"
            encode(reference, settings(RATES_KBPS[1]), out, workdir)
            runs.append((vmaf(out, reference)[0], out.stat().st_size))
            out.unlink()
        found[name] = {"identical": runs[0] == runs[1], "runs": runs}
    reference.unlink()
    return found


def timing(source: Path, start: float, seconds: float, workdir: Path, sizes: list[str], runs: int) -> dict:
    """Encode time only, several runs each, at the middle bitrate. Run with nothing else on the machine."""
    found: dict = {}
    for ratio in sizes:
        size = SIZES[ratio]
        reference = make_reference(source, start, seconds, size, workdir / "reference.mkv")
        kbps = rates_for(size)[len(RATES_KBPS) // 2]
        found[ratio] = {}
        for name, settings in ENCODERS.items():
            times = [
                round(encode(reference, settings(kbps), workdir / "encoded.mp4", workdir), 2)
                for _ in range(runs)
            ]
            found[ratio][name] = {"kbps": kbps, "seconds": times}
            print(source.name, ratio, name, times, flush=True)
        (workdir / "encoded.mp4").unlink(missing_ok=True)
        reference.unlink()
    return found


def summarize(path: Path) -> None:
    """Recompute the rate differences from a saved results file and print them."""
    results = json.loads(path.read_text())
    high = f"bd_rate_vs_current_above_vmaf_{HIGH_QUALITY:g}"
    for key, source in results["sources"].items():
        for ratio, entry in source.items():
            points = entry["points"]
            base = next(iter(points))
            others = [name for name in points if name != base]
            entry["bd_rate_vs_current"] = {name: bd_rate(points[base], points[name]) for name in others}
            entry[high] = {name: bd_rate(points[base], points[name], HIGH_QUALITY) for name in others}
            for name in others:
                print(f"{key:<28} {ratio:<8} {name:<26} all {entry['bd_rate_vs_current'][name]!s:>6}%  "
                      f"VMAF>={HIGH_QUALITY:g} {entry[high][name]!s:>6}%")  # fmt: skip
    path.write_text(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workdir", type=Path, nargs="?")
    parser.add_argument("sources", nargs="*")
    parser.add_argument(
        "--summarize", type=Path, metavar="RESULTS", help="recompute rate differences from a file"
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--sizes", default="9:16,16:9", help=f"comma-separated, from {', '.join(SIZES)}")
    parser.add_argument("--timing", type=int, metavar="RUNS", help="only time the encoders, RUNS times each")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.workdir or not args.sources:
        parser.error("WORKDIR and at least one SOURCE are needed")
    args.workdir.mkdir(parents=True, exist_ok=True)
    results: dict = {"rates_kbps": RATES_KBPS, "seconds": args.seconds, "sources": {}}
    try:
        for spec in args.sources:
            path, _, start = spec.partition("@")
            source = Path(path).expanduser()
            key = f"{source.name}@{start or 0}"
            if args.timing:
                sizes = args.sizes.split(",")
                results["sources"][key] = timing(
                    source, float(start or 0), args.seconds, args.workdir, sizes, args.timing
                )
            elif args.repeat_check:
                results["sources"][key] = repeat_check(source, float(start or 0), args.seconds, args.workdir)
            else:
                sizes = args.sizes.split(",")
                results["sources"][key] = run_source(
                    source, float(start or 0), args.seconds, args.workdir, sizes
                )
            if args.out:
                args.out.write_text(json.dumps(results, indent=1))
    finally:
        shutil.rmtree(args.workdir, ignore_errors=True)
    if args.out and not (args.timing or args.repeat_check):
        summarize(args.out)


if __name__ == "__main__":
    main()
