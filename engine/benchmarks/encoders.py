#!/usr/bin/env python3
"""Compare encoder settings by VMAF quality, file size and speed on the same source clips."""

import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path.home() / "clipforge-work"
FF = str(WORK / "tools" / "bin" / "ffmpeg")
BASE_NV = ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-spatial-aq", "1"]
SETTINGS = {
    "h264_nvenc current": ["-c:v", "h264_nvenc", *BASE_NV],
    "h264_nvenc tuned": [
        "-c:v",
        "h264_nvenc",
        *BASE_NV,
        "-temporal-aq",
        "1",
        "-rc-lookahead",
        "32",
        "-bf",
        "3",
        "-b_ref_mode",
        "middle",
    ],
    "h264_nvenc p7 tuned": [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "19",
        "-b:v",
        "0",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-rc-lookahead",
        "32",
        "-bf",
        "3",
        "-b_ref_mode",
        "middle",
    ],
    "hevc_nvenc current": ["-c:v", "hevc_nvenc", *BASE_NV],
    "hevc_nvenc tuned": [
        "-c:v",
        "hevc_nvenc",
        *BASE_NV,
        "-temporal-aq",
        "1",
        "-rc-lookahead",
        "32",
        "-bf",
        "3",
        "-b_ref_mode",
        "middle",
    ],
    "libx264 medium crf18": ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
    "libx265 medium crf20": ["-c:v", "libx265", "-preset", "medium", "-crf", "20"],
}


def vmaf(encoded: Path, reference: Path, threads: int = 8) -> tuple[float, float]:
    log = encoded.with_suffix(".json")
    graph = f"[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]libvmaf=log_fmt=json:log_path={log}:n_threads={threads}"
    subprocess.run(
        [
            FF,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(encoded),
            "-i",
            str(reference),
            "-lavfi",
            graph,
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    pooled = json.loads(log.read_text())["pooled_metrics"]["vmaf"]
    log.unlink()
    return round(pooled["mean"], 2), round(pooled["harmonic_mean"], 2)


def run(source: Path, tmp: Path) -> dict:
    results = {}
    for name, args in SETTINGS.items():
        out = tmp / (name.replace(" ", "_") + ".mp4")
        start = time.time()
        code = subprocess.run(
            [
                FF,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                *args,
                "-pix_fmt",
                "yuv420p",
                str(out),
            ]
        ).returncode
        took = time.time() - start
        if code != 0:
            results[name] = {"error": "encoder refused these settings"}
            continue
        mean, harmonic = vmaf(out, source)
        results[name] = {
            "vmaf": mean,
            "vmaf_harmonic": harmonic,
            "mb": round(out.stat().st_size / 1e6, 1),
            "seconds": round(took, 1),
        }
        print(source.name, name, results[name], flush=True)
        out.unlink()
    return results


if __name__ == "__main__":
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    print(json.dumps({Path(p).name: run(Path(p), tmp) for p in sys.argv[2:]}, indent=1))
