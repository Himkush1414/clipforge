#!/usr/bin/env python3
"""Compare scene detectors on the same known-answer videos as engine/benchmarks/run_benchmark.py.

Runs in the benchmark environment, which has PySceneDetect. Usage: scene_tools.py out.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine" / "benchmarks"))
import run_benchmark as rb
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector, HashDetector, HistogramDetector


def pyscenedetect(path: Path, make) -> list[int]:
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(make())
    manager.detect_scenes(video, show_progress=False)
    return [scene[0].frame_num for scene in manager.get_scene_list()][1:]


def scdet(path: Path) -> list[int]:
    out = subprocess.run(
        [
            rb.FF,
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "scale=320:-2,scdet=threshold=10:sc_pass=0,metadata=print:file=-",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stdout
    fps = rb.probe(path).fps
    return [round(float(line.split("=")[1]) * fps) for line in out.splitlines() if "lavfi.scd.time" in line]


def main() -> int:
    out = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="scene-") as folder:
        tmp = Path(folder)
        hard, motion, fades = tmp / "hard.mp4", tmp / "motion.mp4", tmp / "fades.mp4"
        truth = rb.scene_test_video(hard)
        rb.fast_motion_video(motion)
        fade_truth = rb.fades_video(fades)
        tools = {
            "clipforge": rb.own_cuts,
            "PySceneDetect Content": lambda p: pyscenedetect(p, ContentDetector),
            "PySceneDetect Adaptive": lambda p: pyscenedetect(p, AdaptiveDetector),
            "PySceneDetect Hash": lambda p: pyscenedetect(p, HashDetector),
            "PySceneDetect Histogram": lambda p: pyscenedetect(p, HistogramDetector),
            "FFmpeg scdet": scdet,
        }
        results = {}
        for name, find in tools.items():
            start = time.time()
            found = find(hard)
            took = time.time() - start
            results[name] = {
                "hard_cuts_exact": rb.score_cuts(found, truth["cuts"], 0),
                "fps": round(truth["frames"] / took),
                "false_cuts_on_motion_and_flash": len(find(motion)),
                "dissolves": rb.score_cuts(find(fades), fade_truth, 20),
            }
            print(name, json.dumps(results[name]), flush=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
