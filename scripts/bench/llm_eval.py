#!/usr/bin/env python3
"""Compare local models against the editor alone, using the pick_eval test sets.

Ratings are cached per model and clip text, so trying a different blend takes seconds. The held-out set
is scored only with --heldout, once the model and weight have been chosen from the tuning set.

Usage: llm_eval.py OUT.json [--models a.gguf,b.gguf] [--weights 0.3,0.5,0.7,1.0] [--heldout]
Runs in the app's environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

from clipforge.analysis import llm_score
from clipforge.analysis.scoring import rank_moments, select_moments

BENCH = Path(__file__).parent
spec = importlib.util.spec_from_file_location("pick_eval", BENCH / "pick_eval.py")
assert spec and spec.loader
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

CACHE = Path.home() / "clipforge-work" / "bench" / "pick-eval" / "llm-ratings.json"
GENERATED = Path.home() / "clipforge-work" / "bench" / "pick-eval" / "generated" / "generated-picks.json"


class VramWatch:
    def __enter__(self):
        self.peak, self._stop = 0, False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if out.isdigit():
                self.peak = max(self.peak, int(out))
            time.sleep(0.3)

    def __exit__(self, *_):
        self._stop = True
        self._thread.join(timeout=2)


def load_set(path: Path) -> list[dict]:
    items = []
    for video in json.loads(path.read_text(encoding="utf-8"))["videos"]:
        analysis = pe.load_analysis(Path(video["path"]).expanduser(), video.get("limit_seconds"))
        picks = [(pe.seconds(p["start"]), pe.seconds(p["end"])) for p in video["picks"]]
        ranked = rank_moments(
            analysis["words"],
            analysis["audio"],
            analysis["scenes"],
            analysis["duration"],
            video.get("genre", "other"),
            video["clip_seconds"],
            analysis["language"],
        )
        items.append({"name": Path(video["path"]).name, "picks": picks, "ranked": ranked})
    return items


def key_of(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{llm_score.PROMPT}|{text}".encode()).hexdigest()[:24]


def rate_all(model: str, sets: list[list[dict]], ratings: dict) -> dict:
    """Rate every shortlisted candidate. Returns timing and memory figures."""
    todo = []
    for items in sets:
        for item in items:
            passing = [m for m in item["ranked"] if m.passed]
            todo += [m.text for m in llm_score.shortlist(passing) if key_of(model, m.text) not in ratings]
    stats = {"rated": 0, "seconds_per_clip": None, "peak_vram_mb": None, "load_seconds": None, "failed": 0}
    if not todo:
        return stats
    path = llm_score.model_path(model)
    runtime = llm_score.runtime_path()
    assert path and runtime
    with VramWatch() as vram:
        start = time.time()
        with llm_score.LocalModel(path, runtime) as local:
            stats["load_seconds"] = round(time.time() - start, 1)
            start = time.time()
            for text in todo:
                try:
                    ratings[key_of(model, text)] = local.rate(text)
                except Exception as exc:
                    stats["failed"] += 1
                    print("  failed:", repr(exc)[:120], flush=True)
            took = time.time() - start
    stats |= {
        "rated": len(todo),
        "seconds_per_clip": round(took / len(todo), 2),
        "peak_vram_mb": vram.peak,
    }
    CACHE.write_text(json.dumps(ratings))
    return stats


def evaluate(items: list[dict], model: str | None, weight: float, ratings: dict) -> dict:
    scores = []
    for item in items:
        ranked = copy.deepcopy(item["ranked"])
        passing = [m for m in ranked if m.passed]
        if model:
            passing = llm_score.shortlist(passing)
            for moment in passing:
                rating = ratings.get(key_of(model, moment.text))
                if rating:
                    llm_score.blend(moment, rating, weight)
            passing.sort(key=lambda m: m.score, reverse=True)
        chosen = select_moments(passing, len(item["picks"]))
        scores.append(pe.compare([(m.start, m.end) for m in chosen], item["picks"]))
    return {key: round(statistics.mean(s[key] for s in scores), 3) for key in scores[0]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", default=llm_score.MODEL_FILE)
    parser.add_argument("--weights", default="0.3,0.5,0.7,1.0")
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args()
    ratings = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    tuning = load_set(BENCH / "nasa_picks.json")
    generated = load_set(GENERATED)
    sets = [tuning, generated]
    heldout = load_set(BENCH / "nasa_heldout_picks.json") if args.heldout else []
    if heldout:
        sets.append(heldout)
    models = args.models.split(",")
    weights = [float(w) for w in args.weights.split(",")]
    results: dict = {
        "editor alone": {
            "tuning": evaluate(tuning, None, 0, ratings),
            "generated": evaluate(generated, None, 0, ratings),
        }
    }
    if heldout:
        results["editor alone"]["heldout"] = evaluate(heldout, None, 0, ratings)
    print("editor alone", results["editor alone"], flush=True)
    for model in models:
        stats = rate_all(model, sets, ratings)
        row: dict = {"stats": stats}
        print(model, stats, flush=True)
        for weight in weights:
            cell = {
                "tuning": evaluate(tuning, model, weight, ratings),
                "generated": evaluate(generated, model, weight, ratings),
            }
            if heldout:
                cell["heldout"] = evaluate(heldout, model, weight, ratings)
            row[str(weight)] = cell
            print(f"  weight {weight}", cell, flush=True)
        results[model] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
