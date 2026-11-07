# Engine benchmark

Measures each stage against known answers. Everything is generated, so nothing here depends on footage with unknown rights.

```bash
cd engine
python benchmarks/run_benchmark.py --out ../docs/benchmark-results.json
```

`--only cuts,audio,captions,loudness,render,scenes,silence,scoring` runs some sections. Speech sections need extra tools and live in `scripts/bench/`. Results and how to read them are in `docs/ENGINE.md`.
