# Engine audit

Which tool does each job, why, and what it measured. This page holds numbers from `engine/benchmarks/`, run on the development machine (RTX 3050 8 GB, 28 threads, 7.6 GB RAM under WSL2). Raw results are in [docs/benchmarks](benchmarks/).

The test material is generated, so the correct answers are known exactly. That makes these numbers precise about mechanics. They say little about taste, or about footage unlike the test material.

The full table arrives as each stage is evaluated.

## Face detection (reframing)

YuNet replaced OpenCV's Haar cascade. MediaPipe's BlazeFace was tested and rejected. Measured by `engine/benchmarks/faces.py` on public domain NASA portraits pasted at known positions, at clipforge's sampling size of 320x180. A detection counts when its centre is within 8% of frame width. Raw results: [benchmarks/faces.json](benchmarks/faces.json).

| Case | Haar (before) | YuNet (now) | BlazeFace |
|---|---|---|---|
| Large face | 17/17 | 17/17 | 17/17 |
| Medium face | 17/17 | 17/17 | 16/17 |
| Small face | 2/17 | **17/17** | 0/17 |
| Tilted 20 degrees | 14/34 | **34/34** | **34/34** |
| Tilted 40 degrees | 1/34 | 25/34 | **33/34** |
| Dark | 32/34 | **34/34** | 31/34 |
| Blurred | 33/34 | **34/34** | 33/34 |
| False alarms on frames with no face | 0/30 | 0/30 | 0/30 |
| Time per frame | 12.1 ms | 10.6 ms | 9.9 ms |

BlazeFace was rejected because it misses every small face, which matters in wide shots, and needs a large extra dependency. Its one clear win is strong tilt.

Limits: the portraits were chosen using YuNet at full size, which favours it a little. They are formal portraits of 17 people, so real footage will score lower. Anime faces were not measured, because no suitably licensed test material was available, and likely fall back to motion tracking.
