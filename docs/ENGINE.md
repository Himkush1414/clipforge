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

## Word timing (captions)

Whisper hears the words well but places them loosely. Clipforge now keeps Whisper's text and finds each word's edges with CTC forced alignment: a small wav2vec2 model scores every letter every 20 ms, and the best path through the known letters gives the times. It runs on the CPU through ONNX Runtime, which faster-whisper already installs, so it adds no PyTorch. Measured by `scripts/bench/`. Raw results: [benchmarks/word-timing.json](benchmarks/word-timing.json).

Ranges are three runs, because Whisper's own times vary between runs. Lower is better.

| Sample | Measure | Whisper (before) | Aligned (now) |
|---|---|---|---|
| English, exact reference | Start, median | 203-214 ms | **25-29 ms** |
| | Start, worst | 543-813 ms | **120-303 ms** |
| | End, median | 115-130 ms | **46-50 ms** |
| Spanish, exact reference | Start, median | 163-177 ms | **24-28 ms** |
| | Start, worst | 390 ms | **69-120 ms** |
| | End, median | 80-83 ms | **21-28 ms** |
| Arabic, exact reference (11-13 words) | Start, median | 130-179 ms | **39-75 ms** |
| | Start, worst | 631-804 ms | **200-234 ms** |
| Hindi, exact reference (20-25 words) | Start, median | 119-138 ms | **37-45 ms** |
| | Start, within 100 ms | 20-33% | **95-100%** |
| | End, median | 35-62 ms | **9-20 ms** |
| English natural, approximate reference | Start, median | 47.5 ms | **37-40 ms** |
| | Start, worst | 560-875 ms | **119-147 ms** |
| | End, median | 45-47.5 ms | 47-48 ms |
| | End, 90th percentile | 230-248 ms | 248-250 ms |

The English natural sample's reference is Kokoro's own reported timing, whose sentence-final ends run about 200 ms past where the voice stops, as seen in the audio's energy. That inflates the end figures for both columns, and they are the only medians or percentiles here that did not improve.

Hindi's worst case is left out of the table because one run in three had a word about 7 seconds off in both columns (7260 ms before, 7054 ms after). The aligner cannot fix that, since the error comes from Whisper's word placement or from matching repeated text when scoring. In the other two runs the worst start fell from 333-551 ms to 85-130 ms. One Hindi word's end got worse in one run, from a worst of 455 ms to 550 ms.

**Real speech.** Ten minutes of a NASA podcast, which has no exact reference, scored at 53 places where the voice starts after a pause of at least 300 ms:

| Measure | Whisper | Aligned |
|---|---|---|
| Start, median | 150 ms | **37 ms** |
| Start, 90th percentile | 340 ms | **162 ms** |
| Start, worst | 640 ms | 626 ms |
| Within 100 ms | 21% | **81%** |

**Cost.** 70 minutes of real speech took 204 seconds on the CPU, 21 times faster than real time, after Whisper has finished. Peak memory rose by 0.41 GB. Models are loaded per job and released afterwards.

**When Whisper's times stay.** A segment keeps them if its text does not match its audio. Correct text scored 0.27 or more on the generated samples, wrong text and noise 0.084 or less, and the gate is 0.15. On the podcast, 7 of 113 segments stayed on Whisper's times. A missing or damaged model also keeps them, and never fails a job.

**Languages.** English, Spanish, Hindi and Arabic have models, about 1.1 GB in all, and setup downloads them. Other languages keep Whisper's times. ONNX models exist for some of them, including Japanese, but they were not measured, and the benchmark has no timing reference for Japanese.

Hindi and Arabic rest on few scored words, 20-25 and 11-13, because Whisper's text and the script rarely match exactly in those samples. Treat them as weak evidence. Only English was checked on real speech.

**Why not WhisperX.** It scored about the same in one reference run on the natural sample, at 139 ms worst start. It would add several GB of dependencies to a machine with 7.6 GB of RAM, and its default Spanish model is licensed CC BY-NC, which forbids commercial use.

Limits: the exact samples are generated voices, which are cleaner than real recordings. The start and end shifts were set from the exact samples' median offsets, and the end shift was also chosen to avoid worsening the natural sample, so that sample is not fully independent of it.
