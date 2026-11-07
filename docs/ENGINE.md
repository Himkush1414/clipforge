# Engine audit

Which tool does each job, why, and what it measured. This page holds numbers from `engine/benchmarks/` and `scripts/bench/`, run on the development machine (RTX 3050 8 GB, 28 threads, 7.6 GB RAM under WSL2). Raw results are in [docs/benchmarks](benchmarks/).

Most test material is generated, so the correct answers are known exactly. That makes these numbers precise about mechanics. They say little about taste, or about footage unlike the test material. Current results are in [current.json](benchmarks/current.json). `baseline.json` is the record from before any replacement, and its noise was not yet seeded.

## Summary

| Stage | Tool | Measured against | Result | Decision |
|---|---|---|---|---|
| Speech to text | faster-whisper `large-v3-turbo` | `large-v3` | About twice as fast, with matching or mixed accuracy | Kept |
| Word timing | CTC forced alignment | Whisper's own times, WhisperX | Start errors fall from about 200 ms to 25 ms on exact English samples | **Replaced** |
| Face detection | YuNet | Haar, BlazeFace | 178 of 187 faces found, up from 116 | **Replaced** |
| Scene cuts | Own detector | PySceneDetect, FFmpeg `scdet` | Only tool with all hard cuts and no false cuts. Misses dissolves | Kept, and fixed |
| Encoding | NVENC | Tuned NVENC, x264, x265 | Tuning gains at most 0.1 VMAF | Kept |
| Loudness | Two-pass `loudnorm` | Earlier version of itself | Speech within 0.0 LU, clipped noise 1.0 LU off | Kept, and fixed |
| Frame cuts | FFmpeg, own planning | Known answers only | 44 of 44 exact | Kept |
| Audio sync | FFmpeg, own planning | Known answers only | Worst 15 ms over 40 cuts | Kept |
| Captions | libass, own layout | Known answers only | 84 renders, none outside the safe area | Kept |
| Clip selection | Own scoring plus editor pass | Random windows, chosen clips | Held-out recall 0.62, up from 0.48, against 0.34 for random | **Added editor** |
| Silence | Own threshold | Known answers only | 2 false gaps and a 200 ms edge error | Kept, **open finding** |

"Known answers only" means no alternative tool was compared, so those rows show that the stage works, not that it is the best choice.

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

## Speech to text

`large-v3-turbo` stays over `large-v3`. Two runs, from `scripts/bench/speech_timing.py`, on generated voices with the script as the reference. The two models' accuracy varies between runs, so these are ranges.

| Language | Turbo error | `large-v3` error | Turbo speed | `large-v3` speed |
|---|---|---|---|---|
| English | 1.2% WER | 1.2% WER | 17x real time | 11x |
| Spanish | 1.9% WER | 1.9% WER | 18-19x | 8-10x |
| Hindi | 11.8-25.5% WER | 19.6-25.5% WER | 10-12x | 4x |
| Arabic | 2.6-5.1% WER | 2.6% WER | 27-28x | 11-12x |
| Japanese | 1.8-3.5% CER | 3.5-5.3% CER | 26x | 11x |

Language detection was right every time, for both models. Raw results: [benchmarks/whisper-models.json](benchmarks/whisper-models.json).

Limits: these are generated voices, and Hindi error is high in both models. Real non-English speech was not measured, nor was the CPU fallback model, `small`.

## Scene cuts

Known-answer videos from `scripts/bench/scene_tools.py`. Raw results: [benchmarks/scene-tools.json](benchmarks/scene-tools.json).

| Tool | Hard cuts exact (of 6) | False cuts on hard-cut video | False cuts on fast motion and a flash | Dissolves found (of 2) |
|---|---|---|---|---|
| clipforge | **6** | **0** | **0** | 0 |
| PySceneDetect Content | 5 | **0** | **0** | 0 |
| PySceneDetect Adaptive | 5 | **0** | **0** | 0 |
| PySceneDetect Histogram | 5 | **0** | 1 | **2**, with 2 false |
| PySceneDetect Hash | 5 | 2 | 11 | 0 |
| FFmpeg `scdet` | **6** | **0** | 3 | 0 |

Dissolves stay undetected. Clips cut at sentences, so the harm is small. Speed is not compared, because it varied too much between runs of the same tool.

## Encoding

NVENC stays at its current settings. Compared with `h264_nvenc` as it is now, on three NASA clips and one generated clip. Raw results: [benchmarks/encoders.json](benchmarks/encoders.json).

| Setting | VMAF difference | File size | Time |
|---|---|---|---|
| `h264_nvenc` tuned | +0.03 to +0.10 | 0.98-1.13x | 1.2x |
| `hevc_nvenc` | -3.02 to -0.14 | 0.61-1.00x | 1.0-1.1x |
| `libx264` medium, CRF 18 | -1.21 to +1.41 | 0.48-1.08x | 1.5-2.7x |
| `libx265` medium, CRF 20 | -1.88 to +0.18 | 0.22-0.95x | 3.4-4.7x |

A 0.1 VMAF gain is not visible, so tuning is not worth 20% more time. x264 makes files about half the size on real footage, for up to about a point of VMAF and about twice the time. NVENC stays for its speed, and because platforms re-encode uploads anyway. If file size starts to matter, x264 is the option to revisit.

## Render

Time for a 30-second clip, five runs. The 7.9x in the baseline was a difference in the machine's state: code from before YuNet gets 7.4-7.5x today.

| Output | Encoder | Speed | Peak RAM |
|---|---|---|---|
| 9:16 1080p | `h264_nvenc` | 7.3-7.6x real time | 0.73-0.81 GB |
| 16:9 4K | `h264_nvenc` | 2.3x | 0.94-1.02 GB |
| 9:16 8K | `hevc_nvenc` | 0.6x | 1.78-1.87 GB |

## Cuts, sync, captions, loudness

| Stage | Result |
|---|---|
| Frame cuts | 44 of 44 exact, at 23.976, 29.97, 30 and 60 fps |
| Audio sync | Worst offset 15 ms and drift 10 ms, across 40 cuts |
| Captions | 84 renders, none outside the safe area, tightest margin 10 px |
| Loudness, generated speech | 0.0 LU from -14 LUFS |
| Loudness, other material | 0.3 to 1.0 LU off, worst on clipped noise |

## Clip selection

Two passes. The scorer finds lively moments, then an editor pass (`engine/clipforge/analysis/editor.py`) asks four questions of each candidate: does it open with a hook, make sense with no earlier context, finish its thought, and hold to one topic? Filler at either end is trimmed, as an editor would trim it, and candidates that still fail are dropped. If nothing passes, the best one is kept with a reason that says what is wrong. Word lists exist for English and Spanish. Other languages are judged on structure alone.

Measured by `scripts/bench/pick_eval.py`, as overlap with chosen clips. Raw results: [benchmarks/selection.json](benchmarks/selection.json).

| Test set | Measure | Random windows | Scorer alone | With editor |
|---|---|---|---|---|
| **Held-out: 2 NASA episodes** | **Pick recall** | **0.34** | **0.48** | **0.62** |
| | Mean IoU | 0.21 | 0.20 | **0.37** |
| Tuning set: 3 NASA episodes | Pick recall | 0.27 | 0.25 | **0.48** |
| | Mean IoU | 0.17 | 0.13 | **0.33** |
| Generated talk (8 good, 8 bad passages) | Pick recall | 0.49 | 0.50 | **1.00** |
| | Mean IoU | 0.35 | 0.49 | **0.95** |

The held-out episodes are the honest figure. They were labelled before any scorer ran on them and never used for tuning, while the editor's word lists were adjusted after seeing the tuning set. They hold only 13 picks, so treat the gain as promising, not settled.

The picks were chosen by Claude from the transcripts, so they are one AI's taste, not a person's. The earlier figure of 80% of clips catching a planted moment overstated quality, since those moments differ from plain talk in ways real speech does not. On that old test, capture rose from 0.80 to 0.83. The editor adds about 2 seconds on 2 hours of speech.

## Silence (open finding)

On pink noise with 4 known gaps, the detector finds all 4, plus 2 false gaps, with a worst edge error of 200 ms. The threshold sits between the signal's own 10th and 90th percentiles, so dips inside steady noise can count as silence.

This is left as it is because silences are only used as approximate marks in sections with no speech. Cuts in speech follow word times. Steady music or noise could gain some extra clip boundaries. No alternative has been measured, so this stage is not yet shown to be the best choice. The baseline's better figures, 4 gaps and 150 ms, came from unseeded noise, and an earlier unseeded run also found 6. The code has not changed.

## Not measured

- Real footage for clip scoring, and any measure of taste
- Anime faces, for lack of suitably licensed test material
- Real non-English speech, in transcription or word timing
- Real uploads to any platform, which need the owner's credentials
