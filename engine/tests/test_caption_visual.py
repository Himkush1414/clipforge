"""Visual regression: every preset at every ratio, compared with reference frames.

The references were rendered by the development machine's static FFmpeg build. CI uses Ubuntu's FFmpeg,
and on 2026-11-10 (FFmpeg 8, libass 0.17.4) it drew all 126 frames pixel for pixel the same. A future
libass, FreeType or HarfBuzz may move edges by a pixel, so the comparison ignores differences that small
and fails on what a viewer would see: text in a different place or size, a missing or extra word, a
wrong colour, or empty boxes.

Regenerate after an intended visual change:
  CLIPFORGE_UPDATE_VISUAL=1 pytest tests/test_caption_visual.py
then look at the changed PNGs before committing them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipforge.analysis.transcribe import Word
from clipforge.captions.presets import BUILTIN_PRESETS, CJK_FILE, cjk_path, fonts_dir
from clipforge.captions.preview import PREVIEW_SIZES, render_preview

REFERENCES = Path(__file__).parent / "visual"
FAILURES = Path(__file__).parent / "visual-failures"
UPDATE = os.environ.get("CLIPFORGE_UPDATE_VISUAL") == "1"
METRICS_LOG = os.environ.get("CLIPFORGE_VISUAL_METRICS")
# Mid-tone, so light text and dark outlines and shadows both stand out from it.
BACKGROUND = (104, 116, 132)
CJK_SOURCE = os.environ.get("CLIPFORGE_TEST_CJK_FONT") or str(
    Path.home() / "clipforge-work" / "data" / "fonts" / CJK_FILE
)

SCENES: dict[str, list[str]] = {
    "one-word": ["Incredible!"],
    # One word wider than the safe area at full size, so it must shrink to fit.
    "long-word": ["Counterrevolutionaries"],
    # Two wide words at the group's character limit, which wraps to two lines in the larger presets.
    "wide-pair": ["Wonderful", "strangers!"],
    "numbers": ["$1,250", "in", "2026"],
    "mixed-scripts": ["Mumbai", "नमस्ते", "مرحبا"],
    "cjk": ["東京", "Tokyo", "2026"],
}
STEP = 0.8

# Limits for what counts as the same picture; see compare(). CI measured 0 on all three, so these leave
# room for a few pixels of drift in a future libass, and were checked against deliberate breakages.
MAX_EDGE_SHIFT = 0.008  # fraction of the frame height: 4 px at 540, 8 px at 960
MAX_INK_CHANGE = 0.08  # fraction of the reference's text pixels
MAX_CHANGED_CELLS = 0.03  # fraction of 4x4 cells holding text whose colour changed clearly


def scene_words(tokens: list[str]) -> list[Word]:
    return [
        Word(text, round(0.1 + i * STEP, 3), round(0.1 + (i + 1) * STEP - 0.05, 3))
        for i, text in enumerate(tokens)
    ]


def render(preset_name: str, ratio: str, scene: str, out: Path) -> Path:
    words = scene_words(SCENES[scene])
    width, height = PREVIEW_SIZES[ratio]
    r, g, b = BACKGROUND
    # Half a second into the last word: entrance and pop have both finished, so timing jitter is harmless.
    at = words[-1].start + 0.5
    return render_preview(
        BUILTIN_PRESETS[preset_name],
        out,
        ratio,
        at=at,
        words=words,
        size=(width, height),
        background=f"color=c=0x{r:02x}{g:02x}{b:02x}:s={width}x{height}",
    )


def _ink(arr: np.ndarray) -> np.ndarray:
    return np.abs(arr - np.array(BACKGROUND)).max(axis=2) > 48


def compare(reference: np.ndarray, rendered: np.ndarray) -> dict[str, float]:
    """How far apart two frames are, in ways that survive a different libass build.

    edge_shift     largest move of any side of the text's bounding box, as a fraction of frame height
    ink_change     change in the number of text pixels, as a fraction of the reference's
    changed_cells  share of 4x4 cells holding text whose average colour moved by more than 48 levels
    """
    if reference.shape != rendered.shape:
        return {"edge_shift": 1.0, "ink_change": 1.0, "changed_cells": 1.0}
    height = reference.shape[0]
    a, b = _ink(reference), _ink(rendered)
    if not a.any() or not b.any():
        same = a.any() == b.any()
        return {"edge_shift": 0.0 if same else 1.0, "ink_change": 0.0 if same else 1.0, "changed_cells": 0.0}
    boxes = []
    for mask in (a, b):
        ys, xs = np.where(mask)
        boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()]))
    edge_shift = float(np.abs(boxes[0] - boxes[1]).max()) / height
    ink_change = abs(int(b.sum()) - int(a.sum())) / int(a.sum())
    k = 4
    h, w = (reference.shape[0] // k) * k, (reference.shape[1] // k) * k

    def cells(arr: np.ndarray) -> np.ndarray:
        return arr[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))

    colour = np.abs(cells(reference.astype(float)) - cells(rendered.astype(float))).max(axis=2)
    texty = (cells(a[..., None].astype(float))[..., 0] > 0) | (cells(b[..., None].astype(float))[..., 0] > 0)
    changed = float((colour[texty] > 48).sum()) / max(1, int(texty.sum()))
    return {
        "edge_shift": round(edge_shift, 4),
        "ink_change": round(ink_change, 4),
        "changed_cells": round(changed, 4),
    }


def within_limits(metrics: dict[str, float]) -> bool:
    return (
        metrics["edge_shift"] <= MAX_EDGE_SHIFT
        and metrics["ink_change"] <= MAX_INK_CHANGE
        and metrics["changed_cells"] <= MAX_CHANGED_CELLS
    )


CASES = [
    (name, ratio, scene)
    for name in sorted(BUILTIN_PRESETS)
    for ratio in ("9:16", "1:1", "16:9")
    for scene in SCENES
]


@pytest.mark.parametrize(
    "name,ratio,scene", CASES, ids=[f"{n}-{r.replace(':', 'x')}-{s}" for n, r, s in CASES]
)
def test_caption_frame_matches_reference(name, ratio, scene, tmp_path):
    if scene == "cjk":
        if not Path(CJK_SOURCE).exists():
            pytest.skip("CJK font not downloaded")
        fonts_dir()
        cjk_path().symlink_to(CJK_SOURCE)
    reference_path = REFERENCES / name / f"{ratio.replace(':', 'x')}-{scene}.png"
    out = render(name, ratio, scene, tmp_path / "frame.png")
    rendered = np.asarray(Image.open(out).convert("RGB")).astype(int)
    if UPDATE:
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rendered.astype(np.uint8)).save(reference_path, optimize=True)
        pytest.skip("reference written")
    assert reference_path.exists(), (
        f"no reference for {reference_path.name}; run with CLIPFORGE_UPDATE_VISUAL=1"
    )
    reference = np.asarray(Image.open(reference_path).convert("RGB")).astype(int)
    metrics = compare(reference, rendered)
    if METRICS_LOG:
        with open(METRICS_LOG, "a") as log:
            log.write(json.dumps({"case": f"{name}/{ratio}/{scene}", **metrics}) + "\n")
    if not within_limits(metrics):
        FAILURES.mkdir(exist_ok=True)
        stem = f"{name}-{ratio.replace(':', 'x')}-{scene}"
        Image.fromarray(rendered.astype(np.uint8)).save(FAILURES / f"{stem}.png")
        diff = np.abs(reference - rendered).max(axis=2).clip(0, 255).astype(np.uint8)
        Image.fromarray(diff).save(FAILURES / f"{stem}-diff.png")
    assert within_limits(metrics), f"{name} {ratio} {scene}: {metrics}"


def test_comparison_ignores_a_one_pixel_shift_but_not_a_moved_word():
    frame = np.zeros((200, 100, 3), dtype=int) + np.array(BACKGROUND)
    frame[90:110, 20:80] = 255
    nudged = np.roll(frame, 1, axis=1)
    assert within_limits(compare(frame, nudged))
    moved = np.roll(frame, 12, axis=0)
    assert not within_limits(compare(frame, moved))
    recoloured = frame.copy()
    recoloured[90:110, 20:80] = (255, 220, 0)
    assert not within_limits(compare(frame, recoloured))
