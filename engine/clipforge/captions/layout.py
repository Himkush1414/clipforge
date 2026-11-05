"""Word grouping, emphasis and text measurement.

libass treats a font size as the height of the font's cell (winAscent + winDescent), not the em
size. Measuring with the same rule is what lets words be placed by hand with exact spacing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from fontTools.ttLib import TTFont
from PIL import ImageFont

from clipforge.analysis.scoring import EMOTION_WORDS, STOPWORDS
from clipforge.analysis.transcribe import Word
from clipforge.captions.presets import FONT_FILES, FONTS_DIR, CaptionPreset

# Fractions of the frame: everything visible must stay inside these. The bottom of a 9:16 frame
# and its right edge are covered by the app's own buttons on Shorts, Reels and TikTok.
SAFE_MARGINS: dict[str, dict[str, float]] = {
    "9:16": {"top": 0.12, "bottom": 0.24, "side": 0.08, "lower": 0.66},
    "1:1": {"top": 0.08, "bottom": 0.12, "side": 0.07, "lower": 0.80},
    "16:9": {"top": 0.08, "bottom": 0.10, "side": 0.06, "lower": 0.84},
}
RATIO_SIZE_FACTOR = {"9:16": 1.0, "1:1": 0.9, "16:9": 0.72}
GROUP_CHAR_LIMIT = {"9:16": 20, "1:1": 24, "16:9": 34}

BREAK_AFTER = re.compile(r"[.!?…,;:。！？，、][\"')\]]*$")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass
class StyledWord:
    text: str
    start: float
    end: float
    kind: str = "normal"  # "emphasis", "normal" or "filler"


@dataclass
class WordGroup:
    words: list[StyledWord]
    start: float
    end: float
    index: int = 0


@dataclass
class PlacedWord:
    word: StyledWord
    font: str
    size: float
    x: float  # horizontal centre of the word
    y: float  # vertical centre of the word's cell
    width: float
    height: float


@dataclass
class LaidOutGroup:
    group: WordGroup
    words: list[PlacedWord] = field(default_factory=list)
    scale: float = 1.0


@lru_cache(maxsize=32)
def cell_ratio(font: str) -> tuple[float, float, float]:
    """Return (em per size unit, baseline offset ratio, cell ascent ratio) for a bundled font."""
    tt = TTFont(FONTS_DIR / FONT_FILES[font], lazy=True)
    upm = tt["head"].unitsPerEm
    os2 = tt["OS/2"]
    ascent, descent = os2.usWinAscent, os2.usWinDescent
    if ascent + descent == 0:
        ascent, descent = tt["hhea"].ascent, abs(tt["hhea"].descent)
    cell = ascent + descent
    tt.close()
    return upm / cell, descent / cell, ascent / cell


@lru_cache(maxsize=256)
def _pil_font(font: str, em_px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS_DIR / FONT_FILES[font]), em_px)


def measure(text: str, font: str, size: float) -> float:
    """Width in pixels of text drawn by libass at this ASS font size."""
    em = size * cell_ratio(font)[0]
    # Measure at a fixed large size and scale, so rounding does not depend on the output size.
    reference = _pil_font(font, 512)
    return reference.getlength(text) * em / 512


def font_for_text(text: str, font: str) -> str:
    """Latin display fonts have no Devanagari, so those words use the bundled Hind Bold."""
    return "Hind" if DEVANAGARI.search(text) else font


def classify_word(word: Word) -> str:
    clean = re.sub(r"[^\w']", "", word.text).lower()
    if not clean:
        return "filler"
    if any(ch.isdigit() for ch in clean) or clean in EMOTION_WORDS:
        return "emphasis"
    if word.text.rstrip("\"')").endswith("!"):
        return "emphasis"
    if len(clean) > 2 and word.text.strip(".,!?\"'").isupper():
        return "emphasis"
    if clean in STOPWORDS:
        return "filler"
    return "emphasis" if len(clean) >= 8 else "normal"


def _end_group(words: list[Word], preset: CaptionPreset) -> list[StyledWord]:
    styled = [StyledWord(w.text, w.start, w.end, classify_word(w)) for w in words]
    # One big word per group: several just reads as one big line.
    emphasized = [s for s in styled if s.kind == "emphasis"]
    for extra in sorted(emphasized, key=lambda s: len(s.text), reverse=True)[1:]:
        extra.kind = "normal"
    if preset.uppercase:
        for s in styled:
            s.text = s.text.upper()
    return styled


def group_words(words: list[Word], preset: CaptionPreset, ratio: str, hold: float = 0.3) -> list[WordGroup]:
    """1 to max_words per group. Breaks at punctuation, pauses and width. Groups never overlap."""
    limit = GROUP_CHAR_LIMIT[ratio]
    raw: list[list[Word]] = []
    current: list[Word] = []
    for i, word in enumerate(words):
        length = sum(len(w.text) + 1 for w in current) + len(word.text)
        if current and (len(current) >= preset.max_words or length > limit):
            raw.append(current)
            current = []
        current.append(word)
        next_word = words[i + 1] if i + 1 < len(words) else None
        if BREAK_AFTER.search(word.text) or (next_word and next_word.start - word.end >= 0.35):
            raw.append(current)
            current = []
    if current:
        raw.append(current)

    groups: list[WordGroup] = []
    for index, chunk in enumerate(raw):
        next_start = raw[index + 1][0].start if index + 1 < len(raw) else None
        end = chunk[-1].end + hold
        if next_start is not None:
            # A tiny gap would make captions flicker, so the group holds until the next one starts.
            end = next_start if next_start - chunk[-1].end < 0.2 else min(end, next_start)
        groups.append(WordGroup(_end_group(chunk, preset), chunk[0].start, end, index))
    return groups


def base_size(preset: CaptionPreset, width: int, height: int, ratio: str) -> float:
    return preset.size * min(width, height) * RATIO_SIZE_FACTOR[ratio]


def anchor_y(preset: CaptionPreset, height: int, ratio: str) -> float:
    margins = SAFE_MARGINS[ratio]
    fraction = {"lower": margins["lower"], "center": 0.5, "upper": margins["top"] + 0.08}[preset.position]
    return (fraction + preset.y_offset) * height


def layout_group(
    group: WordGroup, preset: CaptionPreset, width: int, height: int, ratio: str
) -> LaidOutGroup:
    """Place each word. Text is shrunk, and then wrapped, to stay inside the safe area."""
    margins = SAFE_MARGINS[ratio]
    max_width = width * (1 - 2 * margins["side"])
    top, bottom = margins["top"] * height, (1 - margins["bottom"]) * height
    size = base_size(preset, width, height, ratio)

    def word_size(word: StyledWord, scale: float) -> float:
        factor = {"emphasis": preset.emphasis_scale, "filler": preset.filler_scale}.get(word.kind, 1.0)
        return size * factor * scale

    def word_font(word: StyledWord) -> str:
        font = preset.emphasis_font if word.kind == "emphasis" and preset.emphasis_font else preset.font
        return font_for_text(word.text, font)

    def line_width(words: list[StyledWord], scale: float) -> float:
        total = sum(measure(w.text, word_font(w), word_size(w, scale)) for w in words)
        total += preset.word_gap * size * scale * (len(words) - 1)
        # Room for the outline and the pop, so nothing touches the safe margin at peak scale.
        return total * preset.active_pop + size * scale * preset.outline * 2

    lines = [group.words]
    scale = min(1.0, max_width / max(1.0, line_width(group.words, 1.0)))
    if scale < 0.72 and len(group.words) > 1:
        split = _best_split(group.words, lambda w: line_width(w, 1.0))
        lines = [group.words[:split], group.words[split:]]
        scale = min(1.0, max_width / max(line_width(line, 1.0) for line in lines))

    laid = LaidOutGroup(group, scale=scale)
    heights = [max(word_size(w, scale) for w in line) * 1.02 for line in lines]
    total_height = sum(heights)
    # Peak pop grows from the centre, so keep half the growth inside the safe area too.
    grow = total_height * (preset.active_pop - 1) / 2
    center = anchor_y(preset, height, ratio)
    center = max(top + total_height / 2 + grow, min(bottom - total_height / 2 - grow, center))
    line_top = center - total_height / 2
    for line, line_height in zip(lines, heights, strict=True):
        baseline = line_top + line_height * 0.82
        gap = preset.word_gap * size * scale
        widths = [measure(w.text, word_font(w), word_size(w, scale)) for w in line]
        x = width / 2 - (sum(widths) + gap * (len(line) - 1)) / 2
        for word, word_width in zip(line, widths, strict=True):
            word_px = word_size(word, scale)
            font = word_font(word)
            descent_ratio, ascent_ratio = cell_ratio(font)[1:]
            # Words share a baseline, so a big word and a small word sit on the same line.
            cell_center = baseline + word_px * descent_ratio - word_px / 2
            laid.words.append(
                PlacedWord(word, font, word_px, x + word_width / 2, cell_center, word_width, word_px)
            )
            x += word_width + gap
        line_top += line_height
    return laid


def _best_split(words: list[StyledWord], width_of) -> int:
    return min(range(1, len(words)), key=lambda i: max(width_of(words[:i]), width_of(words[i:])))
