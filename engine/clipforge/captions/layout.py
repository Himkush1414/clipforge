"""Word grouping, emphasis and text measurement.

libass treats a font size as the height of the font's cell (winAscent + winDescent), not the em
size. Measuring with the same rule is what lets words be placed by hand with exact spacing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from fontTools.ttLib import TTFont
from PIL import ImageFont

from clipforge.analysis.scoring import EMOTION_WORDS
from clipforge.analysis.scoring_text import STOPWORDS
from clipforge.analysis.transcribe import Word
from clipforge.captions.presets import CJK_FACES, FALLBACK_FILES, CaptionPreset, cjk_path, font_location

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
HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
KANA = re.compile(r"[\u3040-\u30ff]")


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
    """Return (em per size unit, baseline offset ratio, cell ascent ratio) for a font."""
    path, index = font_location(font)
    tt = TTFont(path, lazy=True, fontNumber=index)
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
    path, index = font_location(font)
    return ImageFont.truetype(str(path), em_px, index=index, layout_engine=ImageFont.Layout.RAQM)


def is_rtl(text: str) -> bool:
    """Decided by the first letter with a strong direction, as the Unicode bidi algorithm does."""
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in ("R", "AL"):
            return True
        if direction == "L":
            return False
    return False


def measure(text: str, font: str, size: float) -> float:
    """Width in pixels of text drawn by libass at this ASS font size."""
    em = size * cell_ratio(font)[0]
    # Measure at a fixed large size and scale, so rounding does not depend on the output size.
    reference = _pil_font(font, 512)
    # Raqm shapes text the way libass does, which matters for joined scripts such as Arabic.
    return reference.getlength(text, direction="rtl" if is_rtl(text) else "ltr") * em / 512


@lru_cache(maxsize=16)
def _font_chars(font: str) -> frozenset[int]:
    path, index = font_location(font)
    tt = TTFont(path, lazy=True, fontNumber=index)
    chars = frozenset(tt.getBestCmap())
    tt.close()
    return chars


def covers(font: str, text: str) -> bool:
    """Whether the font can draw every letter and digit. Punctuation does not decide the font."""
    chars = _font_chars(font)
    return all(ord(ch) in chars for ch in text if ch.isalnum())


def _cjk_family(text: str, language: str | None) -> str:
    """Han characters are drawn differently by region, so the detected language picks the face."""
    if HANGUL.search(text):
        return CJK_FACES["ko"][0]
    if KANA.search(text):
        return CJK_FACES["ja"][0]
    return CJK_FACES.get(language or "ja", CJK_FACES["ja"])[0]


def font_for_text(text: str, font: str, language: str | None = None) -> str:
    """Display fonts only cover Latin. Other scripts use a fallback that really has the letters."""
    if covers(font, text):
        return font
    for fallback in ("Hind", *FALLBACK_FILES):
        if covers(fallback, text):
            return fallback
    if cjk_path().exists():
        family = _cjk_family(text, language)
        if covers(family, text):
            return family
    return font


def undrawable_words(words: list[Word], preset: CaptionPreset, language: str | None = None) -> list[str]:
    """Words no available font can draw. They would appear as empty boxes, so the job warns."""
    missing = []
    for word in words:
        if not covers(font_for_text(word.text, preset.font, language), word.text):
            missing.append(word.text)
    return missing


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
    group: WordGroup, preset: CaptionPreset, width: int, height: int, ratio: str, language: str | None = None
) -> LaidOutGroup:
    """Place each word. Text is shrunk, and then wrapped, to stay inside the safe area."""
    margins = SAFE_MARGINS[ratio]
    max_width = width * (1 - 2 * margins["side"])
    top, bottom = margins["top"] * height, (1 - margins["bottom"]) * height
    size = base_size(preset, width, height, ratio)

    def word_font(word: StyledWord) -> str:
        font = preset.emphasis_font if word.kind == "emphasis" and preset.emphasis_font else preset.font
        return font_for_text(word.text, font, language)

    def word_size(word: StyledWord, scale: float) -> float:
        factor = {"emphasis": preset.emphasis_scale, "filler": preset.filler_scale}.get(word.kind, 1.0)
        # Some fallback fonts reserve a tall line box, which would draw their letters small.
        match_letters = cell_ratio(preset.font)[0] / cell_ratio(word_font(word))[0]
        return size * factor * scale * min(1.6, max(1.0, match_letters))

    def gap_between(left: StyledWord, right: StyledWord, scale: float) -> float:
        """Enough room that neither word touches the other at the peak of its pop, outline included.

        Only one word pops at a time, so the larger of the two growths is what must fit.
        """
        left_w = measure(left.text, word_font(left), word_size(left, scale))
        right_w = measure(right.text, word_font(right), word_size(right, scale))
        growth = max(left_w, right_w) * (preset.active_pop - 1) / 2
        outlines = (word_size(left, scale) + word_size(right, scale)) * preset.outline
        return preset.word_gap * size * scale + growth + outlines

    def line_width(words: list[StyledWord], scale: float) -> float:
        widths = [measure(w.text, word_font(w), word_size(w, scale)) for w in words]
        gaps = sum(gap_between(a, b, scale) for a, b in zip(words, words[1:], strict=False))
        # The first and last words can also pop outward, so the safe margin must allow for that.
        edges = (widths[0] + widths[-1]) * (preset.active_pop - 1) / 2
        outline = (word_size(words[0], scale) + word_size(words[-1], scale)) * preset.outline
        return sum(widths) + gaps + edges + outline

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
        # Right-to-left lines read from the right, so the first spoken word is placed last in x order.
        if sum(is_rtl(w.text) for w in line) * 2 > len(line):
            line = list(reversed(line))
        widths = [measure(w.text, word_font(w), word_size(w, scale)) for w in line]
        gaps = [gap_between(a, b, scale) for a, b in zip(line, line[1:], strict=False)]
        x = width / 2 - (sum(widths) + sum(gaps)) / 2
        for position, (word, word_width) in enumerate(zip(line, widths, strict=True)):
            word_px = word_size(word, scale)
            font = word_font(word)
            descent_ratio, _ascent_ratio = cell_ratio(font)[1:]
            # Words share a baseline, so a big word and a small word sit on the same line.
            cell_center = baseline + word_px * descent_ratio - word_px / 2
            laid.words.append(
                PlacedWord(word, font, word_px, x + word_width / 2, cell_center, word_width, word_px)
            )
            x += word_width + (gaps[position] if position < len(gaps) else 0.0)
        line_top += line_height
    return laid


def _best_split(words: list[StyledWord], width_of) -> int:
    return min(range(1, len(words)), key=lambda i: max(width_of(words[:i]), width_of(words[i:])))
