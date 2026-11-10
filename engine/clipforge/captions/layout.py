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
from clipforge.analysis.scoring_text import END_MARK, STOPWORDS
from clipforge.analysis.transcribe import Word
from clipforge.captions.presets import CJK_FACES, FALLBACK_FILES, CaptionPreset, cjk_path, font_location

# Fractions of the frame: everything visible must stay inside these. The bottom of a 9:16 frame
# and its right edge are covered by the app's own buttons on Shorts, Reels and TikTok.
# "lower" is where the centre of lower captions sits: lower-middle, above the platform's own titles and
# buttons. Text may move up to POSITION_BAND from there to fit, and never past the margins.
SAFE_MARGINS: dict[str, dict[str, float]] = {
    "9:16": {"top": 0.12, "bottom": 0.22, "side": 0.08, "lower": 0.65},
    "1:1": {"top": 0.08, "bottom": 0.14, "side": 0.07, "lower": 0.70},
    "16:9": {"top": 0.08, "bottom": 0.10, "side": 0.06, "lower": 0.74},
}
POSITION_BAND = 0.03
RATIO_SIZE_FACTOR = {"9:16": 1.0, "1:1": 0.9, "16:9": 0.72}
# Share of a preset's fill used at each ratio: a landscape line filling 80% of the width would be huge.
RATIO_FILL = {"9:16": 1.0, "1:1": 0.85, "16:9": 0.55}
GROUP_CHAR_LIMIT = {"9:16": 20, "1:1": 24, "16:9": 34}
# A typical three-word caption. A lone short word may grow to SHORT_WORD_GAIN times the size this line
# gets, so one word reads big without filling the screen.
REFERENCE_LINE = "THE BEST PART"
SHORT_WORD_GAIN = 2.0
# Presets that show one word at a time compare against one typical word instead.
REFERENCE_WORD = "PERFECT"
ONE_WORD_GAIN = 1.3
# Words that lead into the next one. A group cut after one of these reads as a broken phrase.
PHRASE_LEAD = frozenset(
    "a an the of to in on at for with from by my your his her its our their and or but".split()
)
# A lone word is merged into a neighbouring group across a pause up to this long.
ORPHAN_MERGE_GAP = 1.0
# Gaps between groups shorter than this are bridged, so captions never blink off and on.
MIN_EMPTY_GAP = 0.15

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


# Words that say little on their own, on top of the stopwords. They are never the big word.
GENERIC_WORDS = frozenset(
    """time times know knew think thought thing things kind sort lot lots way ways going go goes went
    gone say said says tell told want wanted mean means see saw look looked little bit something anything
    stuff actually probably maybe pretty well right yes sure make made take took come came back much many
    able also even still let's that's it's i'm there's what's they're we're you're don't didn't can't
    wasn't isn't doing done being been one start started put getting""".split()
)
NUMBER_WORDS = frozenset(
    """two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen twenty thirty
    forty fifty hundred thousand million billion first second third half""".split()
)
LONG_WORD = 7


def _clean(text: str) -> str:
    return re.sub(r"[^\w']", "", text).lower()


def emphasis_strength(word: Word, sentence_start: bool = True) -> int:
    """2 for a word that should be big, 1 for a long content word that may be, 0 for neither.

    Strong: numbers, names (a capital letter mid-sentence), strong words, a closing "!" and all-caps words.
    Chosen against hand labels of real podcast captions (scripts/bench/emphasis_eval.py).
    """
    clean = _clean(word.text)
    number = any(ch.isdigit() for ch in clean) or clean in NUMBER_WORDS
    if not number and (
        len(clean) < 3 or clean in STOPWORDS or clean in GENERIC_WORDS or clean.startswith("i'")
    ):
        return 0
    letters = word.text.lstrip("\"'(-¿¡")
    name = not sentence_start and bool(letters) and letters[0].isupper()
    shouted = word.text.rstrip("\"')").endswith("!") or word.text.strip(".,!?\"'").isupper()
    if number or name or shouted or clean in EMOTION_WORDS:
        return 2
    return 1 if len(clean) >= LONG_WORD else 0


def classify_word(word: Word, sentence_start: bool = True) -> str:
    """'emphasis' for a word that could be the big one, 'filler' for small words, else 'normal'."""
    if emphasis_strength(word, sentence_start):
        return "emphasis"
    clean = _clean(word.text)
    return "filler" if not clean or clean in STOPWORDS else "normal"


def _end_group(words: list[Word], starts: list[bool], preset: CaptionPreset) -> list[StyledWord]:
    styled = [
        StyledWord(w.text, w.start, w.end, classify_word(w, s)) for w, s in zip(words, starts, strict=True)
    ]
    # One big word per group: several just reads as one big line. A strong word beats a long one, the last
    # strong word wins (the end of a name), and among long words the longest.
    strength = [emphasis_strength(w, s) for w, s in zip(words, starts, strict=True)]
    big = None
    if 2 in strength:
        big = max(i for i, v in enumerate(strength) if v == 2)
    elif 1 in strength:
        big = max((i for i, v in enumerate(strength) if v == 1), key=lambda i: (len(words[i].text), -i))
    for i, s in enumerate(styled):
        if s.kind == "emphasis" and i != big:
            s.kind = "normal"
    if preset.uppercase:
        for s in styled:
            s.text = s.text.upper()
    return styled


def group_words(words: list[Word], preset: CaptionPreset, ratio: str, hold: float = 0.3) -> list[WordGroup]:
    """1 to max_words per group, broken at punctuation, pauses, width and phrase boundaries.

    A lone word is merged into a neighbour unless it is a strong word that deserves the screen to itself.
    Groups never overlap, and a gap between them is either at least MIN_EMPTY_GAP or bridged.
    """
    limit = GROUP_CHAR_LIMIT[ratio]
    # Positions in words, so each word knows whether it opens a sentence (a capital there is not a name).
    raw: list[list[int]] = []
    current: list[int] = []
    for i, word in enumerate(words):
        length = sum(len(words[j].text) + 1 for j in current) + len(word.text)
        if current and (len(current) >= preset.max_words or length > limit):
            carry = None
            # "THAT BECAME THE | CLASS" reads worse than "THAT BECAME | THE CLASS".
            if len(current) >= 2 and _clean(words[current[-1]].text) in PHRASE_LEAD:
                carry = current.pop()
            raw.append(current)
            current = [carry] if carry is not None else []
        current.append(i)
        next_word = words[i + 1] if i + 1 < len(words) else None
        if BREAK_AFTER.search(word.text) or (next_word and next_word.start - word.end >= 0.35):
            raw.append(current)
            current = []
    if current:
        raw.append(current)
    if preset.max_words > 1:
        raw = _merge_orphans(raw, words, preset.max_words, limit)

    groups: list[WordGroup] = []
    for index, positions in enumerate(raw):
        chunk = [words[i] for i in positions]
        starts = [i == 0 or bool(END_MARK.search(words[i - 1].text)) for i in positions]
        next_start = words[raw[index + 1][0]].start if index + 1 < len(raw) else None
        end = chunk[-1].end + hold
        if next_start is not None:
            # A short empty gap would make captions blink, so the group holds until the next one starts.
            end = next_start if next_start - end < MIN_EMPTY_GAP else end
        groups.append(WordGroup(_end_group(chunk, starts, preset), chunk[0].start, end, index))
    return groups


def _merge_orphans(raw: list[list[int]], words: list[Word], max_words: int, limit: int) -> list[list[int]]:
    """Give a lone ordinary word company: join it to the phrase it belongs to, or borrow a word."""

    def chars(group: list[int]) -> int:
        return sum(len(words[j].text) + 1 for j in group) - 1

    def gap(left: list[int], right: list[int]) -> float:
        return words[right[0]].start - words[left[-1]].end

    def ends_sentence(group: list[int]) -> bool:
        return bool(END_MARK.search(words[group[-1]].text))

    raw = [list(g) for g in raw]
    index = 0
    while index < len(raw):
        group = raw[index]
        prev = raw[index - 1] if index > 0 else None
        nxt = raw[index + 1] if index + 1 < len(raw) else None
        if len(group) != 1:
            index += 1
            continue
        at = group[0]
        opens = at == 0 or bool(END_MARK.search(words[at - 1].text))
        if emphasis_strength(words[at], opens) > 0:
            index += 1
            continue
        joins_prev = prev is not None and not ends_sentence(prev) and gap(prev, group) <= ORPHAN_MERGE_GAP
        # A comma does not stop "UNLESS," joining "DID SOMETHING"; the end of a sentence does.
        joins_next = (
            nxt is not None and not END_MARK.search(words[at].text) and gap(group, nxt) <= ORPHAN_MERGE_GAP
        )
        if joins_prev and prev and len(prev) < max_words and chars(prev + group) <= limit:
            prev.extend(group)
            del raw[index]
            continue
        if joins_next and nxt and len(nxt) < max_words and chars(group + nxt) <= limit:
            nxt[:0] = group
            del raw[index]
            continue
        if joins_prev and prev and len(prev) >= 2 and not BREAK_AFTER.search(words[prev[-1]].text):
            group.insert(0, prev.pop())  # 3 + 1 becomes 2 + 2
        elif joins_next and nxt and len(nxt) >= 2 and not BREAK_AFTER.search(words[nxt[0]].text):
            group.append(nxt.pop(0))
        index += 1
    return raw


def base_size(preset: CaptionPreset, width: int, height: int, ratio: str) -> float:
    return preset.size * min(width, height) * RATIO_SIZE_FACTOR[ratio]


def anchor_y(preset: CaptionPreset, height: int, ratio: str) -> float:
    margins = SAFE_MARGINS[ratio]
    fraction = {"lower": margins["lower"], "center": 0.5, "upper": margins["top"] + 0.08}[preset.position]
    return (fraction + preset.y_offset) * height


def layout_group(
    group: WordGroup, preset: CaptionPreset, width: int, height: int, ratio: str, language: str | None = None
) -> LaidOutGroup:
    """Place each word. A group is scaled to fill preset.fill of the safe width (fit to width).

    A lone short word stops at SHORT_WORD_GAIN times a typical three-word line, and text is never taller
    than the position band allows. A line that would drop below the preset's minimum size wraps to two
    lines instead. Nothing ever leaves the safe area.
    """
    margins = SAFE_MARGINS[ratio]
    max_width = width * (1 - 2 * margins["side"])
    top, bottom = margins["top"] * height, (1 - margins["bottom"]) * height
    floor = base_size(preset, width, height, ratio)
    target = max_width * preset.fill * RATIO_FILL[ratio]

    def word_font(word: StyledWord) -> str:
        font = preset.emphasis_font if word.kind == "emphasis" and preset.emphasis_font else preset.font
        return font_for_text(word.text, font, language)

    def word_size(word: StyledWord, size: float) -> float:
        factor = {"emphasis": preset.emphasis_scale, "filler": preset.filler_scale}.get(word.kind, 1.0)
        # Some fallback fonts reserve a tall line box, which would draw their letters small.
        match_letters = cell_ratio(preset.font)[0] / cell_ratio(word_font(word))[0]
        return size * factor * min(1.6, max(1.0, match_letters))

    def gap_between(left: StyledWord, right: StyledWord, size: float) -> float:
        """Enough room that neither word touches the other at the peak of its pop, outline included.

        Only one word pops at a time, so the larger of the two growths is what must fit.
        """
        left_w = measure(left.text, word_font(left), word_size(left, size))
        right_w = measure(right.text, word_font(right), word_size(right, size))
        growth = max(left_w, right_w) * (preset.active_pop - 1) / 2
        outlines = (word_size(left, size) + word_size(right, size)) * preset.outline
        return preset.word_gap * size + growth + outlines

    def line_width(words: list[StyledWord], size: float) -> float:
        widths = [measure(w.text, word_font(w), word_size(w, size)) for w in words]
        gaps = sum(gap_between(a, b, size) for a, b in zip(words, words[1:], strict=False))
        # The first and last words can also pop outward, so the safe margin must allow for that.
        edges = (widths[0] + widths[-1]) * (preset.active_pop - 1) / 2
        outline = (word_size(words[0], size) + word_size(words[-1], size)) * preset.outline
        return sum(widths) + gaps + edges + outline

    def line_height(line: list[StyledWord], size: float) -> float:
        return max(word_size(w, size) for w in line) * 1.02

    # Every width and height above is proportional to size, so one measurement at size 1 is enough.
    one_at_a_time = preset.max_words == 1
    sample, gain = (REFERENCE_WORD, ONE_WORD_GAIN) if one_at_a_time else (REFERENCE_LINE, SHORT_WORD_GAIN)
    reference = [StyledWord(t if preset.uppercase else t.lower(), 0, 0) for t in sample.split()]
    short_word_cap = gain * target / line_width(reference, 1.0)
    anchor = anchor_y(preset, height, ratio)
    band = POSITION_BAND * height
    # Peak pop grows from the centre, so half the growth must also stay inside.
    room = max(height * 0.02, min(bottom - (anchor - band), (anchor + band) - top))

    def fitted(lines: list[list[StyledWord]]) -> float:
        widest = max(line_width(line, 1.0) for line in lines)
        tallest = sum(line_height(line, 1.0) for line in lines) * preset.active_pop / 2
        return min(target / widest, short_word_cap, room / tallest)

    lines = [group.words]
    size = fitted(lines)
    if size < floor and len(group.words) > 1:
        split = _best_split(group.words, lambda w: line_width(w, 1.0))
        lines = [group.words[:split], group.words[split:]]
        size = fitted(lines)

    laid = LaidOutGroup(group, scale=size / floor if floor else 1.0)
    heights = [line_height(line, size) for line in lines]
    total_height = sum(heights)
    grow = total_height * (preset.active_pop - 1) / 2
    center = max(top + total_height / 2 + grow, min(bottom - total_height / 2 - grow, anchor))
    line_top = center - total_height / 2
    for line, height_of_line in zip(lines, heights, strict=True):
        baseline = line_top + height_of_line * 0.82
        # Right-to-left lines read from the right, so the first spoken word is placed last in x order.
        if sum(is_rtl(w.text) for w in line) * 2 > len(line):
            line = list(reversed(line))
        widths = [measure(w.text, word_font(w), word_size(w, size)) for w in line]
        gaps = [gap_between(a, b, size) for a, b in zip(line, line[1:], strict=False)]
        x = width / 2 - (sum(widths) + sum(gaps)) / 2
        for position, (word, word_width) in enumerate(zip(line, widths, strict=True)):
            word_px = word_size(word, size)
            font = word_font(word)
            descent_ratio, _ascent_ratio = cell_ratio(font)[1:]
            # Words share a baseline, so a big word and a small word sit on the same line.
            cell_center = baseline + word_px * descent_ratio - word_px / 2
            laid.words.append(
                PlacedWord(word, font, word_px, x + word_width / 2, cell_center, word_width, word_px)
            )
            x += word_width + (gaps[position] if position < len(gaps) else 0.0)
        line_top += height_of_line
    return laid


def _best_split(words: list[StyledWord], width_of) -> int:
    return min(range(1, len(words)), key=lambda i: max(width_of(words[:i]), width_of(words[i:])))
