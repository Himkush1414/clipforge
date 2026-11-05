"""Write animated, word-by-word captions as an ASS file for libass.

Each word is its own positioned event. Inline size changes would make libass reflow the line, so
a popping word would shove its neighbours around; separate events keep every other word still.
"""

from __future__ import annotations

from pathlib import Path

from clipforge.analysis.transcribe import Word
from clipforge.captions.layout import LaidOutGroup, PlacedWord, group_words, is_rtl, layout_group
from clipforge.captions.presets import CaptionPreset

ENTRANCE_MS = 90


def ass_color(hex_color: str) -> str:
    """#RRGGBB -> &HBBGGRR& (ASS stores colours backwards)."""
    r, g, b = hex_color[1:3], hex_color[3:5], hex_color[5:7]
    return f"&H{b}{g}{r}&".upper()


def ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def escape_text(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _header(preset: CaptionPreset, width: int, height: int) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: None\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Word,{preset.font},100,{ass_color(preset.text_color)},{ass_color(preset.text_color)},"
        f"{ass_color(preset.outline_color)},{ass_color(preset.shadow_color)},"
        "0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _entrance(preset: CaptionPreset) -> tuple[str, str]:
    """Return (static tags, animated tags) for how a word first appears."""
    if preset.animation == "pop":
        return "\\fscx82\\fscy82", f"\\t(0,{ENTRANCE_MS},\\fscx100\\fscy100)"
    if preset.animation == "bounce":
        return (
            "\\fscx70\\fscy70",
            f"\\t(0,80,\\fscx112\\fscy112)\\t(80,{ENTRANCE_MS + 60},\\fscx100\\fscy100)",
        )
    if preset.animation == "fade":
        return "\\fad(70,50)", ""
    return "", ""


def _word_events(laid: LaidOutGroup, placed: PlacedWord, position: int, preset: CaptionPreset) -> list[str]:
    group = laid.group
    word = placed.word
    active_from = max(0, round((word.start - group.start) * 1000))
    next_start = group.words[position + 1].start if position + 1 < len(group.words) else group.end
    active_to = max(active_from + 40, round((min(next_start, group.end) - group.start) * 1000))

    base_color = preset.text_color
    if word.kind == "emphasis" and preset.emphasis_color:
        base_color = preset.emphasis_color

    tags = [
        "\\an5",
        # -1 lets libass find the base direction, so punctuation lands on the correct side of RTL words.
        "\\fe-1" if is_rtl(word.text) else "",
        f"\\fn{placed.font}",
        f"\\fs{placed.size:.1f}",
        f"\\1c{ass_color(base_color)}",
        f"\\bord{placed.size * preset.outline:.1f}",
        f"\\shad{placed.size * preset.shadow:.1f}",
    ]
    if preset.tilt:
        sign = 1 if (group.index + position) % 2 == 0 else -1
        tags.append(f"\\frz{sign * preset.tilt:.1f}")

    x, y = placed.x, placed.y
    if preset.animation == "slide":
        rise = placed.size * 0.18
        tags.append(f"\\move({x:.1f},{y + rise:.1f},{x:.1f},{y:.1f},0,110)")
    else:
        tags.append(f"\\pos({x:.1f},{y:.1f})")

    static, entrance = _entrance(preset)
    tags.append(static)

    # A \\t with a zero end time means "the whole event" in libass, so ranges are never zero length.
    active_color = ass_color(preset.active_color)
    highlight = f"\\t({active_from},{active_from + 1},\\1c{active_color})"
    restore = f"\\t({active_to},{active_to + 1},\\1c{ass_color(base_color)})"
    pop = ""
    if preset.active_pop > 1.0:
        peak = round(preset.active_pop * 100)
        rise_ms = min(70, max(20, (active_to - active_from) // 3))
        # The entrance also animates scale, so the pop waits for it to finish.
        begin = max(active_from, ENTRANCE_MS + 60) if preset.animation in ("pop", "bounce") else active_from
        settle = begin + rise_ms
        pop = f"\\t({begin},{settle},\\fscx{peak}\\fscy{peak})\\t({settle},{settle + 90},\\fscx100\\fscy100)"

    text = escape_text(word.text)
    start, end = ass_time(group.start), ass_time(group.end)
    main = f"{{{''.join(tags)}{entrance}{highlight}{restore}{pop}}}{text}"
    events = [f"Dialogue: 2,{start},{end},Word,,0,0,0,,{main}"]

    if preset.glow_color and preset.glow > 0:
        glow_tags = [t for t in tags if not t.startswith(("\\bord", "\\shad", "\\1c"))]
        blur = placed.size * preset.glow
        glow = (
            f"{{{''.join(glow_tags)}\\1c{ass_color(preset.glow_color)}\\3c{ass_color(preset.glow_color)}"
            f"\\bord{blur * 0.6:.1f}\\shad0\\blur{blur:.1f}{entrance}}}{text}"
        )
        events.append(f"Dialogue: 0,{start},{end},Word,,0,0,0,,{glow}")

    if preset.active_style == "box":
        events.append(
            _box_event(placed, preset, group.start + active_from / 1000, group.start + active_to / 1000)
        )
    return events


def _box_event(placed: PlacedWord, preset: CaptionPreset, start: float, end: float) -> str:
    """A rounded rectangle behind the active word, drawn on a lower layer."""
    pad_x, pad_y = placed.size * 0.16, placed.size * 0.04
    w = placed.width + pad_x * 2
    h = placed.size * 0.86 + pad_y * 2
    r = min(w, h) * 0.22
    k = r * 0.45
    path = (
        f"m {r:.1f} 0 l {w - r:.1f} 0 b {w - k:.1f} 0 {w:.1f} {k:.1f} {w:.1f} {r:.1f} "
        f"l {w:.1f} {h - r:.1f} b {w:.1f} {h - k:.1f} {w - k:.1f} {h:.1f} {w - r:.1f} {h:.1f} "
        f"l {r:.1f} {h:.1f} b {k:.1f} {h:.1f} 0 {h - k:.1f} 0 {h - r:.1f} "
        f"l 0 {r:.1f} b 0 {k:.1f} {k:.1f} 0 {r:.1f} 0"
    )
    tags = (
        f"\\an7\\pos({placed.x - w / 2:.1f},{placed.y - h / 2:.1f})"
        f"\\1c{ass_color(preset.box_color)}\\bord0\\shad0\\p1"
    )
    return f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Word,,0,0,0,,{{{tags}}}{path}{{\\p0}}"


def build_ass(
    words: list[Word], preset: CaptionPreset, width: int, height: int, ratio: str, language: str | None = None
) -> str:
    lines = [_header(preset, width, height)]
    for group in group_words(words, preset, ratio):
        laid = layout_group(group, preset, width, height, ratio, language)
        for placed in laid.words:
            # Spoken position, for timing. Right-to-left lines are stored in on-screen order.
            position = group.words.index(placed.word)
            lines.extend(event + "\n" for event in _word_events(laid, placed, position, preset))
    return "".join(lines)


def write_ass(
    path: Path,
    words: list[Word],
    preset: CaptionPreset,
    width: int,
    height: int,
    ratio: str,
    language: str | None = None,
) -> Path:
    path.write_text(build_ass(words, preset, width, height, ratio, language), encoding="utf-8")
    return path
