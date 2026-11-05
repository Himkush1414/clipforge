import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from clipforge.analysis.transcribe import Word
from clipforge.captions.ass import ass_color, ass_time, build_ass, escape_text
from clipforge.captions.layout import (
    SAFE_MARGINS,
    classify_word,
    font_for_text,
    group_words,
    layout_group,
    measure,
)
from clipforge.captions.presets import BUILTIN_PRESETS, FONT_FILES, FONTS_DIR, CaptionPreset
from clipforge.captions.preview import render_preview, sample_words

BG = "color=c=black:s={w}x{h}"


def words_from(text, start=0.0, step=0.4):
    out = []
    t = start
    for token in text.split():
        out.append(Word(token, round(t, 3), round(t + step - 0.05, 3)))
        t += step
    return out


def plain(**kw):
    return CaptionPreset(outline=0, shadow=0, animation="none", active_pop=1.0, **kw)


def bbox(path, threshold=60):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(int)
    ys, xs = np.where(arr.max(axis=2) > threshold)
    assert xs.size, "nothing was drawn"
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_all_fonts_exist_and_have_a_license_file():
    for filename in FONT_FILES.values():
        assert (FONTS_DIR / filename).exists()
    assert len(list(FONTS_DIR.glob("*.OFL.txt"))) >= 7


def test_all_builtin_presets_are_valid_and_cover_styles():
    assert len(BUILTIN_PRESETS) >= 5
    assert {p.animation for p in BUILTIN_PRESETS.values()} >= {"pop", "fade", "bounce"}


@pytest.mark.parametrize(
    "bad",
    [{"font": "Comic Sans"}, {"active_color": "red"}, {"max_words": 4}, {"max_words": 0}, {"size": 2}],
)
def test_bad_preset_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        CaptionPreset(**bad)


def test_ass_color_is_reversed_and_time_is_centiseconds():
    assert ass_color("#FF8000") == "&H0080FF&"
    assert ass_time(3661.256) == "1:01:01.26"
    assert ass_time(-1) == "0:00:00.00"


def test_text_cannot_inject_override_tags():
    assert escape_text(r"a{\pos(0,0)}b") == "a(pos(0,0))b"


@pytest.mark.parametrize("max_words", [1, 2, 3])
def test_groups_respect_word_limit_and_never_overlap(max_words):
    preset = CaptionPreset(max_words=max_words)
    words = words_from("one two three four five six seven eight nine ten eleven twelve")
    groups = group_words(words, preset, "9:16")
    assert all(1 <= len(g.words) <= max_words for g in groups)
    assert sum(len(g.words) for g in groups) == len(words)
    for a, b in zip(groups, groups[1:], strict=False):
        assert a.end <= b.start + 1e-6
        assert a.start < a.end


def test_groups_break_at_punctuation_and_pauses():
    words = [Word("Stop.", 0, 0.3), Word("Go", 0.4, 0.6), Word("now", 2.0, 2.3)]
    groups = group_words(words, CaptionPreset(), "9:16")
    assert [[w.text for w in g.words] for g in groups] == [["STOP."], ["GO"], ["NOW"]]


def test_group_holds_briefly_after_last_word_but_not_past_next():
    words = [Word("hello.", 0, 0.5), Word("world", 0.6, 1.0)]
    groups = group_words(words, CaptionPreset(), "9:16")
    assert groups[0].end == 0.6
    assert groups[1].end == pytest.approx(1.3)


def test_word_kinds():
    assert classify_word(Word("the", 0, 1)) == "filler"
    assert classify_word(Word("incredible!", 0, 1)) == "emphasis"
    assert classify_word(Word("2026", 0, 1)) == "emphasis"
    assert classify_word(Word("boat", 0, 1)) == "normal"


def test_only_one_big_word_in_a_group():
    groups = group_words(words_from("incredible unbelievable amazing"), CaptionPreset(), "9:16")
    assert all(sum(w.kind == "emphasis" for w in g.words) <= 1 for g in groups)


def test_mixed_sizes_big_emphasis_small_filler():
    preset = CaptionPreset()
    group = group_words(words_from("the incredible boat"), preset, "9:16")[0]
    sizes = {p.word.kind: p.size for p in layout_group(group, preset, 1080, 1920, "9:16").words}
    assert sizes["emphasis"] > sizes["normal"] > sizes["filler"]


def test_devanagari_uses_a_font_that_has_it():
    assert font_for_text("नमस्ते", "Anton") == "Hind"
    assert font_for_text("HELLO", "Anton") == "Anton"


@pytest.mark.parametrize("font", sorted(set(FONT_FILES) - {"Hind"}))
def test_measured_width_matches_what_libass_draws(font, tmp_path):
    """Hand placement depends on this: predicted width must match the real render."""
    preset = plain(font=font, size=0.1, uppercase=True)
    words = [Word("MEASURE", 0.0, 2.0)]
    width, height = 1080, 1920
    out = render_preview(
        preset,
        tmp_path / "m.png",
        "9:16",
        at=1.0,
        words=words,
        size=(width, height),
        background=BG.format(w=width, h=height),
    )
    x0, _, x1, _ = bbox(out)
    group = group_words(words, preset, "9:16")[0]
    placed = layout_group(group, preset, width, height, "9:16").words[0]
    drawn = x1 - x0 + 1
    assert abs(drawn - measure("MEASURE", placed.font, placed.size)) / drawn < 0.05
    assert abs((x0 + x1) / 2 - width / 2) < 8


@pytest.mark.parametrize(
    "ratio,size", [("9:16", (1080, 1920)), ("1:1", (1080, 1080)), ("16:9", (1920, 1080))]
)
@pytest.mark.parametrize("name", sorted(BUILTIN_PRESETS))
def test_long_words_stay_inside_safe_margins_at_every_ratio(name, ratio, size, tmp_path):
    preset = BUILTIN_PRESETS[name]
    width, height = size
    words = [
        Word("EXTRAORDINARILY", 0.0, 1.0),
        Word("uncomfortable", 1.0, 2.0),
        Word("CONVERSATIONS", 2.0, 3.0),
    ]
    out = render_preview(
        preset.model_copy(update={"glow": 0, "glow_color": None}),
        tmp_path / "s.png",
        ratio,
        at=1.5,
        words=words,
        size=size,
        background=BG.format(w=width, h=height),
    )
    x0, y0, x1, y1 = bbox(out)
    margin = SAFE_MARGINS[ratio]
    tolerance = height * 0.012  # outline and shadow reach slightly past the text box
    assert x0 >= width * margin["side"] - tolerance
    assert x1 <= width * (1 - margin["side"]) + tolerance
    assert y0 >= height * margin["top"] - tolerance
    assert y1 <= height * (1 - margin["bottom"]) + tolerance


def _has_color(path, rgb, min_pixels=30):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return (np.abs(arr - np.array(rgb)).max(axis=2) < 40).sum() >= min_pixels


def test_active_color_follows_the_spoken_word(tmp_path):
    preset = CaptionPreset(
        active_color="#00FF00",
        text_color="#FF0000",
        outline=0,
        shadow=0,
        animation="none",
        active_pop=1.0,
        filler_scale=1.0,
        emphasis_scale=1.0,
    )
    words = [Word("LEFT", 0.2, 1.0), Word("RIGHT", 1.0, 2.0)]
    size = (1080, 1920)
    bg = BG.format(w=1080, h=1920)

    def x_of_green(at):
        out = render_preview(
            preset, tmp_path / f"{at}.png", "9:16", at=at, words=words, size=size, background=bg
        )
        arr = np.asarray(Image.open(out).convert("RGB")).astype(int)
        green = (arr[:, :, 1] > 180) & (arr[:, :, 0] < 90)
        return np.where(green)[1]

    during_first, during_second = x_of_green(0.6), x_of_green(1.5)
    assert during_first.size and during_second.size
    assert during_first.mean() < 540 < during_second.mean()


def test_every_word_active_exactly_when_spoken_in_the_ass_text():
    words = sample_words("alpha beta gamma")
    ass = build_ass(words, CaptionPreset(animation="none", active_pop=1.0), 1080, 1920, "9:16")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert len(events) == 3
    first_ms = round((words[1].start - words[0].start) * 1000)
    assert f"\\t({first_ms},{first_ms + 1}," in events[1]


def test_no_zero_length_transform_that_libass_reads_as_whole_event():
    ass = build_ass(sample_words("alpha beta"), CaptionPreset(), 1080, 1920, "9:16")
    assert "\\t(0,0," not in ass


def test_preview_clip_is_real_video(tmp_path):
    from clipforge.media.probe import probe

    out = render_preview(
        BUILTIN_PRESETS["neon-glow"], tmp_path / "p.mp4", "1:1", at=None, words=sample_words("short sample")
    )
    info = probe(out)
    assert (info.width, info.height) == (720, 720)
