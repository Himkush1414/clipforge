import os
from pathlib import Path

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
    is_rtl,
    layout_group,
    measure,
    undrawable_words,
)
from clipforge.captions.presets import (
    BUILTIN_PRESETS,
    CJK_FILE,
    FONT_FILES,
    FONTS_DIR,
    CaptionPreset,
    cjk_path,
    fonts_dir,
)
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


def big_word(text: str) -> str | None:
    """The big word of a one-group caption, from a whole sentence so names can be told apart."""
    words = words_from(text)
    groups = group_words(words, CaptionPreset(uppercase=False, max_words=3), "16:9")
    big = [w.text for g in groups for w in g.words if w.kind == "emphasis"]
    assert len(big) <= len(groups)
    return big[-1] if big else None


def test_a_name_mid_sentence_is_the_big_word():
    assert big_word("studying Mars or") == "Mars"
    # The end of a two-word name.
    assert big_word("with Scott Kelly") == "Kelly"


def test_a_capital_at_a_sentence_start_is_not_a_name():
    assert big_word("Today the") is None
    assert big_word("done. Then the") is None


def test_capital_i_contractions_are_not_names():
    assert big_word("and then I'll go") is None


def test_numbers_win_over_long_words():
    assert big_word("been eight years") == "eight"
    assert big_word("Episode 378,") == "378,"


def test_generic_words_are_never_big():
    assert big_word("something actually") is None
    assert classify_word(Word("probably", 0, 1)) == "normal"


def test_long_content_word_is_big_and_short_one_is_not():
    assert big_word("an integral member") == "integral"
    assert big_word("in the park") is None


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


CJK_SOURCE = os.environ.get("CLIPFORGE_TEST_CJK_FONT") or str(
    Path.home() / "clipforge-work" / "data" / "fonts" / CJK_FILE
)
needs_cjk = pytest.mark.skipif(not Path(CJK_SOURCE).exists(), reason="CJK font not downloaded")


@pytest.fixture
def cjk_font():
    fonts_dir()
    cjk_path().symlink_to(CJK_SOURCE)


def test_without_the_cjk_font_japanese_is_reported_not_silently_boxed():
    words = [Word("今日は", 0, 1), Word("hello", 1, 2)]
    assert undrawable_words(words, CaptionPreset(), "ja") == ["今日は"]


def test_arabic_and_hebrew_have_fonts():
    assert font_for_text("مرحبا", "Anton") == "Noto Sans Arabic"
    assert font_for_text("שלום", "Anton") == "Noto Sans Hebrew"
    assert font_for_text("سلام", "Anton") == "Noto Sans Arabic"  # Persian and Urdu share this script.
    assert undrawable_words([Word("مرحبا", 0, 1), Word("שלום", 1, 2)], CaptionPreset()) == []


def test_text_no_font_can_draw_is_reported():
    assert undrawable_words([Word("ሰላም", 0, 1)], CaptionPreset()) == ["ሰላም"]  # Amharic


@pytest.mark.parametrize(
    "text,rtl",
    [
        ("مرحبا", True),
        ("שלום", True),
        ("hello", False),
        ("123", False),
        ("", False),
        ("(مرحبا)", True),
        ("٣٤", False),
    ],
)
def test_direction_detection(text, rtl):
    assert is_rtl(text) is rtl


RTL_WORDS = {
    "arabic": ["مرحبا", "بكم", "اليوم"],
    "hebrew": ["שלום", "לכולם", "היום"],
}


@pytest.mark.parametrize("language", sorted(RTL_WORDS))
def test_rtl_words_are_placed_right_to_left_and_keep_spoken_order(language):
    # A low minimum size keeps the three words on one line, so the order across the line can be checked.
    preset = CaptionPreset(uppercase=True, size=0.04)
    tokens = RTL_WORDS[language]
    words = [Word(t, i * 0.5, i * 0.5 + 0.45) for i, t in enumerate(tokens)]
    group = group_words(words, preset, "9:16")[0]
    placed = layout_group(group, preset, 1080, 1920, "9:16").words
    by_text = {p.word.text: p.x for p in placed}
    assert by_text[tokens[0]] > by_text[tokens[1]] > by_text[tokens[2]]


def _green_center(path):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(int)
    green = (arr[:, :, 1] > 180) & (arr[:, :, 0] < 90) & (arr[:, :, 2] < 90)
    return np.where(green)[1].mean() if green.any() else None


@pytest.mark.parametrize("language", sorted(RTL_WORDS))
def test_rtl_active_word_moves_from_right_to_left_on_real_renders(language, tmp_path):
    preset = CaptionPreset(
        active_color="#00FF00", text_color="#FF0000", outline=0, shadow=0, animation="none", active_pop=1.0
    )
    tokens = RTL_WORDS[language]
    words = [Word(t, i * 0.6, i * 0.6 + 0.55) for i, t in enumerate(tokens)]
    centers = []
    for index in range(3):
        out = render_preview(
            preset,
            tmp_path / f"{index}.png",
            "9:16",
            at=0.2 + index * 0.6,
            words=words,
            size=(1080, 1920),
            background=BG.format(w=1080, h=1920),
        )
        centers.append(_green_center(out))
    assert all(c is not None for c in centers)
    assert centers[0] > centers[1] > centers[2]


@pytest.mark.parametrize(
    "text,font_expected",
    [("مرحبا", "Noto Sans Arabic"), ("شكرا", "Noto Sans Arabic"), ("שלום", "Noto Sans Hebrew")],
)
def test_rtl_width_prediction_matches_libass_including_joined_letters(text, font_expected, tmp_path):
    preset = plain(size=0.1, uppercase=False)
    words = [Word(text, 0, 2)]
    out = render_preview(
        preset,
        tmp_path / "w.png",
        "9:16",
        at=1.0,
        words=words,
        size=(1080, 1920),
        background=BG.format(w=1080, h=1920),
    )
    x0, _, x1, _ = bbox(out)
    placed = layout_group(group_words(words, preset, "9:16")[0], preset, 1080, 1920, "9:16").words[0]
    assert placed.font == font_expected
    drawn = x1 - x0 + 1
    assert abs(drawn - placed.width) / drawn < 0.1, (drawn, placed.width)
    # Side bearings and the tails of joined letters grow with the text, so relative to its width.
    assert abs((x0 + x1) / 2 - 540) < 0.1 * drawn


def test_arabic_letters_are_actually_joined(tmp_path):
    """Joined letters draw narrower than the same letters kept apart. Boxes or unshaped text would not."""
    preset = plain(size=0.1, uppercase=False)

    def drawn_width(text):
        out = render_preview(
            preset,
            tmp_path / "a.png",
            "9:16",
            at=1.0,
            words=[Word(text, 0, 2)],
            size=(1080, 1920),
            background=BG.format(w=1080, h=1920),
        )
        x0, _, x1, _ = bbox(out)
        return x1 - x0

    joined = drawn_width("سلم")
    apart = drawn_width("س\u200cل\u200cم")  # Zero-width non-joiners prevent the letters from joining.
    assert joined < apart


@pytest.mark.parametrize("name", sorted(BUILTIN_PRESETS))
def test_rtl_captions_stay_inside_safe_margins(name, tmp_path):
    preset = BUILTIN_PRESETS[name].model_copy(update={"glow": 0, "glow_color": None})
    words = [Word("والمستخدمين", 0.0, 1.0), Word("المعلومات", 1.0, 2.0), Word("التكنولوجيا", 2.0, 3.0)]
    out = render_preview(
        preset,
        tmp_path / "s.png",
        "9:16",
        at=1.5,
        words=words,
        size=(1080, 1920),
        background=BG.format(w=1080, h=1920),
    )
    x0, _, x1, _ = bbox(out)
    assert x0 >= 1080 * SAFE_MARGINS["9:16"]["side"] - 24
    assert x1 <= 1080 * (1 - SAFE_MARGINS["9:16"]["side"]) + 24


@needs_cjk
@pytest.mark.parametrize(
    "text,language,family",
    [
        ("今日は", "ja", "Noto Sans CJK JP"),
        ("오늘은", "ko", "Noto Sans CJK KR"),
        ("今天", "zh", "Noto Sans CJK SC"),
        ("今日", "ja", "Noto Sans CJK JP"),
        ("カタカナ", None, "Noto Sans CJK JP"),
        ("HELLO", "ja", "Anton"),
    ],
)
def test_cjk_words_pick_the_right_regional_face(cjk_font, text, language, family):
    assert font_for_text(text, "Anton", language) == family


@needs_cjk
def test_cjk_captions_draw_real_letters_not_empty_boxes(cjk_font, tmp_path):
    """Every missing glyph draws the same box, so two different letters must look different."""
    preset = CaptionPreset(outline=0, shadow=0, animation="none", active_pop=1.0, active_color="#FFFFFF")

    def draw(letter):
        out = render_preview(
            preset,
            tmp_path / f"{ord(letter)}.png",
            "9:16",
            at=0.5,
            words=[Word(letter, 0, 2)],
            size=(540, 960),
            background=BG.format(w=540, h=960),
        )
        return np.asarray(Image.open(out).convert("L")).astype(int)

    first, second = draw("今"), draw("鬱")
    assert first.max() > 200
    assert np.abs(first - second).sum() > 50_000


@needs_cjk
def test_cjk_width_prediction_matches_libass(cjk_font, tmp_path):
    preset = plain(size=0.1)
    words = [Word("今日はいい天気", 0, 2)]
    out = render_preview(
        preset,
        tmp_path / "w.png",
        "9:16",
        at=1.0,
        words=words,
        size=(1080, 1920),
        background=BG.format(w=1080, h=1920),
    )
    x0, _, x1, _ = bbox(out)
    placed = layout_group(group_words(words, preset, "9:16")[0], preset, 1080, 1920, "9:16", "ja").words[0]
    assert placed.font == "Noto Sans CJK JP"
    drawn = x1 - x0 + 1
    assert abs(drawn - measure(words[0].text.upper(), placed.font, placed.size)) / drawn < 0.05


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
    assert abs((x0 + x1) / 2 - width / 2) < max(8, 0.02 * drawn)  # side bearings grow with the text


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


@pytest.mark.parametrize(
    "ratio,size", [("9:16", (1080, 1920)), ("1:1", (1080, 1080)), ("16:9", (1920, 1080))]
)
@pytest.mark.parametrize("name", sorted(BUILTIN_PRESETS))
def test_words_never_collide_even_at_peak_pop(name, ratio, size):
    """Found on a real render: a big popping word covered its neighbours."""
    preset = BUILTIN_PRESETS[name]
    width, height = size
    if preset.max_words == 1:
        return
    words = [Word("really", 0.0, 0.5), Word("wondered", 0.5, 1.0), Word("how", 1.0, 1.5)]
    group = group_words(words, preset, ratio)[0]
    placed = layout_group(group, preset, width, height, ratio).words
    assert len(placed) >= 2
    pop = preset.active_pop
    same_line = [(a, b) for a, b in zip(placed, placed[1:], strict=False) if abs(a.y - b.y) < a.size]
    assert same_line
    for a, b in same_line:
        for popping_left in (True, False):
            left_half = a.width / 2 * (pop if popping_left else 1.0) + a.size * preset.outline
            right_half = b.width / 2 * (1.0 if popping_left else pop) + b.size * preset.outline
            assert a.x + left_half <= b.x - right_half + 0.5, (a.word.text, b.word.text)


def _word_columns(path, threshold=120):
    """Columns holding drawn text. Runs separated by an empty column are separate words."""
    arr = np.asarray(Image.open(path).convert("RGB")).astype(int)
    used = np.where(arr.max(axis=2).max(axis=0) > threshold)[0]
    return np.split(used, np.where(np.diff(used) > 1)[0] + 1)


def test_big_word_leaves_visible_space_on_the_rendered_frame_at_peak_pop(tmp_path):
    """Measured on real pixels at the moment the middle word is at its largest."""
    # A white outline on black: letters merge into word blobs, so only gaps between words remain.
    # A low minimum size keeps the three words on one line, where their gaps can be seen.
    preset = BUILTIN_PRESETS["manga-punch"].model_copy(
        update={"tilt": 0.0, "shadow": 0.0, "outline_color": "#FFFFFF", "size": 0.05}
    )
    words = [Word("really", 0.0, 0.5), Word("wondered", 0.5, 1.0), Word("how", 1.0, 1.5)]
    out = render_preview(
        preset,
        tmp_path / "peak.png",
        "9:16",
        at=0.56,
        words=words,
        size=(1080, 1920),
        background=BG.format(w=1080, h=1920),
    )
    assert len(_word_columns(out)) == 3


# ---------- fit to width (anime clipper look) ----------
ANIME = BUILTIN_PRESETS["anime-impact"]
SAFE_W_916 = 1080 * (1 - 2 * SAFE_MARGINS["9:16"]["side"])


def _laid(text, preset=ANIME, ratio="9:16", size=(1080, 1920)):
    words = words_from(text)
    group = group_words(words, preset, ratio)[0]
    return layout_group(group, preset, *size, ratio)


def _span(laid):
    return max(p.x + p.width / 2 for p in laid.words) - min(p.x - p.width / 2 for p in laid.words)


def _block_center(laid):
    top = min(p.y - p.height / 2 for p in laid.words)
    bottom = max(p.y + p.height / 2 for p in laid.words)
    return (top + bottom) / 2


@pytest.mark.parametrize("text", ["find the answers", "time flies fast", "we never stop"])
def test_a_three_word_line_fills_70_to_85_percent_of_the_safe_width(text):
    laid = _laid(text)
    # Words on one line overlap vertically; words on two lines do not.
    assert max(p.y - p.height / 2 for p in laid.words) < min(p.y + p.height / 2 for p in laid.words)
    assert 0.70 <= _span(laid) / SAFE_W_916 <= 0.85


def test_a_single_short_word_is_about_twice_a_three_word_line():
    single = _laid("mad?").words[0].size
    line = max(p.size for p in _laid("time flies fast").words if p.word.kind == "normal")
    assert 1.6 <= single / line <= 2.2


def test_emphasis_words_are_bigger_than_the_rest_of_their_line():
    sizes = {p.word.kind: p.size for p in _laid("the 2026 launch").words}
    assert sizes["emphasis"] > sizes["normal"] > sizes["filler"]


def test_letters_are_at_least_60_px_tall_at_1080x1920():
    """Anton capitals are 0.496 of the libass size. The preset minimum keeps them at least 60 px.

    Only a single word too long for the safe width at that size is drawn smaller, since it cannot wrap.
    """
    for text in [
        "find the answers",
        "unbelievably",
        "unbelievable wonderful stuff",
        "mad?",
        "time flies fast",
    ]:
        laid = _laid(text)
        normal = [p for p in laid.words if p.word.kind != "filler"]
        assert min(p.size for p in normal) * 0.496 >= 60, text


@pytest.mark.parametrize(
    "text", ["find the answers", "mad?", "unbelievable wonderful stuff", "counterrevolutionaries"]
)
def test_lower_captions_are_centred_62_to_68_percent_down_a_9_16_frame(text):
    assert 0.62 <= _block_center(_laid(text)) / 1920 <= 0.68


def test_no_group_ends_on_a_word_that_leads_into_the_next():
    words = words_from("we saw the dragon fly over the river")
    groups = group_words(words, ANIME, "9:16")
    assert all(g.words[-1].text.lower() not in {"the", "of", "a"} for g in groups[:-1])


def test_a_lone_ordinary_word_is_given_company_but_a_strong_word_may_stand_alone():
    texts = [[w.text for w in g.words] for g in group_words(words_from("we went home today."), ANIME, "9:16")]
    assert all(len(t) > 1 for t in texts), texts
    texts = [
        [w.text for w in g.words]
        for g in group_words(words_from("it was over. unless, did it work"), ANIME, "9:16")
    ]
    assert ["UNLESS,", "DID"] in [g[:2] for g in texts], texts  # a comma does not leave "unless," alone
    texts = [[w.text for w in g.words] for g in group_words(words_from("ready? 2026."), ANIME, "9:16")]
    assert ["2026."] in [[t.lower() for t in g] for g in texts]


def test_groups_switch_on_word_boundaries_with_no_short_blank_gaps():
    # Gaps of 0.30 s (bridged) and 0.90 s (left, longer than 0.15 s once the hold ends).
    words = [Word("one.", 0.0, 0.4), Word("two.", 0.7, 1.1), Word("three.", 2.0, 2.4)]
    groups = group_words(words, ANIME, "9:16")
    assert [g.start for g in groups] == [0.0, 0.7, 2.0]  # each group starts with its first word
    for a, b in zip(groups, groups[1:], strict=False):
        assert a.end <= b.start
        assert a.end == b.start or b.start - a.end >= 0.15


def test_anime_impact_pops_in_85_105_100_and_has_a_soft_shadow_layer():
    ass = build_ass(words_from("hello there"), ANIME, 1080, 1920, "9:16")
    main = [line for line in ass.splitlines() if line.startswith("Dialogue: 2")][0]
    assert "\\fscx85\\fscy85" in main and "\\t(0,60,\\fscx105\\fscy105)\\t(60,110,\\fscx100\\fscy100)" in main
    assert "\\shad0" in main
    shadow = [line for line in ass.splitlines() if line.startswith("Dialogue: 1")][0]
    assert "\\1a&HFF&" in shadow and "\\blur" in shadow and "\\4a&H" in shadow
