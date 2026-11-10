"""Generated titles, descriptions and hashtags never carry blocked words from the dialogue."""

import pytest

from clipforge.metadata import build_metadata, make_hashtags, make_summary, make_title
from clipforge.wordfilter import contains_blocked, is_blocked

DIALOGUE = (
    "Am I really the only survivor? My leg! Now get to work, woman. Bear your breasts. "
    "The dragon is waking. The dragon will not wait for us."
)


@pytest.mark.parametrize(
    "word",
    ["breasts", "Breasts.", "BREAST", "fucking", "Shit!", "killed", "rapist", "चूतिया", "madarchod", "gaand"],
)
def test_blocked_words_are_caught_whatever_their_case_or_punctuation(word):
    assert is_blocked(word.strip(".!"))
    assert contains_blocked(f"well {word} then")


@pytest.mark.parametrize("word", ["assume", "class", "skill", "breakfast", "therapist", "dragon", "survivor"])
def test_ordinary_words_are_not_blocked(word):
    assert not is_blocked(word)


def test_breasts_never_reach_the_title_description_or_hashtags():
    title, description = build_metadata(DIALOGUE, DIALOGUE, "anime", 2, "short", "private_test", "")
    assert "breast" not in title.lower() and "breast" not in description.lower()
    assert not contains_blocked(title) and not contains_blocked(description)
    assert title.endswith("| Part 2")
    tags = [w for w in description.split() if w.startswith("#")]
    assert tags[:3] == ["#shorts", "#anime", "#part2"]


def test_hashtags_are_format_genre_part_and_recurring_topics_only():
    tags = make_hashtags(DIALOGUE, DIALOGUE, "anime", 1, "short")
    assert tags == ["#shorts", "#anime", "#part1", "#dragon"]  # "survivor" is said once, so not a topic
    assert make_hashtags("Just one line here.", "Just one line here.", "podcast", None, "long") == [
        "#podcast"
    ]


def test_a_title_skips_lines_with_blocked_words():
    text = "What the fuck was that? The bridge is gone."
    assert make_title(text, "movie") == "The bridge is gone"
    assert make_title("Shit! Damn it!", "anime") == "Anime highlight"


def test_hindi_lines_with_blocked_words_are_dropped():
    text = "तू चूतिया है क्या? हमें आज रात नदी पार करनी है।"
    assert "चूतिया" not in make_summary(text)
    assert "चूतिया" not in make_title(text, "series")
