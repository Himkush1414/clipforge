import numpy as np
import pytest

from clipforge.analysis.editor import (
    ENGLISH,
    Verdict,
    explain,
    inverse_document_frequency,
    judge,
    judge_complete,
    judge_hook,
    judge_self_contained,
    lexicon_for,
    split_thoughts,
    trim_filler,
)
from clipforge.analysis.scoring import find_moments, rank_moments
from clipforge.analysis.transcribe import Word


def say(text: str, start: float = 0.0, step: float = 0.3) -> list[Word]:
    return [
        Word(token, round(start + i * step, 3), round(start + i * step + step - 0.02, 3))
        for i, token in enumerate(text.split())
    ]


def test_fragments_at_breaths_join_into_one_thought():
    words = say("And") + say("I grew really interested in", 0.9) + say("tiny life underground.", 2.9)
    thoughts = split_thoughts(words)
    assert len(thoughts) == 1
    assert thoughts[0].text == "And I grew really interested in tiny life underground."


def test_a_thought_ends_at_an_end_mark_or_a_long_pause():
    words = say("First one here.") + say("no end mark at all", 1.0) + say("then more", 4.0)
    assert [t.text for t in split_thoughts(words)] == ["First one here.", "no end mark at all", "then more"]


def test_a_long_unpunctuated_stretch_breaks_at_a_pause_only_when_asked():
    first = say("word " * 25)
    second = say("more " * 10, first[-1].end + 0.3)
    assert len(split_thoughts(first + second)) == 1
    assert [len(t.words) for t in split_thoughts(first + second, fine=True)] == [25, 10]


def test_a_short_breath_never_breaks_a_thought():
    first = say("word " * 5)
    second = say("more " * 5, first[-1].end + 0.3)
    assert len(split_thoughts(first + second, fine=True)) == 1


def test_an_endless_run_is_capped():
    thoughts = split_thoughts(say("word " * 300))
    assert len(thoughts) > 1
    assert all(t.end - t.start <= 36 for t in thoughts)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("And so I left.", "I left."),
        ("Um, you know, the rain stopped.", "the rain stopped."),
        ("I mean nobody knew.", "nobody knew."),
        ("We won the game and", "We won the game"),
        ("It rained because", "It rained"),
        ("That is exactly right.", "That is exactly right."),
    ],
)
def test_filler_is_trimmed_from_both_ends(text, expected):
    assert " ".join(w.text for w in trim_filler(say(text), ENGLISH)) == expected


def test_trimming_only_filler_leaves_nothing():
    assert trim_filler(say("um so and"), ENGLISH) == []


def test_languages_without_a_word_list_are_not_trimmed():
    words = say("And so I left")
    assert trim_filler(words, None) == words


def test_word_lists_exist_only_where_written():
    assert lexicon_for("en") is ENGLISH
    assert lexicon_for("es") is not None
    assert lexicon_for("hi") is None
    assert lexicon_for(None) is ENGLISH


def test_a_question_makes_a_stronger_hook_than_a_flat_statement():
    question, why = judge_hook(say("Why do cats always land on their feet?"), 0.5, 0.2, ENGLISH)
    flat, _ = judge_hook(say("The meeting took place in a room near the office."), 0.5, 0.2, ENGLISH)
    assert question > flat
    assert "opens with a question" in why


def test_hook_is_judged_without_a_word_list():
    score, why = judge_hook(say("यह एक वाक्य है"), 0.5, 0.5, None)
    assert 0 < score <= 1
    assert why == []


@pytest.mark.parametrize(
    "text", ["He said the same thing about it.", "That was the problem with it.", "They kept going."]
)
def test_opening_on_a_pronoun_is_not_self_contained(text):
    score, problems = judge_self_contained(say(text), False, ENGLISH)
    assert score <= 0.5
    assert "points at something said earlier" in problems[0]


def test_leaning_on_earlier_context_is_caught_anywhere_in_the_clip():
    score, problems = judge_self_contained(say("Rain fell, and as I said, nobody came."), False, ENGLISH)
    assert score < 1.0
    assert "earlier context" in problems[0]


def test_show_introductions_are_not_clips():
    score, problems = judge_self_contained(say("Welcome to the show on this podcast today."), False, ENGLISH)
    assert score < 0.5
    assert "introduction" in problems[0]


def test_a_clean_opening_is_self_contained():
    assert judge_self_contained(say("Rain fell across the valley all night."), False, ENGLISH) == (1.0, [])


def test_a_finished_sentence_is_complete():
    score, problems = judge_complete(say("The boat reached the shore."), 0.8, "next", ENGLISH)
    assert score >= 0.75
    assert problems == []


def test_stopping_mid_sentence_fails():
    score, problems = judge_complete(say("And that leads to the"), 0.1, "next", ENGLISH)
    assert score < 0.4
    assert "mid-sentence" in problems[0]


def test_ending_on_a_question_fails():
    _, problems = judge_complete(say("So what happened next?"), 0.8, "well", ENGLISH)
    assert "unanswered question" in problems[0]


def test_stopping_right_before_because_fails():
    _, problems = judge_complete(say("The boat sank."), 0.2, "because", ENGLISH)
    assert "continues the thought" in problems[0]


def test_a_payoff_counts_for_the_ending():
    plain, _ = judge_complete(say("The boat reached the shore."), 0.1, "next", ENGLISH)
    payoff, _ = judge_complete(say("That's why the boat reached the shore."), 0.1, "next", ENGLISH)
    assert payoff > plain


def _judged(text: str, language=ENGLISH) -> Verdict:
    words = say(text, 10.0)
    thoughts = split_thoughts(words)
    return judge(
        words, thoughts, 0, len(thoughts) - 1, inverse_document_frequency(thoughts), 0.5, 100.0, language
    )


def test_a_good_clip_passes_and_a_bad_one_does_not():
    assert _judged("Why do cats land on their feet? A reflex turns the head, then the spine.").passed
    assert not _judged("He said it was the same as the last one, as I said, and").passed


def test_the_reason_names_what_is_good_and_stays_short():
    reason = explain(_judged("Why do cats land on their feet? A reflex turns the head, then the spine."), {})
    assert reason.startswith("Opens with a question")
    assert reason.endswith(".")
    assert len(reason) < 140


def test_the_reason_is_honest_about_a_weak_clip():
    reason = explain(_judged("He said it was the same as the last one, as I said, and"), {})
    assert reason.startswith("Best available, though it")


def _analysis(duration: float):
    audio = type(
        "Audio",
        (),
        {
            "hop": 0.05,
            "rms_db": np.full(int(duration / 0.05), -30.0, dtype=np.float32),
            "flux": np.zeros(int(duration / 0.05), dtype=np.float32),
            "silences": [],
        },
    )()
    scenes = type(
        "Scenes", (), {"fps": 30.0, "motion": np.zeros(int(duration * 30), dtype=np.float32), "cut_times": []}
    )()
    return audio, scenes


GOOD = "Why do cats always land on their feet? A fast reflex turns the head first and the spine follows. That is why a cat can walk away from a long fall."
BAD = "So he said it was the same as the last one. They did that again, as I said. Those were the other ones and they were all the same as before, and"


def _talk(parts: list[str]) -> tuple[list[Word], float]:
    words: list[Word] = []
    t = 1.0
    for text in parts:
        passage = say(text, t)
        words += passage
        t = passage[-1].end + 1.5
    return words, t + 1


def test_selection_prefers_the_passages_that_work_as_clips():
    words, duration = _talk([BAD, GOOD, BAD, GOOD, BAD])
    audio, scenes = _analysis(duration)
    chosen = find_moments(words, audio, scenes, duration, "podcast", 9, 2)
    assert len(chosen) == 2
    assert all(m.text.startswith("Why do cats") for m in chosen)
    assert all(m.why for m in chosen)


def test_clips_never_start_or_end_inside_a_word():
    words, duration = _talk([BAD, GOOD, BAD, GOOD])
    audio, scenes = _analysis(duration)
    starts, ends = {w.start for w in words}, {w.end for w in words}
    for moment in rank_moments(words, audio, scenes, duration, "podcast", 9):
        assert moment.start in starts
        assert moment.end in ends


def test_passing_candidates_rank_ahead_of_failing_ones():
    words, duration = _talk([BAD, GOOD, BAD])
    audio, scenes = _analysis(duration)
    passed = [m.passed for m in rank_moments(words, audio, scenes, duration, "podcast", 9)]
    assert passed == sorted(passed, reverse=True)


def test_a_video_where_nothing_passes_still_yields_a_clip_with_an_honest_reason():
    words, duration = _talk([BAD, BAD, BAD])
    audio, scenes = _analysis(duration)
    chosen = find_moments(words, audio, scenes, duration, "podcast", 9, 2)
    assert chosen
    assert all(m.why.startswith("Best available") for m in chosen)


def test_the_old_scorer_can_still_run():
    words, duration = _talk([BAD, GOOD])
    audio, scenes = _analysis(duration)
    assert find_moments(words, audio, scenes, duration, "podcast", 9, 1, editor=False)


def test_a_language_without_a_word_list_still_chooses_clips():
    words, duration = _talk(["यह एक लंबा वाक्य है जो यहाँ तक चलता है और फिर रुकता है। " * 2] * 4)
    audio, scenes = _analysis(duration)
    assert find_moments(words, audio, scenes, duration, "podcast", 9, 2, "hi")


def test_speechless_video_gets_a_plain_reason():
    audio, scenes = _analysis(100.0)
    chosen = find_moments([], audio, scenes, 100.0, "gameplay", 20, 2)
    assert chosen
    assert all(m.why for m in chosen)
