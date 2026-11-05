import subprocess

import numpy as np
import pytest

from clipforge.analysis.audio import analyze_audio, find_silences
from clipforge.analysis.boundaries import refine, remove_pauses
from clipforge.analysis.scenes import analyze_scenes, detect_cuts
from clipforge.analysis.scoring import (
    FeatureCurves,
    Moment,
    Sentence,
    find_moments,
    generate_candidates,
    robust_norm,
    select_moments,
    split_sentences,
    topic_change_curve,
)
from clipforge.analysis.transcribe import Word, extract_audio, sanitize_words
from clipforge.media.ffmpeg import find_tool


def words_from(text, start=0.0, step=0.4):
    out = []
    t = start
    for token in text.split():
        out.append(Word(token, t, t + step - 0.05))
        t += step
    return out


def test_sanitize_fixes_overlap_zero_length_and_blank():
    fixed = sanitize_words([Word(" a", 1.0, 1.0), Word("b", 0.9, 1.1), Word("  ", 2, 3), Word("c", 1.2, 1.3)])
    assert [w.text for w in fixed] == ["a", "b", "c"]
    for prev, cur in zip(fixed, fixed[1:], strict=False):
        assert cur.start >= prev.end
    assert all(w.end > w.start for w in fixed)


def test_scene_cuts_land_on_the_exact_frames(scene_video):
    result = analyze_scenes(scene_video, fps=30, duration=6)
    assert result.frame_count == 180
    assert result.cut_frames == [60, 120]
    assert result.cut_times == [2.0, 4.0]


def test_gradual_motion_is_not_a_cut():
    diffs = np.full(120, 12.0, dtype=np.float32) + np.sin(np.arange(120)) * 2
    assert detect_cuts(diffs, fps=30) == []


def test_silence_and_energy_from_real_audio(scene_video, tmp_path):
    wav = extract_audio(scene_video, tmp_path / "a.wav")
    audio = analyze_audio(wav)
    assert len(audio.silences) == 1
    start, end = audio.silences[0]
    assert abs(start - 2.0) < 0.12 and abs(end - 3.0) < 0.12
    assert audio.rms_at(0.5, 1.5) > audio.rms_at(2.2, 2.8) + 30


def test_find_silences_handles_trailing_silence():
    rms = np.array([-20.0] * 10 + [-70.0] * 10)
    assert find_silences(rms, 0.05, -40.0) == [(0.5, 1.0)]


def test_sentences_split_on_punctuation_and_gaps():
    words = words_from("Hello there friend. This is great") + [Word("again", 10.0, 10.3)]
    sentences = split_sentences(words)
    assert [s.text for s in sentences] == ["Hello there friend.", "This is great", "again"]


def test_endless_run_on_is_capped():
    sentences = split_sentences(words_from("word " * 200))
    assert max(s.end - s.start for s in sentences) <= 20.5


def test_topic_curve_peaks_at_real_topic_change():
    cats = [Sentence(i * 3, i * 3 + 2, words_from("cats whiskers purring kittens")) for i in range(6)]
    cars = [Sentence(18 + i * 3, 20 + i * 3, words_from("engines pistons turbo gearbox")) for i in range(6)]
    curve = topic_change_curve(cats + cars, 40)
    assert int(np.argmax(curve)) == 18


def test_robust_norm_constant_is_zero():
    assert robust_norm(np.ones(5)).sum() == 0


def _curves(seconds, hot=(0, 0)):
    speech = np.full(seconds, 0.2, dtype=np.float32)
    speech[hot[0] : hot[1]] = 0.9
    zero = np.zeros(seconds, dtype=np.float32)
    return FeatureCurves(speech, zero.copy(), zero.copy(), zero.copy(), np.ones(seconds, dtype=np.float32))


def test_candidates_start_and_end_only_on_sentence_edges():
    sentences = [Sentence(i * 5.0, i * 5.0 + 4.5, words_from("a b c", i * 5.0)) for i in range(20)]
    weights = {"speech": 1.0, "emotion": 0, "topic": 0, "action": 0}
    moments = generate_candidates(sentences, _curves(100), weights, target=20)
    starts, ends = {s.start for s in sentences}, {s.end for s in sentences}
    assert moments
    for moment in moments:
        assert moment.start in starts and moment.end in ends
        assert 14 <= moment.end - moment.start <= 26


def test_hot_section_wins_selection():
    sentences = [Sentence(i * 5.0, i * 5.0 + 4.5, []) for i in range(40)]
    weights = {"speech": 1.0, "emotion": 0, "topic": 0, "action": 0}
    moments = generate_candidates(sentences, _curves(200, hot=(100, 125)), weights, target=20)
    best = select_moments(moments, 1)[0]
    assert 95 <= best.start <= 110


def test_selection_never_overlaps_and_is_chronological():
    moments = [
        Moment(s, s + 20, score) for s, score in [(50, 0.9), (55, 0.8), (0, 0.7), (100, 0.6), (21, 0.5)]
    ]
    chosen = select_moments(moments, 4)
    assert [m.start for m in chosen] == [0, 21, 50, 100]
    assert len(select_moments(moments, 2)) == 2


class _Audio:
    hop = 0.05
    silences = []
    peaks = []


def _fake_analysis(duration):
    audio = _Audio()
    audio.rms_db = np.full(int(duration / 0.05), -20.0, dtype=np.float32)
    audio.flux = np.zeros(int(duration / 0.05), dtype=np.float32)
    scenes = type("S", (), {})()
    scenes.fps, scenes.motion, scenes.cut_times = 30, np.zeros(int(duration * 30), dtype=np.float32), []
    return audio, scenes


def test_full_finder_never_cuts_mid_sentence():
    words = []
    t = 0.0
    for i in range(30):
        words += words_from(f"Sentence number {i} has several plain words inside it.", t)
        t = words[-1].end + 0.3
    audio, scenes = _fake_analysis(t)
    moments = find_moments(words, audio, scenes, t, "podcast", 15, 3)
    sentences = split_sentences(words)
    starts, ends = {s.start for s in sentences}, {s.end for s in sentences}
    assert len(moments) == 3
    for m in moments:
        assert m.start in starts and m.end in ends


def test_no_speech_video_still_gets_moments():
    audio, scenes = _fake_analysis(120)
    scenes.cut_times = [float(x) for x in range(5, 120, 5)]
    moments = find_moments([], audio, scenes, 120, "gameplay", 20, 3)
    assert len(moments) == 3
    assert all(m.end <= 120 for m in moments)


def test_refine_never_touches_neighbouring_words():
    words = [
        Word("before", 9.0, 9.95),
        Word("first", 10.0, 10.5),
        Word("last", 19.5, 20.0),
        Word("next", 20.05, 21),
    ]
    start, end = refine(10.0, 20.0, words, [], fps=30, duration=60)
    assert 9.95 - 1 / 30 <= start <= 10.0
    assert 20.0 <= end <= 20.05 + 1 / 30
    assert round(start * 30, 6).is_integer() and round(end * 30, 6).is_integer()


def test_refine_snaps_to_a_nearby_scene_cut():
    words = [Word("first", 10.0, 10.5), Word("last", 19.5, 20.0)]
    start, end = refine(10.0, 20.0, words, [9.9, 20.2], fps=30, duration=60)
    assert start == pytest.approx(9.9, abs=1 / 30)
    assert end == pytest.approx(20.2, abs=1 / 30)


def test_refine_stays_inside_video():
    start, end = refine(0.05, 5.95, [Word("a", 0.05, 5.95)], [], fps=30, duration=6)
    assert start == 0 and end <= 6


def test_remove_pauses_keeps_every_word():
    words = [Word("a", 1, 2), Word("b", 2.1, 3), Word("c", 6, 7), Word("d", 7.1, 8)]
    segments = remove_pauses(1, 8, words)
    assert segments == [(1, 3.2), (5.8, 8)]
    for word in words:
        assert any(a <= word.start and word.end <= b for a, b in segments)


def test_static_silent_video_still_gets_clips_of_about_the_right_length():
    audio, scenes = _fake_analysis(120)
    audio.silences = [(0.0, 120.0)]
    moments = find_moments([], audio, scenes, 120, "other", 20, 3)
    assert len(moments) == 3
    assert all(14 <= m.end - m.start <= 26 for m in moments)


def test_very_long_sentences_fall_back_to_whole_sentences():
    """Target 5 s, every sentence 15 s: still whole sentences, never a mid-sentence cut."""
    words = []
    t = 0.0
    for _ in range(6):
        words += words_from("word " * 37 + "end.", t)
        t = words[-1].end + 1.0
    audio, scenes = _fake_analysis(t)
    moments = find_moments(words, audio, scenes, t, "other", 5, 2)
    sentences = split_sentences(words)
    assert moments
    assert all(
        m.start in {s.start for s in sentences} and m.end in {s.end for s in sentences} for m in moments
    )


def test_flash_is_not_a_cut_but_a_very_short_shot_is(tmp_path):
    video = tmp_path / "flash.mp4"
    graph = (
        "color=c=0x404040:s=320x180:r=30:d=4,"
        "drawbox=c=white:t=fill:enable='between(n,30,31)',"  # A two-frame flash.
        "drawbox=c=0xc02020:t=fill:enable='gte(n,60)',"  # A real cut at frame 60.
        "drawbox=c=0x2020c0:t=fill:enable='gte(n,90)'"  # And a real cut at frame 90.
    )
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            graph,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    assert analyze_scenes(video, fps=30, duration=4).cut_frames == [60, 90]


def test_eight_frame_shot_keeps_both_cuts(tmp_path):
    video = tmp_path / "short.mp4"
    graph = "color=c=0x404040:s=320x180:r=30:d=3,drawbox=c=0xc02020:t=fill:enable='between(n,30,37)',drawbox=c=0x20c020:t=fill:enable='gte(n,38)'"
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            graph,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    assert analyze_scenes(video, fps=30, duration=3).cut_frames == [30, 38]
