import importlib.util
import json
import random
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench"
spec = importlib.util.spec_from_file_location("pick_eval", BENCH / "pick_eval.py")
assert spec and spec.loader
pick_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick_eval)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12.5, 12.5), (90, 90.0), ("754.5", 754.5), ("12:34.5", 754.5), ("1:02:03", 3723.0)],
)
def test_times_can_be_seconds_minutes_or_hours(value, expected):
    assert pick_eval.seconds(value) == expected


def test_perfect_clips_score_one():
    picks = [(10.0, 20.0), (50.0, 60.0)]
    assert pick_eval.compare(picks, picks) == {"pick_recall": 1.0, "clip_precision": 1.0, "mean_iou": 1.0}


def test_clips_far_from_every_pick_score_zero():
    scores = pick_eval.compare([(100.0, 110.0)], [(10.0, 20.0)])
    assert scores == {"pick_recall": 0.0, "clip_precision": 0.0, "mean_iou": 0.0}


def test_recall_and_precision_count_separately():
    scores = pick_eval.compare([(10.0, 20.0), (100.0, 110.0)], [(10.0, 20.0), (50.0, 60.0), (70.0, 80.0)])
    assert scores["pick_recall"] == pytest.approx(1 / 3)
    assert scores["clip_precision"] == pytest.approx(1 / 2)


def test_a_clip_lands_on_a_pick_when_they_share_half_of_the_shorter():
    assert pick_eval.lands_on((0.0, 60.0), (20.0, 30.0))
    assert pick_eval.lands_on((0.0, 10.0), (5.0, 100.0))
    assert not pick_eval.lands_on((0.0, 10.0), (6.0, 100.0))


def test_no_clips_or_no_picks_score_zero_instead_of_failing():
    assert pick_eval.compare([], [(1.0, 2.0)])["pick_recall"] == 0.0
    assert pick_eval.compare([(1.0, 2.0)], [])["clip_precision"] == 0.0


def test_random_windows_do_not_overlap_and_stay_inside_the_video():
    windows = pick_eval.random_windows(100.0, 10.0, 5, random.Random(1))
    assert len(windows) == 5
    assert all(0 <= a and b <= 100.0 for a, b in windows)
    ordered = sorted(windows)
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:], strict=False))


def test_chance_of_covering_a_whole_video_is_certain():
    picks = [(0.0, 100.0)]
    assert pick_eval.chance(100.0, [(0.0, 10.0)], picks)["pick_recall"] == 1.0


@pytest.mark.parametrize("name", ["my_picks.template.json", "nasa_picks.json"])
def test_picks_files_are_valid_with_ordered_times(name):
    videos = json.loads((BENCH / name).read_text(encoding="utf-8"))["videos"]
    assert videos
    for video in videos:
        assert video["clip_seconds"] > 0
        for pick in video["picks"]:
            assert pick_eval.seconds(pick["start"]) < pick_eval.seconds(pick["end"])
