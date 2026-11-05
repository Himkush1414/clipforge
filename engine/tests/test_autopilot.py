import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from clipforge import autopilot, jobs, pipeline
from clipforge.api.app import create_app
from clipforge.autopilot import (
    Signals,
    animation_score,
    choose_count,
    choose_duration,
    choose_format,
    choose_genre,
    choose_ratio,
    choose_resolution,
    decide,
)
from clipforge.captions.presets import BUILTIN_PRESETS
from clipforge.media.ffmpeg import find_tool
from clipforge.media.probe import MediaInfo


def signals(**over):
    base = dict(
        duration=1800,
        speech_share=0.5,
        words_per_minute=120,
        cuts_per_minute=5,
        motion=3,
        animation=0.2,
        sentence_seconds=4,
    )
    return Signals(**{**base, **over})


def info(width=1920, height=1080, duration=1800.0):
    return MediaInfo(
        duration, width, height, 30.0, 30, 1, "h264", True, "aac", 48000, int(duration * 30), 1000, 1000
    )


@pytest.mark.parametrize(
    "over,expected",
    [
        (dict(speech_share=0.8, cuts_per_minute=0.5), "podcast"),
        (dict(animation=0.8), "anime"),
        (dict(speech_share=0.1, motion=20), "gameplay"),
        (dict(cuts_per_minute=20, duration=7200), "movie"),
        (dict(cuts_per_minute=20, duration=2400), "series"),
        (dict(speech_share=0.4, cuts_per_minute=4), "other"),
    ],
)
def test_genre_rules(over, expected):
    genre, why = choose_genre(signals(**over))
    assert genre == expected
    assert why


def test_long_format_only_for_long_speech_led_video():
    assert choose_format(signals(duration=3600, speech_share=0.7), "podcast")[0] == "long"
    assert choose_format(signals(duration=1200, speech_share=0.7), "podcast")[0] == "short"
    assert choose_format(signals(duration=3600, speech_share=0.1), "gameplay")[0] == "short"


@pytest.mark.parametrize("seconds", [2400, 3600, 5400, 7200, 14400])
def test_long_parts_are_10_to_15_minutes(seconds):
    length, _ = choose_duration(signals(duration=seconds), "podcast", "long")
    assert 600 <= length <= 900
    count, _ = choose_count(signals(duration=seconds), length, "long")
    assert count * length <= seconds


def test_short_lengths_follow_pacing():
    assert choose_duration(signals(), "podcast", "short")[0] == 50
    assert choose_duration(signals(cuts_per_minute=30), "movie", "short")[0] == 25
    assert choose_duration(signals(duration=10), "podcast", "short")[0] == 10


def test_count_scales_with_length_and_always_fits():
    assert choose_count(signals(duration=60), 25, "short")[0] == 1
    assert choose_count(signals(duration=1800), 35, "short")[0] == 10
    count, _ = choose_count(signals(duration=36000), 35, "short")
    assert count == 12


def test_ratio_respects_vertical_source():
    assert choose_ratio(info(1080, 1920), "long")[0] == "9:16"
    assert choose_ratio(info(), "long")[0] == "16:9"
    assert choose_ratio(info(), "short")[0] == "9:16"


def test_resolution_matches_source_and_never_picks_8k():
    assert choose_resolution(info(1280, 720))[0] == "1080p"
    assert "upscaled" in choose_resolution(info(1280, 720))[1]
    assert choose_resolution(info(1920, 1080))[0] == "1080p"
    assert choose_resolution(info(7680, 4320))[0] in ("2K", "4K")


def _video(tmp_path, source, name):
    out = tmp_path / name
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
            source,
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    return out


def test_animation_score_separates_flat_art_from_camera_noise(tmp_path):
    flat = _video(tmp_path, "testsrc=s=640x360:r=10", "flat.mp4")
    noisy = _video(tmp_path, "color=c=gray:s=640x360:r=10,noise=alls=40:allf=t", "noise.mp4")
    assert animation_score(flat, 3) > 0.62 > animation_score(noisy, 3)


def test_blank_video_scores_zero(tmp_path):
    blank = _video(tmp_path, "color=c=black:s=320x180:r=10", "blank.mp4")
    assert animation_score(blank, 3) == 0.0


def test_channel_defaults_override_format_and_ratio():
    audio = type("A", (), {"hop": 0.05})()
    scenes = type("S", (), {"cut_frames": [], "motion": np.zeros(10)})()
    result = decide(
        [], audio, scenes, info(), None, {"format": "long", "ratio": "1:1", "preset_id": "neon-glow"}
    )
    assert (result.format, result.ratio, result.preset_id) == ("long", "1:1", "neon-glow")
    assert "channel" in result.reasons["format"]


def test_every_decision_has_a_reason():
    audio = type("A", (), {"hop": 0.05})()
    scenes = type("S", (), {"cut_frames": [], "motion": np.zeros(10)})()
    report = decide([], audio, scenes, info(), None).report()
    assert set(report["reasons"]) == {
        "genre",
        "format",
        "ratio",
        "resolution",
        "clip_duration",
        "clip_count",
        "preset",
    }
    assert all(report["reasons"].values())
    assert all(
        autopilot.GENRE_PRESETS[g] for g in ("anime", "gameplay", "podcast", "movie", "series", "other")
    )


def test_auto_job_end_to_end_still_requires_rights(scene_video, monkeypatch):
    words = [
        {"text": t, "start": i * 0.4, "end": i * 0.4 + 0.35, "probability": 1.0}
        for i, t in enumerate("Nobody expected this. What a wild little test video it was!".split())
    ]
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: {"language": "en", "words": words})
    rendered_presets = []
    real_render = pipeline.render_clip

    def spying_render(spec, **kwargs):
        rendered_presets.append(spec.preset.name)
        return real_render(spec, **kwargs)

    monkeypatch.setattr(pipeline, "render_clip", spying_render)
    with TestClient(create_app()) as client:
        with open(scene_video, "rb") as handle:
            source = client.put(
                "/api/sources/upload", params={"filename": "a.mp4"}, content=handle.read()
            ).json()
        assert client.post("/api/jobs", json={"source_id": source["id"], "mode": "auto"}).status_code == 422
        res = client.post(
            "/api/jobs",
            json={
                "source_id": source["id"],
                "mode": "auto",
                "rights": {"type": "own"},
                "edit": {"fit": "crop"},
            },
        )
        assert res.status_code == 201, res.text
        job_id = jobs.claim_next_job()
        jobs.run_one(job_id, pipeline.run_job)
        job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    assert job["decisions"]["reasons"]["genre"]
    assert job["clips"]
    clip = job["clips"][0]
    assert clip["ratio"] == job["decisions"]["ratio"]
    assert clip["preset_id"] == job["decisions"]["preset"]
    # What was really rendered, not only what was recorded: this once differed.
    assert rendered_presets == [BUILTIN_PRESETS[job["decisions"]["preset"]].name] * len(job["clips"])
