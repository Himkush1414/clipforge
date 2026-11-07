import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from clipforge import jobs, pipeline, review
from clipforge.analysis.scoring import Moment
from clipforge.analysis.transcribe import Word
from clipforge.api.app import create_app
from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.media.ffmpeg import find_tool
from clipforge.media.probe import MediaInfo
from clipforge.models import Clip, Job, JobStatus
from clipforge.pipeline import run_job


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def talk_video(tmp_path_factory):
    """30 s, 30 fps, long enough for the transcript below."""
    out = tmp_path_factory.mktemp("media") / "talk.mp4"
    subprocess.run(
        [
            find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=30",
            "-f", "lavfi", "-i", "sine=f=440:d=30:r=48000",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


TRANSCRIPT = (
    "Why do cats land on their feet? A fast reflex turns the head first. That is why cats walk away. "
    "Nobody expected the oven to fail. The baker left the bread in far too long. That mistake made him famous. "
    "Here is the strangest thing about honey. It never goes bad at all. That is why jars last for years."
)


def transcript_words() -> list[dict]:
    words, t = [], 0.5
    for token in TRANSCRIPT.split():
        words.append({"text": token, "start": round(t, 3), "end": round(t + 0.3, 3), "probability": 0.9})
        t += 0.33
        if token.endswith((".", "?")):
            t += 0.4
    return words


@pytest.fixture
def fake_transcript(monkeypatch):
    def fake(wav, duration, cache=None, **kw):
        result = {"language": "en", "words": transcript_words()}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(pipeline, "transcribe", fake)


def add_source(client, video):
    with open(video, "rb") as handle:
        res = client.put("/api/sources/upload", params={"filename": "My Video.mp4"}, content=handle.read())
    assert res.status_code == 200, res.text
    return res.json()


def run_queued():
    job_id = jobs.claim_next_job()
    assert job_id
    jobs.run_one(job_id, run_job)
    return job_id


def start_job(client, video, mode="manual", count=2, duration=6, **over):
    body = {
        "source_id": add_source(client, video)["id"],
        "mode": mode,
        "rights": {"type": "own"},
        "review": True,
        "edit": {"fit": "crop", "captions": False},
        **over,
    }
    if mode == "manual":
        body["manual"] = {
            "genre": "podcast",
            "clip_duration": duration,
            "clip_count": count,
            "part_numbering": False,
            "format": "short",
            "ratio": "9:16",
            "resolution": "1080p",
        }
    res = client.post("/api/jobs", json=body)
    assert res.status_code == 201, res.text
    return run_queued()


def test_a_reviewed_job_stops_before_rendering_anything(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "review"
    assert job["clips"] == []
    plan = client.get(f"/api/jobs/{job_id}/review").json()
    assert len(plan["clips"]) == 2
    assert all(c["why"] and c["text"] for c in plan["clips"])
    assert plan["source_url"]


def test_auto_mode_stops_for_review_too(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video, mode="auto")
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "review"
    assert client.get(f"/api/jobs/{job_id}/review").json()["clips"]


def test_a_job_without_review_goes_straight_through(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video, review=False)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["clips"]


def test_approved_clips_render_exactly_the_adjusted_frames(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    plan = client.get(f"/api/jobs/{job_id}/review").json()
    first = plan["clips"][0]
    res = client.patch(
        f"/api/jobs/{job_id}/review/{first['key']}",
        json={"start_frame": first["start_frame"] + 1, "end_frame": first["start_frame"] + 121},
    )
    assert res.status_code == 200, res.text
    assert client.post(f"/api/jobs/{job_id}/review/approve").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    assert run_queued() == job_id
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    assert len(job["clips"]) == 2
    frames = sorted(c["segments"][0] for c in job["clips"])
    assert [first["start_frame"] + 1, first["start_frame"] + 121] in frames
    assert any(c["frames"] == 120 for c in job["clips"])
    assert all(c["reasons"]["why"] for c in job["clips"])


def test_the_rights_record_stays_through_review(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video, rights={"type": "public_domain"})
    client.post(f"/api/jobs/{job_id}/review/approve")
    run_queued()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["rights_type"] == "public_domain"
    assert all(c["rights_type"] == "public_domain" for c in job["clips"])


def test_export_still_refuses_a_job_with_no_rights_record(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    with session_scope() as s:
        row = s.get(Job, job_id)
        row.rights_type = ""
        s.add(row)
        s.commit()
    client.post(f"/api/jobs/{job_id}/review/approve")
    run_queued()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "rights" in job["error"]
    assert job["clips"] == []


def test_a_rejected_clip_is_not_exported(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    key = client.get(f"/api/jobs/{job_id}/review").json()["clips"][0]["key"]
    res = client.post(f"/api/jobs/{job_id}/review/{key}/reject", json={"rejected": True})
    assert res.json()["clips"][0]["rejected"] is True
    client.post(f"/api/jobs/{job_id}/review/approve")
    run_queued()
    assert len(client.get(f"/api/jobs/{job_id}").json()["clips"]) == 1


def test_a_rejected_clip_can_be_restored(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    key = client.get(f"/api/jobs/{job_id}/review").json()["clips"][0]["key"]
    client.post(f"/api/jobs/{job_id}/review/{key}/reject", json={"rejected": True})
    res = client.post(f"/api/jobs/{job_id}/review/{key}/reject", json={"rejected": False})
    assert res.json()["clips"][0]["rejected"] is False


def test_rejecting_everything_cannot_be_exported(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    for clip in client.get(f"/api/jobs/{job_id}/review").json()["clips"]:
        client.post(f"/api/jobs/{job_id}/review/{clip['key']}/reject", json={"rejected": True})
    res = client.post(f"/api/jobs/{job_id}/review/approve")
    assert res.status_code == 422
    assert "Keep at least one" in res.json()["detail"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "review"


def test_swap_brings_in_a_different_clip_that_overlaps_nothing(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video, count=1)
    plan = client.get(f"/api/jobs/{job_id}/review").json()
    before = plan["clips"][0]
    res = client.post(f"/api/jobs/{job_id}/review/{before['key']}/swap")
    assert res.status_code == 200, res.text
    after = res.json()["clips"][0]
    assert after["key"] == before["key"]
    assert (after["start_frame"], after["end_frame"]) != (before["start_frame"], before["end_frame"])


def test_swapping_twice_can_bring_the_first_clip_back(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video, count=1)
    plan = client.get(f"/api/jobs/{job_id}/review").json()
    original = plan["clips"][0]["start_frame"]
    seen = []
    for _ in range(plan["alternates_left"] + 1):
        res = client.post(f"/api/jobs/{job_id}/review/{plan['clips'][0]['key']}/swap")
        assert res.status_code == 200
        seen.append(res.json()["clips"][0]["start_frame"])
    assert original in seen


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (5, 10**7, "past the last frame"),
        (50, 50, "after the start"),
        (60, 50, "after the start"),
        (-1, 50, "first frame"),
    ],
)
def test_impossible_frames_are_refused_and_change_nothing(
    client, talk_video, fake_transcript, start, end, message
):
    job_id = start_job(client, talk_video)
    clip = client.get(f"/api/jobs/{job_id}/review").json()["clips"][0]
    res = client.patch(
        f"/api/jobs/{job_id}/review/{clip['key']}", json={"start_frame": start, "end_frame": end}
    )
    assert res.status_code == 422
    assert message in res.json()["detail"]
    after = client.get(f"/api/jobs/{job_id}/review").json()["clips"][0]
    assert after["start_frame"] == clip["start_frame"]


def test_a_clip_outside_the_review_is_not_found(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    assert client.post(f"/api/jobs/{job_id}/review/nope/reject", json={}).status_code == 404
    assert client.post(f"/api/jobs/{job_id}/review/nope/swap").status_code == 409


def test_review_actions_refuse_a_job_not_in_review(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video, review=False)
    assert client.get(f"/api/jobs/{job_id}/review").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/review/approve").status_code == 409


def test_a_review_can_be_cancelled(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert client.get(f"/api/jobs/{job_id}/review").status_code == 409


def test_a_video_under_review_cannot_be_deleted(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    source_id = client.get(f"/api/jobs/{job_id}").json()["source_id"]
    assert client.delete(f"/api/sources/{source_id}").status_code == 409


def test_a_review_survives_a_restart(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    assert jobs.requeue_interrupted_jobs() == 0
    assert jobs.claim_next_job() is None
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == JobStatus.review.value


def test_the_source_can_be_previewed_with_seeking(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    url = client.get(f"/api/jobs/{job_id}/review").json()["source_url"]
    res = client.get(url, headers={"range": "bytes=0-99"})
    assert res.status_code == 206
    assert len(res.content) == 100


def _plan() -> dict:
    return {
        "fps": 10.0,
        "frame_count": 1000,
        "options": {},
        "decisions": {},
        "clips": [
            {"key": "a", "start_frame": 0, "end_frame": 100, "rejected": False, "warnings": [], "text": ""},
            {"key": "b", "start_frame": 200, "end_frame": 300, "rejected": False, "warnings": [], "text": ""},
        ],
        "alternates": [
            {"key": "x", "start_frame": 250, "end_frame": 350, "rejected": False, "warnings": [], "text": ""},
            {"key": "y", "start_frame": 500, "end_frame": 600, "rejected": False, "warnings": [], "text": ""},
        ],
    }


def test_swap_skips_a_candidate_that_overlaps_a_kept_clip():
    plan = _plan()
    swapped = review.swap(plan, "a")
    assert swapped["start_frame"] == 500
    assert swapped["key"] == "a"


def test_swap_says_so_when_nothing_fits():
    plan = _plan()
    plan["alternates"] = plan["alternates"][:1]
    with pytest.raises(review.ReviewError, match="no other candidate"):
        review.swap(plan, "a")


def test_a_rejected_clip_does_not_block_a_swap():
    plan = _plan()
    plan["clips"][1]["rejected"] = True
    assert review.swap(plan, "a")["start_frame"] == 250


def test_editing_warns_when_a_cut_goes_through_a_word():
    plan = _plan()
    words = [Word("hello", 1.0, 2.0), Word("there", 2.5, 3.0)]
    clip = review.adjust(plan, "a", 15, 40, words)
    assert clip["warnings"] == ['The start cuts through the word "hello".']
    assert clip["text"] == "there"


def test_editing_warns_about_overlapping_another_clip():
    plan = _plan()
    assert "overlaps another" in review.adjust(plan, "a", 0, 250, [])["warnings"][0]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, 10, "before the first"),
        (0, 1001, "past the last"),
        (50, 50, "after the start"),
        (60, 50, "after the start"),
    ],
)
def test_impossible_edits_are_refused(start, end, message):
    with pytest.raises(review.ReviewError, match=message):
        review.adjust(_plan(), "a", start, end, [])


def test_approved_clips_come_in_story_order_without_the_rejected():
    plan = _plan()
    plan["clips"].reverse()
    plan["clips"].append({"key": "c", "start_frame": 700, "end_frame": 800, "rejected": True})
    assert [c["key"] for c in review.approved_clips(plan)] == ["a", "b"]


def test_a_plan_offers_alternates_that_overlap_nothing_and_each_other():
    info = MediaInfo.__new__(MediaInfo)
    object.__setattr__(info, "fps", 10.0)
    object.__setattr__(info, "frame_count", 1000)
    chosen = [Moment(0.0, 10.0, 1.0, {}, "a", "why a")]
    ranked = [
        Moment(5.0, 15.0, 0.9, {}, "overlaps the chosen clip", "w"),
        Moment(20.0, 30.0, 0.8, {}, "fine", "w"),
        Moment(25.0, 35.0, 0.7, {}, "overlaps the alternate", "w"),
        Moment(40.0, 50.0, 0.6, {}, "also fine", "w"),
    ]
    plan = review.build_plan(chosen, ranked, info, {}, {})
    assert [a["text"] for a in plan["alternates"]] == ["fine", "also fine"]


def test_no_clip_row_exists_until_approval(client, talk_video, fake_transcript):
    job_id = start_job(client, talk_video)
    with session_scope() as s:
        assert s.get(Job, job_id) is not None
        from sqlmodel import select

        assert s.exec(select(Clip).where(Clip.job_id == job_id)).all() == []
    assert get_settings().unassigned_dir.exists() is False or not list(
        get_settings().unassigned_dir.glob("*.mp4")
    )
