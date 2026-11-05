import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from clipforge import jobs, pipeline
from clipforge.api.app import create_app, windows_to_wsl
from clipforge.db import session_scope
from clipforge.metadata import build_metadata, credit_line, make_title
from clipforge.models import Clip, Job, utcnow
from clipforge.schemas import JobCreate

SPOKEN = "This is the first line. Nobody expected what happened next! The rest was ordinary work anyway."


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def fake_transcript(monkeypatch):
    """Real render, real analysis. Only the Whisper call is replaced, since CI has no model."""
    words = []
    t = 0.2
    for token in SPOKEN.split():
        words.append({"text": token, "start": round(t, 3), "end": round(t + 0.3, 3), "probability": 1.0})
        t += 0.33

    def fake(wav, duration, cache=None, **kw):
        return {"language": "en", "words": words}

    monkeypatch.setattr(pipeline, "transcribe", fake)


def add_source(client, video):
    with open(video, "rb") as handle:
        res = client.put("/api/sources/upload", params={"filename": "My Video.mp4"}, content=handle.read())
    assert res.status_code == 200, res.text
    return res.json()


def manual(**over):
    return {
        "genre": "podcast",
        "clip_duration": 3,
        "clip_count": 1,
        "part_numbering": True,
        "format": "short",
        "ratio": "9:16",
        "resolution": "1080p",
        **over,
    }


def run_queued():
    job_id = jobs.claim_next_job()
    assert job_id
    jobs.run_one(job_id, pipeline.run_job)
    return job_id


def test_health_and_hardware(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "1080p" in client.get("/api/hardware").json()["allowed_resolutions"]


def test_foreign_origin_and_host_are_refused(client):
    assert client.get("/api/health", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 403
    assert client.get("/api/health", headers={"origin": "http://localhost:3000"}).status_code == 200


def test_upload_rejects_wrong_type_and_fake_video(client):
    assert client.put("/api/sources/upload", params={"filename": "a.exe"}, content=b"x").status_code == 415
    assert client.put("/api/sources/upload", params={"filename": "a.mp4"}, content=b"nope").status_code == 422
    from clipforge.config import get_settings

    assert list(get_settings().media_dir.iterdir()) == []


def test_upload_filename_cannot_escape_media_dir(client, scene_video):
    with open(scene_video, "rb") as handle:
        res = client.put("/api/sources/upload", params={"filename": "../../evil.mp4"}, content=handle.read())
    assert res.status_code == 200
    assert res.json()["filename"] == "evil.mp4"


def test_import_path_and_windows_conversion(client, scene_video):
    assert windows_to_wsl(r"D:\videos\a b.mp4").as_posix() == "/mnt/d/videos/a b.mp4"
    res = client.post("/api/sources/import", json={"path": str(scene_video)})
    assert res.status_code == 200 and res.json()["owned_copy"] is False
    assert client.post("/api/sources/import", json={"path": "/nope/x.mp4"}).status_code == 404
    assert client.post("/api/sources/import", json={"path": "/etc/passwd"}).status_code == 415


def test_deleting_an_imported_source_keeps_the_users_file(client, scene_video):
    source = client.post("/api/sources/import", json={"path": str(scene_video)}).json()
    assert client.delete(f"/api/sources/{source['id']}").status_code == 200
    assert scene_video.exists()
    assert client.get("/api/sources").json() == []


@pytest.mark.parametrize(
    "rights",
    [
        None,
        {},
        {"type": ""},
        {"type": "stolen"},
        {"type": "creative_commons"},
        {"type": "creative_commons", "detail": "   "},
        {"type": "written_permission", "detail": ""},
    ],
)
def test_job_cannot_start_without_valid_rights(client, scene_video, rights):
    source = add_source(client, scene_video)
    body = {"source_id": source["id"], "mode": "manual", "manual": manual()}
    if rights is not None:
        body["rights"] = rights
    assert client.post("/api/jobs", json=body).status_code == 422
    assert client.get("/api/jobs").json() == []


def test_pipeline_refuses_a_job_row_with_no_rights(scene_video, client):
    source = add_source(client, scene_video)
    with session_scope() as s:
        job = Job(source_id=source["id"], rights_type="", settings={"manual": manual(), "edit": {}})
        s.add(job)
        s.commit()
        job_id = job.id
    run_queued()
    assert "rights" in client.get(f"/api/jobs/{job_id}").json()["error"]


def test_manual_mode_requires_all_options():
    with pytest.raises(ValueError):
        JobCreate(source_id="x", mode="manual", rights={"type": "own"})


def test_manual_job_end_to_end(client, scene_video, fake_transcript):
    source = add_source(client, scene_video)
    res = client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "own"},
            "manual": manual(),
            "edit": {"preset_id": "anime-impact", "fit": "crop"},
        },
    )
    assert res.status_code == 201, res.text
    job_id = run_queued()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    assert job["rights_type"] == "own"
    assert len(job["clips"]) == 1
    clip = job["clips"][0]
    assert (clip["width"], clip["height"]) == (1080, 1920)
    assert clip["qc"]["checks"]
    assert 86000 < clip["seconds_left"] <= 86400
    assert "Nobody expected" in clip["title"] or "first line" in clip["title"]

    video = client.get(clip["file_url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    part = client.get(clip["file_url"], headers={"range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert client.get(clip["thumb_url"]).headers["content-type"] == "image/jpeg"

    assert client.patch(f"/api/clips/{clip['id']}", json={"title": "New"}).json()["title"] == "New"


def test_creative_commons_credit_reaches_description(client, scene_video, fake_transcript):
    source = add_source(client, scene_video)
    client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "creative_commons", "detail": "Film by A. Person, CC BY 4.0"},
            "manual": manual(),
            "edit": {"captions": False, "fit": "crop"},
        },
    )
    job = client.get(f"/api/jobs/{run_queued()}").json()
    assert job["status"] == "done", job["error"]
    assert "Film by A. Person, CC BY 4.0" in job["clips"][0]["description"]


def test_frame_cut_job_and_recut_keep_rights(client, scene_video, fake_transcript):
    source = add_source(client, scene_video)
    client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "own"},
            "manual": manual(ratio="16:9"),
            "edit": {"captions": False},
            "frame_cuts": [{"start_frame": 58, "end_frame": 61}],
        },
    )
    job = client.get(f"/api/jobs/{run_queued()}").json()
    assert job["status"] == "done", job["error"]
    assert job["clips"][0]["frames"] == 3

    res = client.post(f"/api/clips/{job['clips'][0]['id']}/recut", json={"start_frame": 10, "end_frame": 15})
    assert res.status_code == 201
    assert res.json()["rights_type"] == "own"
    second = client.get(f"/api/jobs/{run_queued()}").json()
    assert second["clips"][0]["frames"] == 5


def test_bad_frame_cut_rejected(client, scene_video):
    source = add_source(client, scene_video)
    res = client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "own"},
            "manual": manual(),
            "frame_cuts": [{"start_frame": 10, "end_frame": 5}],
        },
    )
    assert res.status_code == 422


def test_second_run_uses_cached_analysis(client, scene_video, monkeypatch, fake_transcript):
    source = add_source(client, scene_video)
    body = {
        "source_id": source["id"],
        "mode": "manual",
        "rights": {"type": "own"},
        "manual": manual(),
        "edit": {"captions": False, "fit": "crop"},
    }
    client.post("/api/jobs", json=body)
    run_queued()

    def must_not_run(*a, **k):
        raise AssertionError("analysis should be cached")

    monkeypatch.setattr(pipeline, "analyze_scenes", must_not_run)
    monkeypatch.setattr(pipeline, "analyze_audio", must_not_run)
    client.post("/api/jobs", json=body)
    job = client.get(f"/api/jobs/{run_queued()}").json()
    assert job["status"] == "done", job["error"]


def test_too_long_duration_gives_a_clear_failure_or_clamps(client, scene_video, fake_transcript):
    source = add_source(client, scene_video)
    client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "own"},
            "manual": manual(clip_duration=600),
            "edit": {"captions": False, "fit": "crop"},
        },
    )
    job = client.get(f"/api/jobs/{run_queued()}").json()
    assert job["status"] in ("done", "failed")
    if job["status"] == "failed":
        assert "clip length" in job["error"]


def test_cancel_queued_job(client, scene_video):
    source = add_source(client, scene_video)
    job = client.post(
        "/api/jobs",
        json={"source_id": source["id"], "mode": "manual", "rights": {"type": "own"}, "manual": manual()},
    ).json()
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert jobs.claim_next_job() is None
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 409


def test_source_in_use_cannot_be_deleted(client, scene_video):
    source = add_source(client, scene_video)
    client.post(
        "/api/jobs",
        json={"source_id": source["id"], "mode": "manual", "rights": {"type": "own"}, "manual": manual()},
    )
    assert client.delete(f"/api/sources/{source['id']}").status_code == 409


def test_expired_clip_disappears_and_its_url_says_so(client, scene_video, fake_transcript):
    source = add_source(client, scene_video)
    client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "own"},
            "manual": manual(),
            "edit": {"captions": False, "fit": "crop"},
        },
    )
    job = client.get(f"/api/jobs/{run_queued()}").json()
    clip_id = job["clips"][0]["id"]
    with session_scope() as s:
        row = s.get(Clip, clip_id)
        row.expires_at = utcnow() - timedelta(seconds=1)
        s.add(row)
        s.commit()
    assert client.post("/api/cleanup/run").json()["removed"] == [clip_id]
    assert client.get("/api/clips").json() == []
    assert client.get(f"/api/clips/{clip_id}/file").status_code == 410


def test_presets_crud_and_builtin_protection(client):
    presets = client.get("/api/presets").json()
    assert {"anime-impact", "karaoke-box"} <= {p["id"] for p in presets}
    assert client.delete("/api/presets/anime-impact").status_code == 409
    data = next(p for p in presets if p["id"] == "anime-impact")["data"]
    created = client.post("/api/presets", json={"name": "Mine", "data": data}).json()
    assert (
        client.put(f"/api/presets/{created['id']}", json={"name": "Mine 2", "data": data}).json()["name"]
        == "Mine 2"
    )
    assert client.put("/api/presets/anime-impact", json={"name": "x", "data": data}).status_code == 409
    assert (
        client.post("/api/presets", json={"name": "Bad", "data": {**data, "font": "Nope"}}).status_code == 422
    )
    assert client.delete(f"/api/presets/{created['id']}").status_code == 200


def test_preset_preview_is_a_real_image_and_cleans_up(client):
    from clipforge.config import get_settings

    data = client.get("/api/presets").json()[0]["data"]
    res = client.post("/api/presets/preview", json={"preset": data, "ratio": "9:16", "at": 0.5})
    assert res.status_code == 200 and res.content[:4] == b"\x89PNG"
    assert list(get_settings().tmp_dir.iterdir()) == []
    clip = client.post(
        "/api/presets/preview", json={"preset": data, "ratio": "1:1", "at": None, "text": "hi there"}
    )
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4"


def test_fonts_only_serve_bundled_files(client):
    fonts = client.get("/api/fonts").json()
    assert client.get(f"/api/fonts/{fonts[0]['file']}").status_code == 200
    assert client.get("/api/fonts/..%2F..%2Fconfig.py").status_code == 404


def test_channels_settings_and_linking(client):
    a = client.post("/api/channels", json={"name": "Main", "platform": "youtube"}).json()
    b = client.post("/api/channels", json={"name": "Reels", "platform": "instagram"}).json()
    assert client.post("/api/channels", json={"name": "x", "platform": "myspace"}).status_code == 422
    updated = client.put(
        f"/api/channels/{a['id']}/settings", json={"ratio": "1:1", "title_template": "{title}!"}
    )
    assert updated.json()["settings"]["ratio"] == "1:1"
    linked = client.post("/api/channels/link", json={"channel_ids": [a["id"], b["id"]]}).json()
    assert linked[0]["link_group"] == linked[1]["link_group"] is not None


def test_channel_templates_are_used_and_keep_the_credit(client, scene_video, fake_transcript):
    channel = client.post(
        "/api/channels",
        json={
            "name": "Main",
            "platform": "youtube",
            "settings": {"title_template": "[CF] {title}", "description_template": "Only this."},
        },
    ).json()
    source = add_source(client, scene_video)
    client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "channel_id": channel["id"],
            "rights": {"type": "creative_commons", "detail": "By Someone, CC BY"},
            "manual": manual(),
            "edit": {"captions": False, "fit": "crop"},
        },
    )
    clip = client.get(f"/api/jobs/{run_queued()}").json()["clips"][0]
    assert clip["title"].startswith("[CF] ")
    assert clip["description"].startswith("Only this.")
    assert "By Someone, CC BY" in clip["description"]


def test_title_logic():
    assert make_title("", "anime") == "Anime highlight"
    assert make_title("Ok. Why did the whole village suddenly vanish? Then we left.", "other").startswith(
        "Why did"
    )
    assert len(make_title("word " * 100, "other")) <= 72


def test_long_title_prefers_a_whole_clause():
    text = (
        "That is incredible when you think about it, you are looking straight into the past of the universe."
    )
    assert make_title(text, "other") == "That is incredible when you think about it"


def test_sentences_without_punctuation_do_not_run_together():
    from clipforge.analysis.scoring import Sentence
    from clipforge.analysis.transcribe import Word

    assert (
        Sentence(0, 1, [Word("into", 0, 0.5), Word("the", 0.5, 0.7), Word("past", 0.7, 1)]).closed_text
        == "into the past."
    )
    assert Sentence(0, 1, [Word("really?", 0, 1)]).closed_text == "really?"


def test_part_number_and_stray_braces():
    title, description = build_metadata(
        "A real sentence here.",
        "a real sentence",
        "other",
        3,
        "short",
        "own",
        "",
        "{title} {oops}",
        "{summary} {unknown}",
    )
    assert title.endswith("| Part 3") and "{oops}" in title
    assert "#shorts" not in description and "{unknown}" in description
    assert credit_line("own", "") == ""


def test_long_format_has_no_shorts_tag():
    _, description = build_metadata("A real sentence here.", "x", "other", None, "long", "own", "")
    assert "#shorts" not in description
    assert json.dumps(description)
