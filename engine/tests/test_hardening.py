import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from clipforge import cleanup, jobs, pipeline
from clipforge.api.app import create_app
from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.export.plan import Segment
from clipforge.export.render import RenderSpec, check_disk_space, render_clip
from clipforge.jobs import Worker, friendly_error
from clipforge.media.ffmpeg import FFmpegError, find_tool
from clipforge.media.probe import probe
from clipforge.models import Clip, Job, Source, utcnow


def ff(*args):
    subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True
    )


def count_frames(path):
    out = subprocess.run(
        [
            find_tool("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(json.loads(out.stdout)["streams"][0]["nb_read_frames"])


def first_frame_color(path, tmp_path):
    out = tmp_path / "f.png"
    ff("-i", path, "-frames:v", "1", out)
    return np.asarray(Image.open(out).convert("RGB")).astype(int).mean(axis=(0, 1))


def manual():
    return {
        "genre": "other",
        "clip_duration": 2,
        "clip_count": 1,
        "part_numbering": False,
        "format": "short",
        "ratio": "9:16",
        "resolution": "1080p",
    }


def test_frame_exact_at_ntsc_rate_far_into_the_file(tmp_path):
    """29.97 fps: frame 600 is the first blue frame. Whole-number rates would hide rounding bugs."""
    video = tmp_path / "ntsc.mp4"
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x240:r=30000/1001:d=20.02",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:r=30000/1001:d=10",
        "-filter_complex",
        "[0][1]concat=n=2:v=1:a=0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "48",
        video,
    )
    info = probe(video)
    assert abs(info.fps - 29.97) < 0.01

    def render(start, end, name):
        spec = RenderSpec(video, info, [Segment(start, end)], "16:9", "1080p", tmp_path / name, fit="crop")
        return render_clip(spec).path

    before = render(599, 602, "a.mp4")
    assert count_frames(before) == 3
    color = first_frame_color(before, tmp_path)
    assert color[0] > 180 and color[2] < 80  # frame 599 is still red
    after = render(600, 603, "b.mp4")
    color = first_frame_color(after, tmp_path)
    assert color[2] > 180 and color[0] < 80  # frame 600 is blue


def test_rotated_phone_video_exports_upright_at_the_right_size(tmp_path):
    video = tmp_path / "rot.mp4"
    ff(
        "-display_rotation",
        "90",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=640x360:r=30:d=2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        video,
    )
    info = probe(video)
    assert info.display_size == (360, 640)
    spec = RenderSpec(video, info, [Segment(0, 30)], "9:16", "1080p", tmp_path / "o.mp4", fit="crop")
    out = probe(render_clip(spec).path)
    assert (out.width, out.height) == (1080, 1920)


def test_job_on_video_without_audio_finishes_and_warns(tmp_path, monkeypatch):
    video = tmp_path / "silent.mp4"
    ff(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=640x360:r=30:d=6",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        video,
    )

    def must_not_run(*a, **k):
        raise AssertionError("no audio, so no transcription")

    monkeypatch.setattr(pipeline, "transcribe", must_not_run)
    with TestClient(create_app()) as client:
        source = client.post("/api/sources/import", json={"path": str(video)}).json()
        client.post(
            "/api/jobs",
            json={
                "source_id": source["id"],
                "mode": "manual",
                "rights": {"type": "own"},
                "manual": manual(),
                "edit": {"fit": "crop"},
            },
        )
        job_id = jobs.claim_next_job()
        jobs.run_one(job_id, pipeline.run_job)
        job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    states = {c["name"]: c["status"] for c in job["clips"][0]["qc"]["checks"]}
    assert states["audio"] == "warn"


def test_restarted_job_resumes_without_duplicate_clips(scene_video, monkeypatch):
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: {"language": "en", "words": []})
    with TestClient(create_app()) as client:
        source = client.post("/api/sources/import", json={"path": str(scene_video)}).json()
        client.post(
            "/api/jobs",
            json={
                "source_id": source["id"],
                "mode": "manual",
                "rights": {"type": "own"},
                "manual": manual(),
                "edit": {"fit": "crop"},
            },
        )
        job_id = jobs.claim_next_job()
        jobs.run_one(job_id, pipeline.run_job)
        first = client.get(f"/api/jobs/{job_id}").json()["clips"]
        assert first

        def must_not_render(*a, **k):
            raise AssertionError("finished clip rendered again")

        monkeypatch.setattr(pipeline, "render_clip", must_not_render)
        with session_scope() as s:
            row = s.get(Job, job_id)
            row.status = "running"
            s.add(row)
            s.commit()
        assert jobs.requeue_interrupted_jobs() == 1
        again = jobs.claim_next_job()
        jobs.run_one(again, pipeline.run_job)
        job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    assert len(job["clips"]) == len(first)


def test_worker_survives_a_loop_error(monkeypatch):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database hiccup")
        return None

    monkeypatch.setattr(jobs, "claim_next_job", flaky)
    monkeypatch.setattr(jobs, "requeue_interrupted_jobs", lambda: 0)
    worker = Worker(lambda ctx: None, poll_seconds=0.01)
    worker.start()
    deadline = time.time() + 3
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert worker.is_alive()
    assert len(calls) >= 3
    worker.stop()
    worker.join(timeout=2)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("write error: No space left on device", "disk is full"),
        ("x.mp4: Invalid data found when processing input", "damaged"),
        ("Cannot allocate memory", "out of memory"),
        ("plain", "plain"),
    ],
)
def test_friendly_errors(message, expected):
    assert expected in friendly_error(RuntimeError(message))


def test_disk_check_refuses_before_rendering(monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _p: type("U", (), {"free": 100})())
    with pytest.raises(FFmpegError, match="Not enough disk space"):
        check_disk_space(1920, 1080, 60)


def test_unhandled_error_is_json_and_keeps_cors_headers(monkeypatch):
    app = create_app()

    @app.get("/api/boom")
    def boom():
        raise ValueError("secret detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/api/boom", headers={"origin": "http://localhost:3000"})
    assert res.status_code == 500
    assert "secret detail" not in res.text
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_preflight_from_the_ui_is_allowed_and_from_others_is_not():
    with TestClient(create_app()) as client:
        ok = client.options(
            "/api/jobs", headers={"origin": "http://localhost:3000", "access-control-request-method": "POST"}
        )
        bad = client.options(
            "/api/jobs", headers={"origin": "https://evil.example", "access-control-request-method": "POST"}
        )
    assert ok.status_code == 200
    assert "access-control-allow-origin" not in bad.headers


def test_job_list_counts_clips_without_a_query_per_job():
    with session_scope() as s:
        source = Source(filename="a.mp4", path="/x/a.mp4")
        s.add(source)
        s.commit()
        job = Job(source_id=source.id, rights_type="own")
        s.add(job)
        s.commit()
        s.add(Clip(job_id=job.id, start=0, end=1, expires_at=utcnow()))
        s.commit()
    with TestClient(create_app()) as client:
        row = client.get("/api/jobs").json()[0]
    assert row["clip_count"] == 1
    assert row["source_filename"] == "a.mp4"


def test_crash_leftovers_are_removed_only_when_old():
    renders = get_settings().renders_dir
    (renders / "job1").mkdir(parents=True)
    old_part = renders / "job1" / ".clip.part.mp4"
    old_part.write_bytes(b"x")
    new_dir = renders / "job2"
    new_dir.mkdir()
    new_part = new_dir / ".clip.part.mp4"
    new_part.write_bytes(b"x")
    old = time.time() - 7 * 3600
    os.utime(old_part, (old, old))
    cleanup.cleanup_stale_temp()
    assert not old_part.exists()
    assert new_part.exists()
    assert Path(new_dir).exists()
