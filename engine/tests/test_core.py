from datetime import timedelta

import pytest
from sqlmodel import select

from clipforge import cleanup, jobs
from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.hardware import RESOLUTIONS, detect
from clipforge.media.ffmpeg import (
    Cancelled,
    FFmpegError,
    assert_allowed_graph,
    escape_filter_path,
    run_ffmpeg,
)
from clipforge.media.probe import probe
from clipforge.models import Clip, ClipStatus, Job, JobStatus, Source, Upload, UploadStatus, utcnow


def _job(session, status=JobStatus.queued.value, source_id=None):
    if source_id is None:
        source = Source(filename="a.mp4", path="/x/a.mp4")
        session.add(source)
        session.commit()
        source_id = source.id
    job = Job(source_id=source_id, rights_type="own", status=status)
    session.add(job)
    session.commit()
    return job


def test_probe_reads_exact_stream_facts(scene_video):
    info = probe(scene_video)
    assert (info.width, info.height) == (1280, 720)
    assert info.fps == 30
    assert info.frame_count == 180
    assert info.has_audio
    assert abs(info.duration - 6) < 0.05


def test_probe_rejects_non_media(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(FFmpegError):
        probe(bad)


def test_ffmpeg_error_carries_stderr(tmp_path):
    with pytest.raises(FFmpegError) as err:
        run_ffmpeg(["-i", tmp_path / "missing.mp4", "-f", "null", "-"])
    assert "missing.mp4" in err.value.stderr


def test_ffmpeg_reports_progress_and_finishes_at_one(scene_video):
    seen = []
    run_ffmpeg(["-i", scene_video, "-f", "null", "-"], duration=6, on_progress=seen.append)
    assert seen[-1] == 1.0


def test_ffmpeg_cancel_kills_process(scene_video):
    with pytest.raises(Cancelled):
        run_ffmpeg(["-stream_loop", "200", "-i", scene_video, "-f", "null", "-"], should_cancel=lambda: True)


@pytest.mark.parametrize("graph", ["[0:v]hflip[v]", "atempo=1.05", "asetrate=44100*1.02", "scale=1:1,vflip"])
def test_evasion_filters_are_refused(graph):
    with pytest.raises(FFmpegError):
        assert_allowed_graph(["-filter_complex", graph])


def test_normal_filters_pass():
    assert_allowed_graph(["-filter_complex", "[0:v]scale=1080:1920,ass=a.ass[v];loudnorm=I=-14"])


def test_filter_path_escaping():
    assert escape_filter_path("/a b/c:d'e.ass") == "/a b/c\\:d\\'e.ass"


def test_hardware_only_offers_resolutions_an_encoder_passed():
    hw = detect()
    assert "1080p" in hw.allowed_resolutions
    top = max(hw.h264_max_height, hw.hevc_max_height)
    assert all(RESOLUTIONS[r] <= top for r in hw.allowed_resolutions)


def test_claim_is_oldest_first_and_exclusive():
    with session_scope() as s:
        first = _job(s)
        _job(s, source_id=first.source_id)
    assert jobs.claim_next_job() == first.id
    with session_scope() as s:
        assert s.get(Job, first.id).status == JobStatus.running.value
    second = jobs.claim_next_job()
    assert second and second != first.id
    assert jobs.claim_next_job() is None


def test_interrupted_jobs_are_requeued():
    with session_scope() as s:
        job = _job(s, status=JobStatus.running.value)
    assert jobs.requeue_interrupted_jobs() == 1
    with session_scope() as s:
        assert s.get(Job, job.id).status == JobStatus.queued.value


def test_run_one_records_failure_and_cancel():
    with session_scope() as s:
        failed, cancelled, ok = _job(s), _job(s), _job(s)

    def boom(ctx):
        raise ValueError("bad input")

    def cancel(ctx):
        raise Cancelled()

    jobs.run_one(failed.id, boom)
    jobs.run_one(cancelled.id, cancel)
    jobs.run_one(ok.id, lambda ctx: ctx.update(stage="x", progress=0.5))
    with session_scope() as s:
        assert s.get(Job, failed.id).status == "failed"
        assert "bad input" in s.get(Job, failed.id).error
        assert s.get(Job, cancelled.id).status == "cancelled"
        assert s.get(Job, ok.id).status == "done"
        assert s.get(Job, ok.id).progress == 1.0


def _clip(session, job_id, tmp_path, name, expires_delta):
    path = tmp_path / name
    path.write_bytes(b"x")
    thumb = tmp_path / f"{name}.jpg"
    thumb.write_bytes(b"x")
    clip = Clip(
        job_id=job_id,
        start=0,
        end=1,
        path=str(path),
        thumb_path=str(thumb),
        expires_at=utcnow() + expires_delta,
    )
    session.add(clip)
    session.commit()
    return clip


def test_cleanup_removes_only_expired_clips(tmp_path):
    with session_scope() as s:
        job = _job(s)
        old = _clip(s, job.id, tmp_path, "old.mp4", timedelta(seconds=-1))
        fresh = _clip(s, job.id, tmp_path, "fresh.mp4", timedelta(hours=23))
    assert cleanup.cleanup_expired_clips() == [old.id]
    assert not (tmp_path / "old.mp4").exists()
    assert not (tmp_path / "old.mp4.jpg").exists()
    assert (tmp_path / "fresh.mp4").exists()
    with session_scope() as s:
        assert s.get(Clip, old.id).status == ClipStatus.expired.value
        assert s.get(Clip, old.id).path is None
        assert s.get(Clip, fresh.id).status == ClipStatus.ready.value
    assert cleanup.cleanup_expired_clips() == []


def test_cleanup_waits_for_active_upload(tmp_path):
    with session_scope() as s:
        job = _job(s)
        clip = _clip(s, job.id, tmp_path, "busy.mp4", timedelta(seconds=-1))
        up = Upload(clip_id=clip.id, channel_id="c", status=UploadStatus.uploading.value)
        s.add(up)
        s.commit()
    assert cleanup.cleanup_expired_clips() == []
    assert (tmp_path / "busy.mp4").exists()
    with session_scope() as s:
        row = s.get(Upload, up.id)
        row.status = UploadStatus.published.value
        s.add(row)
        s.commit()
    assert cleanup.cleanup_expired_clips() == [clip.id]


def test_cleanup_removes_only_stale_temp():
    import os
    import time

    tmp = get_settings().tmp_dir
    stale = tmp / "stale"
    stale.mkdir()
    (stale / "frame.png").write_bytes(b"x")
    old = time.time() - 7 * 3600
    os.utime(stale, (old, old))
    (tmp / "active").mkdir()
    assert cleanup.cleanup_stale_temp() == ["stale"]
    assert (tmp / "active").exists()


def test_db_has_expected_tables():
    with session_scope() as s:
        assert s.exec(select(Clip)).all() == []


def test_stale_database_stops_startup_with_a_clear_message_and_changes_nothing():
    import sqlite3

    from clipforge import db

    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE upload (id TEXT PRIMARY KEY, clip_id TEXT, keepme TEXT)")
    conn.execute("INSERT INTO upload (id, keepme) VALUES ('a', 'data')")
    conn.commit()
    conn.close()
    db.get_engine.cache_clear()
    with pytest.raises(db.SchemaMismatch, match=r"older clipforge version .*upload\.made_for_kids"):
        db.get_engine()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT keepme FROM upload").fetchone() == ("data",)
    conn.close()
    db.get_engine.cache_clear()
