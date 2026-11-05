import logging
import os
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge import cleanup, clip_files, folders, jobs, pipeline
from clipforge.api.app import create_app
from clipforge.appsettings import AppSettings, load_app_settings, save_app_settings
from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.folders import (
    FolderError,
    LowDisk,
    delete_by_user,
    delete_from_unassigned,
    is_inside,
    storage_report,
    validate_saved_dir,
)
from clipforge.models import Channel, Clip, Job, Source, Upload, UploadStatus, utcnow
from clipforge.uploads import service
from clipforge.uploads.base import RemoteState, save_tokens

HOUR = timedelta(hours=1)


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def make_clip(folder="unassigned", expires=-HOUR, name=None, rights="own", detail=""):
    """A real file in the real folder, with a matching clip row."""
    directory = folders.folder_dir(folder)
    directory.mkdir(parents=True, exist_ok=True)
    with session_scope() as s:
        source = Source(filename="a.mp4", path="/x/a.mp4")
        s.add(source)
        s.commit()
        job = Job(source_id=source.id, rights_type=rights, rights_detail=detail)
        s.add(job)
        s.commit()
        clip = Clip(
            job_id=job.id,
            start=0,
            end=1,
            folder=folder,
            duration=10.0,
            frames=300,
            width=1080,
            height=1920,
            qc={"status": "ok"},
            expires_at=(utcnow() + expires) if expires is not None else None,
        )
        path = directory / f"{name or clip.id}.mp4"
        path.write_bytes(b"video-bytes")
        thumb = directory / f"{name or clip.id}.jpg"
        thumb.write_bytes(b"thumb")
        clip.path, clip.thumb_path = str(path), str(thumb)
        s.add(clip)
        s.commit()
        return clip.id, path, thumb


def row(clip_id):
    with session_scope() as s:
        return s.get(Clip, clip_id)


# ---------- the cleanup job only deletes inside Unassigned ----------
def test_expired_unassigned_clip_is_deleted_and_logged(caplog):
    clip_id, path, thumb = make_clip("unassigned")
    with caplog.at_level(logging.INFO, logger="clipforge.deletions"):
        assert cleanup.cleanup_expired_clips() == [clip_id]
    assert not path.exists() and not thumb.exists()
    assert row(clip_id).status == "expired"
    logged = caplog.text
    assert f"DELETED {path}" in logged and f"DELETED {thumb}" in logged


def test_fresh_unassigned_clip_survives():
    clip_id, path, _ = make_clip("unassigned", expires=HOUR)
    assert cleanup.cleanup_expired_clips() == []
    assert path.exists()


@pytest.mark.parametrize("folder", ["saved", "queued"])
def test_saved_and_queued_files_survive_every_cleanup_pass(folder):
    """Even with an expiry time in the past, which should never happen, they are not touched."""
    expired_row, path, thumb = make_clip(folder, expires=-HOUR)
    no_timer, path2, _ = make_clip(folder, expires=None)
    old = os.path.getmtime(path) - 30 * 24 * 3600
    os.utime(path, (old, old))
    for _ in range(3):
        assert cleanup.cleanup_expired_clips() == []
        cleanup.cleanup_orphans()
        cleanup.cleanup_stale_temp()
    assert path.read_bytes() == b"video-bytes" and thumb.exists() and path2.exists()
    assert row(expired_row).status == "ready" and row(no_timer).status == "ready"


def test_unassigned_row_pointing_at_saved_is_refused(caplog):
    """A corrupted row must not be able to aim the cleanup at another folder."""
    _, saved_file, _ = make_clip("saved", expires=None)
    clip_id, own, thumb = make_clip("unassigned")
    with session_scope() as s:
        clip = s.get(Clip, clip_id)
        clip.path = str(saved_file)
        s.add(clip)
        s.commit()
    with caplog.at_level(logging.INFO, logger="clipforge.deletions"):
        assert cleanup.cleanup_expired_clips() == []
    assert saved_file.exists()
    assert "REFUSED" in caplog.text
    assert row(clip_id).status == "ready"


@pytest.mark.parametrize("where", ["outside", "repo", "queued", "dotdot"])
def test_paths_outside_unassigned_are_never_deleted(tmp_path, caplog, where):
    settings = get_settings()
    if where == "outside":
        victim = tmp_path / "victim.mp4"
    elif where == "repo":
        victim = folders.REPO_ROOT / "engine" / "pyproject.toml"
    elif where == "queued":
        settings.queued_dir.mkdir(parents=True, exist_ok=True)
        victim = settings.queued_dir / "victim.mp4"
    else:
        victim = tmp_path / "victim.mp4"
    if where != "repo":
        victim.write_bytes(b"keep")
    before = victim.read_bytes()
    target = (
        settings.unassigned_dir / ".." / ".." / ".." / ".." / victim.name if where == "dotdot" else victim
    )
    if where == "dotdot":
        target = Path(os.path.relpath(victim, settings.unassigned_dir))
        target = settings.unassigned_dir / target
        assert ".." in str(target)
    with caplog.at_level(logging.ERROR, logger="clipforge.deletions"):
        assert delete_from_unassigned(target, "test") is False
    assert victim.read_bytes() == before
    assert "REFUSED" in caplog.text


def test_symlink_in_unassigned_is_not_followed_or_removed(tmp_path, caplog):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"keep")
    root = get_settings().unassigned_dir
    root.mkdir(parents=True, exist_ok=True)
    link = root / "link.mp4"
    link.symlink_to(outside)
    with caplog.at_level(logging.ERROR, logger="clipforge.deletions"):
        assert delete_from_unassigned(link, "test") is False
    assert outside.read_bytes() == b"keep"
    assert link.is_symlink()
    assert "symlink" in caplog.text


def test_file_reached_through_a_symlinked_folder_is_refused(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    victim = outside_dir / "victim.mp4"
    victim.write_bytes(b"keep")
    root = get_settings().unassigned_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "sneaky").symlink_to(outside_dir)
    assert delete_from_unassigned(root / "sneaky" / "victim.mp4", "test") is False
    assert victim.exists()


def test_orphan_sweep_ignores_symlinks_and_recent_files_and_known_clips(tmp_path):
    root = get_settings().unassigned_dir
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"keep")
    (root / "link.mp4").symlink_to(outside)
    orphan = root / "orphan.mp4"
    orphan.write_bytes(b"x")
    recent = root / "recent.mp4"
    recent.write_bytes(b"x")
    _, known, _ = make_clip("unassigned", expires=HOUR)
    old = os.path.getmtime(orphan) - 40 * 3600
    for path in (orphan, known):
        os.utime(path, (old, old))
    assert cleanup.cleanup_orphans() == ["orphan.mp4"]
    assert recent.exists() and known.exists() and outside.exists() and (root / "link.mp4").is_symlink()


@pytest.mark.parametrize("bad", ["symlinked", "saved_inside"])
def test_nothing_is_deleted_if_unassigned_itself_is_unsafe(tmp_path, bad, caplog):
    settings = get_settings()
    root = settings.unassigned_dir
    victim = None
    if bad == "symlinked":
        target = tmp_path / "elsewhere"
        target.mkdir()
        victim = target / "v.mp4"
        victim.write_bytes(b"keep")
        os.rmdir(root) if root.exists() else root.parent.mkdir(parents=True, exist_ok=True)
        root.symlink_to(target)
        path = root / "v.mp4"
    else:
        root.mkdir(parents=True, exist_ok=True)
        save_app_settings(AppSettings(saved_dir=str(root / "saved")))
        path = root / "v.mp4"
        path.write_bytes(b"keep")
        victim = path
    with caplog.at_level(logging.ERROR, logger="clipforge.deletions"):
        assert delete_from_unassigned(path, "test") is False
    assert victim.exists()
    assert "REFUSED" in caplog.text


def test_user_delete_only_works_inside_clip_folders(tmp_path):
    outside = tmp_path / "o.mp4"
    outside.write_bytes(b"keep")
    assert delete_by_user(outside) is False
    assert outside.exists()
    _, saved_file, _ = make_clip("saved", expires=None)
    assert delete_by_user(saved_file) is True
    assert not saved_file.exists()


def test_is_inside_rejects_the_folder_itself_and_lookalike_names(tmp_path):
    (tmp_path / "unassigned").mkdir()
    (tmp_path / "unassigned-evil").mkdir()
    assert is_inside(tmp_path / "unassigned" / "a", tmp_path / "unassigned")
    assert not is_inside(tmp_path / "unassigned", tmp_path / "unassigned")
    assert not is_inside(tmp_path / "unassigned-evil" / "a", tmp_path / "unassigned")


def test_temp_sweep_never_follows_a_symlink_out(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    keep = outside / "keep.mp4"
    keep.write_bytes(b"keep")
    tmp = get_settings().tmp_dir
    tmp.mkdir(parents=True, exist_ok=True)
    link = tmp / "link"
    link.symlink_to(outside)
    old = os.path.getmtime(outside) - 7 * 3600
    os.utime(link, (old, old), follow_symlinks=False)
    cleanup.cleanup_stale_temp()
    assert keep.exists()


# ---------- moving clips ----------
def test_move_to_saved_moves_files_and_stops_the_timer():
    clip_id, path, thumb = make_clip("unassigned", expires=HOUR)
    moved = clip_files.move_clip(clip_id, "saved")
    assert moved.folder == "saved" and moved.expires_at is None
    assert not path.exists() and not thumb.exists()
    assert Path(moved.path).parent == folders.saved_dir()
    assert Path(moved.path).read_bytes() == b"video-bytes"
    assert Path(moved.thumb_path).read_bytes() == b"thumb"
    assert cleanup.cleanup_expired_clips() == []


def test_move_to_queued_pauses_the_timer_even_if_almost_expired():
    clip_id, _, _ = make_clip("unassigned", expires=timedelta(seconds=30))
    moved = clip_files.move_clip(clip_id, "queued")
    assert moved.expires_at is None and Path(moved.path).parent == get_settings().queued_dir


def test_cannot_move_while_uploading_or_after_expiry():
    clip_id, _, _ = make_clip("unassigned", expires=HOUR)
    with session_scope() as s:
        s.add(Upload(clip_id=clip_id, channel_id="c", status=UploadStatus.uploading.value))
        s.commit()
    with pytest.raises(FolderError, match="uploading"):
        clip_files.move_clip(clip_id, "saved")
    gone, _, _ = make_clip("unassigned", expires=-HOUR)
    cleanup.cleanup_expired_clips()
    with pytest.raises(FolderError, match="expired"):
        clip_files.move_clip(gone, "saved")


def test_move_with_low_disk_refuses_and_leaves_the_clip_alone(monkeypatch):
    clip_id, path, _ = make_clip("unassigned", expires=HOUR)
    monkeypatch.setattr("shutil.disk_usage", lambda _p: type("U", (), {"free": 10})())
    with pytest.raises(LowDisk, match="Not enough disk space"):
        clip_files.move_clip(clip_id, "saved")
    assert path.exists() and row(clip_id).folder == "unassigned"


def test_failed_copy_leaves_the_source_and_no_partial(monkeypatch):
    clip_id, path, _ = make_clip("unassigned", expires=HOUR)

    def broken(src, dst):
        Path(dst).write_bytes(b"short")

    monkeypatch.setattr("shutil.copyfile", broken)
    with pytest.raises(FolderError, match="sizes do not match"):
        clip_files.move_clip(clip_id, "saved")
    assert path.read_bytes() == b"video-bytes"
    assert list(folders.saved_dir().glob(".*")) == []


# ---------- Saved path setting ----------
def test_saved_path_can_be_configured_and_windows_paths_work(client, tmp_path):
    target = tmp_path / "my saved"
    res = client.put(
        "/api/settings",
        json={"saved_dir": str(target), "queued_limit_gb": 10, "saved_limit_gb": 20, "low_disk_gb": 1},
    )
    assert res.status_code == 200, res.text
    assert res.json()["saved_path"] == str(target.resolve())
    clip_id, _, _ = make_clip("unassigned", expires=HOUR)
    moved = clip_files.move_clip(clip_id, "saved")
    assert Path(moved.path).parent == target.resolve()
    assert folders.windows_to_wsl(r"D:\clipforge saved").as_posix() == "/mnt/d/clipforge saved"
    assert client.put("/api/settings", json={"saved_dir": ""}).json()["saved_path"] == str(
        get_settings().default_saved_dir
    )


@pytest.mark.parametrize(
    "bad",
    ["/", "/etc/clipforge", "/usr/x", "relative/path", "/mnt/d", "D:\\"],
)
def test_unsafe_saved_paths_are_refused(bad):
    with pytest.raises(FolderError):
        validate_saved_dir(bad)


def test_saved_cannot_overlap_repo_or_other_folders():
    settings = get_settings()
    for bad in (
        folders.REPO_ROOT / "saved",
        settings.unassigned_dir / "x",
        settings.queued_dir,
        settings.media_dir,
        settings.work_dir / "tmp" / "s",
    ):
        with pytest.raises(FolderError, match="cannot be inside"):
            validate_saved_dir(str(bad))
    assert load_app_settings().saved_dir is None


def test_changing_saved_path_keeps_existing_saved_clips_where_they_are(tmp_path):
    clip_id, _, _ = make_clip("unassigned", expires=HOUR)
    first = clip_files.move_clip(clip_id, "saved").path
    save_app_settings(AppSettings(saved_dir=str(tmp_path / "new")))
    assert Path(first).exists() and row(clip_id).path == first


# ---------- disk guard ----------
def test_storage_warns_past_limits_and_when_disk_is_low(monkeypatch):
    make_clip("queued", expires=None)
    make_clip("saved", expires=None)
    save_app_settings(AppSettings(queued_limit_gb=1e-9, saved_limit_gb=1e-9, low_disk_gb=5))
    monkeypatch.setattr("shutil.disk_usage", lambda _p: type("U", (), {"free": 1024**3})())
    report = storage_report()
    text = " ".join(report["warnings"])
    assert "Queued is" in text and "Saved folder is" in text and "New jobs will not start" in text
    assert report["folders"]["queued"] > 0


def test_no_warnings_when_everything_is_fine():
    assert storage_report()["warnings"] == []


def test_low_disk_refuses_a_job_with_a_clear_message(client, scene_video, monkeypatch):
    source = client.post("/api/sources/import", json={"path": str(scene_video)}).json()
    real = __import__("shutil").disk_usage
    monkeypatch.setattr(
        "shutil.disk_usage", lambda p: type("U", (), {"free": 1024**2, "total": real(p).total})()
    )
    res = client.post(
        "/api/jobs", json={"source_id": source["id"], "mode": "auto", "rights": {"type": "own"}}
    )
    assert res.status_code == 507
    assert "Not enough disk space" in res.json()["detail"]


def test_disk_filling_mid_render_is_a_visible_job_failure(client, scene_video, monkeypatch):
    source = client.post("/api/sources/import", json={"path": str(scene_video)}).json()
    client.post(
        "/api/jobs",
        json={
            "source_id": source["id"],
            "mode": "manual",
            "rights": {"type": "own"},
            "edit": {"captions": False, "fit": "crop"},
            "manual": {
                "genre": "other",
                "clip_duration": 2,
                "clip_count": 1,
                "format": "short",
                "ratio": "9:16",
                "resolution": "1080p",
            },
        },
    )
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: {"language": "en", "words": []})
    monkeypatch.setattr(
        "clipforge.export.render.check_disk_space",
        lambda *a: (_ for _ in ()).throw(
            __import__("clipforge.media.ffmpeg", fromlist=["FFmpegError"]).FFmpegError(
                "write error: No space left on device"
            )
        ),
    )
    job_id = jobs.claim_next_job()
    jobs.run_one(job_id, pipeline.run_job)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and "disk is full" in job["error"]


# ---------- jobs choose a folder ----------
def _job(client, video, monkeypatch, **extra):
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: {"language": "en", "words": []})
    source = client.post("/api/sources/import", json={"path": str(video)}).json()
    body = {
        "source_id": source["id"],
        "mode": "manual",
        "rights": {"type": "creative_commons", "detail": "By A, CC BY"},
        "edit": {"captions": False, "fit": "crop"},
        "manual": {
            "genre": "other",
            "clip_duration": 2,
            "clip_count": 1,
            "format": "short",
            "ratio": "9:16",
            "resolution": "1080p",
        },
        **extra,
    }
    assert client.post("/api/jobs", json=body).status_code == 201
    job_id = jobs.claim_next_job()
    jobs.run_one(job_id, pipeline.run_job)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    return job["clips"][0]


def test_default_job_goes_to_unassigned_with_a_24_hour_timer(client, scene_video, monkeypatch):
    clip = _job(client, scene_video, monkeypatch)
    assert clip["folder"] == "unassigned"
    assert 86000 < clip["seconds_left"] <= 86400
    assert Path(row(clip["id"]).path).parent == get_settings().unassigned_dir
    assert folders.is_inside(row(clip["id"]).thumb_path, get_settings().unassigned_dir)


def test_make_now_upload_later_goes_to_queued_with_no_timer(client, scene_video, monkeypatch):
    clip = _job(client, scene_video, monkeypatch, destination="queued")
    assert clip["folder"] == "queued" and clip["seconds_left"] is None
    assert Path(row(clip["id"]).path).parent == get_settings().queued_dir
    assert cleanup.cleanup_expired_clips() == []


def test_rejected_destination(client):
    res = client.post(
        "/api/jobs",
        json={"source_id": "x", "mode": "auto", "rights": {"type": "own"}, "destination": "saved"},
    )
    assert res.status_code == 422


def test_no_partial_files_ever_appear_in_a_clip_folder(client, scene_video, monkeypatch):
    clip = _job(client, scene_video, monkeypatch)
    names = {p.name for p in get_settings().unassigned_dir.iterdir()}
    assert names == {f"{clip['id']}.mp4", f"{clip['id']}.jpg"}


# ---------- rights travel with the clip ----------
@pytest.mark.parametrize("target", ["saved", "queued"])
def test_rights_record_stays_with_moved_clips(client, target):
    clip_id, _, _ = make_clip("unassigned", expires=HOUR, rights="creative_commons", detail="By A, CC BY")
    moved = client.post(f"/api/clips/{clip_id}/move", json={"folder": target}).json()
    assert moved["rights_type"] == "creative_commons" and moved["rights_detail"] == "By A, CC BY"
    listed = next(c for c in client.get("/api/clips", params={"folder": target}).json() if c["id"] == clip_id)
    assert listed["rights_detail"] == "By A, CC BY"


def _channel():
    with session_scope() as s:
        channel = Channel(name="Y", platform="youtube", connected=True)
        s.add(channel)
        s.commit()
        save_tokens(channel.id, {"access_token": "T", "refresh_token": "R", "expires_at": 9e12})
        return channel.id


def test_uploading_a_queued_clip_later_still_requires_rights():
    clip_id, _, _ = make_clip("queued", expires=None, rights="")
    with pytest.raises(service.UploadBlocked, match="rights"):
        service.queue_upload(clip_id, [_channel()], "private", False)


def test_queued_credit_is_added_at_upload_time():
    clip_id, _, _ = make_clip("queued", expires=None, rights="creative_commons", detail="By A, CC BY")
    assert "By A, CC BY" in service.queue_upload(clip_id, [_channel()], "private", False)[0].description


# ---------- after upload ----------
def test_queued_clip_is_deleted_24_hours_after_a_successful_upload_not_before():
    clip_id, _, _ = make_clip("queued", expires=None)
    upload = service.queue_upload(clip_id, [_channel()], "private", False)[0]
    service.apply_state(upload.id, RemoteState("processing", remote_id="V"))
    assert row(clip_id).folder == "queued" and row(clip_id).expires_at is None
    service.apply_state(upload.id, RemoteState("published", remote_id="V"))
    clip = row(clip_id)
    assert clip.folder == "unassigned"
    assert 86000 < (clip.expires_at - utcnow()).total_seconds() <= 86400
    assert folders.is_inside(clip.path, get_settings().unassigned_dir)
    assert cleanup.cleanup_expired_clips() == []
    with session_scope() as s:
        c = s.get(Clip, clip_id)
        c.expires_at = utcnow() - timedelta(seconds=1)
        s.add(c)
        s.commit()
    assert cleanup.cleanup_expired_clips() == [clip_id]


def test_failed_upload_keeps_the_clip_queued():
    clip_id, _, _ = make_clip("queued", expires=None)
    upload = service.queue_upload(clip_id, [_channel()], "private", False)[0]
    service.apply_state(upload.id, RemoteState("failed", error="nope"))
    assert row(clip_id).folder == "queued" and row(clip_id).expires_at is None


def test_claimed_upload_keeps_the_clip_queued():
    clip_id, _, _ = make_clip("queued", expires=None)
    upload = service.queue_upload(clip_id, [_channel()], "private", False)[0]
    service.apply_state(upload.id, RemoteState("claimed", remote_id="V", claim={"reason": "copyright"}))
    assert row(clip_id).folder == "queued" and row(clip_id).expires_at is None


def test_linked_channels_start_the_timer_only_after_the_last_finishes():
    clip_id, _, _ = make_clip("queued", expires=None)
    first, second = service.queue_upload(clip_id, [_channel(), _channel()], "private", False)
    service.apply_state(first.id, RemoteState("published", remote_id="A"))
    assert row(clip_id).folder == "queued"
    service.apply_state(second.id, RemoteState("published", remote_id="B"))
    assert row(clip_id).folder == "unassigned"


def test_uploading_a_saved_clip_never_starts_a_timer():
    clip_id, path, _ = make_clip("saved", expires=None)
    upload = service.queue_upload(clip_id, [_channel()], "private", False)[0]
    service.apply_state(upload.id, RemoteState("published", remote_id="V"))
    assert row(clip_id).folder == "saved" and row(clip_id).expires_at is None and path.exists()


# ---------- multi-select ----------
def test_bulk_move_and_delete_report_each_clip(client):
    good, _, _ = make_clip("unassigned", expires=HOUR)
    busy, _, _ = make_clip("unassigned", expires=HOUR)
    with session_scope() as s:
        s.add(Upload(clip_id=busy, channel_id="c", status=UploadStatus.uploading.value))
        s.commit()
    res = client.post(
        "/api/clips/bulk", json={"clip_ids": [good, busy, "missing", good], "action": "move_saved"}
    ).json()
    assert res["done"] == [good]
    assert {f["clip_id"] for f in res["failed"]} == {busy, "missing"}
    assert row(good).folder == "saved"
    res = client.post("/api/clips/bulk", json={"clip_ids": [good], "action": "delete"}).json()
    assert res["done"] == [good] and row(good).status == "deleted"
    assert client.post("/api/clips/bulk", json={"clip_ids": [], "action": "delete"}).status_code == 422
    assert client.post("/api/clips/bulk", json={"clip_ids": [good], "action": "shred"}).status_code == 422


def test_bulk_upload_reports_each_clip(client):
    ok, _, _ = make_clip("queued", expires=None)
    no_rights, _, _ = make_clip("queued", expires=None, rights="")
    channel = _channel()
    res = client.post("/api/uploads/bulk", json={"clip_ids": [ok, no_rights], "channel_id": channel})
    assert res.status_code == 201
    body = res.json()
    assert body["done"] == [ok]
    assert body["failed"][0]["clip_id"] == no_rights and "rights" in body["failed"][0]["error"]
    assert client.post("/api/uploads/bulk", json={"clip_ids": [ok]}).status_code == 422


def test_clip_list_filters_by_folder(client):
    a, _, _ = make_clip("unassigned", expires=HOUR)
    b, _, _ = make_clip("queued", expires=None)
    c, _, _ = make_clip("saved", expires=None)
    assert [x["id"] for x in client.get("/api/clips", params={"folder": "queued"}).json()] == [b]
    assert [x["id"] for x in client.get("/api/clips", params={"folder": "saved"}).json()] == [c]
    assert {x["id"] for x in client.get("/api/clips").json()} == {a, b, c}
    assert next(x for x in client.get("/api/clips").json() if x["id"] == c)["seconds_left"] is None
