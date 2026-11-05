"""24-hour clip cleanup. It can only delete inside the Unassigned folder, and logs every deletion.

Queued and Saved have no timer. The temp sweep is a separate job, confined to the temp folder.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time

from sqlmodel import select

from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.folders import delete_by_user, delete_from_unassigned, file_lock, is_inside
from clipforge.models import Clip, ClipFolder, ClipStatus, Upload, UploadStatus, utcnow

log = logging.getLogger("clipforge.cleanup")
deletions = logging.getLogger("clipforge.deletions")

ORPHAN_GRACE_HOURS = 1.0


def remove_clip_files(clip: Clip) -> None:
    """The user pressed Delete on this clip."""
    for value in (clip.path, clip.thumb_path):
        if value:
            delete_by_user(value)
    clip.path = None
    clip.thumb_path = None


def _busy_clip_ids(session) -> set[str]:
    rows = session.exec(select(Upload).where(Upload.status == UploadStatus.uploading.value)).all()
    return {row.clip_id for row in rows}


def cleanup_expired_clips() -> list[str]:
    """Delete expired Unassigned clips. A clip mid-upload is kept until the next pass."""
    removed = []
    now = utcnow()
    with file_lock, session_scope() as session:
        due = session.exec(
            select(Clip).where(
                Clip.status == ClipStatus.ready.value,
                Clip.folder == ClipFolder.unassigned.value,
                Clip.expires_at <= now,  # type: ignore[operator]
            )
        ).all()
        busy = _busy_clip_ids(session)
        for clip in due:
            if clip.id in busy:
                log.info("Clip %s expired but is uploading; retrying next pass", clip.id)
                continue
            files = [value for value in (clip.path, clip.thumb_path) if value]
            if not all(delete_from_unassigned(value, f"clip {clip.id} expired") for value in files):
                log.error("Clip %s expired but its files were refused; see the deletion log", clip.id)
                continue
            clip.path = None
            clip.thumb_path = None
            clip.status = ClipStatus.expired.value
            session.add(clip)
            removed.append(clip.id)
        session.commit()
    if removed:
        log.info("Cleanup removed %d expired clips", len(removed))
    return removed


def cleanup_orphans() -> list[str]:
    """A crash between render and database write leaves a file no clip row knows about."""
    settings = get_settings()
    cutoff = time.time() - (settings.clip_ttl_hours + ORPHAN_GRACE_HOURS) * 3600
    removed: list[str] = []
    root = settings.unassigned_dir
    if not root.exists():
        return removed
    with file_lock, session_scope() as session:
        known = {
            value
            for clip_path, thumb in session.exec(select(Clip.path, Clip.thumb_path)).all()
            for value in (clip_path, thumb)
            if value
        }
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_file() or str(entry) in known:
                continue
            if entry.stat().st_mtime < cutoff and delete_from_unassigned(entry, "orphan with no clip record"):
                removed.append(entry.name)
    return removed


def cleanup_stale_temp() -> list[str]:
    """Remove temp left by a crash. Normal renders remove their own temp at once.

    Confined to the temp folder. A symlink is removed as a link and never followed.
    """
    settings = get_settings()
    cutoff = time.time() - settings.temp_max_age_hours * 3600
    removed: list[str] = []
    if not settings.tmp_dir.exists():
        return removed
    for entry in settings.tmp_dir.iterdir():
        if not is_inside(entry, settings.tmp_dir) and not entry.is_symlink():
            deletions.error("REFUSED %s (temp sweep): outside the temp folder", entry)
            continue
        if entry.lstat().st_mtime >= cutoff:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        deletions.info("DELETED %s (stale temp)", entry)
        removed.append(entry.name)
    return removed


class CleanupThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="clipforge-cleanup")
        self._stop_event = threading.Event()

    def run(self) -> None:
        interval = get_settings().cleanup_interval_seconds
        while not self._stop_event.is_set():
            try:
                cleanup_expired_clips()
                cleanup_orphans()
                cleanup_stale_temp()
            except Exception:
                log.exception("Cleanup pass failed")
            self._stop_event.wait(interval)

    def stop(self) -> None:
        self._stop_event.set()
