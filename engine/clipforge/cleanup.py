"""24-hour clip cleanup and stale temp removal."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

from sqlmodel import select

from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.models import Clip, ClipStatus, Upload, UploadStatus, utcnow

log = logging.getLogger("clipforge.cleanup")


def remove_clip_files(clip: Clip) -> None:
    for value in (clip.path, clip.thumb_path):
        if value:
            Path(value).unlink(missing_ok=True)
    clip.path = None
    clip.thumb_path = None


def _busy_clip_ids(session) -> set[str]:
    rows = session.exec(select(Upload).where(Upload.status == UploadStatus.uploading.value)).all()
    return {row.clip_id for row in rows}


def cleanup_expired_clips() -> list[str]:
    """Delete expired clip files. A clip mid-upload is kept until the next pass."""
    removed = []
    now = utcnow()
    with session_scope() as session:
        due = session.exec(
            select(Clip).where(Clip.status == ClipStatus.ready.value, Clip.expires_at <= now)
        ).all()
        busy = _busy_clip_ids(session)
        for clip in due:
            if clip.id in busy:
                log.info("Clip %s expired but is uploading; retrying next pass", clip.id)
                continue
            remove_clip_files(clip)
            clip.status = ClipStatus.expired.value
            session.add(clip)
            removed.append(clip.id)
        session.commit()
    if removed:
        log.info("Cleanup removed %d expired clips", len(removed))
    return removed


def _remove_render_leftovers(cutoff: float) -> None:
    """A crash mid-render leaves a hidden partial file. Empty job folders also go."""
    renders = get_settings().renders_dir
    if not renders.exists():
        return
    for folder in renders.iterdir():
        if not folder.is_dir():
            continue
        for partial in folder.glob(".*.part.*"):
            if partial.stat().st_mtime < cutoff:
                partial.unlink(missing_ok=True)
        if not any(folder.iterdir()) and folder.stat().st_mtime < cutoff:
            folder.rmdir()


def cleanup_stale_temp() -> list[str]:
    """Remove temp dirs left by a crash. Normal renders remove their own temp at once."""
    settings = get_settings()
    cutoff = time.time() - settings.temp_max_age_hours * 3600
    removed: list[str] = []
    _remove_render_leftovers(cutoff)
    if not settings.tmp_dir.exists():
        return removed
    for entry in settings.tmp_dir.iterdir():
        if entry.stat().st_mtime < cutoff:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
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
                cleanup_stale_temp()
            except Exception:
                log.exception("Cleanup pass failed")
            self._stop_event.wait(interval)

    def stop(self) -> None:
        self._stop_event.set()
