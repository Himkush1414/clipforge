"""Move clips between folders. Moving changes the timer; the rights record stays on the job."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from sqlmodel import select

from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.folders import FolderError, file_lock, folder_dir, move_file
from clipforge.models import Clip, ClipFolder, ClipStatus, Upload, UploadStatus, utcnow

log = logging.getLogger("clipforge.folders")


def move_clip(clip_id: str, folder: str) -> Clip:
    """Move to Saved or Queued. Either one stops the 24-hour timer."""
    if folder not in (ClipFolder.saved.value, ClipFolder.queued.value):
        raise FolderError("Clips can be moved to Saved or Queued.")
    with file_lock, session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise FolderError("Clip not found.")
        if clip.status != ClipStatus.ready.value or not clip.path:
            raise FolderError("This clip has expired and was deleted.")
        uploading = session.exec(
            select(Upload).where(Upload.clip_id == clip_id, Upload.status == UploadStatus.uploading.value)
        ).first()
        if uploading:
            raise FolderError("This clip is uploading. Move it when the upload has finished.")
        if clip.folder != folder:
            _relocate(clip, folder)
        clip.folder = folder
        clip.expires_at = None
        session.add(clip)
        session.commit()
        return clip


def _relocate(clip: Clip, folder: str) -> None:
    target = folder_dir(folder)
    assert clip.path
    clip.path = str(move_file(Path(clip.path), target))
    if clip.thumb_path and Path(clip.thumb_path).exists():
        clip.thumb_path = str(move_file(Path(clip.thumb_path), target))


def start_timer_after_upload(clip_id: str) -> bool:
    """A Queued clip that uploaded goes to Unassigned, to be deleted 24 hours from now."""
    with file_lock, session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None or clip.folder != ClipFolder.queued.value or not clip.path:
            return False
        waiting = session.exec(
            select(Upload).where(
                Upload.clip_id == clip_id,
                Upload.status.in_(  # type: ignore[attr-defined]
                    [UploadStatus.pending.value, UploadStatus.uploading.value, UploadStatus.processing.value]
                ),
            )
        ).first()
        if waiting:
            return False  # A linked channel is still on its way. The last one starts the timer.
        _relocate(clip, ClipFolder.unassigned.value)
        clip.folder = ClipFolder.unassigned.value
        clip.expires_at = utcnow() + timedelta(hours=get_settings().clip_ttl_hours)
        session.add(clip)
        session.commit()
        log.info("Clip %s uploaded from Queued; it will be deleted in 24 hours", clip_id)
        return True
