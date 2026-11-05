"""Upload rules live here. A copyright claim stops everything and nothing retries by itself."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import httpx
from sqlmodel import col, select

from clipforge.db import session_scope
from clipforge.metadata import credit_line
from clipforge.models import Channel, Clip, ClipStatus, Job, RightsType, Upload, UploadStatus, utcnow
from clipforge.uploads import meta, youtube
from clipforge.uploads.base import RemoteState, UploadError, UploadRequest

log = logging.getLogger("clipforge.uploads")

RESOLUTIONS = ("trim", "replace_audio", "dispute")
POLL_SECONDS = 30
POLL_ATTEMPTS = 10


class UploadBlocked(UploadError):
    """A rule says no. Shown to the user, and never retried."""


def http_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(60.0, write=600.0), follow_redirects=False)


def source_of(session, clip: Clip) -> str | None:
    job = session.get(Job, clip.job_id)
    return job.source_id if job else None


def clip_ids_for_source(session, source_id: str) -> list[str]:
    job_ids = session.exec(select(Job.id).where(Job.source_id == source_id)).all()
    if not job_ids:
        return []
    return list(session.exec(select(Clip.id).where(col(Clip.job_id).in_(job_ids))).all())


def open_claim_for_source(session, source_id: str) -> Upload | None:
    """A claim with no recorded decision blocks every upload from the same source video."""
    clip_ids = clip_ids_for_source(session, source_id)
    if not clip_ids:
        return None
    return session.exec(
        select(Upload).where(
            col(Upload.clip_id).in_(clip_ids),
            Upload.status == UploadStatus.claimed.value,
            col(Upload.claim_resolution).is_(None),
        )
    ).first()


def description_with_credit(clip: Clip, job: Job) -> str:
    """Editing the description in the UI must not be able to remove a credit the licence requires."""
    credit = credit_line(job.rights_type, job.rights_detail)
    if credit and credit not in clip.description:
        return f"{clip.description}\n\n{credit}".strip()
    return clip.description


def queue_upload(clip_id: str, channel_ids: list[str], privacy: str, made_for_kids: bool) -> list[Upload]:
    with session_scope() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise UploadBlocked("Clip not found.")
        if clip.status != ClipStatus.ready.value or not clip.path or not Path(clip.path).exists():
            raise UploadBlocked("This clip has expired and was deleted.")
        job = session.get(Job, clip.job_id)
        if job is None or job.rights_type not in {r.value for r in RightsType}:
            raise UploadBlocked("This clip has no source rights record, so it cannot be uploaded.")
        if clip.qc.get("status") == "fail":
            raise UploadBlocked("This clip failed its quality checks. Fix or recut it first.")
        claimed_before = session.exec(
            select(Upload).where(Upload.clip_id == clip_id, Upload.status == UploadStatus.claimed.value)
        ).first()
        if claimed_before:
            raise UploadBlocked(
                "This exact clip received a copyright claim, so it will not be uploaded again. "
                "Trim the claimed part or replace the audio to make a new version."
            )
        blocking = open_claim_for_source(session, job.source_id)
        if blocking:
            raise UploadBlocked(
                "Uploads from this video are stopped by a copyright claim. Choose what to do about it first."
            )
        rows = []
        for channel_id in channel_ids:
            channel = session.get(Channel, channel_id)
            if channel is None:
                raise UploadBlocked("Channel not found.")
            if not channel.connected:
                raise UploadBlocked(f"{channel.name} is not connected.")
            active = session.exec(
                select(Upload).where(
                    Upload.clip_id == clip_id,
                    Upload.channel_id == channel_id,
                    col(Upload.status).in_(
                        [
                            UploadStatus.pending.value,
                            UploadStatus.uploading.value,
                            UploadStatus.processing.value,
                            UploadStatus.published.value,
                        ]
                    ),
                )
            ).first()
            if active:
                raise UploadBlocked(f"This clip is already on {channel.name} or on its way there.")
            rows.append(
                Upload(
                    clip_id=clip_id,
                    channel_id=channel_id,
                    privacy=privacy if channel.platform == "youtube" else "public",
                    made_for_kids=made_for_kids,
                    title=clip.title,
                    description=description_with_credit(clip, job),
                )
            )
        session.add_all(rows)
        session.commit()
        return rows


def _send(http: httpx.Client, channel: Channel, request: UploadRequest) -> RemoteState:
    if channel.platform == "youtube":
        return youtube.upload(http, channel.id, request)
    if channel.platform == "facebook":
        return meta.upload_facebook(http, channel.id, request)
    if channel.platform == "instagram":
        return meta.upload_instagram(http, channel.id, request)
    raise UploadError(f"Unknown platform: {channel.platform}")


def _check(http: httpx.Client, channel: Channel, upload: Upload) -> RemoteState:
    assert upload.remote_id
    if channel.platform == "youtube":
        return youtube.check(http, channel.id, upload.remote_id)
    if channel.platform == "facebook":
        return meta.check_facebook(http, channel.id, upload.remote_id, upload.detail)
    if channel.platform == "instagram":
        return meta.check_instagram(http, channel.id, upload.remote_id, upload.detail)
    raise UploadError(f"Unknown platform: {channel.platform}")


def apply_state(upload_id: str, state: RemoteState) -> None:
    with session_scope() as session:
        row = session.get(Upload, upload_id)
        assert row is not None
        if state.remote_id:
            row.remote_id = state.remote_id
        if state.url:
            row.url = state.url
        if state.detail:
            row.detail = state.detail
        row.error = state.error
        row.status = {
            "processing": UploadStatus.processing.value,
            "published": UploadStatus.published.value,
            "failed": UploadStatus.failed.value,
            "claimed": UploadStatus.claimed.value,
        }[state.state]
        row.updated_at = utcnow()
        session.add(row)
        session.commit()
    if state.state == "claimed":
        record_claim(upload_id, state.claim)


def record_claim(upload_id: str, claim: dict) -> None:
    """Stop: log the claim, hold every waiting upload from this source, and wait for the user."""
    with session_scope() as session:
        row = session.get(Upload, upload_id)
        assert row is not None
        row.status = UploadStatus.claimed.value
        row.claim = {**claim, "reported_at": utcnow().isoformat()}
        row.updated_at = utcnow()
        session.add(row)
        clip = session.get(Clip, row.clip_id)
        source_id = source_of(session, clip) if clip else None
        held = 0
        if source_id:
            clip_ids = clip_ids_for_source(session, source_id)
            waiting = session.exec(
                select(Upload).where(
                    col(Upload.clip_id).in_(clip_ids), Upload.status == UploadStatus.pending.value
                )
            ).all()
            for other in waiting:
                other.status = UploadStatus.stopped.value
                other.error = "Stopped because another clip from this video received a copyright claim."
                session.add(other)
                held += 1
        session.commit()
    log.warning(
        "Copyright claim on upload %s: %s. Held %d waiting uploads.", upload_id, claim.get("reason"), held
    )


def resolve_claim(upload_id: str, resolution: str, note: str) -> Upload:
    if resolution not in RESOLUTIONS:
        raise UploadBlocked("Choose trim, replace_audio or dispute.")
    with session_scope() as session:
        row = session.get(Upload, upload_id)
        if row is None or row.status != UploadStatus.claimed.value:
            raise UploadBlocked("This upload has no open claim.")
        if resolution == "dispute" and not note.strip():
            raise UploadBlocked("A dispute needs a note about the licence you hold.")
        row.claim_resolution = resolution
        row.claim_note = note.strip()
        row.updated_at = utcnow()
        session.add(row)
        session.commit()
        return row


def run_upload(upload_id: str) -> None:
    """Send one upload. Any claim or failure ends it here: no retry, and no re-edit."""
    with session_scope() as session:
        row = session.get(Upload, upload_id)
        if row is None or row.status != UploadStatus.pending.value:
            return
        clip = session.get(Clip, row.clip_id)
        channel = session.get(Channel, row.channel_id)
        if clip is None or channel is None or not clip.path:
            row.status = UploadStatus.failed.value
            row.error = "The clip or channel no longer exists."
            session.add(row)
            session.commit()
            return
        row.status = UploadStatus.uploading.value
        session.add(row)
        session.commit()
        request = UploadRequest(
            Path(clip.path),
            row.title,
            row.description,
            row.privacy,
            row.made_for_kids,
            clip.width,
            clip.height,
            clip.duration,
            clip.frames / clip.duration if clip.duration else 30.0,
            Path(clip.path).stat().st_size,
        )
    try:
        with http_client() as http:
            state = _send(http, channel, request)
            apply_state(upload_id, state)
    except (UploadError, httpx.HTTPError, OSError) as exc:
        _fail(upload_id, str(exc))
    except Exception:
        log.exception("Upload %s crashed", upload_id)
        _fail(upload_id, "The upload failed unexpectedly. See the log for details.")


def recover_interrupted_uploads() -> int:
    """After a crash, an "uploading" row would block its clip's 24-hour cleanup forever.

    It is marked failed and never resumed automatically: you decide whether to send it again.
    """
    with session_scope() as session:
        rows = session.exec(select(Upload).where(Upload.status == UploadStatus.uploading.value)).all()
        for row in rows:
            row.status = UploadStatus.failed.value
            row.error = (
                "clipforge closed during this upload. Check the platform, then send it again if needed."
            )
            row.updated_at = utcnow()
            session.add(row)
        session.commit()
        return len(rows)


def _fail(upload_id: str, message: str) -> None:
    with session_scope() as session:
        row = session.get(Upload, upload_id)
        if row is not None:
            row.status = UploadStatus.failed.value
            row.error = message[:800]
            row.updated_at = utcnow()
            session.add(row)
            session.commit()


def refresh_upload(upload_id: str) -> Upload:
    with session_scope() as session:
        row = session.get(Upload, upload_id)
        if row is None:
            raise UploadBlocked("Upload not found.")
        channel = session.get(Channel, row.channel_id)
    if row.status != UploadStatus.processing.value or not row.remote_id or channel is None:
        return row
    try:
        with http_client() as http:
            apply_state(upload_id, _check(http, channel, row))
    except httpx.HTTPError as exc:
        raise UploadError(f"Could not reach {channel.platform}: {exc}") from exc
    with session_scope() as session:
        fresh = session.get(Upload, upload_id)
        assert fresh is not None
        return fresh


class UploadWorker(threading.Thread):
    """One upload at a time. After sending, it checks status a few times, then leaves it to Refresh."""

    def __init__(self, poll_seconds: float = 2.0, status_seconds: float = POLL_SECONDS):
        super().__init__(daemon=True, name="clipforge-uploads")
        self.poll_seconds = poll_seconds
        self.status_seconds = status_seconds
        self._stop_event = threading.Event()
        self._last_status_check = 0.0

    def run(self) -> None:
        recover_interrupted_uploads()
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception:
                log.exception("Upload worker error; continuing")
            self._stop_event.wait(self.poll_seconds)

    def step(self) -> None:
        with session_scope() as session:
            pending = session.exec(
                select(Upload)
                .where(Upload.status == UploadStatus.pending.value)
                .order_by(col(Upload.created_at))
            ).first()
            pending_id = pending.id if pending else None
        if pending_id:
            run_upload(pending_id)
        if time.time() - self._last_status_check < self.status_seconds:
            return
        self._last_status_check = time.time()
        with session_scope() as session:
            processing = session.exec(
                select(Upload.id).where(Upload.status == UploadStatus.processing.value)
            ).all()
        for upload_id in processing:
            try:
                refresh_upload(upload_id)
            except UploadError as exc:
                log.warning("Status check failed for %s: %s", upload_id, exc)

    def stop(self) -> None:
        self._stop_event.set()
