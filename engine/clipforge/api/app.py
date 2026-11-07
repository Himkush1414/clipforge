"""Local API. Binds to loopback and refuses requests from other sites."""

from __future__ import annotations

import html
import logging
import secrets
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import Session, col, select, update

from clipforge import __version__, review
from clipforge.appsettings import AppSettings, load_app_settings, save_app_settings
from clipforge.captions.layout import cell_ratio
from clipforge.captions.presets import BUILTIN_PRESETS, FONT_FILES, FONTS_DIR, CaptionPreset
from clipforge.captions.preview import PREVIEW_SIZES, render_preview, sample_words
from clipforge.cleanup import CleanupThread, cleanup_expired_clips, remove_clip_files
from clipforge.clip_files import move_clip
from clipforge.config import get_settings
from clipforge.db import get_session, session_scope
from clipforge.export.render import replace_audio
from clipforge.folders import (
    FolderError,
    LowDisk,
    low_disk_message,
    saved_dir,
    storage_report,
    validate_saved_dir,
    windows_to_wsl,
)
from clipforge.hardware import detect
from clipforge.jobs import Worker
from clipforge.media.ffmpeg import FFmpegError
from clipforge.media.probe import probe
from clipforge.models import (
    Channel,
    Clip,
    ClipStatus,
    Job,
    JobStatus,
    Preset,
    Source,
    Upload,
    UploadStatus,
    new_id,
    utcnow,
)
from clipforge.pipeline import run_job
from clipforge.qc import run_qc
from clipforge.schemas import (
    BulkClipAction,
    BulkUpload,
    ChannelCreate,
    ChannelSettings,
    ClaimReport,
    ClaimResolve,
    ClipMove,
    FrameCut,
    JobCreate,
    PlatformCredentials,
    TargetChoice,
    UploadCreate,
)
from clipforge.uploads import meta, service, youtube
from clipforge.uploads.base import (
    UploadError,
    delete_platform_credentials,
    delete_tokens,
    load_tokens,
    platform_credentials,
    redirect_uri,
    save_platform_credentials,
    save_tokens,
)

log = logging.getLogger("clipforge.api")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".m4v",
    ".ts",
    ".mts",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
DISK_RESERVE_BYTES = 2 * 1024**3


def seed_presets() -> None:
    with session_scope() as session:
        for key, preset in BUILTIN_PRESETS.items():
            row = session.get(Preset, key)
            if row is None:
                session.add(Preset(id=key, name=preset.name, builtin=True, data=preset.model_dump()))
            else:
                row.name, row.data = preset.name, preset.model_dump()
                session.add(row)
        session.commit()


def source_json(source: Source) -> dict:
    return {
        "id": source.id,
        "filename": source.filename,
        "owned_copy": source.owned_copy,
        "probe": source.probe,
        "created_at": source.created_at,
    }


def clip_json(clip: Clip, job: Job | None = None) -> dict:
    left = None if clip.expires_at is None else round(max(0.0, (clip.expires_at - utcnow()).total_seconds()))
    return {
        **clip.model_dump(exclude={"path", "thumb_path"}),
        "seconds_left": left,
        "rights_type": job.rights_type if job else None,
        "rights_detail": job.rights_detail if job else None,
        "file_url": f"/api/clips/{clip.id}/file" if clip.path else None,
        "thumb_url": f"/api/clips/{clip.id}/thumb" if clip.thumb_path else None,
    }


def job_json(job: Job, clip_count: int = 0) -> dict:
    return {**job.model_dump(), "clip_count": clip_count}


def require(session: Session, model, item_id: str, label: str):
    row = session.get(model, item_id)
    if row is None:
        raise HTTPException(404, f"{label} not found")
    return row


class ImportRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class ReviewCut(BaseModel):
    """Unchecked here, so the review step can explain a bad edit in plain words."""

    start_frame: int
    end_frame: int


class ReviewReject(BaseModel):
    rejected: bool = True


class ClipPatch(BaseModel):
    title: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=5000)


class PresetBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    data: CaptionPreset


class PreviewRequest(BaseModel):
    preset: CaptionPreset
    ratio: str = "9:16"
    at: float | None = Field(default=0.5, ge=0, le=60)
    text: str | None = Field(default=None, max_length=200)


class LinkRequest(BaseModel):
    channel_ids: list[str] = Field(min_length=1, max_length=20)
    link_group: str | None = Field(default=None, max_length=80)


OAUTH_STATE_SECONDS = 900
OAUTH_PAGE = (
    "<!doctype html><meta charset=utf-8><title>clipforge</title>"
    "<body style='font-family:system-ui;background:#0c0d10;color:#eceef2;padding:3rem'>"
    "<h1>{title}</h1><p>{message}</p><p>You can close this tab.</p>"
)


def create_app() -> FastAPI:
    settings = get_settings()
    oauth_states: dict[str, dict] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        seed_presets()
        worker, cleaner, uploader = None, None, None
        if settings.start_background:
            # Encoder tests take seconds, so run them now and not during the first page load.
            threading.Thread(target=detect, daemon=True).start()
            worker, cleaner, uploader = Worker(run_job), CleanupThread(), service.UploadWorker()
            worker.start()
            cleaner.start()
            uploader.start()
        yield
        if worker and cleaner and uploader:
            worker.stop()
            cleaner.stop()
            uploader.stop()

    app = FastAPI(title="clipforge", version=__version__, lifespan=lifespan)
    allowed_hosts = {"127.0.0.1", "localhost", "testserver"}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # A web page you visit could otherwise send requests to this API (CSRF, DNS rebinding).
        host = request.headers.get("host", "").rsplit(":", 1)[0]
        origin = request.headers.get("origin")
        if host not in allowed_hosts:
            return JSONResponse({"detail": "Host not allowed"}, status_code=403)
        if origin and origin not in settings.web_origins:
            return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        try:
            return await call_next(request)
        except Exception:
            # Answered here, inside CORS, so the browser sees this message and not a network error.
            log.exception("Unhandled error on %s %s", request.method, request.url.path)
            return JSONResponse({"detail": "Something went wrong. See the log for details."}, status_code=500)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.web_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LowDisk)
    async def low_disk_error(_request: Request, exc: LowDisk):
        return JSONResponse({"detail": str(exc)}, status_code=507)

    @app.exception_handler(FolderError)
    async def folder_error(_request: Request, exc: FolderError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(UploadError)
    async def upload_error(_request: Request, exc: UploadError):
        return JSONResponse({"detail": str(exc)[:800]}, status_code=409)

    @app.exception_handler(FFmpegError)
    async def ffmpeg_error(_request: Request, exc: FFmpegError):
        return JSONResponse({"detail": str(exc)[:500]}, status_code=422)

    # ---------- system ----------
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/hardware")
    def hardware() -> dict:
        return detect().to_dict()

    @app.get("/api/storage")
    def storage() -> dict:
        return storage_report()

    @app.get("/api/settings")
    def read_settings() -> dict:
        values = load_app_settings()
        return {**values.model_dump(), "saved_path": str(saved_dir())}

    @app.put("/api/settings")
    def write_settings(body: AppSettings) -> dict:
        update = body.model_copy()
        if body.saved_dir and body.saved_dir.strip():
            # Clips already in Saved stay where they are. Only new moves use the new path.
            update.saved_dir = str(validate_saved_dir(body.saved_dir))
        else:
            update.saved_dir = None
        save_app_settings(update)
        return {**update.model_dump(), "saved_path": str(saved_dir())}

    @app.post("/api/cleanup/run")
    def run_cleanup() -> dict:
        return {"removed": cleanup_expired_clips()}

    # ---------- sources ----------
    @app.put("/api/sources/upload")
    async def upload_source(request: Request, filename: str) -> dict:
        """Raw body streamed straight to disk: multipart would copy huge files twice."""
        name = PurePosixPath(filename.replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type: {suffix or 'none'}")
        length = int(request.headers.get("content-length") or 0)
        if length and shutil.disk_usage(settings.work_dir).free < length + DISK_RESERVE_BYTES:
            raise HTTPException(507, "Not enough free disk space for this video.")
        source_id = new_id()
        target = settings.media_dir / f"{source_id}{suffix}"
        settings.media_dir.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
            info = probe(target)
        except FFmpegError as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"This file is not a playable video: {str(exc)[:200]}") from exc
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        with session_scope() as session:
            source = Source(
                id=source_id, filename=name, path=str(target), owned_copy=True, probe=info.to_dict()
            )
            session.add(source)
            session.commit()
            return source_json(source)

    @app.post("/api/sources/import")
    def import_source(body: ImportRequest, session: Session = Depends(get_session)) -> dict:
        """Use a file where it already is, without copying. It is never deleted by clipforge."""
        path = windows_to_wsl(body.path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(415, "That path is not a video file.")
        if not path.is_file():
            raise HTTPException(404, f"File not found: {path}")
        info = probe(path)
        source = Source(filename=path.name, path=str(path), owned_copy=False, probe=info.to_dict())
        session.add(source)
        session.commit()
        return source_json(source)

    @app.get("/api/sources")
    def list_sources(session: Session = Depends(get_session)) -> list[dict]:
        rows = session.exec(select(Source).order_by(col(Source.created_at).desc())).all()
        return [source_json(s) for s in rows if s.path]

    @app.delete("/api/sources/{source_id}")
    def delete_source(source_id: str, session: Session = Depends(get_session)) -> dict:
        source = require(session, Source, source_id, "Source")
        active = session.exec(
            select(Job).where(
                Job.source_id == source_id,
                col(Job.status).in_(
                    [JobStatus.queued.value, JobStatus.running.value, JobStatus.review.value]
                ),
            )
        ).first()
        if active:
            raise HTTPException(409, "A job is still using this video.")
        if source.owned_copy and source.path:
            Path(source.path).unlink(missing_ok=True)
        shutil.rmtree(settings.analysis_dir / source_id, ignore_errors=True)
        source.path = ""  # The row stays because jobs and their rights records refer to it.
        session.add(source)
        session.commit()
        return {"deleted": source_id}

    # ---------- jobs ----------
    @app.post("/api/jobs", status_code=201)
    def create_job(body: JobCreate, session: Session = Depends(get_session)) -> dict:
        source = require(session, Source, body.source_id, "Source")
        if not source.path:
            raise HTTPException(410, "That video was deleted.")
        if body.channel_id:
            require(session, Channel, body.channel_id, "Channel")
        if body.mode == "auto" and body.frame_cuts:
            raise HTTPException(422, "Frame cuts need manual mode.")
        if body.manual and body.manual.resolution not in detect().allowed_resolutions:
            raise HTTPException(422, f"This machine cannot export {body.manual.resolution}.")
        if body.edit.preset_id and session.get(Preset, body.edit.preset_id) is None:
            raise HTTPException(422, "Caption preset not found.")
        low = low_disk_message()
        if low:
            raise HTTPException(507, low)
        job = Job(
            source_id=body.source_id,
            channel_id=body.channel_id,
            mode=body.mode,
            rights_type=body.rights.type.value,
            rights_detail=body.rights.detail,
            settings={
                **body.model_dump(mode="json", include={"manual", "edit", "frame_cuts", "destination"}),
                "wants_review": body.review,
            },
        )
        session.add(job)
        session.commit()
        return job_json(job)

    @app.get("/api/jobs")
    def list_jobs(session: Session = Depends(get_session)) -> list[dict]:
        rows = session.exec(select(Job).order_by(col(Job.created_at).desc()).limit(100)).all()
        counts = dict(session.exec(select(Clip.job_id, func.count()).group_by(Clip.job_id)).all())
        names = dict(session.exec(select(Source.id, Source.filename)).all())
        return [
            {**job_json(job, counts.get(job.id, 0)), "source_filename": names.get(job.source_id)}
            for job in rows
        ]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, session: Session = Depends(get_session)) -> dict:
        job = require(session, Job, job_id, "Job")
        clips = session.exec(select(Clip).where(Clip.job_id == job_id)).all()
        return {**job_json(job, len(clips)), "clips": [clip_json(c, job) for c in clips]}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, session: Session = Depends(get_session)) -> dict:
        require(session, Job, job_id, "Job")
        # Atomic, because the worker may be claiming this same job right now.
        queued = session.exec(
            update(Job)
            .where(
                col(Job.id) == job_id, col(Job.status).in_([JobStatus.queued.value, JobStatus.review.value])
            )
            .values(status=JobStatus.cancelled.value, stage="cancelled", updated_at=utcnow())
        )
        if queued.rowcount == 0:
            running = session.exec(
                update(Job)
                .where(col(Job.id) == job_id, col(Job.status) == JobStatus.running.value)
                .values(cancel_requested=True, updated_at=utcnow())
            )
            if running.rowcount == 0:
                session.rollback()
                current = require(session, Job, job_id, "Job")
                raise HTTPException(409, f"Job is already {current.status}.")
        session.commit()
        session.expire_all()
        return job_json(require(session, Job, job_id, "Job"))

    # ---------- review ----------
    def job_in_review(session: Session, job_id: str) -> Job:
        job = require(session, Job, job_id, "Job")
        if job.status != JobStatus.review.value or not job.settings.get("review"):
            raise HTTPException(409, f"This job is not waiting for review. It is {job.status}.")
        return job

    def save_plan(session: Session, job: Job, plan: dict) -> None:
        job.settings = {**job.settings, "review": plan}  # A new object, so the JSON column is saved.
        job.updated_at = utcnow()
        session.add(job)
        session.commit()

    def review_json(session: Session, job: Job) -> dict:
        plan = job.settings["review"]
        source = session.get(Source, job.source_id)
        return {
            "job_id": job.id,
            "mode": job.mode,
            "status": job.status,
            "fps": plan["fps"],
            "frame_count": plan["frame_count"],
            "options": plan["options"],
            "clips": plan["clips"],
            "alternates_left": len(plan["alternates"]),
            "source_url": f"/api/sources/{job.source_id}/file" if source and source.path else None,
        }

    @app.get("/api/sources/{source_id}/file")
    def source_file(source_id: str, session: Session = Depends(get_session)):
        source = require(session, Source, source_id, "Source")
        if not source.path or not Path(source.path).exists():
            raise HTTPException(410, "That video was deleted.")
        return FileResponse(source.path)

    @app.get("/api/jobs/{job_id}/review")
    def get_review(job_id: str, session: Session = Depends(get_session)) -> dict:
        return review_json(session, job_in_review(session, job_id))

    @app.patch("/api/jobs/{job_id}/review/{key}")
    def adjust_review_clip(
        job_id: str, key: str, cut: ReviewCut, session: Session = Depends(get_session)
    ) -> dict:
        job = job_in_review(session, job_id)
        plan = deepcopy(job.settings["review"])
        try:
            review.adjust(plan, key, cut.start_frame, cut.end_frame, review.words_of_job(job))
        except review.ReviewError as exc:
            raise HTTPException(422, str(exc)) from exc
        save_plan(session, job, plan)
        return review_json(session, job)

    @app.post("/api/jobs/{job_id}/review/{key}/reject")
    def reject_review_clip(
        job_id: str, key: str, body: ReviewReject, session: Session = Depends(get_session)
    ) -> dict:
        job = job_in_review(session, job_id)
        plan = deepcopy(job.settings["review"])
        try:
            review.find_clip(plan, key)["rejected"] = body.rejected
        except review.ReviewError as exc:
            raise HTTPException(404, str(exc)) from exc
        save_plan(session, job, plan)
        return review_json(session, job)

    @app.post("/api/jobs/{job_id}/review/{key}/swap")
    def swap_review_clip(job_id: str, key: str, session: Session = Depends(get_session)) -> dict:
        job = job_in_review(session, job_id)
        plan = deepcopy(job.settings["review"])
        try:
            review.swap(plan, key)
        except review.ReviewError as exc:
            raise HTTPException(409, str(exc)) from exc
        save_plan(session, job, plan)
        return review_json(session, job)

    @app.post("/api/jobs/{job_id}/review/approve")
    def approve_review(job_id: str, session: Session = Depends(get_session)) -> dict:
        """Export what the review kept, through the normal render path."""
        job = job_in_review(session, job_id)
        low = low_disk_message()
        if low:
            raise HTTPException(507, low)
        try:
            approved = review.approved_clips(job.settings["review"])
        except review.ReviewError as exc:
            raise HTTPException(422, str(exc)) from exc
        job.settings = {**job.settings, "approved": approved}
        job.status = JobStatus.queued.value
        job.stage = "queued"
        job.updated_at = utcnow()
        session.add(job)
        session.commit()
        return job_json(job)

    # ---------- clips ----------
    def job_of(session: Session, clip: Clip) -> Job | None:
        return session.get(Job, clip.job_id)

    @app.get("/api/clips")
    def list_clips(
        include_expired: bool = False, folder: str | None = None, session: Session = Depends(get_session)
    ) -> list[dict]:
        query = select(Clip).order_by(col(Clip.created_at).desc())
        if not include_expired:
            query = query.where(Clip.status == ClipStatus.ready.value)
        if folder:
            query = query.where(Clip.folder == folder)
        rows = session.exec(query.limit(500)).all()
        jobs = {
            j.id: j for j in session.exec(select(Job).where(col(Job.id).in_({c.job_id for c in rows}))).all()
        }
        return [clip_json(c, jobs.get(c.job_id)) for c in rows]

    @app.get("/api/clips/{clip_id}")
    def get_clip(clip_id: str, session: Session = Depends(get_session)) -> dict:
        clip = require(session, Clip, clip_id, "Clip")
        return clip_json(clip, job_of(session, clip))

    @app.patch("/api/clips/{clip_id}")
    def patch_clip(clip_id: str, body: ClipPatch, session: Session = Depends(get_session)) -> dict:
        clip = require(session, Clip, clip_id, "Clip")
        if body.title is not None:
            clip.title = body.title
        if body.description is not None:
            clip.description = body.description
        session.add(clip)
        session.commit()
        return clip_json(clip, job_of(session, clip))

    @app.post("/api/clips/{clip_id}/move")
    def move_one_clip(clip_id: str, body: ClipMove, session: Session = Depends(get_session)) -> dict:
        moved = move_clip(clip_id, body.folder)
        return clip_json(moved, job_of(session, moved))

    def delete_one(session: Session, clip_id: str) -> None:
        clip = require(session, Clip, clip_id, "Clip")
        uploading = session.exec(
            select(Upload).where(Upload.clip_id == clip_id, Upload.status == UploadStatus.uploading.value)
        ).first()
        if uploading:
            raise FolderError("This clip is uploading. Delete it when the upload has finished.")
        remove_clip_files(clip)
        clip.status = ClipStatus.deleted.value
        session.add(clip)
        session.commit()

    @app.post("/api/clips/bulk")
    def bulk_clips(body: BulkClipAction, session: Session = Depends(get_session)) -> dict:
        """Each clip succeeds or fails on its own, so one problem cannot hide the others."""
        done, failed = [], []
        for clip_id in dict.fromkeys(body.clip_ids):
            try:
                if body.action == "delete":
                    delete_one(session, clip_id)
                else:
                    move_clip(clip_id, "saved" if body.action == "move_saved" else "queued")
                done.append(clip_id)
            except (FolderError, HTTPException, OSError) as exc:
                failed.append({"clip_id": clip_id, "error": str(getattr(exc, "detail", exc))[:300]})
        return {"done": done, "failed": failed}

    @app.post("/api/uploads/bulk", status_code=201)
    def bulk_upload(body: BulkUpload, session: Session = Depends(get_session)) -> dict:
        if body.link_group:
            channel_ids = list(
                session.exec(select(Channel.id).where(Channel.link_group == body.link_group)).all()
            )
            if not channel_ids:
                raise HTTPException(404, "No channels in that group.")
        else:
            assert body.channel_id
            channel_ids = [body.channel_id]
        done, failed = [], []
        for clip_id in dict.fromkeys(body.clip_ids):
            try:
                service.queue_upload(clip_id, channel_ids, body.privacy, body.made_for_kids)
                done.append(clip_id)
            except UploadError as exc:
                failed.append({"clip_id": clip_id, "error": str(exc)[:300]})
        return {"done": done, "failed": failed}

    @app.get("/api/clips/{clip_id}/file")
    def clip_file(clip_id: str, download: bool = False, session: Session = Depends(get_session)):
        clip = require(session, Clip, clip_id, "Clip")
        if not clip.path or not Path(clip.path).exists():
            raise HTTPException(410, "This clip has expired and was deleted.")
        name = f"{clip.id}.mp4" if download else None
        return FileResponse(clip.path, media_type="video/mp4", filename=name)

    @app.get("/api/clips/{clip_id}/thumb")
    def clip_thumb(clip_id: str, session: Session = Depends(get_session)):
        clip = require(session, Clip, clip_id, "Clip")
        if not clip.thumb_path or not Path(clip.thumb_path).exists():
            raise HTTPException(410, "Thumbnail deleted.")
        return FileResponse(clip.thumb_path, media_type="image/jpeg")

    @app.delete("/api/clips/{clip_id}")
    def delete_clip(clip_id: str, session: Session = Depends(get_session)) -> dict:
        delete_one(session, clip_id)
        return {"deleted": clip_id}

    @app.post("/api/clips/{clip_id}/recut", status_code=201)
    def recut_clip(clip_id: str, cut: FrameCut, session: Session = Depends(get_session)) -> dict:
        """Re-render one clip from exact frames, with the same settings and rights record."""
        clip = require(session, Clip, clip_id, "Clip")
        old = require(session, Job, clip.job_id, "Job")
        options = old.settings.get("manual") or {
            "genre": old.decisions.get("genre", "other"),
            "clip_duration": clip.duration,
            "clip_count": 1,
            "part_numbering": False,
            "format": clip.format,
            "ratio": clip.ratio,
            "resolution": clip.resolution,
        }
        job = Job(
            source_id=old.source_id,
            channel_id=old.channel_id,
            mode="manual",
            rights_type=old.rights_type,
            rights_detail=old.rights_detail,
            settings={
                "manual": {
                    **options,
                    "ratio": clip.ratio,
                    "resolution": clip.resolution,
                    "format": clip.format,
                },
                "edit": old.settings.get("edit", {}),
                "frame_cuts": [cut.model_dump()],
                "destination": "queued" if clip.folder == "queued" else "unassigned",
            },
        )
        session.add(job)
        session.commit()
        return job_json(job)

    # ---------- presets ----------
    @app.get("/api/presets")
    def list_presets(session: Session = Depends(get_session)) -> list[dict]:
        rows = session.exec(select(Preset).order_by(col(Preset.builtin).desc(), Preset.name)).all()
        return [r.model_dump() for r in rows]

    @app.post("/api/presets", status_code=201)
    def create_preset(body: PresetBody, session: Session = Depends(get_session)) -> dict:
        row = Preset(name=body.name, data=body.data.model_copy(update={"name": body.name}).model_dump())
        session.add(row)
        session.commit()
        return row.model_dump()

    @app.put("/api/presets/{preset_id}")
    def update_preset(preset_id: str, body: PresetBody, session: Session = Depends(get_session)) -> dict:
        row = require(session, Preset, preset_id, "Preset")
        if row.builtin:
            raise HTTPException(409, "Built-in presets cannot be changed. Save a copy instead.")
        row.name, row.data = body.name, body.data.model_copy(update={"name": body.name}).model_dump()
        session.add(row)
        session.commit()
        return row.model_dump()

    @app.delete("/api/presets/{preset_id}")
    def delete_preset(preset_id: str, session: Session = Depends(get_session)) -> dict:
        row = require(session, Preset, preset_id, "Preset")
        if row.builtin:
            raise HTTPException(409, "Built-in presets cannot be deleted.")
        session.delete(row)
        session.commit()
        return {"deleted": preset_id}

    @app.post("/api/presets/preview")
    def preview_preset(body: PreviewRequest, tasks: BackgroundTasks):
        """The real libass render, so the preview matches the export exactly."""
        if body.ratio not in PREVIEW_SIZES:
            raise HTTPException(422, "Unknown ratio")
        settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="preview-out-", dir=settings.tmp_dir))
        tasks.add_task(shutil.rmtree, folder, ignore_errors=True)
        suffix = ".png" if body.at is not None else ".mp4"
        words = sample_words(body.text) if body.text and body.text.strip() else None
        out = render_preview(body.preset, folder / f"preview{suffix}", body.ratio, body.at, words)
        return FileResponse(out, media_type="image/png" if suffix == ".png" else "video/mp4")

    @app.get("/api/fonts")
    def list_fonts() -> list[dict]:
        return [
            {"family": family, "file": file, "em_ratio": round(cell_ratio(family)[0], 4)}
            for family, file in FONT_FILES.items()
        ]

    @app.get("/api/fonts/{filename}")
    def font_file(filename: str):
        if filename not in FONT_FILES.values():  # Only bundled names, so no path can be smuggled in.
            raise HTTPException(404, "Font not found")
        return FileResponse(FONTS_DIR / filename, media_type="font/ttf")

    # ---------- channels ----------
    @app.get("/api/channels")
    def list_channels(session: Session = Depends(get_session)) -> list[dict]:
        return [c.model_dump() for c in session.exec(select(Channel).order_by(Channel.name)).all()]

    @app.post("/api/channels", status_code=201)
    def create_channel(body: ChannelCreate, session: Session = Depends(get_session)) -> dict:
        row = Channel(
            name=body.name,
            platform=body.platform,
            link_group=body.link_group,
            settings=body.settings.model_dump(),
        )
        session.add(row)
        session.commit()
        return row.model_dump()

    @app.put("/api/channels/{channel_id}/settings")
    def update_channel(
        channel_id: str, body: ChannelSettings, session: Session = Depends(get_session)
    ) -> dict:
        row = require(session, Channel, channel_id, "Channel")
        row.settings = body.model_dump()
        session.add(row)
        session.commit()
        return row.model_dump()

    @app.post("/api/channels/link")
    def link_channels(body: LinkRequest, session: Session = Depends(get_session)) -> list[dict]:
        rows = [require(session, Channel, cid, "Channel") for cid in body.channel_ids]
        group = body.link_group or new_id()
        for row in rows:
            row.link_group = group
            session.add(row)
        session.commit()
        return [r.model_dump() for r in rows]

    @app.delete("/api/channels/{channel_id}")
    def delete_channel(channel_id: str, session: Session = Depends(get_session)) -> dict:
        row = require(session, Channel, channel_id, "Channel")
        session.delete(row)
        session.commit()
        return {"deleted": channel_id}

    # ---------- platform connections ----------
    @app.get("/api/platforms")
    def list_platforms() -> list[dict]:
        """Secrets are write-only: this says whether they exist, and never returns them."""
        return [
            {
                "group": group,
                "configured": platform_credentials(group) is not None,
                "redirect_uri": redirect_uri(),
            }
            for group in ("google", "meta")
        ]

    @app.put("/api/platforms/{group}/credentials")
    def set_credentials(group: Literal["google", "meta"], body: PlatformCredentials) -> dict:
        save_platform_credentials(group, body.client_id.strip(), body.client_secret.strip())
        return {"group": group, "configured": True}

    @app.delete("/api/platforms/{group}/credentials")
    def remove_credentials(group: Literal["google", "meta"]) -> dict:
        delete_platform_credentials(group)
        return {"group": group, "configured": False}

    @app.post("/api/channels/{channel_id}/connect")
    def connect_channel(channel_id: str, session: Session = Depends(get_session)) -> dict:
        channel = require(session, Channel, channel_id, "Channel")
        now = time.time()
        for key in [k for k, v in oauth_states.items() if now - v["created"] > OAUTH_STATE_SECONDS]:
            del oauth_states[key]
        state = secrets.token_urlsafe(32)
        if channel.platform == "youtube":
            url, verifier = youtube.begin_auth(state)
        else:
            url, verifier = meta.begin_auth(channel.platform, state)
        oauth_states[state] = {"channel_id": channel_id, "verifier": verifier, "created": now}
        return {"auth_url": url}

    @app.get("/api/oauth/callback", response_class=HTMLResponse)
    def oauth_callback(
        state: str = "", code: str = "", error: str = "", session: Session = Depends(get_session)
    ) -> HTMLResponse:
        pending = oauth_states.pop(state, None)
        if pending is None or time.time() - pending["created"] > OAUTH_STATE_SECONDS:
            page = OAUTH_PAGE.format(
                title="Sign-in link expired", message="Start again from the Channels page."
            )
            return HTMLResponse(page, status_code=400)
        if error or not code:
            message = html.escape(error or "No authorization code came back.")
            return HTMLResponse(
                OAUTH_PAGE.format(title="Sign-in cancelled", message=message), status_code=400
            )
        channel = require(session, Channel, pending["channel_id"], "Channel")
        try:
            with service.http_client() as http:
                if channel.platform == "youtube":
                    name = youtube.finish_auth(http, channel.id, code, pending["verifier"])
                else:
                    name = meta.finish_auth(http, channel.platform, channel.id, code)
        except (UploadError, httpx.HTTPError, KeyError) as exc:
            delete_tokens(channel.id)
            log.warning("Sign-in failed for channel %s: %s", channel.id, type(exc).__name__)
            message = (
                html.escape(str(exc)[:300])
                if isinstance(exc, UploadError)
                else "The platform's reply was not usable."
            )
            return HTMLResponse(OAUTH_PAGE.format(title="Sign-in failed", message=message), status_code=400)
        channel.connected = True
        channel.account_name = name
        session.add(channel)
        session.commit()
        return HTMLResponse(
            OAUTH_PAGE.format(title="Connected", message=f"{html.escape(name)} is ready to use.")
        )

    @app.post("/api/channels/{channel_id}/disconnect")
    def disconnect_channel(channel_id: str, session: Session = Depends(get_session)) -> dict:
        channel = require(session, Channel, channel_id, "Channel")
        delete_tokens(channel_id)
        channel.connected = False
        channel.account_name = None
        session.add(channel)
        session.commit()
        return channel.model_dump()

    @app.get("/api/channels/{channel_id}/targets")
    def channel_targets(channel_id: str, session: Session = Depends(get_session)) -> dict:
        require(session, Channel, channel_id, "Channel")
        tokens = load_tokens(channel_id) or {}
        return {
            "selected": tokens.get("target"),
            "targets": [{"id": t["id"], "name": t["name"]} for t in tokens.get("targets", [])],
        }

    @app.post("/api/channels/{channel_id}/target")
    def choose_target(channel_id: str, body: TargetChoice, session: Session = Depends(get_session)) -> dict:
        channel = require(session, Channel, channel_id, "Channel")
        tokens = load_tokens(channel_id) or {}
        target = next((t for t in tokens.get("targets", []) if t["id"] == body.target_id), None)
        if target is None:
            raise HTTPException(404, "That account is not available on this connection.")
        tokens["target"] = target["id"]
        save_tokens(channel_id, tokens)
        channel.account_name = target["name"]
        session.add(channel)
        session.commit()
        return channel.model_dump()

    # ---------- uploads ----------
    @app.post("/api/clips/{clip_id}/upload", status_code=201)
    def create_upload(
        clip_id: str, body: UploadCreate, session: Session = Depends(get_session)
    ) -> list[dict]:
        if body.link_group:
            rows = session.exec(select(Channel.id).where(Channel.link_group == body.link_group)).all()
            channel_ids = list(rows)
            if not channel_ids:
                raise HTTPException(404, "No channels in that group.")
        else:
            assert body.channel_id
            channel_ids = [body.channel_id]
        return [
            u.model_dump()
            for u in service.queue_upload(clip_id, channel_ids, body.privacy, body.made_for_kids)
        ]

    @app.get("/api/uploads")
    def list_uploads(session: Session = Depends(get_session)) -> list[dict]:
        rows = session.exec(select(Upload).order_by(col(Upload.created_at).desc()).limit(200)).all()
        names = {c.id: c for c in session.exec(select(Channel)).all()}
        return [
            {
                **u.model_dump(exclude={"detail"}),
                "channel_name": names[u.channel_id].name if u.channel_id in names else None,
                "platform": names[u.channel_id].platform if u.channel_id in names else None,
            }
            for u in rows
        ]

    @app.post("/api/uploads/{upload_id}/refresh")
    def refresh_upload(upload_id: str) -> dict:
        return service.refresh_upload(upload_id).model_dump(exclude={"detail"})

    @app.post("/api/uploads/{upload_id}/report-claim")
    def report_claim(upload_id: str, body: ClaimReport, session: Session = Depends(get_session)) -> dict:
        """Some claims are invisible to the API (Instagram, or YouTube when the video stays live)."""
        require(session, Upload, upload_id, "Upload")
        service.record_claim(
            upload_id, {"platform": "manual", "reason": "reported by you", "note": body.details}
        )
        session.expire_all()
        return require(session, Upload, upload_id, "Upload").model_dump(exclude={"detail"})

    @app.post("/api/uploads/{upload_id}/resolve")
    def resolve_upload(upload_id: str, body: ClaimResolve) -> dict:
        return service.resolve_claim(upload_id, body.resolution, body.note).model_dump(exclude={"detail"})

    @app.put("/api/clips/{clip_id}/replace-audio", status_code=201)
    async def replace_clip_audio(
        clip_id: str, request: Request, filename: str, licence: str, session: Session = Depends(get_session)
    ) -> dict:
        """Only after a claim, and only with a track the user states a licence for. Makes a new clip."""
        clip = require(session, Clip, clip_id, "Clip")
        if not licence.strip():
            raise HTTPException(422, "Say what licence you hold for this audio.")
        if not clip.path or not Path(clip.path).exists():
            raise HTTPException(410, "This clip has expired and was deleted.")
        claimed = session.exec(
            select(Upload).where(Upload.clip_id == clip_id, Upload.status == UploadStatus.claimed.value)
        ).first()
        if claimed is None:
            raise HTTPException(409, "Replacing audio is offered only after a copyright claim on this clip.")
        suffix = Path(PurePosixPath(filename.replace("\\", "/")).name).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise HTTPException(415, "Use an audio file such as WAV, MP3, M4A, FLAC or AAC.")
        settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="audio-", dir=settings.tmp_dir))
        try:
            audio = folder / f"track{suffix}"
            with audio.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
            new_id_value = new_id()
            output = Path(clip.path).with_name(f"{new_id_value}.mp4")
            await run_in_threadpool(replace_audio, Path(clip.path), audio, output, clip.duration)
            # The audio changed, so the old quality results no longer describe this file.
            quality = await run_in_threadpool(
                run_qc, output, clip.width, clip.height, clip.duration, clip.frames / clip.duration, True
            )
            thumb = None
            if clip.thumb_path and Path(clip.thumb_path).exists():
                # Its own copy: each clip's cleanup deletes its own thumbnail.
                thumb = output.with_suffix(".jpg")
                shutil.copyfile(clip.thumb_path, thumb)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        now = utcnow()
        fresh = Clip(
            **clip.model_dump(
                exclude={
                    "id",
                    "path",
                    "thumb_path",
                    "qc",
                    "size_bytes",
                    "description",
                    "reasons",
                    "created_at",
                    "expires_at",
                }
            ),  # fmt: skip
            id=new_id_value,
            thumb_path=str(thumb) if thumb else None,
            qc={**quality, "encoder": "video copied"},
            path=str(output),
            size_bytes=output.stat().st_size,
            description=f"{clip.description}\n\nAudio: {licence.strip()}",
            reasons={**clip.reasons, "replaced_audio_licence": licence.strip(), "from_clip": clip_id},
            created_at=now,
            expires_at=now + timedelta(hours=settings.clip_ttl_hours)
            if clip.folder == "unassigned"
            else None,
        )
        session.add(fresh)
        session.commit()
        return clip_json(fresh, job_of(session, fresh))

    return app


def setup_logging() -> None:
    settings = get_settings()
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_dir / "clipforge.log", maxBytes=5_000_000, backupCount=3),
        ],
    )
    # Every deletion and refused deletion gets its own permanent record as well.
    deletion_log = RotatingFileHandler(log_dir / "deletions.log", maxBytes=5_000_000, backupCount=5)
    deletion_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("clipforge.deletions").addHandler(deletion_log)
    # httpx logs full request URLs at INFO, and sign-in URLs must never reach a log file.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    import uvicorn

    setup_logging()
    settings = get_settings()
    uvicorn.run(create_app(), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
