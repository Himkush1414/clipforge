"""SQLite models. All timestamps are stored in UTC."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class RightsType(StrEnum):
    own = "own"
    public_domain = "public_domain"
    creative_commons = "creative_commons"
    written_permission = "written_permission"


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class ClipStatus(StrEnum):
    ready = "ready"
    expired = "expired"
    deleted = "deleted"


class UploadStatus(StrEnum):
    pending = "pending"
    uploading = "uploading"
    processing = "processing"
    published = "published"
    failed = "failed"
    claimed = "claimed"


class Source(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    filename: str
    path: str
    owned_copy: bool = True  # False for imported paths: never delete the user's original file.
    probe: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    source_id: str = Field(foreign_key="source.id", index=True)
    channel_id: str | None = Field(default=None)
    mode: str = "manual"
    status: str = Field(default=JobStatus.queued.value, index=True)
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    rights_type: str
    rights_detail: str = ""
    settings: dict = Field(default_factory=dict, sa_column=Column(JSON))
    decisions: dict = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    cancel_requested: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Clip(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    part: int | None = None
    start: float
    end: float
    segments: list = Field(default_factory=list, sa_column=Column(JSON))
    score: float = 0.0
    reasons: dict = Field(default_factory=dict, sa_column=Column(JSON))
    title: str = ""
    description: str = ""
    format: str = "short"
    ratio: str = "9:16"
    resolution: str = "1080p"
    width: int = 0
    height: int = 0
    duration: float = 0.0
    frames: int = 0
    size_bytes: int = 0
    path: str | None = None
    thumb_path: str | None = None
    preset_id: str | None = None
    qc: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default=ClipStatus.ready.value, index=True)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime = Field(index=True)


class Preset(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    builtin: bool = False
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class Channel(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    platform: str
    link_group: str | None = Field(default=None, index=True)
    settings: dict = Field(default_factory=dict, sa_column=Column(JSON))
    connected: bool = False
    account_name: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Upload(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    clip_id: str = Field(index=True)
    channel_id: str = Field(index=True)
    status: str = UploadStatus.pending.value
    privacy: str = "private"
    title: str = ""
    description: str = ""
    remote_id: str | None = None
    url: str | None = None
    error: str | None = None
    claim: dict = Field(default_factory=dict, sa_column=Column(JSON))
    claim_resolution: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
