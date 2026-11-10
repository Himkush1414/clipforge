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
    # Footage with no upload rights, used only to try the app. Its clips can never be queued or uploaded.
    private_test = "private_test"


def uploadable(rights_type: str | None) -> bool:
    """Whether clips from a job with these rights may ever be queued or uploaded."""
    return rights_type in {r.value for r in RightsType} and rights_type != RightsType.private_test.value


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    review = "review"  # Waiting for a person to approve the clips. Nothing has been rendered yet.
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class ClipFolder(StrEnum):
    unassigned = "unassigned"  # Deleted 24 hours after creation.
    queued = "queued"  # Timer paused. Deleted 24 hours after a successful upload.
    saved = "saved"  # Never deleted automatically.


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
    stopped = "stopped"  # Held back because another upload of the same source was claimed.


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
    folder: str = Field(default=ClipFolder.unassigned.value, index=True)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = Field(default=None, index=True)  # None means the timer is off.


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
    made_for_kids: bool = False
    title: str = ""
    description: str = ""
    remote_id: str | None = None
    url: str | None = None
    error: str | None = None
    claim: dict = Field(default_factory=dict, sa_column=Column(JSON))
    claim_resolution: str | None = None
    claim_note: str = ""
    detail: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
