"""Request and settings models shared by the API and the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from clipforge.models import RightsType

Genre = Literal["anime", "movie", "series", "gameplay", "podcast", "other"]
Format = Literal["short", "long"]
Ratio = Literal["9:16", "1:1", "16:9"]
Resolution = Literal["1080p", "2K", "4K", "8K"]


class Rights(BaseModel):
    """Required on every job. A job cannot start with this empty."""

    type: RightsType
    detail: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _detail_where_needed(self) -> Rights:
        self.detail = self.detail.strip()
        if self.type is RightsType.creative_commons and not self.detail:
            raise ValueError("Creative Commons footage needs the credit text.")
        if self.type is RightsType.written_permission and not self.detail:
            raise ValueError("Written permission needs a link or a note.")
        return self


class ManualOptions(BaseModel):
    genre: Genre
    clip_duration: float = Field(gt=0, le=1800)
    clip_count: int = Field(ge=1, le=50)
    part_numbering: bool = True
    format: Format
    ratio: Ratio
    resolution: Resolution


class EditOptions(BaseModel):
    preset_id: str | None = None
    captions: bool = True
    fit: Literal["smart", "crop", "blur"] = "smart"
    remove_pauses: bool = False
    transition: Literal["cut", "crossfade"] = "cut"
    language: str | None = Field(default=None, max_length=8)


class FrameCut(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def _order(self) -> FrameCut:
        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be after start_frame")
        return self


class JobCreate(BaseModel):
    source_id: str
    mode: Literal["manual", "auto"]
    rights: Rights
    manual: ManualOptions | None = None
    edit: EditOptions = EditOptions()
    channel_id: str | None = None
    frame_cuts: list[FrameCut] = Field(default_factory=list, max_length=50)
    destination: Literal["unassigned", "queued"] = "unassigned"
    review: bool = False  # Stop for a person to review the clips before anything is rendered.

    @model_validator(mode="after")
    def _manual_needs_options(self) -> JobCreate:
        if self.mode == "manual" and self.manual is None:
            raise ValueError("Manual mode needs genre, duration, count, format, ratio and resolution.")
        return self


class ChannelSettings(BaseModel):
    format: Format = "short"
    ratio: Ratio = "9:16"
    preset_id: str | None = None
    title_template: str = Field(default="{title}", max_length=300)
    description_template: str = Field(default="{summary}\n\n{hashtags}", max_length=4000)


class ChannelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    platform: Literal["youtube", "instagram", "facebook"]
    settings: ChannelSettings = ChannelSettings()
    link_group: str | None = Field(default=None, max_length=80)


class PlatformCredentials(BaseModel):
    client_id: str = Field(min_length=1, max_length=500)
    client_secret: str = Field(min_length=1, max_length=500)


class UploadCreate(BaseModel):
    channel_id: str | None = None
    link_group: str | None = None
    privacy: Literal["private", "unlisted", "public"] = "private"
    made_for_kids: bool = False

    @model_validator(mode="after")
    def _one_target(self) -> UploadCreate:
        if bool(self.channel_id) == bool(self.link_group):
            raise ValueError("Choose one channel or one linked group.")
        return self


class ClaimResolve(BaseModel):
    resolution: Literal["trim", "replace_audio", "dispute"]
    note: str = Field(default="", max_length=2000)


class ClaimReport(BaseModel):
    details: str = Field(min_length=1, max_length=4000)


class TargetChoice(BaseModel):
    target_id: str = Field(min_length=1, max_length=200)


class ClipMove(BaseModel):
    folder: Literal["saved", "queued"]


class BulkClipAction(BaseModel):
    clip_ids: list[str] = Field(min_length=1, max_length=500)
    action: Literal["move_saved", "move_queued", "delete"]


class BulkUpload(BaseModel):
    clip_ids: list[str] = Field(min_length=1, max_length=500)
    channel_id: str | None = None
    link_group: str | None = None
    privacy: Literal["private", "unlisted", "public"] = "private"
    made_for_kids: bool = False

    @model_validator(mode="after")
    def _one_target(self) -> BulkUpload:
        if bool(self.channel_id) == bool(self.link_group):
            raise ValueError("Choose one channel or one linked group.")
        return self
