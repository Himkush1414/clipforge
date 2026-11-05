"""Settings the user changes in the UI. Stored as JSON in the work folder."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from clipforge.config import get_settings


class AppSettings(BaseModel):
    saved_dir: str | None = None  # None means the default folder under the work folder.
    queued_limit_gb: float = Field(default=50.0, gt=0, le=100_000)
    saved_limit_gb: float = Field(default=200.0, gt=0, le=100_000)
    low_disk_gb: float = Field(default=5.0, ge=0, le=100_000)


def load_app_settings() -> AppSettings:
    path = get_settings().app_settings_path
    if not path.exists():
        return AppSettings()
    return AppSettings(**json.loads(path.read_text()))


def save_app_settings(values: AppSettings) -> None:
    path = get_settings().app_settings_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(values.model_dump_json(indent=2))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
