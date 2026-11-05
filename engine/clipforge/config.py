"""Paths and settings. All media, models and temp files live outside the repo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIPFORGE_")

    work_dir: Path = Path.home() / "clipforge-work"
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    web_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    clip_ttl_hours: float = 24.0
    cleanup_interval_seconds: int = 60
    temp_max_age_hours: float = 6.0
    whisper_model_gpu: str = "large-v3-turbo"
    whisper_model_cpu: str = "small"
    force_cpu: bool = False
    start_background: bool = True

    @property
    def media_dir(self) -> Path:
        return self.work_dir / "media"

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def renders_dir(self) -> Path:
        return self.work_dir / "renders"

    @property
    def tmp_dir(self) -> Path:
        return self.work_dir / "tmp"

    @property
    def data_dir(self) -> Path:
        return self.work_dir / "data"

    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"

    @property
    def secrets_dir(self) -> Path:
        return self.work_dir / "secrets"

    @property
    def tools_bin(self) -> Path:
        return self.work_dir / "tools" / "bin"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipforge.sqlite"

    def ensure_dirs(self) -> None:
        for path in (
            self.media_dir,
            self.models_dir,
            self.renders_dir,
            self.tmp_dir,
            self.analysis_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.secrets_dir.mkdir(parents=True, exist_ok=True, mode=0o700)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
