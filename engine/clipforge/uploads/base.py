"""Shared upload types and the secret store. Secrets live in files outside the repo, mode 0600."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.config import get_settings

CHUNK_BYTES = 8 * 1024 * 1024


class UploadError(RuntimeError):
    """A failure to show the user as is. It never contains a token."""


class NotConfigured(UploadError):
    pass


@dataclass
class UploadRequest:
    path: Path
    title: str
    description: str
    privacy: str
    made_for_kids: bool
    width: int
    height: int
    duration: float
    fps: float
    size_bytes: int


@dataclass
class RemoteState:
    """state: processing, published, failed or claimed."""

    state: str
    remote_id: str | None = None
    url: str | None = None
    error: str | None = None
    claim: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


def _write_secret(path: Path, data: dict) -> None:
    """Atomic, and never readable by other users, even for a moment."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
        os.chmod(tmp, 0o600)
        # Windows drives under /mnt ignore chmod. Refusing beats leaving tokens readable by everyone.
        if stat.S_IMODE(os.stat(tmp).st_mode) != 0o600:
            raise UploadError(
                "This drive cannot protect secret files. Keep the clipforge work folder inside WSL, "
                "not under /mnt."
            )
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_secret(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def platform_credentials(platform_group: str) -> dict | None:
    return _read_secret(get_settings().secrets_dir / f"app-{platform_group}.json")


def save_platform_credentials(platform_group: str, client_id: str, client_secret: str) -> None:
    _write_secret(
        get_settings().secrets_dir / f"app-{platform_group}.json",
        {"client_id": client_id, "client_secret": client_secret},
    )


def delete_platform_credentials(platform_group: str) -> None:
    (get_settings().secrets_dir / f"app-{platform_group}.json").unlink(missing_ok=True)


def load_tokens(channel_id: str) -> dict | None:
    return _read_secret(get_settings().secrets_dir / f"channel-{channel_id}.json")


def save_tokens(channel_id: str, tokens: dict) -> None:
    _write_secret(get_settings().secrets_dir / f"channel-{channel_id}.json", tokens)


def delete_tokens(channel_id: str) -> None:
    (get_settings().secrets_dir / f"channel-{channel_id}.json").unlink(missing_ok=True)


def redirect_uri() -> str:
    settings = get_settings()
    return f"http://127.0.0.1:{settings.api_port}/api/oauth/callback"
