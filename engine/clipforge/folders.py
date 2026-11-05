"""Clip folders and every file deletion. Automatic deletion is only possible inside Unassigned.

All checks use real paths, so a symlink can never lead a deletion somewhere else.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import threading
from pathlib import Path

from sqlmodel import select

from clipforge.appsettings import load_app_settings
from clipforge.config import get_settings
from clipforge.models import ClipFolder

log = logging.getLogger("clipforge.folders")
deletions = logging.getLogger("clipforge.deletions")

REPO_ROOT = Path(__file__).resolve().parents[2]
MOVE_RESERVE_BYTES = 512 * 1024**2
SYSTEM_DIRS = (
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/proc",
    "/root",
    "/run",
    "/sbin",
    "/sys",
    "/usr",
    "/var",
)
# One lock for everything that moves or deletes clip files, so cleanup cannot race a move.
file_lock = threading.RLock()


class FolderError(RuntimeError):
    """A rule refused a file operation. The message is shown to the user."""


class LowDisk(FolderError):
    pass


def windows_to_wsl(raw: str) -> Path:
    """Accept a pasted Windows path such as D:\\videos\\a.mp4."""
    text = raw.strip().strip('"')
    match = re.match(r"^([A-Za-z]):[\\/]?(.*)$", text)
    if match:
        return Path(f"/mnt/{match.group(1).lower()}") / match.group(2).replace("\\", "/")
    return Path(text)


def _real(path: Path | str) -> Path:
    return Path(os.path.realpath(path))


def is_inside(path: Path | str, root: Path | str) -> bool:
    """True only when path's real location is strictly below root's real location."""
    real_path, real_root = _real(path), _real(root)
    return real_path != real_root and real_path.is_relative_to(real_root)


def _overlap(a: Path | str, b: Path | str) -> bool:
    real_a, real_b = _real(a), _real(b)
    return real_a == real_b or real_a.is_relative_to(real_b) or real_b.is_relative_to(real_a)


def saved_dir() -> Path:
    configured = load_app_settings().saved_dir
    return Path(configured) if configured else get_settings().default_saved_dir


def folder_dir(folder: str) -> Path:
    settings = get_settings()
    if folder == ClipFolder.unassigned.value:
        return settings.unassigned_dir
    if folder == ClipFolder.queued.value:
        return settings.queued_dir
    if folder == ClipFolder.saved.value:
        return saved_dir()
    raise FolderError(f"Unknown folder: {folder}")


def _protected_roots() -> list[Path]:
    settings = get_settings()
    return [
        REPO_ROOT,
        settings.queued_dir,
        saved_dir(),
        settings.media_dir,
        settings.models_dir,
        settings.secrets_dir,
        settings.data_dir,
        settings.tools_bin.parent,
    ]


def unassigned_problem() -> str | None:
    """Why automatic deletion must not run right now, or None when it is safe."""
    root = get_settings().unassigned_dir
    if root.is_symlink():
        return f"{root} is a symlink"
    for protected in _protected_roots():
        if _overlap(root, protected):
            return f"{root} overlaps {protected}"
    return None


def delete_from_unassigned(path: Path | str, reason: str) -> bool:
    """The only automatic file deletion for clips. Anything doubtful is refused and logged."""
    target = Path(path)
    with file_lock:
        problem = unassigned_problem()
        if problem:
            deletions.error("REFUSED %s (%s): %s", target, reason, problem)
            return False
        if target.is_symlink():
            deletions.error("REFUSED %s (%s): it is a symlink", target, reason)
            return False
        if not is_inside(target, get_settings().unassigned_dir):
            deletions.error("REFUSED %s (%s): not inside the Unassigned folder", target, reason)
            return False
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            return True  # Already gone, which is the outcome we wanted.
        if not stat.S_ISREG(mode):
            deletions.error("REFUSED %s (%s): not a regular file", target, reason)
            return False
        target.unlink()
        deletions.info("DELETED %s (%s)", target, reason)
        return True


def delete_by_user(path: Path | str) -> bool:
    """The user pressed Delete. Allowed in any clip folder, and nowhere else."""
    target = Path(path)
    with file_lock:
        if target.is_symlink():
            deletions.error("REFUSED %s (user delete): it is a symlink", target)
            return False
        roots = [folder_dir(folder.value) for folder in ClipFolder]
        if not any(is_inside(target, root) for root in roots):
            deletions.error("REFUSED %s (user delete): not inside a clip folder", target)
            return False
        target.unlink(missing_ok=True)
        deletions.info("DELETED %s (user delete)", target)
        return True


def validate_saved_dir(raw: str) -> Path:
    """Saved may be anywhere you can write, such as the D drive, except where it would be unsafe."""
    path = windows_to_wsl(raw)
    if not path.is_absolute():
        raise FolderError("Use a full path, such as D:\\clipforge-saved.")
    settings = get_settings()
    real = _real(path)
    if real == Path("/") or real == Path.home():
        raise FolderError("Choose a dedicated folder, not a whole drive root or your home folder.")
    if len(real.parts) <= 3 and real.is_relative_to("/mnt"):
        raise FolderError("Choose a folder on that drive, not the drive itself.")
    for system in SYSTEM_DIRS:
        if real.is_relative_to(system):
            raise FolderError("That is a system folder.")
    forbidden = {
        "the clipforge repo": REPO_ROOT,
        "the Unassigned folder": settings.unassigned_dir,
        "the Queued folder": settings.queued_dir,
        "the source videos folder": settings.media_dir,
        "the temp folder": settings.tmp_dir,
        "the data folder": settings.data_dir,
    }
    for name, root in forbidden.items():
        if _overlap(real, root):
            raise FolderError(f"Saved cannot be inside or around {name}.")
    try:
        real.mkdir(parents=True, exist_ok=True)
        probe = real / ".clipforge-write-test"
        probe.write_text("x")
        probe.unlink()
    except OSError as exc:
        raise FolderError(f"Cannot write to that folder: {exc.strerror or exc}") from exc
    return real


def _check_space(target_dir: Path, needed: int) -> None:
    free = shutil.disk_usage(target_dir).free
    if free < needed + MOVE_RESERVE_BYTES:
        raise LowDisk(
            f"Not enough disk space to move this clip: it needs {needed / 1024**2:.0f} MB "
            f"and {free / 1024**2:.0f} MB is free at {target_dir}."
        )


def move_file(source: Path, target_dir: Path) -> Path:
    """Copy, verify the size, swap into place, then remove the source. Safe across drives."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source.is_symlink() or not source.is_file():
        raise FolderError(f"{source.name} is missing, so it cannot be moved.")
    size = source.stat().st_size
    _check_space(target_dir, size)
    partial = target_dir / f".{source.name}.moving"
    try:
        shutil.copyfile(source, partial)
        if partial.stat().st_size != size:
            raise FolderError(f"Copying {source.name} failed: the sizes do not match.")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    source.unlink()
    deletions.info("MOVED %s to %s", source, target)
    return target


def folder_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if not entry.is_symlink() and entry.is_file():
            total += entry.stat().st_size
    return total


def clip_bytes_by_folder() -> dict[str, int]:
    """Measured from the clip records: Saved clips can sit in an earlier Saved path too."""
    from clipforge.db import session_scope  # Imported here: the database layer loads the models first.
    from clipforge.models import Clip, ClipStatus

    sizes = {folder.value: 0 for folder in ClipFolder}
    with session_scope() as session:
        rows = session.exec(
            select(Clip.folder, Clip.path, Clip.thumb_path).where(Clip.status == ClipStatus.ready.value)
        )
        for folder, *paths in rows.all():
            for value in paths:
                if value and os.path.exists(value):
                    sizes[folder] = sizes.get(folder, 0) + os.path.getsize(value)
    return sizes


def storage_report() -> dict:
    """Sizes, free space and plain-language warnings for the UI."""
    settings = get_settings()
    app = load_app_settings()
    gib = 1024**3
    sizes = clip_bytes_by_folder()
    saved = saved_dir()
    saved.mkdir(parents=True, exist_ok=True)
    work_free = shutil.disk_usage(settings.work_dir).free
    saved_free = shutil.disk_usage(saved).free
    warnings = []
    if sizes["queued"] > app.queued_limit_gb * gib:
        warnings.append(
            f"Queued is {sizes['queued'] / gib:.1f} GB, past your {app.queued_limit_gb:g} GB limit. "
            "Upload or delete some clips."
        )
    if sizes["saved"] > app.saved_limit_gb * gib:
        warnings.append(
            f"The Saved folder is {sizes['saved'] / gib:.1f} GB, past your {app.saved_limit_gb:g} GB limit. "
            "Nothing there is deleted automatically, so it needs your attention."
        )
    if work_free < app.low_disk_gb * gib:
        warnings.append(
            f"Only {work_free / gib:.1f} GB is free on the clipforge work drive. New jobs will not start "
            f"below {app.low_disk_gb:g} GB."
        )
    if saved_free < app.low_disk_gb * gib:
        warnings.append(f"Only {saved_free / gib:.1f} GB is free on the drive holding the Saved folder.")
    return {
        "folders": sizes,
        "saved_path": str(saved),
        "free_bytes": work_free,
        "saved_free_bytes": saved_free,
        "media_bytes": folder_bytes(settings.media_dir),
        "tmp_bytes": folder_bytes(settings.tmp_dir),
        "warnings": warnings,
    }


def low_disk_message() -> str | None:
    """A reason not to start a job, or None. Shown to the user: it never fails silently."""
    free = shutil.disk_usage(get_settings().work_dir).free
    limit = load_app_settings().low_disk_gb
    if free < limit * 1024**3:
        return (
            f"Not enough disk space to start: {free / 1024**3:.1f} GB is free, under {limit:g} GB limit. "
            "Free some space or lower the limit in Settings."
        )
    return None
