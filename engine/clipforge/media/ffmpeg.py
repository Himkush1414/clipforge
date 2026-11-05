"""Thin FFmpeg wrapper: binary lookup, error capture, progress and cancellation."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from clipforge.config import get_settings

# Filters that change pitch, speed or orientation are how people try to dodge Content ID.
# clipforge does not offer them. Every render graph is checked against this list.
BANNED_FILTERS = (
    "hflip",
    "vflip",
    "atempo",
    "asetrate",
    "rubberband",
    "aresample=async",
    "transpose",
)


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


class Cancelled(RuntimeError):
    pass


def find_tool(name: str) -> str:
    local = get_settings().tools_bin / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise FFmpegError(f"{name} not found. Run scripts/setup.sh or install FFmpeg.")
    return found


def assert_allowed_graph(args: Sequence[str]) -> None:
    joined = " ".join(str(a) for a in args)
    for banned in BANNED_FILTERS:
        if re.search(rf"(?<![\w]){re.escape(banned)}(?![\w])", joined):
            raise FFmpegError(f"Filter '{banned}' is not allowed in clipforge render graphs.")


def run_ffmpeg(
    args: Sequence[str | Path],
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    """Run ffmpeg, return its stderr. Reports progress 0..1 when a duration is given."""
    str_args = [str(a) for a in args]
    assert_allowed_graph(str_args)
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        *str_args,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    stderr_tail: deque[str] = deque(maxlen=400)

    def drain_stderr() -> None:
        assert proc.stderr
        for line in proc.stderr:
            stderr_tail.append(line)

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    assert proc.stdout
    cancelled = False
    for line in proc.stdout:
        if should_cancel and should_cancel():
            cancelled = True
            proc.kill()
            break
        if on_progress and duration and line.startswith("out_time_us="):
            value = line.split("=", 1)[1].strip()
            if value.lstrip("-").isdigit():
                on_progress(min(1.0, max(0.0, int(value) / 1e6 / duration)))
    code = proc.wait()
    reader.join(timeout=5)
    stderr = "".join(stderr_tail)
    if cancelled:
        raise Cancelled("Cancelled")
    if code != 0:
        raise FFmpegError(f"ffmpeg failed (exit {code}): {stderr[-600:]}", stderr)
    if on_progress and duration:
        on_progress(1.0)
    return stderr


def escape_filter_path(path: Path | str) -> str:
    r"""Escape a path for a filter option such as ass=filename=...

    FFmpeg parses twice, so two levels are needed: the option value, then the filter graph.
    """
    value = str(path)
    for char in ("\\", "'", ":"):
        value = value.replace(char, "\\" + char)
    for char in ("\\", "'", "[", "]", ",", ";"):
        value = value.replace(char, "\\" + char)
    return value
