import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

SHARED_TOOLS = Path.home() / "clipforge-work" / "tools" / "bin"


def _chmod_capable_temp() -> Path:
    """Secrets need a disk that honours chmod. /mnt drives under WSL do not, so use the work folder."""
    folder = Path(tempfile.mkdtemp(prefix="cf-test-"))
    probe = folder / "probe"
    probe.write_text("x")
    probe.chmod(0o600)
    if stat.S_IMODE(probe.stat().st_mode) == 0o600:
        return folder
    shutil.rmtree(folder, ignore_errors=True)
    fallback = Path.home() / "clipforge-work" / "tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="cf-test-", dir=fallback))


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path, monkeypatch):
    """Each test gets its own work dir and database, but reuses the installed FFmpeg."""
    from clipforge import config, db, hardware

    monkeypatch.setenv("CLIPFORGE_WORK_DIR", str(tmp_path / "work"))
    secrets_root = _chmod_capable_temp()
    monkeypatch.setenv("CLIPFORGE_SECRETS_DIR", str(secrets_root / "secrets"))
    monkeypatch.setenv("CLIPFORGE_START_BACKGROUND", "false")
    if SHARED_TOOLS.exists():
        monkeypatch.setenv("PATH", f"{SHARED_TOOLS}{os.pathsep}{os.environ['PATH']}")
    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    yield
    shutil.rmtree(secrets_root, ignore_errors=True)
    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    hardware.detect.cache_clear()


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg") or str(SHARED_TOOLS / "ffmpeg")
    return found


@pytest.fixture(scope="session")
def scene_video(tmp_path_factory) -> Path:
    """6 s, 30 fps, 1280x720 video. Hard cuts at frames 60 and 120. Tone with a silent gap at 2-3 s."""
    out = tmp_path_factory.mktemp("media") / "scenes.mp4"
    tools = os.pathsep.join([str(SHARED_TOOLS), os.environ["PATH"]])
    filt = (
        "color=c=red:s=1280x720:r=30:d=2[a];color=c=blue:s=1280x720:r=30:d=2[b];"
        "testsrc2=s=1280x720:r=30:d=2[c];[a][b][c]concat=n=3:v=1:a=0[v];"
        "sine=f=440:d=6:r=48000,volume='if(between(t,2,3),0,1)':eval=frame[au]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            filt,
            "-map",
            "[v]",
            "-map",
            "[au]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-c:a",
            "aac",
            str(out),
        ],
        check=True,
        env={**os.environ, "PATH": tools},
    )
    return out
