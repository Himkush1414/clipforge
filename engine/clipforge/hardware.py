"""Hardware and encoder detection. Encoders are proven with a real test encode, not assumed."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import psutil

from clipforge.config import get_settings
from clipforge.media.ffmpeg import find_tool

# name -> (width, height) for 16:9 landscape; other ratios are derived from the short side.
RESOLUTIONS = {"1080p": 1080, "2K": 1440, "4K": 2160, "8K": 4320}


@dataclass
class Hardware:
    gpu_name: str | None
    gpu_memory_mb: int | None
    cpu_threads: int
    ram_gb: float
    disk_free_gb: float
    h264_encoder: str
    hevc_encoder: str
    h264_max_height: int
    hevc_max_height: int
    whisper_device: str
    allowed_resolutions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _gpu() -> tuple[str | None, int | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if out.returncode != 0 or not out.stdout.strip():
        return None, None
    name, memory = out.stdout.strip().splitlines()[0].split(",")
    return name.strip(), int(float(memory))


def _encoder_works(encoder: str, height: int) -> bool:
    width = height * 16 // 9
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r=30:d=0.2",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _max_height(encoder: str) -> int:
    best = 0
    for height in sorted(RESOLUTIONS.values()):
        if not _encoder_works(encoder, height):
            break
        best = height
    return best


def load_gpu_runtime() -> bool:
    """Preload CUDA math libraries from the pip packages so CTranslate2 can find them."""
    try:
        import nvidia.cublas  # type: ignore
    except ImportError:
        try:
            ctypes.CDLL("libcublas.so.12")
            return True
        except OSError:
            return False
    lib_dir = Path(next(iter(nvidia.cublas.__path__))) / "lib"
    try:
        ctypes.CDLL(str(lib_dir / "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(lib_dir / "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


def _whisper_device(gpu_name: str | None) -> str:
    if get_settings().force_cpu or not gpu_name:
        return "cpu"
    try:
        import ctranslate2
    except ImportError:
        return "cpu"
    if ctranslate2.get_cuda_device_count() > 0 and load_gpu_runtime():
        return "cuda"
    return "cpu"


@lru_cache
def detect() -> Hardware:
    settings = get_settings()
    gpu_name, gpu_memory = (None, None) if settings.force_cpu else _gpu()
    h264, hevc = "libx264", "libx265"
    if gpu_name and _encoder_works("h264_nvenc", 1080):
        h264 = "h264_nvenc"
    if gpu_name and _encoder_works("hevc_nvenc", 1080):
        hevc = "hevc_nvenc"
    h264_max = _max_height(h264) if h264.endswith("nvenc") else RESOLUTIONS["4K"]
    hevc_max = _max_height(hevc) if hevc.endswith("nvenc") else RESOLUTIONS["4K"]
    threads = psutil.cpu_count() or 1
    if not hevc.endswith("nvenc") and threads >= 16 and psutil.virtual_memory().total >= 16 * 1024**3:
        hevc_max = RESOLUTIONS["8K"]  # CPU 8K is slow, only offered on a strong machine.
    top = max(h264_max, hevc_max)
    allowed = [name for name, height in RESOLUTIONS.items() if height <= top]
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    return Hardware(
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory,
        cpu_threads=threads,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        disk_free_gb=round(psutil.disk_usage(str(settings.work_dir)).free / 1024**3, 1),
        h264_encoder=h264,
        hevc_encoder=hevc,
        h264_max_height=h264_max,
        hevc_max_height=hevc_max,
        whisper_device=_whisper_device(gpu_name),
        allowed_resolutions=allowed,
    )
