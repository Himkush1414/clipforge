"""Ask a small local model how good each candidate clip is. Fully offline.

Not used by jobs yet: it joins the pipeline only once scripts/bench/llm_eval.py shows it beats the
editor alone on the held-out picks. Until then this module exists so that comparison can be run.

llama.cpp runs as a child process, on the loopback address, with a random key, for one job only, and is
stopped afterwards so its memory is free for rendering. If the program or the model is missing, or the
model misbehaves, nothing changes: the clips chosen by the scorer and the editor stand.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

from clipforge.analysis.scoring import Moment
from clipforge.config import get_settings
from clipforge.media.ffmpeg import Cancelled

log = logging.getLogger("clipforge.llm")

MODEL_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
LLM_WEIGHT = 0.5
MAX_CANDIDATES = 60
MAX_WORDS = 260
START_TIMEOUT = 120.0
REPLY_TIMEOUT = 60.0
KEYS = ("hook", "standalone", "payoff", "interest")

PROMPT = """You are an experienced editor choosing short clips from a long video.
Below is the transcript of one candidate clip. It may be in any language.

Rate it from 0 to 10 on each of these:
- hook: do the first few seconds make a stranger want to keep watching?
- standalone: does it make sense to someone who has seen nothing else?
- payoff: does it deliver a satisfying point, surprise or story by the end?
- interest: is this genuinely interesting, not routine introductions, logistics or small talk?

Be a tough judge. Most clips deserve 3 to 6, and only a rare one deserves 9 or 10.

Transcript:
\"\"\"{text}\"\"\""""

SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "integer", "minimum": 0, "maximum": 10} for key in KEYS},
    "required": list(KEYS),
}


def runtime_path() -> Path | None:
    """The llama-server program, wherever setup unpacked it."""
    root = get_settings().work_dir / "tools" / "llama"
    return next(iter(sorted(root.glob("*/**/llama-server"))), None) if root.exists() else None


def library_dirs(runtime: Path) -> list[Path]:
    """Folders holding the libraries llama-server loads.

    The CUDA build ships its CUDA runtime in a separate archive, unpacked beside the program's folder,
    and libgomp sits in tools/llama/lib when the system lacks it. Without them on the library path the
    CUDA backend does not load and the model would quietly run on the CPU.
    """
    release, root = runtime.parent.parent, runtime.parent.parent.parent
    found = [runtime.parent]
    found += sorted(
        d for d in release.iterdir() if d.is_dir() and d != runtime.parent and any(d.glob("*.so*"))
    )
    if (root / "lib").is_dir():
        found.append(root / "lib")
    return found


def model_path(name: str = MODEL_FILE) -> Path | None:
    path = get_settings().models_dir / "llm" / name
    return path if path.exists() else None


def available() -> bool:
    return runtime_path() is not None and model_path() is not None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class LocalModel:
    """A llama.cpp server for the length of a with-block."""

    def __init__(self, model: Path, runtime: Path, context: int = 4096, gpu_layers: int = 99):
        self.model, self.runtime, self.context, self.gpu_layers = model, runtime, context, gpu_layers
        self.key = secrets.token_urlsafe(24)
        self.port = _free_port()
        self.process: subprocess.Popen | None = None
        self.log_file = tempfile.TemporaryFile()  # the server's own output, for errors
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}",
            headers={"Authorization": f"Bearer {self.key}"},
            timeout=REPLY_TIMEOUT,
            trust_env=False,  # Loopback only, whatever proxy the machine has configured.
        )

    def __enter__(self) -> LocalModel:
        libraries = ":".join(str(d) for d in library_dirs(self.runtime))
        env = {**os.environ, "LD_LIBRARY_PATH": f"{libraries}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        self.process = subprocess.Popen(
            [
                str(self.runtime),
                "--model",
                str(self.model),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--api-key",
                self.key,
                "--ctx-size",
                str(self.context),
                "--n-gpu-layers",
                str(self.gpu_layers),
                "--parallel",
                "1",
                "--offline",
                "--reasoning",
                "off",
            ],  # fmt: skip
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server stopped with code {self.process.returncode}: {self.log_tail()}"
                )
            try:
                if self.client.get("/health").status_code == 200:
                    return self
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        raise RuntimeError(f"llama-server did not start in time: {self.log_tail()}")

    def log_tail(self, size: int = 600) -> str:
        """The end of the server's own output, which says why it failed."""
        self.log_file.seek(0, os.SEEK_END)
        self.log_file.seek(max(0, self.log_file.tell() - size))
        return self.log_file.read().decode(errors="replace").strip()

    def log_text(self) -> str:
        self.log_file.seek(0)
        return self.log_file.read().decode(errors="replace")

    def __exit__(self, *_exc) -> None:
        self.client.close()
        self.log_file.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def rate(self, text: str) -> dict[str, int]:
        words = text.split()
        prompt = PROMPT.format(text=" ".join(words[:MAX_WORDS]))
        reply = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "seed": 7,
                "max_tokens": 48,
                "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
            },
        )
        reply.raise_for_status()
        rating = json.loads(reply.json()["choices"][0]["message"]["content"])
        return {key: max(0, min(10, int(rating[key]))) for key in KEYS}


def rating_score(rating: dict[str, int]) -> float:
    """0 to 1. Interest and payoff count most, since the editor already checks the structure."""
    return (rating["hook"] + rating["standalone"] + 2 * rating["payoff"] + 2 * rating["interest"]) / 60


def shortlist(moments: Iterable[Moment], limit: int = MAX_CANDIDATES) -> list[Moment]:
    """The best candidates that are not near-copies of each other, as rating costs real time."""
    chosen: list[Moment] = []
    for moment in moments:
        if len(chosen) >= limit:
            break
        length = moment.end - moment.start
        if all(min(moment.end, c.end) - max(moment.start, c.start) <= 0.5 * length for c in chosen):
            chosen.append(moment)
    return chosen


def blend(moment: Moment, rating: dict[str, int], weight: float = LLM_WEIGHT) -> None:
    moment.reasons |= {f"llm_{key}": rating[key] for key in KEYS}
    moment.score = round((1 - weight) * moment.score + weight * rating_score(rating), 4)


def rescore(
    moments: list[Moment],
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[Moment], str]:
    """Candidates rated by the model, best first. Returns (moments, how they were scored).

    Only the shortlist is kept when the model runs, since unrated clips could not be compared fairly.
    Never raises for a model problem: the editor's ranking is a good answer on its own.
    """
    runtime, model = runtime_path(), model_path()
    passing = [m for m in moments if m.passed]
    if runtime is None or model is None or not passing:
        return moments, "editor"
    short = shortlist(passing)
    originals = [m.score for m in short]
    try:
        with LocalModel(model, runtime) as local:
            for index, moment in enumerate(short):
                if should_cancel and should_cancel():
                    raise Cancelled("Cancelled")
                blend(moment, local.rate(moment.text))
                if on_progress:
                    on_progress((index + 1) / len(short))
    except Cancelled:
        raise
    except Exception as exc:  # A bad model file, out of memory, or an odd reply must not fail the job.
        log.warning("The local model could not rate clips, keeping the editor's ranking: %s", exc)
        for moment, score in zip(short, originals, strict=True):
            moment.score = score
            for key in KEYS:
                moment.reasons.pop(f"llm_{key}", None)
        return moments, "editor"
    return sorted(short, key=lambda m: m.score, reverse=True), "local model"
