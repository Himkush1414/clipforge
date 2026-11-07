"""Job queue: one worker thread, state kept in SQLite so it survives restarts."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from sqlmodel import col, select, update

from clipforge.db import session_scope
from clipforge.media.ffmpeg import Cancelled
from clipforge.models import Job, JobStatus, utcnow
from clipforge.review import AwaitingReview

log = logging.getLogger("clipforge.jobs")

JobRunner = Callable[["JobContext"], None]


class JobContext:
    """Handed to the pipeline to report progress and check for cancellation."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def update(
        self, stage: str | None = None, progress: float | None = None, message: str | None = None
    ) -> None:
        with session_scope() as session:
            job = session.get(Job, self.job_id)
            if job is None:
                return
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = round(max(0.0, min(1.0, progress)), 4)
            if message is not None:
                job.message = message
            job.updated_at = utcnow()
            session.add(job)
            session.commit()

    def should_cancel(self) -> bool:
        with session_scope() as session:
            job = session.get(Job, self.job_id)
            return job is None or job.cancel_requested

    def check_cancel(self) -> None:
        if self.should_cancel():
            raise Cancelled("Cancelled")


def claim_next_job() -> str | None:
    """Atomically move the oldest queued job to running."""
    with session_scope() as session:
        candidate = session.exec(
            select(Job).where(Job.status == JobStatus.queued.value).order_by(col(Job.created_at))
        ).first()
        if candidate is None:
            return None
        result = session.exec(
            update(Job)
            .where(col(Job.id) == candidate.id, col(Job.status) == JobStatus.queued.value)
            .values(status=JobStatus.running.value, stage="starting", updated_at=utcnow())
        )
        session.commit()
        return candidate.id if result.rowcount == 1 else None


def requeue_interrupted_jobs() -> int:
    """After a crash or restart, running jobs restart. Analysis is cached, so little is lost."""
    with session_scope() as session:
        jobs = session.exec(select(Job).where(Job.status == JobStatus.running.value)).all()
        for job in jobs:
            job.status = JobStatus.queued.value
            job.stage = "queued"
            job.progress = 0.0
            job.message = "Restarted after the app was closed"
            session.add(job)
        session.commit()
        return len(jobs)


def friendly_error(exc: Exception) -> str:
    """Say what to do about common failures. The full details are in the log file."""
    text = str(exc)
    lowered = text.lower()
    if "no space left" in lowered:
        return "The disk is full. Free some space, then run this job again."
    if "invalid data found" in lowered or "moov atom not found" in lowered:
        return "This video file looks damaged, so FFmpeg could not read it."
    if isinstance(exc, MemoryError) or "cannot allocate memory" in lowered:
        return "The computer ran out of memory. Close other programs, then run this job again."
    return text[:2000]


def run_one(job_id: str, runner: JobRunner) -> None:
    ctx = JobContext(job_id)
    final_status, error = JobStatus.done, None
    try:
        runner(ctx)
    except AwaitingReview:
        final_status = JobStatus.review
    except Cancelled:
        final_status = JobStatus.cancelled
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        final_status, error = JobStatus.failed, friendly_error(exc)
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = final_status.value
        job.error = error
        job.stage = final_status.value
        if final_status is JobStatus.done:
            job.progress = 1.0
        job.updated_at = utcnow()
        session.add(job)
        session.commit()


class Worker(threading.Thread):
    def __init__(self, runner: JobRunner, poll_seconds: float = 1.0):
        super().__init__(daemon=True, name="clipforge-worker")
        self.runner = runner
        self.poll_seconds = poll_seconds
        self._stop_event = threading.Event()

    def run(self) -> None:
        requeue_interrupted_jobs()
        while not self._stop_event.is_set():
            try:
                job_id = claim_next_job()
                if job_id is None:
                    self._stop_event.wait(self.poll_seconds)
                    continue
                run_one(job_id, self.runner)
            except Exception:
                # The worker must outlive any one failure, or every later job would sit queued forever.
                log.exception("Worker loop error; continuing")
                self._stop_event.wait(self.poll_seconds)

    def stop(self) -> None:
        self._stop_event.set()
