"""The Review clips step: a plan a person can change before anything is rendered.

The plan lives on the job, so it survives a restart. Approving it requeues the job, which then renders
exactly the approved frames through the normal path, with the usual rights check and quality checks.
"""

from __future__ import annotations

import json
from pathlib import Path

from clipforge.analysis.scoring import Moment
from clipforge.analysis.transcribe import Word, words_from_dicts
from clipforge.config import get_settings
from clipforge.export.plan import to_frame_segments
from clipforge.media.probe import MediaInfo
from clipforge.models import Job

MAX_ALTERNATES = 30


class AwaitingReview(Exception):
    """Raised by the pipeline to stop a job at the review step. It is not a failure."""


class ReviewError(ValueError):
    """Something a person asked for that cannot be done, with a message that says why."""


def entry_for(moment: Moment, key: str, info: MediaInfo) -> dict:
    segment = to_frame_segments([(moment.start, moment.end)], info.fps, info.frame_count)[0]
    return {
        "key": key,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "original": [segment.start_frame, segment.end_frame],
        "score": moment.score,
        "reasons": moment.reasons,
        "why": moment.why,
        "text": moment.text,
        "rejected": False,
        "warnings": [],
    }


def overlaps(a: dict, b: dict) -> bool:
    return a["start_frame"] < b["end_frame"] and b["start_frame"] < a["end_frame"]


def build_plan(
    chosen: list[Moment], ranked: list[Moment], info: MediaInfo, options: dict, decisions: dict
) -> dict:
    """The chosen clips, plus the next-best candidates ready to swap in."""
    clips = [entry_for(m, f"clip-{i}", info) for i, m in enumerate(chosen)]
    alternates: list[dict] = []
    for index, moment in enumerate(ranked):
        if len(alternates) >= MAX_ALTERNATES:
            break
        entry = entry_for(moment, f"alt-{index}", info)
        # Near-copies of each other are not real choices.
        if not any(overlaps(entry, other) for other in [*clips, *alternates]):
            alternates.append(entry)
    return {
        "fps": info.fps,
        "frame_count": info.frame_count,
        "options": options,
        "decisions": decisions,
        "clips": clips,
        "alternates": alternates,
    }


def words_of_job(job: Job) -> list[Word]:
    """The cached transcript. Missing means no text to show, which must not block a person's edit."""
    language = job.settings.get("edit", {}).get("language")
    path = get_settings().analysis_dir / job.source_id / f"transcript-{language or 'auto'}.json"
    if not Path(path).exists():
        return []
    return words_from_dicts(json.loads(Path(path).read_text(encoding="utf-8"))["words"])


def describe_range(words: list[Word], start: float, end: float) -> tuple[str, list[str]]:
    """The text inside a range, and a warning for any word it cuts through."""
    warnings = [
        f'{label} cuts through the word "{word.text.strip()}".'
        for label, at in (("The start", start), ("The end", end))
        for word in words
        if word.start < at < word.end
    ]
    return " ".join(w.text for w in words if start <= w.start and w.end <= end), warnings


def find_clip(plan: dict, key: str) -> dict:
    for clip in plan["clips"]:
        if clip["key"] == key:
            return clip
    raise ReviewError("That clip is not part of this review.")


def adjust(plan: dict, key: str, start_frame: int, end_frame: int, words: list[Word]) -> dict:
    clip = find_clip(plan, key)
    if start_frame < 0:
        raise ReviewError("The start cannot be before the first frame.")
    if end_frame > plan["frame_count"]:
        raise ReviewError(f"The end cannot be past the last frame ({plan['frame_count']}).")
    if end_frame <= start_frame:
        raise ReviewError("The end must come after the start.")
    clip["start_frame"], clip["end_frame"] = start_frame, end_frame
    clip["text"], clip["warnings"] = describe_range(words, start_frame / plan["fps"], end_frame / plan["fps"])
    if any(overlaps(clip, o) for o in plan["clips"] if o["key"] != key and not o["rejected"]):
        clip["warnings"].append("This clip now overlaps another one.")
    return clip


def swap(plan: dict, key: str) -> dict:
    """Replace a clip with the best candidate that overlaps nothing still in use."""
    clip = find_clip(plan, key)
    others = [c for c in plan["clips"] if c["key"] != key and not c["rejected"]]
    for index, alternate in enumerate(plan["alternates"]):
        if not any(overlaps(alternate, other) for other in others):
            plan["alternates"].pop(index)
            plan["alternates"].append(
                {**clip, "key": f"returned-{clip['key']}", "warnings": [], "rejected": False}
            )
            position = plan["clips"].index(clip)
            plan["clips"][position] = {**alternate, "key": key}
            return plan["clips"][position]
    raise ReviewError("There is no other candidate that fits without overlapping your clips.")


def approved_clips(plan: dict) -> list[dict]:
    """The clips to render, in story order, as part numbers should run."""
    kept = [c for c in plan["clips"] if not c["rejected"]]
    if not kept:
        raise ReviewError("Every clip is rejected. Keep at least one, or cancel the job.")
    return sorted(kept, key=lambda c: c["start_frame"])
