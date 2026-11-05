"""Find the best moments: sentences, feature curves, candidates, selection.

Every clip starts on a sentence start and ends on a sentence end, so nothing is cut mid-sentence.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from clipforge.analysis.audio import AudioAnalysis
from clipforge.analysis.scenes import SceneAnalysis
from clipforge.analysis.transcribe import Word

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being but by
    can could did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its just me more most my no nor not now of off on once only or
    other our out over own same she should so some such than that the their them then there these they
    this those through to too under until up very was we were what when where which while who whom why
    will with would you your yeah okay ok um uh like gonna wanna got get really""".split()
)

EMOTION_WORDS = frozenset(
    """amazing awesome incredible insane crazy unbelievable love hate never always impossible secret
    shocking terrible horrible beautiful perfect wrong dead die kill death fight power war win lose won
    lost destroy best worst stop run help wow no damn truth lie betray promise fear afraid angry scared
    remember forget finally everything nothing""".split()
)

SENTENCE_END = re.compile(r"[.!?…。！？]+[\"')\]]*$")


@dataclass
class Sentence:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def closed_text(self) -> str:
        """Text with an end mark. Whisper often drops it, and joined sentences then run together."""
        text = self.text.strip()
        return f"{text}." if text and not SENTENCE_END.search(text) else text


@dataclass
class Moment:
    start: float
    end: float
    score: float
    reasons: dict[str, float] = field(default_factory=dict)
    text: str = ""


GENRE_WEIGHTS: dict[str, dict[str, float]] = {
    "anime": {"speech": 0.20, "emotion": 0.30, "topic": 0.05, "action": 0.45},
    "movie": {"speech": 0.25, "emotion": 0.35, "topic": 0.10, "action": 0.30},
    "series": {"speech": 0.30, "emotion": 0.35, "topic": 0.15, "action": 0.20},
    "gameplay": {"speech": 0.20, "emotion": 0.25, "topic": 0.05, "action": 0.50},
    "podcast": {"speech": 0.35, "emotion": 0.30, "topic": 0.30, "action": 0.05},
    "other": {"speech": 0.30, "emotion": 0.30, "topic": 0.20, "action": 0.20},
}


def split_sentences(words: list[Word], gap: float = 0.6, max_seconds: float = 20.0) -> list[Sentence]:
    """Split on end punctuation or a pause. A run with neither is capped at max_seconds.

    Whisper sometimes drops punctuation, so a short pause before a capitalized word also counts.
    """
    sentences: list[Sentence] = []
    current: list[Word] = []
    for i, word in enumerate(words):
        current.append(word)
        next_word = words[i + 1] if i + 1 < len(words) else None
        at_end = next_word is None
        pause = next_word.start - word.end if next_word else 0.0
        capital_next = next_word is not None and next_word.text.strip()[:1].isupper()
        long_gap = pause >= gap or (capital_next and pause >= 0.25)
        too_long = word.end - current[0].start >= max_seconds
        if at_end or long_gap or SENTENCE_END.search(word.text) or too_long:
            sentences.append(Sentence(current[0].start, word.end, current))
            current = []
    return sentences


def robust_norm(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 by 5th to 95th percentile, so one spike does not flatten everything."""
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _per_second(curve: np.ndarray, hop: float, seconds: int) -> np.ndarray:
    out = np.zeros(seconds, dtype=np.float32)
    per = 1.0 / hop
    for s in range(seconds):
        chunk = curve[int(s * per) : int((s + 1) * per)]
        out[s] = float(chunk.mean()) if chunk.size else 0.0
    return out


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def topic_change_curve(sentences: list[Sentence], seconds: int, window: int = 4) -> np.ndarray:
    """Cosine distance between the sentences before and after each point (a TF-IDF TextTiling)."""
    curve = np.zeros(seconds, dtype=np.float32)
    if len(sentences) < window * 2:
        return curve
    tokens = [tokenize(s.text) for s in sentences]
    doc_freq: Counter[str] = Counter(t for sent in tokens for t in set(sent))
    total = len(sentences)
    idf = {t: math.log((1 + total) / (1 + df)) + 1 for t, df in doc_freq.items()}

    def vector(chunk: list[list[str]]) -> dict[str, float]:
        counts = Counter(t for sent in chunk for t in sent)
        return {t: c * idf[t] for t, c in counts.items()}

    for i in range(window, len(sentences) - window + 1):
        before, after = vector(tokens[i - window : i]), vector(tokens[i : i + window])
        dot = sum(v * after.get(t, 0.0) for t, v in before.items())
        norm = math.sqrt(sum(v * v for v in before.values())) * math.sqrt(sum(v * v for v in after.values()))
        distance = 1.0 - dot / norm if norm else 0.0
        at = int(sentences[i].start)
        if at < seconds:
            curve[at] = max(curve[at], distance)
    return curve


def lexicon_curve(words: list[Word], seconds: int) -> np.ndarray:
    curve = np.zeros(seconds, dtype=np.float32)
    for word in words:
        at = int(word.start)
        if at >= seconds:
            continue
        clean = re.sub(r"[^\w']", "", word.text.lower())
        if clean in EMOTION_WORDS:
            curve[at] += 1.0
        if word.text.rstrip("\"')").endswith(("!", "?")):
            curve[at] += 0.7
        if len(clean) > 2 and word.text.isupper():
            curve[at] += 0.5
    return curve


@dataclass
class FeatureCurves:
    speech: np.ndarray
    emotion: np.ndarray
    topic: np.ndarray
    action: np.ndarray
    voice: np.ndarray  # share of each second holding speech (not normalized)


def build_curves(
    words: list[Word], sentences: list[Sentence], audio: AudioAnalysis, scenes: SceneAnalysis, duration: float
) -> FeatureCurves:
    seconds = max(1, math.ceil(duration))
    voice = np.zeros(seconds, dtype=np.float32)
    rate = np.zeros(seconds, dtype=np.float32)
    for word in words:
        at = int(word.start)
        if at < seconds:
            rate[at] += 1.0
            voice[at] = min(1.0, voice[at] + (word.end - word.start))
    loud = _per_second(audio.rms_db, audio.hop, seconds)
    flux = _per_second(audio.flux, audio.hop, seconds)
    speech = robust_norm(loud) * 0.6 + robust_norm(rate) * 0.4

    loud_var = np.array(
        [
            float(np.std(audio.rms_db[int(s / audio.hop) : int((s + 2) / audio.hop)]) if s < seconds else 0.0)
            for s in range(seconds)
        ],
        dtype=np.float32,
    )
    emotion = (
        robust_norm(loud_var) * 0.35
        + robust_norm(flux) * 0.25
        + np.minimum(lexicon_curve(words, seconds), 1.0) * 0.40
    )

    motion_s = _per_second(scenes.motion, 1.0 / scenes.fps, seconds)
    cut_density = np.zeros(seconds, dtype=np.float32)
    for t in scenes.cut_times:
        if int(t) < seconds:
            cut_density[int(t)] += 1.0
    action = robust_norm(motion_s) * 0.6 + robust_norm(cut_density) * 0.4
    topic = topic_change_curve(sentences, seconds)
    return FeatureCurves(speech, emotion, topic, action, voice)


def _nonverbal_units(audio: AudioAnalysis, scenes: SceneAnalysis, duration: float) -> list[Sentence]:
    """Video with little or no speech: natural units are the spans between silences and cuts."""
    marks = sorted({0.0, duration, *scenes.cut_times, *(s[1] for s in audio.silences)})
    merged = [marks[0]]
    for mark in marks[1:]:
        if mark - merged[-1] >= 1.5 or mark == duration:
            merged.append(mark)
    return [Sentence(a, b, []) for a, b in zip(merged, merged[1:], strict=False) if b > a]


def score_moment(curves: FeatureCurves, weights: dict[str, float], start: float, end: float) -> Moment:
    i, j = int(start), max(int(start) + 1, math.ceil(end))
    reasons = {
        name: float(getattr(curves, name)[i:j].mean()) if getattr(curves, name)[i:j].size else 0.0
        for name in ("speech", "emotion", "topic", "action")
    }
    # A topic shift counts by its strongest point; an average would wash it out.
    topic_slice = curves.topic[i:j]
    reasons["topic"] = float(topic_slice.max()) if topic_slice.size else 0.0
    base = sum(weights[k] * reasons[k] for k in weights)
    hook_slice = (
        curves.speech[i : min(j, i + 3)]
        + curves.emotion[i : min(j, i + 3)]
        + curves.action[i : min(j, i + 3)]
    )
    hook = float(hook_slice.mean()) / 3.0 if hook_slice.size else 0.0
    silence = 1.0 - float(curves.voice[i:j].mean()) if curves.voice.any() else 0.0
    score = base + 0.15 * hook - 0.15 * silence
    reasons |= {"hook": hook, "silence": silence}
    return Moment(start, end, round(score, 4), {k: round(v, 3) for k, v in reasons.items()})


def generate_candidates(
    units: list[Sentence],
    curves: FeatureCurves,
    weights: dict[str, float],
    target: float,
    tolerance: float = 0.3,
) -> list[Moment]:
    """From each start, take the unit end nearest the target within the tolerance band."""
    low, high = target * (1 - tolerance), target * (1 + tolerance)
    moments: list[Moment] = []
    for i, first in enumerate(units):
        best_index = None
        best_gap = math.inf
        for j in range(i, len(units)):
            length = units[j].end - first.start
            if length > high:
                break
            if length >= low and abs(length - target) < best_gap:
                best_index, best_gap = j, abs(length - target)
        if best_index is None:
            continue
        moment = score_moment(curves, weights, first.start, units[best_index].end)
        moment.text = " ".join(u.closed_text for u in units[i : best_index + 1]).strip()
        moments.append(moment)
    return moments


def select_moments(candidates: list[Moment], count: int, min_gap: float = 1.0) -> list[Moment]:
    """Highest scores first, skipping overlaps, returned in story order for part numbering."""
    chosen: list[Moment] = []
    for moment in sorted(candidates, key=lambda m: m.score, reverse=True):
        if len(chosen) >= count:
            break
        if all(moment.end + min_gap <= c.start or moment.start >= c.end + min_gap for c in chosen):
            chosen.append(moment)
    return sorted(chosen, key=lambda m: m.start)


def find_moments(
    words: list[Word],
    audio: AudioAnalysis,
    scenes: SceneAnalysis,
    duration: float,
    genre: str,
    target_seconds: float,
    count: int,
) -> list[Moment]:
    target_seconds = min(target_seconds, duration)
    sentences = split_sentences(words)
    curves = build_curves(words, sentences, audio, scenes, duration)
    weights = GENRE_WEIGHTS.get(genre, GENRE_WEIGHTS["other"])
    speech_share = sum(s.end - s.start for s in sentences) / max(duration, 1e-3)
    units = sentences if speech_share >= 0.1 else _nonverbal_units(audio, scenes, duration)
    candidates = generate_candidates(units, curves, weights, target_seconds)
    if not candidates:  # Sentences never land in the band: widen it before giving up.
        candidates = generate_candidates(units, curves, weights, target_seconds, tolerance=0.75)
    return select_moments(candidates, count)
