"""Tighten Whisper's word times with CTC forced alignment.

Whisper knows what was said far better than when. A small wav2vec2 model says, every 20 ms, how
likely each letter is. Given the words Whisper already heard, the best path through those letters
is one exact answer, so word edges come from the audio itself and the text never changes.

Only ONNX Runtime and numpy are needed, both already installed with faster-whisper. Whisper's own
times are kept for any language without a model, a damaged model, or text the audio does not match.
Measured in docs/ENGINE.md.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clipforge.analysis.transcribe import Word
from clipforge.config import get_settings
from clipforge.media.ffmpeg import Cancelled

log = logging.getLogger("clipforge.align")

RATE = 16000
PAD_SECONDS = 0.2
BLANK_TOKENS = ("<pad>", "[PAD]")
WORD_GAP = "|"

# The raw edges run late at the start of a word and early at the end, as CTC only fires once a letter
# is clearly under way. Set from the median offsets on the exact word-by-word benchmark samples.
START_SHIFT = -0.03
END_SHIFT = 0.03
# Mean letter probability along the path. On the benchmark samples, correct text scored 0.27 or more and
# wrong text or noise 0.084 or less. Below this, the segment keeps Whisper's times.
MIN_CONFIDENCE = 0.15


@dataclass
class Segment:
    """One Whisper segment: the window of audio its words are searched in."""

    start: float
    end: float
    words: list[Word] = field(default_factory=list)


def model_folder(language: str | None) -> Path | None:
    """The installed model folder for a language, or None to keep Whisper's times."""
    if not language:
        return None
    folder = get_settings().models_dir / "align" / language
    return folder if (folder / "model.onnx").exists() and (folder / "vocab.json").exists() else None


def _load(folder: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Memory patterns keep the largest chunk's buffers alive, and chunk sizes vary a lot here.
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(
        str(Path(folder) / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    vocab: dict[str, int] = json.loads((Path(folder) / "vocab.json").read_text(encoding="utf-8"))
    return session, vocab


def letters_of(text: str, vocab: dict[str, int]) -> list[int]:
    """Model tokens for a word. Anything the model cannot spell, such as digits, is dropped."""
    text = unicodedata.normalize("NFC", text)
    tokens: list[int] = []
    for char in text:
        for option in (char, char.lower(), char.upper()):
            if option in vocab and option != WORD_GAP:
                tokens.append(vocab[option])
                break
    return tokens


def blank_id(vocab: dict[str, int]) -> int:
    return next(vocab[token] for token in BLANK_TOKENS if token in vocab)


def best_path(
    log_probs: np.ndarray, tokens: list[int], blank: int, gap: int | None = None
) -> list[tuple[int, int]] | None:
    """Viterbi over the tokens in order. Returns (first frame, end frame) per token.

    Each frame either starts the next token, or stays on the current one as itself or as blank.
    Before the first token and after the last, a word gap also counts as silence.
    Returns None when there are fewer frames than tokens, as no valid path exists.
    """
    frames, count = log_probs.shape[0], len(tokens)
    if count == 0 or frames < count:
        return None
    empty = log_probs[:, blank]
    quiet = empty if gap is None else np.maximum(empty, log_probs[:, gap])
    letter = log_probs[:, tokens]
    score = np.full(count + 1, -np.inf)
    score[0] = 0.0
    moved = np.zeros((frames, count + 1), dtype=bool)
    for t in range(frames):
        stay = score.copy()
        stay[0] += quiet[t]
        stay[1:] += np.maximum(empty[t], letter[t])
        stay[count] = score[count] + max(quiet[t], letter[t, count - 1])
        enter = score[:-1] + letter[t]
        moved[t, 1:] = enter > stay[1:]
        stay[1:] = np.maximum(stay[1:], enter)
        score = stay
    if not np.isfinite(score[count]):
        return None
    firsts = [0] * count
    state = count
    for t in range(frames - 1, -1, -1):
        if state > 0 and moved[t, state]:
            firsts[state - 1] = t
            state -= 1
    if state != 0:
        return None
    spans = []
    for index, first in enumerate(firsts):
        limit = firsts[index + 1] if index + 1 < count else frames
        voiced = np.nonzero(letter[first:limit, index] > empty[first:limit])[0]
        spans.append((first, first + int(voiced[-1]) + 1 if voiced.size else first + 1))
    return spans


def path_confidence(log_probs: np.ndarray, tokens: list[int], spans: list[tuple[int, int]]) -> float:
    """Mean probability of each letter at its best frame in its span."""
    best = [log_probs[first:end, token].max() for (first, end), token in zip(spans, tokens, strict=True)]
    return float(np.exp(np.mean(best)))


def place_words(
    words: list[Word],
    word_letters: list[list[int]],
    spans: list[tuple[int, int]],
    seconds_per_frame: float,
    origin: float,
    has_gaps: bool,
) -> list[Word]:
    """Turn token spans into word times. Words with no spellable letters keep their place between."""
    placed: list[Word] = []
    cursor = 0
    for word, tokens in zip(words, word_letters, strict=True):
        if tokens:
            last = cursor + len(tokens) - 1
            first = spans[cursor][0]
            # A word's fading tail is not confidently any letter, so it runs up to the gap that follows.
            end = spans[last + 1][0] if has_gaps and last + 1 < len(spans) else spans[last][1]
            cursor = last + 1 + has_gaps
            placed.append(
                Word(
                    word.text,
                    origin + first * seconds_per_frame + START_SHIFT,
                    origin + end * seconds_per_frame + END_SHIFT,
                    word.probability,
                )
            )
        else:
            placed.append(Word(word.text, -1.0, -1.0, word.probability))
    return _fill_unspelled(placed, words)


def _fill_unspelled(placed: list[Word], original: list[Word]) -> list[Word]:
    """Words like "1984" have no letters to align. Fit Whisper's time into the gap the others left."""
    for index, word in enumerate(placed):
        if word.start >= 0:
            continue
        after = next((w.start for w in placed[index + 1 :] if w.start >= 0), original[index].end)
        before = placed[index - 1].end if index else original[index].start
        word.start = max(before, min(original[index].start, after))
        word.end = max(word.start, min(original[index].end, after))
    return placed


def emissions(session, chunk: np.ndarray) -> np.ndarray:
    """Letter log probabilities per frame for one piece of audio."""
    normal = (chunk - chunk.mean()) / max(float(chunk.std()), 1e-7)
    feed = {"input_values": normal[None, :].astype(np.float32)}
    names = {item.name for item in session.get_inputs()}
    if "attention_mask" in names:
        feed["attention_mask"] = np.ones((1, chunk.size), dtype=np.int64)
    logits = session.run(None, feed)[0][0].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def align_segment(
    session, vocab: dict[str, int], samples: np.ndarray, segment: Segment, floor: float = 0.0
) -> list[Word] | None:
    """New word times for one segment, or None to keep Whisper's.

    The window never reaches back before `floor`, where the previous segment's words ended. Whisper's
    segments often start early, and the end of the last word could otherwise pass for this one's start.
    """
    origin = max(floor, segment.start - PAD_SECONDS)
    chunk = samples[int(origin * RATE) : int((segment.end + PAD_SECONDS) * RATE)]
    word_letters = [letters_of(word.text, vocab) for word in segment.words]
    gap = vocab.get("|")
    tokens: list[int] = []
    for letters in word_letters:
        if letters:
            tokens += ([gap] if tokens and gap is not None else []) + letters
    if not tokens or chunk.size < RATE // 10:
        return None
    log_probs = emissions(session, chunk)
    spans = best_path(log_probs, tokens, blank_id(vocab), gap)
    if spans is None or path_confidence(log_probs, tokens, spans) < MIN_CONFIDENCE:
        return None
    return place_words(
        segment.words, word_letters, spans, chunk.size / RATE / log_probs.shape[0], origin, gap is not None
    )


def align_segments(
    samples: np.ndarray,
    segments: list[Segment],
    language: str | None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[Word], str]:
    """Return (words, method), where method says where the times came from.

    Never raises for a model problem: captions on Whisper's times beat a failed job. The model is
    loaded per call and released afterwards, as RAM is tight on the target machine.
    """
    words = [word for segment in segments for word in segment.words]
    folder = model_folder(language)
    if folder is None:
        return words, "whisper"
    try:
        session, vocab = _load(str(folder))
    except Exception as exc:  # ONNX Runtime raises its own unrelated types for a bad file.
        log.warning("Could not load the alignment model for %s, keeping Whisper's times: %s", language, exc)
        return words, "whisper"
    aligned: list[Word] = []
    kept = 0
    floor = 0.0
    for index, segment in enumerate(segments):
        if should_cancel and should_cancel():
            raise Cancelled("Cancelled")
        try:
            result = align_segment(session, vocab, samples, segment, floor)
        except Exception as exc:  # Same reason. One odd segment must not cost the whole transcript.
            log.warning("Alignment failed on a segment, keeping Whisper's times: %s", exc)
            result = None
        kept += result is None
        aligned += result if result is not None else segment.words
        if aligned:
            floor = aligned[-1].end - (END_SHIFT if result is not None else 0.0)
        if on_progress:
            on_progress((index + 1) / len(segments))
    if kept:
        log.info("Kept Whisper's times on %d of %d segments", kept, len(segments))
    return aligned, "aligned"
