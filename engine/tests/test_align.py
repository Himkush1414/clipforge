import numpy as np
import pytest
from scipy.io import wavfile

from clipforge.analysis import align, transcribe
from clipforge.analysis.align import Segment, align_segments, best_path, letters_of
from clipforge.analysis.transcribe import Word
from clipforge.media.ffmpeg import Cancelled

VOCAB = {"<pad>": 0, "|": 1, "A": 2, "B": 3, "C": 4, "'": 5}
FRAME = 320 / 16000
BLANK, GAP, A, B, C = 0, 1, 2, 3, 4


def emission(frames: list[int], sure: float = 0.97) -> np.ndarray:
    """Log probabilities where each frame is confidently the given token."""
    rest = (1.0 - sure) / (len(VOCAB) - 1)
    probs = np.full((len(frames), len(VOCAB)), rest)
    probs[np.arange(len(frames)), frames] = sure
    return np.log(probs)


class FakeSession:
    """Stands in for ONNX Runtime, returning planned logits for each chunk of audio."""

    def __init__(self, plans: list[list[int]]):
        self.plans = list(plans)
        self.chunks: list[int] = []

    def get_inputs(self):
        return []

    def run(self, _outputs, feed):
        self.chunks.append(feed["input_values"].shape[1])
        return [emission(self.plans.pop(0))[None]]


def use_session(monkeypatch, session):
    monkeypatch.setattr(align, "model_folder", lambda language: "folder")
    monkeypatch.setattr(align, "_load", lambda folder: (session, VOCAB))


def test_path_finds_letters_at_their_known_frames():
    frames = [BLANK, BLANK, A, A, BLANK, B, B, B, GAP, GAP, BLANK]
    spans = best_path(emission(frames), [A, B], BLANK, GAP)
    assert spans == [(2, 4), (5, 8)]


def test_repeated_letters_stay_separate_when_a_blank_divides_them():
    spans = best_path(emission([A, BLANK, A, BLANK]), [A, A], BLANK, GAP)
    assert spans == [(0, 1), (2, 3)]


def test_no_path_when_there_are_fewer_frames_than_letters():
    assert best_path(emission([A]), [A, B], BLANK, GAP) is None
    assert best_path(emission([A, B]), [], BLANK, GAP) is None


def test_spelling_ignores_case_and_drops_what_the_model_cannot_spell():
    assert letters_of("ab", VOCAB) == [A, B]
    assert letters_of("A'b,", VOCAB) == [A, 5, B]
    assert letters_of("1984", VOCAB) == []
    assert letters_of("|", VOCAB) == []


def test_spelling_joins_combining_accents_when_the_model_has_the_whole_letter():
    assert letters_of("é", {"é": 7, "<pad>": 0}) == [7]


def test_words_get_times_from_the_audio_not_from_whisper(monkeypatch):
    # "AB" on frames 10-14, a gap, then "C" on frames 18-19.
    plan = [BLANK] * 10 + [A, A, B, B, GAP, GAP, GAP, GAP, C, C] + [BLANK] * 2
    use_session(monkeypatch, FakeSession([plan]))
    audio = np.zeros(int(16000 * len(plan) * FRAME), np.float32)
    whisper_times = [Word("ab", 0.0, 0.01), Word("c", 0.02, 0.03)]
    words, method = align_segments(audio, [Segment(0.2, 0.24, whisper_times)], "en")
    assert method == "aligned"
    assert [w.text for w in words] == ["ab", "c"]
    assert words[0].start == pytest.approx(10 * FRAME + align.START_SHIFT, abs=1e-6)
    assert words[0].end == pytest.approx(14 * FRAME + align.END_SHIFT, abs=1e-6)
    assert words[1].start == pytest.approx(18 * FRAME + align.START_SHIFT, abs=1e-6)
    assert words[1].end == pytest.approx(20 * FRAME + align.END_SHIFT, abs=1e-6)


def test_a_number_keeps_its_place_between_aligned_words(monkeypatch):
    plan = [A, A, GAP, GAP, GAP, GAP, GAP, GAP, B, B]
    use_session(monkeypatch, FakeSession([plan]))
    audio = np.zeros(int(16000 * len(plan) * FRAME), np.float32)
    whisper_times = [Word("a", 0.0, 0.04), Word("1984", 0.06, 0.12), Word("b", 0.14, 0.2)]
    words, _ = align_segments(audio, [Segment(0.0, 0.2, whisper_times)], "en")
    assert words[1].text == "1984"
    assert words[0].end <= words[1].start <= words[1].end <= words[2].start


def test_a_number_alone_keeps_whispers_times(monkeypatch):
    use_session(monkeypatch, FakeSession([]))
    whisper_times = [Word("1984", 0.1, 0.4)]
    words, _ = align_segments(np.zeros(16000, np.float32), [Segment(0.1, 0.4, whisper_times)], "en")
    assert (words[0].start, words[0].end) == (0.1, 0.4)


def test_text_that_does_not_match_the_audio_keeps_whispers_times(monkeypatch):
    # The audio says nothing but blanks, yet the text claims letters.
    use_session(monkeypatch, FakeSession([[BLANK] * 20]))
    whisper_times = [Word("abc", 0.1, 0.3)]
    words, method = align_segments(np.zeros(6400, np.float32), [Segment(0.1, 0.3, whisper_times)], "en")
    assert method == "aligned"
    assert (words[0].start, words[0].end) == (0.1, 0.3)


def test_a_segment_window_never_reaches_back_over_the_previous_words(monkeypatch):
    session = FakeSession([[A, A, BLANK] * 5, [B, B, BLANK] * 2])
    use_session(monkeypatch, session)
    audio = np.zeros(16000 * 5, np.float32)
    first = Segment(0.0, 0.3, [Word("a", 0.0, 0.3)])
    # Whisper starts this segment right where the first ended, so padding alone would overlap it.
    second = Segment(0.3, 0.42, [Word("b", 0.3, 0.42)])
    words, _ = align_segments(audio, [first, second], "en")
    floor = words[0].end - align.END_SHIFT
    assert session.chunks[1] == int((0.42 + align.PAD_SECONDS) * 16000) - int(floor * 16000)


def test_no_model_for_the_language_keeps_whispers_times():
    words = [Word("hello", 1.0, 1.5)]
    found, method = align_segments(np.zeros(32000, np.float32), [Segment(1.0, 1.5, words)], "xx")
    assert method == "whisper"
    assert found == words


def test_a_damaged_model_file_keeps_whispers_times_instead_of_failing_the_job():
    pytest.importorskip("onnxruntime")
    from clipforge.config import get_settings

    folder = get_settings().models_dir / "align" / "en"
    folder.mkdir(parents=True)
    (folder / "model.onnx").write_bytes(b"this is not an onnx model")
    (folder / "vocab.json").write_text('{"<pad>": 0, "A": 1}')
    words = [Word("a", 1.0, 1.5)]
    found, method = align_segments(np.zeros(32000, np.float32), [Segment(1.0, 1.5, words)], "en")
    assert method == "whisper"
    assert found == words


def test_a_failure_on_one_segment_keeps_the_rest_aligned(monkeypatch):
    class Flaky(FakeSession):
        def run(self, outputs, feed):
            if not self.chunks:
                self.chunks.append(0)
                raise RuntimeError("odd segment")
            return super().run(outputs, feed)

    use_session(monkeypatch, Flaky([[BLANK, A, A, BLANK] * 3]))
    audio = np.zeros(16000 * 3, np.float32)
    first, second = Word("a", 0.1, 0.2), Word("a", 1.0, 1.2)
    words, _ = align_segments(audio, [Segment(0.1, 0.2, [first]), Segment(1.0, 1.2, [second])], "en")
    assert (words[0].start, words[0].end) == (0.1, 0.2)
    assert (words[1].start, words[1].end) != (1.0, 1.2)


def test_cancel_stops_alignment(monkeypatch):
    use_session(monkeypatch, FakeSession([]))
    segment = Segment(0.0, 0.2, [Word("a", 0.0, 0.2)])
    with pytest.raises(Cancelled):
        align_segments(np.zeros(16000, np.float32), [segment], "en", should_cancel=lambda: True)


class FakeWhisper:
    def __init__(self, words):
        self.words = words

    def transcribe(self, *_args, **_kwargs):
        class Item:
            def __init__(self, text, start, end):
                self.word, self.start, self.end, self.probability = text, start, end, 0.9

        class Found:
            start, end = 0.0, 1.0
            words = [Item(*word) for word in self.words]

        class Info:
            language = "en"

        return iter([Found]), Info


def _wav(tmp_path):
    path = tmp_path / "a.wav"
    wavfile.write(path, 16000, np.zeros(16000, np.int16))
    return path


def test_transcript_records_aligned_times_and_reports_progress_to_the_end(monkeypatch, tmp_path):
    plan = [BLANK] * 25 + [A, A] + [BLANK] * 23
    use_session(monkeypatch, FakeSession([plan]))
    monkeypatch.setattr(transcribe, "_load_model", lambda device: FakeWhisper([(" a", 0.0, 0.9)]))
    progress: list[float] = []
    result = transcribe._run(_wav(tmp_path), 1.0, "cpu", None, progress.append, None)
    assert result["timing"] == "aligned"
    assert result["words"][0]["text"] == "a"
    assert result["words"][0]["start"] > 0.2
    assert progress == sorted(progress)
    assert progress[0] == pytest.approx(transcribe.ALIGN_PROGRESS_START)
    assert progress[-1] == pytest.approx(1.0)


def test_transcript_without_a_model_says_so_and_stays_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "_load_model", lambda device: FakeWhisper([(" hi", 0.1, 0.5)]))
    progress: list[float] = []
    result = transcribe._run(_wav(tmp_path), 1.0, "cpu", None, progress.append, None)
    assert result["timing"] == "whisper"
    assert (result["words"][0]["start"], result["words"][0]["end"]) == (0.1, 0.5)
    assert progress[-1] == pytest.approx(1.0)


def test_bad_aligned_times_still_come_out_increasing(monkeypatch, tmp_path):
    # Two words on the same frames still leave non-overlapping, positive times after sanitizing.
    plan = [A, B] + [BLANK] * 48
    use_session(monkeypatch, FakeSession([plan]))
    monkeypatch.setattr(
        transcribe, "_load_model", lambda d: FakeWhisper([(" a", 0.0, 0.5), (" b", 0.5, 0.9)])
    )
    words = transcribe._run(_wav(tmp_path), 1.0, "cpu", None, None, None)["words"]
    assert words[0]["end"] <= words[1]["start"] < words[1]["end"]
    assert words[0]["start"] >= 0.0


def test_a_word_moved_implausibly_far_keeps_whispers_times():
    aligned = [Word("a", 1.0, 1.2), Word("b", 9.0, 9.3), Word("c", 2.1, 2.4)]
    original = [Word("a", 1.1, 1.3), Word("b", 1.6, 1.9), Word("c", 2.0, 2.3)]
    out, kept = align.guard(aligned, original)
    assert kept == 1
    assert [(w.start, w.end) for w in out] == [(1.0, 1.2), (1.6, 1.9), (2.1, 2.4)]


def test_ordinary_corrections_pass_the_guard():
    aligned = [Word("a", 1.0, 1.2), Word("b", 2.9, 3.3)]
    original = [Word("a", 1.4, 1.6), Word("b", 2.0, 2.3)]
    assert align.guard(aligned, original) == (aligned, 0)


def test_the_guard_is_applied_and_logged_during_alignment(monkeypatch, caplog):
    # The audio puts the word near the end of the window, far from where Whisper said it was.
    plan = [BLANK] * 200 + [A, A] + [BLANK] * 5
    use_session(monkeypatch, FakeSession([plan]))
    audio = np.zeros(int(16000 * len(plan) * FRAME), np.float32)
    words, _ = align_segments(audio, [Segment(0.0, 4.0, [Word("a", 0.1, 0.4)])], "en")
    assert (words[0].start, words[0].end) == (0.1, 0.4)
    assert "moved more than" in caplog.text
