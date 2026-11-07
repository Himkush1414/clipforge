import numpy as np
import pytest

from clipforge.analysis.audio import QUIET_HOP, frame_levels, quiet_spans, quiet_threshold

RATE = 16000


def levels(audio: np.ndarray) -> np.ndarray:
    return frame_levels((np.clip(audio, -1, 1) * 32767).astype(np.int16), int(RATE * QUIET_HOP))


def noise(seconds: float, level: float, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * RATE)).astype(np.float32) * level


def test_quiet_gaps_in_noise_are_found_to_the_frame():
    audio = noise(10, 0.2)
    audio[int(3 * RATE) : int(4 * RATE)] = 0
    audio[int(6.5 * RATE) : int(7 * RATE)] = 0
    assert quiet_spans(levels(audio), QUIET_HOP) == [(3.0, 4.0), (6.5, 7.0)]


def test_steady_noise_has_no_silences_however_it_wanders():
    audio = noise(10, 0.2) * np.linspace(0.3, 1.0, 10 * RATE).astype(np.float32)
    assert quiet_threshold(levels(audio)) is None
    assert quiet_spans(levels(audio), QUIET_HOP) == []


def test_room_tone_counts_as_quiet():
    audio = noise(10, 0.2)
    audio[int(4 * RATE) : int(5 * RATE)] = noise(1, 0.002, seed=2)
    assert quiet_spans(levels(audio), QUIET_HOP) == [(4.0, 5.0)]


def test_short_pauses_are_not_silences():
    audio = noise(5, 0.2)
    audio[int(2 * RATE) : int(2.2 * RATE)] = 0
    assert quiet_spans(levels(audio), QUIET_HOP) == []


def test_a_click_inside_a_pause_does_not_split_it():
    audio = noise(10, 0.2)
    audio[int(3 * RATE) : int(5 * RATE)] = 0
    audio[64000:64320] = noise(0.02, 0.5, seed=3)
    assert quiet_spans(levels(audio), QUIET_HOP) == [(3.0, 5.0)]


def test_a_consonant_burst_at_the_end_of_a_pause_does_not_swallow_the_pause_edge():
    audio = noise(10, 0.2)
    audio[int(3 * RATE) : int(5 * RATE)] = 0
    # A short burst, a brief closure, then the word itself: the pause still ends at the burst.
    audio[79200:79520] = noise(0.02, 0.5, seed=4)
    spans = quiet_spans(levels(audio), QUIET_HOP)
    assert spans[0] == (3.0, pytest.approx(4.95, abs=0.011))


def test_digital_silence_alone_has_nothing_to_compare_with():
    assert quiet_spans(levels(np.zeros(5 * RATE, np.float32)), QUIET_HOP) == []
