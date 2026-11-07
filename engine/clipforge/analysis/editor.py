"""The second pass: judge each candidate clip the way an editor would.

The scorer finds lively moments, but a lively moment can still make a poor clip. This pass asks four
questions of each candidate. Does it open with a hook, not filler? Does it make sense with no earlier
context? Does it finish its thought? Is it a whole topic, not a slice of one? A candidate that fails is
trimmed if that fixes it, and dropped if not.

Word lists exist for English and Spanish. Other languages are judged on structure alone (pauses, end
punctuation, energy, topic), since a wrong word list would be worse than none.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from clipforge.analysis.scoring_text import END_MARK, content_tokens
from clipforge.analysis.transcribe import Word

HOOK_SECONDS = 3.0
MAX_THOUGHT_SECONDS = 35.0
HARD_PAUSE = 1.0
GATE = {"self_contained": 0.4, "complete": 0.4}
TOPIC_WINDOW = 3


@dataclass(frozen=True)
class Lexicon:
    # Words that say nothing at the start of a clip, and can simply be trimmed away.
    filler: frozenset[str]
    # Phrases made of filler, matched before single words.
    filler_phrases: tuple[tuple[str, ...], ...]
    # Words a clip cannot end on, since the sentence clearly continues.
    open_end: frozenset[str]
    # Words that point at something said earlier, when they open a clip.
    dangling: frozenset[str]
    # Phrases anywhere in a clip that lean on what came before.
    back_references: tuple[str, ...]
    # Words that create curiosity near the start.
    curiosity: frozenset[str]
    # Words that promise a personal story.
    story: tuple[str, ...]
    # Reactions from a second speaker. A clip should not end on one.
    reactions: frozenset[str]
    # Words that tell the listener the point has arrived.
    payoff: tuple[str, ...]
    # Words that continue the previous thought when the next one opens with them.
    continuing: frozenset[str]
    # Show business rather than content.
    boilerplate: tuple[str, ...]


ENGLISH = Lexicon(
    filler=frozenset(
        """um uh uhm er ah hmm mm so and but well yeah yes yep okay ok right like or then also anyway
        because exactly absolutely sure actually basically honestly oh""".split()
    ),
    filler_phrases=(("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of"), ("i", "guess")),
    open_end=frozenset(
        """and but or so because which that who whom whose the a an to of in on at for with from by as if
        than then when while although though my your his her their our its is are was were be been am
        can could would should will have has had do does did i we you he she they it into about
        like""".split()
    ),
    dangling=frozenset(
        """he she they it that this those these them him her his their its there which who whom whose
        then another other others same such one ones it's that's they're he's she's there's""".split()
    ),
    back_references=(
        "as i said",
        "like i said",
        "as i was saying",
        "i mentioned",
        "you mentioned",
        "we mentioned",
        "we talked about",
        "we were talking",
        "we were just",
        "we covered",
        "we were saying",
        "as we said",
        "you said",
        "i told you",
        "i said earlier",
        "earlier section",
        "last time",
        "the last one",
        "the other one",
        "that other",
        "go back to",
        "come back to",
    ),
    curiosity=frozenset(
        """why never always nobody everyone everybody secret truth mistake strangest best worst only
        problem imagine remember forget wrong impossible surprising biggest hardest""".split()
    ),
    story=(
        "i remember",
        "i will never",
        "i'll never",
        "i never",
        "i wanted",
        "i thought",
        "one time",
        "the day",
        "my mom",
        "my dad",
        "my mother",
        "my father",
    ),
    reactions=frozenset(
        """absolutely amazing exactly right yeah yes okay ok perfect wow interesting beautiful sure
        thanks""".split()
    ),
    payoff=(
        "that's why",
        "that is why",
        "the reason",
        "which means",
        "that means",
        "the point",
        "never forget",
        "the lesson",
        "turns out",
        "in the end",
        "more than",
    ),
    continuing=frozenset("because which but and so although though however".split()),
    boilerplate=(
        "welcome to",
        "welcome back",
        "on this podcast",
        "this podcast",
        "thank you for joining",
        "thanks for joining",
        "thank you so much for",
        "thanks for watching",
        "thank you for watching",
        "subscribe",
        "follow us",
        "our website",
        "your host",
        "your hosts",
        "be one of your",
        "let's get started",
    ),
)

SPANISH = Lexicon(
    filler=frozenset("eh este pues bueno y o pero entonces o sea vale sí claro también además".split()),
    filler_phrases=(("o", "sea"), ("es", "decir"), ("a", "ver")),
    open_end=frozenset(
        "y o pero que porque de del la el los las un una en con por para a al si como cuando mi tu su".split()
    ),
    dangling=frozenset(
        "él ella ellos ellas eso esto esa ese estos esas esos aquello entonces otro otra".split()
    ),
    back_references=("como dije", "como decía", "ya mencioné", "hablamos de", "lo que dije", "antes dije"),
    curiosity=frozenset(
        """por qué cómo qué nunca siempre nadie todos secreto verdad error mejor peor primero único problema
        imagina recuerda""".split()
    ),
    story=("recuerdo", "nunca olvidaré", "una vez", "el día", "mi madre", "mi padre"),
    reactions=frozenset("claro exacto vale bueno sí genial perfecto".split()),
    payoff=("por eso", "la razón", "lo que significa", "al final", "resulta que"),
    continuing=frozenset("porque pero y aunque sin embargo".split()),
    boilerplate=("bienvenidos", "suscríbete", "gracias por ver", "nuestro canal"),
)

LEXICONS = {"en": ENGLISH, "es": SPANISH}


def lexicon_for(language: str | None) -> Lexicon | None:
    """A language with no word list is judged on structure alone. Unknown means assume English."""
    if language is None:
        return ENGLISH
    return LEXICONS.get(language.split("-")[0].lower())


@dataclass
class Thought:
    """One complete utterance, as far as can be told: fragments joined up to an end mark or long pause."""

    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def split_thoughts(words: Sequence[Word]) -> list[Thought]:
    """Join words until an end mark or a long pause.

    Whisper breaks speech at every breath, which gives "sentences" such as "And" or "NASA astronaut."
    A clip built on those starts mid-sentence, however clean it looks.
    """
    thoughts: list[Thought] = []
    current: list[Word] = []
    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else None
        pause = following.start - word.end if following else math.inf
        too_long = word.end - current[0].start >= MAX_THOUGHT_SECONDS
        if END_MARK.search(word.text) or pause >= HARD_PAUSE or too_long:
            thoughts.append(Thought(current))
            current = []
    return thoughts


def _plain(word: Word | str) -> str:
    text = word.text if isinstance(word, Word) else word
    return re.sub(r"^[^\w']+|[^\w']+$", "", text.lower())


def trim_filler(words: list[Word], lexicon: Lexicon | None) -> list[Word]:
    """Drop filler from both ends, as an editor would. "And so I left, and" becomes "I left"."""
    if lexicon is None:
        return words
    begin = 0
    while begin < len(words):
        pair = (_plain(words[begin]), _plain(words[begin + 1])) if begin + 1 < len(words) else None
        if pair in lexicon.filler_phrases:
            begin += 2
        elif _plain(words[begin]) in lexicon.filler:
            begin += 1
        else:
            break
    finish = len(words)
    while finish > begin:
        last = words[finish - 1]
        # A word with an end mark really did close the sentence, so "that's right." stays.
        if not END_MARK.search(last.text) and _plain(last) in lexicon.open_end:
            finish -= 1
        else:
            break
    return words[begin:finish]


@dataclass
class Verdict:
    hook: float
    self_contained: float
    complete: float
    on_topic: float
    problems: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    @property
    def score(self) -> float:
        return 0.45 * self.hook + 0.25 * self.self_contained + 0.2 * self.complete + 0.1 * self.on_topic


def judge_hook(
    words: list[Word], pause_before: float, energy: float, lexicon: Lexicon | None
) -> tuple[float, list[str]]:
    """How strongly do the first seconds pull a listener in?"""
    strengths: list[str] = []
    opening = [w for w in words if w.start < words[0].start + HOOK_SECONDS] or words[:1]
    points = 0.15 * min(energy, 1.0)
    if pause_before >= 0.4:
        points += 0.1
    if lexicon is None:
        return min(1.0, points + 0.35), strengths
    plain = [_plain(w) for w in opening]
    meaningful = [t for t in plain if t and t not in lexicon.filler and len(content_tokens(t)) > 0]
    points += 0.2 * (len(meaningful) / max(len(plain), 1))
    first_sentence: list[Word] = []
    for word in words:
        first_sentence.append(word)
        if END_MARK.search(word.text):
            break
    if (
        first_sentence[-1].text.rstrip("\"')").endswith("?")
        and first_sentence[-1].end - words[0].start <= 8.0
    ):
        points += 0.35
        strengths.append("opens with a question")
    window = [_plain(w) for w in words[:14]]
    if lexicon.curiosity & set(window):
        points += 0.2
        if not strengths:
            strengths.append("opens with a strong claim")
    text = " ".join(window)
    if any(phrase in text for phrase in lexicon.story):
        points += 0.2
        if not strengths:
            strengths.append("opens a personal story")
    if any(re.search(r"\d", w.text) for w in opening):
        points += 0.08
    return min(1.0, points), strengths


def judge_self_contained(
    words: list[Word], at_edge: bool, lexicon: Lexicon | None
) -> tuple[float, list[str]]:
    """Would this make sense to someone who has heard nothing before it?"""
    if lexicon is None:
        return 0.7, []
    score = 1.0
    problems: list[str] = []
    opener = _plain(words[0])
    if opener in lexicon.dangling:
        score -= 0.5
        problems.append(f'opens on "{opener}", which points at something said earlier')
    text = " " + " ".join(_plain(w) for w in words) + " "
    leans = [phrase for phrase in lexicon.back_references if f" {phrase} " in text]
    if leans:
        score -= min(0.6, 0.3 * len(leans))
        problems.append(f'relies on earlier context ("{leans[0]}")')
    if any(phrase in text for phrase in lexicon.boilerplate):
        score -= 0.6
        problems.append("is show introduction or sign-off")
    if at_edge:
        score -= 0.1
    return max(0.0, score), problems


def judge_complete(
    words: list[Word], pause_after: float, next_word: str, lexicon: Lexicon | None
) -> tuple[float, list[str]]:
    """Does it finish its thought, or just stop?"""
    last = words[-1]
    finished = bool(END_MARK.search(last.text))
    score = 0.6 if finished else 0.3
    problems: list[str] = []
    if pause_after >= 0.3:
        score += 0.2
    if last.text.rstrip("\"')").endswith("?"):
        score -= 0.45
        problems.append("ends on an unanswered question")
    if lexicon is not None:
        final = _plain(last)
        if not finished and final in lexicon.open_end:
            score -= 0.4
            problems.append(f'stops mid-sentence on "{final}"')
        if finished and final in lexicon.reactions and len(words) > 1:
            score -= 0.2
        if next_word in lexicon.continuing and pause_after < 0.6:
            score -= 0.3
            problems.append(f'stops just before "{next_word}", which continues the thought')
        ending = " ".join(_plain(w) for w in words[-16:])
        if any(phrase in ending for phrase in lexicon.payoff):
            score += 0.2
    if score < GATE["complete"] and not problems:
        problems.append("does not reach a clear ending")
    return float(max(0.0, min(1.0, score))), problems


def _vector(thoughts: Sequence[Thought], idf: dict[str, float]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for thought in thoughts:
        counts.update(content_tokens(thought.text))
    return Counter({term: count * idf.get(term, 1.0) for term, count in counts.items()})


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def judge_topic(thoughts: Sequence[Thought], first: int, last: int, idf: dict[str, float]) -> float:
    """1 when a clip holds together and its edges are clean, near 0 when it is a slice of a bigger topic."""
    inside = thoughts[first : last + 1]
    opening = _vector(inside[:TOPIC_WINDOW], idf)
    closing = _vector(inside[-TOPIC_WINDOW:], idf)
    before = _vector(thoughts[max(0, first - TOPIC_WINDOW) : first], idf)
    after = _vector(thoughts[last + 1 : last + 1 + TOPIC_WINDOW], idf)
    within = _cosine(opening, closing)
    edges = max(_cosine(before, opening), _cosine(closing, after))
    return float(max(0.0, min(1.0, 0.5 + within - edges)))


def inverse_document_frequency(thoughts: Sequence[Thought]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for thought in thoughts:
        counts.update(set(content_tokens(thought.text)))
    total = max(1, len(thoughts))
    return {term: math.log((1 + total) / (1 + count)) + 1 for term, count in counts.items()}


def judge(
    words: list[Word],
    thoughts: Sequence[Thought],
    first: int,
    last: int,
    idf: dict[str, float],
    energy: float,
    duration: float,
    lexicon: Lexicon | None,
) -> Verdict:
    """All four questions for one candidate, given as trimmed words and the thoughts they came from."""
    previous_end = thoughts[first - 1].end if first > 0 else 0.0
    next_thought = thoughts[last + 1] if last + 1 < len(thoughts) else None
    pause_before = words[0].start - previous_end
    pause_after = next_thought.start - words[-1].end if next_thought else 1.0
    next_word = _plain(next_thought.words[0]) if next_thought else ""
    at_edge = words[0].start < duration * 0.03 or words[-1].end > duration * 0.97
    hook, strengths = judge_hook(words, pause_before, energy, lexicon)
    contained, loose = judge_self_contained(words, at_edge, lexicon)
    complete, unfinished = judge_complete(words, pause_after, next_word, lexicon)
    topic = judge_topic(thoughts, first, last, idf)
    problems: list[str] = []
    if contained < GATE["self_contained"] or (loose and contained <= 0.5):
        problems += loose
    problems += unfinished
    if contained >= 0.9:
        strengths.append("makes sense on its own")
    if complete >= 0.75:
        strengths.append("ends on a complete thought")
    if topic >= 0.7:
        strengths.append("stays on one topic")
    return Verdict(hook, contained, complete, topic, problems, strengths)


def explain(verdict: Verdict, traits: dict[str, float]) -> str:
    """A short plain reason for the clip card."""
    if verdict.problems:
        return f"Best available, though it {verdict.problems[0]}."
    parts = verdict.strengths[:3]
    if not parts:
        parts = ["passes the editor's checks"]
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    sentence = text[0].upper() + text[1:] + "."
    # Topic is left out: most stretches of speech reach a high score there, so it says nothing.
    strongest = max(("speech", "emotion", "action"), key=lambda k: traits.get(k, 0.0))
    extra = {
        "speech": "Lively delivery.",
        "emotion": "Strong emotion.",
        "action": "Lots of on-screen action.",
    }
    return f"{sentence} {extra[strongest]}" if traits.get(strongest, 0.0) >= 0.6 else sentence
