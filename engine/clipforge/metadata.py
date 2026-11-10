"""Titles and descriptions generated locally from the transcript. No network, no LLM."""

from __future__ import annotations

import math
import re
from collections import Counter

from clipforge.analysis.scoring import EMOTION_WORDS, tokenize
from clipforge.models import RightsType
from clipforge.wordfilter import contains_blocked, is_blocked

TITLE_LIMIT = 70
DESCRIPTION_LIMIT = 4500

GENRE_LABEL = {
    "anime": "Anime",
    "movie": "Movie",
    "series": "Series",
    "gameplay": "Gameplay",
    "podcast": "Podcast",
    "other": "Video",
}


def split_text(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def keywords(text: str, corpus: str, limit: int = 5) -> list[str]:
    """Words common in this clip but not across the whole video (TF-IDF style)."""
    counts = Counter(tokenize(text))
    corpus_counts = Counter(tokenize(corpus))
    total = sum(corpus_counts.values()) or 1

    def weight(word: str) -> float:
        return counts[word] * math.log(1 + total / (1 + corpus_counts[word])) * min(len(word), 10)

    return sorted(counts, key=lambda w: (-weight(w), w))[:limit]


def _sentence_score(sentence: str, position: int) -> float:
    words = sentence.split()
    score = 3.0 - position * 0.3  # An early line works as the hook.
    if 4 <= len(words) <= 14:
        score += 2.0
    if sentence.endswith("?"):
        score += 2.0
    if sentence.endswith("!"):
        score += 1.5
    score += sum(1.0 for w in words if re.sub(r"[^\w]", "", w).lower() in EMOTION_WORDS)
    if len(words) < 3:
        score -= 3.0
    return score


def make_title(text: str, genre: str) -> str:
    # A line with a blocked word is never a title, wherever it sits in the clip.
    candidates = [(i, s) for i, s in enumerate(split_text(text)) if not contains_blocked(s)]
    if not candidates:
        return f"{GENRE_LABEL.get(genre, 'Video')} highlight"
    best = max(candidates, key=lambda item: _sentence_score(item[1], item[0]))[1]
    best = best.rstrip(".,;:…- ")
    if len(best) > TITLE_LIMIT:
        # A whole clause reads better than a sentence chopped at a word.
        clause = re.split(r"(?<=[,;:])\s+", best)[0].rstrip(",;: ")
        if 20 <= len(clause) <= TITLE_LIMIT:
            best = clause
        else:
            best = best[:TITLE_LIMIT].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"
    return best[:1].upper() + best[1:]


def make_summary(text: str, limit: int = 2) -> str:
    return " ".join([s for s in split_text(text) if not contains_blocked(s)][:limit])


def topic_tags(text: str, corpus: str, limit: int = 2) -> list[str]:
    """Hashtags from words the clip is about: they recur in it, are plain letters, and are not blocked.

    A word said once is a line of dialogue, not a topic, so it never becomes a hashtag.
    """
    counts = Counter(tokenize(text))
    chosen = []
    for word in keywords(text, corpus, 12):
        if counts[word] >= 2 and word.isalpha() and len(word) >= 4 and not is_blocked(word):
            chosen.append(word)
        if len(chosen) == limit:
            break
    return chosen


def make_hashtags(text: str, corpus: str, genre: str, part: int | None, fmt: str) -> list[str]:
    """Format, genre and part first; topic words only when the clip clearly has one."""
    tags = ["#shorts"] if fmt == "short" else []
    tags.append(f"#{genre}")
    if part is not None:
        tags.append(f"#part{part}")
    tags += [f"#{w}" for w in topic_tags(text, corpus)]
    return [t for t in dict.fromkeys(tags) if not is_blocked(t.lstrip("#"))]


def credit_line(rights_type: str, detail: str) -> str:
    detail = detail.strip()
    if rights_type == RightsType.creative_commons.value:
        return f"Credit: {detail}"
    if rights_type == RightsType.public_domain.value:
        return f"Source: public domain. {detail}".strip()
    if rights_type == RightsType.written_permission.value:
        return "Used with written permission."
    return ""


def _fill(template: str, values: dict[str, str]) -> str:
    # Only known placeholders are replaced, so a stray brace in a template cannot crash a job.
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def build_metadata(
    text: str,
    corpus: str,
    genre: str,
    part: int | None,
    fmt: str,
    rights_type: str,
    rights_detail: str,
    title_template: str = "{title}",
    description_template: str = "{summary}\n\n{hashtags}",
) -> tuple[str, str]:
    title = make_title(text, genre)
    tags = make_hashtags(text, corpus, genre, part, fmt)
    credit = credit_line(rights_type, rights_detail)
    values = {
        "title": title,
        "part": str(part or ""),
        "genre": GENRE_LABEL.get(genre, "Video"),
        "summary": make_summary(text),
        "hashtags": " ".join(tags),
        "credit": credit,
    }
    final_title = _fill(title_template, values).strip()
    if part is not None and "{part}" not in title_template:
        final_title = f"{final_title} | Part {part}"
    description = _fill(description_template, values).strip()
    # The credit is added even when a channel's template forgets it. Licences require it.
    if credit and credit not in description:
        description = f"{description}\n\n{credit}".strip()
    if len(description) > DESCRIPTION_LIMIT:
        keep = DESCRIPTION_LIMIT - len(credit) - 3
        description = f"{description[:keep]}\n\n{credit}" if credit else description[:DESCRIPTION_LIMIT]
    return final_title[:100], description
