"""Small text helpers shared by the scorer and the editor."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being but by
    can could did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its just me more most my no nor not now of off on once only or
    other our out over own same she should so some such than that the their them then there these they
    this those through to too under until up very was we were what when where which while who whom why
    will with would you your yeah okay ok um uh like gonna wanna got get really""".split()
)

END_MARK = re.compile(r"[.!?…。！？]+[\"')\]]*$")


def content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]
