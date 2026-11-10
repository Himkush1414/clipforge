"""Words that must never appear in generated titles, descriptions or hashtags.

Sexual, violent, profane and hateful terms, in English and in Hindi (Devanagari and the usual Latin
spellings). Generated text comes from the transcript, so a line of dialogue could otherwise become a
clip's public title or hashtag. Matching is per word, ignoring case and punctuation; a root also
blocks its longer forms ("fuck" blocks "fucking").
"""

from __future__ import annotations

import re

# Whole words only: these are too short or too common as parts of ordinary words to match as roots.
_WORDS = """
ass asses arse anal anus boob boobs booby breast breasts butt butts tit tits titty titties nipple nipples
penis penises vagina vaginas cock cocks dick dicks pussy pussies cum cums sex sexy sexual nude nudes naked
horny hooker hookers whore whores slut sluts bitch bitches bastard bastards damn crap piss pissed
kill kills killed killer killing murder murders murdered murderer gore gory blood bloody stab stabbed
behead beheaded corpse corpses torture tortured suicide shoot shooting gun guns bomb bombs
nazi nazis kike kikes spic spics chink chinks fag fags gook gooks wetback tranny retard retards retarded
harami haramkhor kamina kamine kutta kutti kutiya saala saali randi gaand gand lauda lund
बलात्कार हत्या खून चूत चूतिया चुतिया मादरचोद बहनचोद भोसड़ी भोसड़ीके भोसडी गांड गाँड
लौड़ा लौडा लंड रंडी हरामी हरामखोर कमीना कमीने कुत्ता कुत्ती कुतिया साला साली
"""
# Roots: blocked with any ending.
_ROOTS = """
fuck shit motherfuck asshole dickhead bullshit porn masturbat orgasm erotic genital rape rapist
slaughter massacre dismember decapitat terroris nigg faggot whor cunt
chutiya chutiye madarchod behenchod bhenchod bhosd bsdk
"""
BLOCKED_WORDS = frozenset(_WORDS.split())
BLOCKED_ROOTS = tuple(sorted(_ROOTS.split()))

# Devanagari vowel signs are not "word" characters to Python, so they are listed explicitly.
_TOKEN = re.compile(r"[\wऀ-ॿ']+")


def words(text: str) -> list[str]:
    return [t.strip("'").lower() for t in _TOKEN.findall(text) if t.strip("'")]


def is_blocked(word: str) -> bool:
    token = word.strip("'").lower()
    token = re.sub(r"'s$", "", token)
    return token in BLOCKED_WORDS or token.startswith(BLOCKED_ROOTS)


def contains_blocked(text: str) -> bool:
    return any(is_blocked(token) for token in words(text))
