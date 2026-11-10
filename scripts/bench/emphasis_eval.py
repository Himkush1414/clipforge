#!/usr/bin/env python3
"""Does the caption engine make the right word big?

Real podcast transcripts are grouped exactly as captions are (group_words, 9:16, 3 words). A sample
of groups was labelled by hand with the one word that should be big, or none. Each rule picks at most
one word per group, the way the caption engine does, and is scored on:
  agreement   share of groups where the rule picks the labelled word, or picks none when the label is none
  precision   share of the rule's big words that are the labelled word
  recall      share of labelled words that the rule picks

Usage:
  emphasis_eval.py --dump OUT.txt        write the sampled groups as text only, for labelling
  emphasis_eval.py [--set tuning|heldout] [--out results.json]
Labels are in emphasis_labels.json next to this script. Runs in the app's environment and reads the
transcripts cached by pick_eval.py, so the audio must be downloaded as described in nasa_picks.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from clipforge.analysis.scoring import EMOTION_WORDS
from clipforge.analysis.scoring_text import STOPWORDS
from clipforge.analysis.transcribe import Word
from clipforge.captions.layout import (
    GENERIC_WORDS,
    LONG_WORD,
    NUMBER_WORDS,
    group_words,
)
from clipforge.captions.presets import CaptionPreset

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pick_eval", HERE / "pick_eval.py")
assert spec and spec.loader
pick_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick_eval)

LABELS = HERE / "emphasis_labels.json"
# The held-out episode is never looked at while choosing a rule.
SETS = {
    "tuning": [(HERE / "nasa_picks.json", index, 50) for index in range(3)],
    "heldout": [(HERE / "nasa_heldout_picks.json", 0, 60)],
}
SEED = 41


def sampled_groups(name: str) -> list[dict]:
    """The same groups every run: a seeded sample from each episode, in time order."""
    # Case is kept: the engine classifies words before a preset upper-cases them.
    preset = CaptionPreset(name="eval", uppercase=False)
    groups = []
    for picks_file, index, count in SETS[name]:
        video = json.loads(picks_file.read_text())["videos"][index]
        path = Path(video["path"]).expanduser()
        analysis = pick_eval.load_analysis(path, video.get("limit_seconds"))
        words = analysis["words"]
        counts = Counter(clean(w.text) for w in words)
        made = group_words(words, preset, "9:16")
        chosen = sorted(
            random.Random(f"{SEED}:{path.stem}").sample(range(len(made)), count)
        )
        for g in chosen:
            group_word_list = [Word(w.text, w.start, w.end) for w in made[g].words]
            engine = [i for i, w in enumerate(made[g].words) if w.kind == "emphasis"]
            first = next(i for i, w in enumerate(words) if w.start == made[g].start)
            groups.append(
                {
                    "id": f"{path.stem}:{g}",
                    "words": group_word_list,
                    "all_words": words,
                    "first": first,
                    "counts": counts,
                    "audio": analysis["audio"],
                    "engine": engine[0] if engine else None,
                }
            )
    return groups


def previous_rule(group: dict) -> int | None:
    """The engine's rule before this benchmark (kept here to compare against): numbers, strong words, "!",
    all caps and words of 8 or more letters are candidates, stopwords never; the longest candidate wins."""
    words: list[Word] = group["words"]
    candidates = []
    for i, word in enumerate(words):
        c = clean(word.text)
        if not c:
            continue
        shouted = word.text.rstrip("\"')").endswith("!") or (
            len(c) > 2 and word.text.strip(".,!?\"'").isupper()
        )
        if (
            any(ch.isdigit() for ch in c)
            or c in EMOTION_WORDS
            or shouted
            or (c not in STOPWORDS and len(c) >= 8)
        ):
            candidates.append(i)
    return (
        max(candidates, key=lambda i: (len(words[i].text), -i)) if candidates else None
    )


def engine_rule(group: dict) -> int | None:
    """What the caption engine does now, read from group_words itself."""
    return group["engine"]


def clean(text: str) -> str:
    return re.sub(r"[^\w']", "", text).lower()


SENTENCE_END = re.compile(r"[.!?][\"')]*$")


def word_features(group: dict, energy: bool) -> list[dict | None]:
    """Per word: None for words that are never big, else the evidence for it."""
    words, all_words, first = group["words"], group["all_words"], group["first"]
    total = sum(group["counts"].values())
    features: list[dict | None] = []
    for i, word in enumerate(words):
        c = clean(word.text)
        number = any(ch.isdigit() for ch in c) or c in NUMBER_WORDS
        if not number and (
            len(c) < 3 or c in STOPWORDS or c in GENERIC_WORDS or c.startswith("i'")
        ):
            features.append(None)
            continue
        at = first + i
        sentence_start = at == 0 or bool(SENTENCE_END.search(all_words[at - 1].text))
        letters = word.text.lstrip("\"'(-")
        proper = bool(letters) and letters[0].isupper() and not sentence_start
        # Last content word of a sentence: the sentence ends here, or only stopwords follow before it does.
        final = False
        for later in all_words[at : at + 4]:
            if later is not all_words[at] and clean(later.text) not in STOPWORDS:
                break
            if SENTENCE_END.search(later.text):
                final = True
                break
        rarity = -math.log(group["counts"][c] / total)
        loud = 0.0
        if energy:
            audio = group["audio"]
            here = audio.rms_at(word.start, word.end)
            around = audio.rms_at(max(0.0, word.start - 3), word.end + 3)
            loud = (here - around) / 6
        features.append(
            {
                "number": number,
                "proper": proper,
                "final": final,
                "rarity": rarity,
                "loud": loud,
                "long": max(0, len(c) - 5),
                "emotion": c in EMOTION_WORDS,
            }
        )
    return features


def make_rule(
    weights: dict[str, float], threshold: float | None, energy: bool = False
) -> Callable[[dict], int | None]:
    """Score each candidate word; the best one is big if its score reaches the threshold."""

    def rule(group: dict) -> int | None:
        best, best_score = None, -math.inf
        for i, f in enumerate(word_features(group, energy)):
            if f is None:
                continue
            value = sum(weights.get(k, 0.0) * float(v) for k, v in f.items())
            if value >= best_score:
                best, best_score = i, value
        if best is None or (threshold is not None and best_score < threshold):
            return None
        return best

    return rule


def stoplist_rule(group: dict) -> int | None:
    """Any word that is not a stopword or filler; the longest."""
    candidates = [i for i, f in enumerate(word_features(group, False)) if f is not None]
    return (
        max(candidates, key=lambda i: len(group["words"][i].text))
        if candidates
        else None
    )


def priority_rule(min_length: int) -> Callable[[dict], int | None]:
    """Strong words first (number, name, strong word, "!", all caps), the last one; else the longest word
    of at least min_length letters that is not a stopword or generic word."""

    def rule(group: dict) -> int | None:
        features, words = word_features(group, False), group["words"]
        candidates = [i for i, f in enumerate(features) if f is not None]
        strong = []
        for i in candidates:
            f, text = features[i], words[i].text
            shouted = (
                text.rstrip("\"')").endswith("!") or text.strip(".,!?\"'").isupper()
            )
            if f is not None and (
                f["number"] or f["proper"] or f["emotion"] or shouted
            ):
                strong.append(i)
        if strong:
            return strong[-1]
        long_words = [i for i in candidates if len(clean(words[i].text)) >= min_length]
        return (
            max(long_words, key=lambda i: (len(words[i].text), -i))
            if long_words
            else None
        )

    return rule


RULES: dict[str, Callable[[dict], int | None]] = {
    "previous": previous_rule,
    "stoplist": stoplist_rule,
    "rarity": make_rule({"rarity": 1.0}, None),
    **{f"priority_len{n}": priority_rule(n) for n in range(4, 9)},
    "engine": engine_rule,
}
# The engine is meant to be priority_len{LONG_WORD}; main() checks that they agree on every group.


def configs(energy: bool):
    keys = ["number", "proper", "final", "long", "emotion"] + (
        ["loud"] if energy else []
    )
    for values in itertools.product([0.0, 1.0, 2.0], repeat=len(keys)):
        weights = {"rarity": 1.0, **dict(zip(keys, values, strict=True))}
        for threshold in [None, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]:
            yield weights, threshold


def search(groups: list[dict], labels: dict[str, int | None], energy: bool) -> dict:
    """Grid over weights and threshold. Each tuning episode is scored by a config chosen on the other two,
    so the reported figure is not just how well the grid fits these labels."""
    episodes = sorted({g["id"].split(":")[0] for g in groups})
    table = []
    for weights, threshold in configs(energy):
        rule = make_rule(weights, threshold, energy)
        hits = {
            e: [
                rule(g) == labels[g["id"]]
                for g in groups
                if g["id"].startswith(e + ":")
            ]
            for e in episodes
        }
        table.append((weights, threshold, hits))
    held = []
    for episode in episodes:
        best = max(
            table,
            key=lambda row: sum(sum(v) for e, v in row[2].items() if e != episode),
        )
        held.extend(best[2][episode])
    overall = max(table, key=lambda row: sum(sum(v) for v in row[2].values()))
    return {
        "cross_validated_agreement": round(sum(held) / len(held), 3),
        "best_fit_agreement": round(
            sum(sum(v) for v in overall[2].values()) / len(groups), 3
        ),
        "weights": overall[0],
        "threshold": overall[1],
    }


def score(
    rule: Callable[[dict], int | None],
    groups: list[dict],
    labels: dict[str, int | None],
) -> dict:
    agree = picked = right = labelled = 0
    for group in groups:
        want = labels[group["id"]]
        got = rule(group)
        agree += got == want
        picked += got is not None
        labelled += want is not None
        right += got is not None and got == want
    return {
        "agreement": round(agree / len(groups), 3),
        "precision": round(right / picked, 3) if picked else None,
        "recall": round(right / labelled, 3) if labelled else None,
        "big_share": round(picked / len(groups), 3),
    }


def load_labels(groups: list[dict]) -> dict[str, int | None]:
    data = json.loads(LABELS.read_text())["groups"]
    labels = {}
    for group in groups:
        entry = data[group["id"]]
        text = " ".join(w.text for w in group["words"])
        if entry["text"] != text:
            raise SystemExit(
                f"{group['id']}: transcript changed ({entry['text']!r} != {text!r}), relabel"
            )
        labels[group["id"]] = entry["key"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--set", choices=sorted(SETS), default="tuning")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--search",
        action="store_true",
        help="grid search the weights (tuning set only)",
    )
    args = parser.parse_args()

    if args.dump:
        lines = []
        for name in SETS:
            for group in sampled_groups(name):
                numbered = " ".join(
                    f"[{i}]{w.text}" for i, w in enumerate(group["words"])
                )
                first, size = group["first"], len(group["words"])
                before = " ".join(
                    w.text for w in group["all_words"][max(0, first - 8) : first]
                )
                after = " ".join(
                    w.text for w in group["all_words"][first + size : first + size + 8]
                )
                lines.append(
                    f"{name}\t{group['id']}\t{numbered}\t... {before} | {after} ..."
                )
        args.dump.write_text("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} groups to {args.dump}")
        return

    groups = sampled_groups(args.set)
    labels = load_labels(groups)
    if args.search:
        if args.set != "tuning":
            raise SystemExit("search only on the tuning set")
        for energy in (False, True):
            print(f"energy={energy}: {search(groups, labels, energy)}")
        return
    mismatch = [
        g["id"] for g in groups if engine_rule(g) != priority_rule(LONG_WORD)(g)
    ]
    if mismatch:
        print(f"engine differs from priority_len{LONG_WORD} on {mismatch}")
    results = {name: score(rule, groups, labels) for name, rule in RULES.items()}
    none_share = sum(v is None for v in labels.values()) / len(labels)
    results["always_none"] = {"agreement": round(none_share, 3)}
    print(f"{args.set}: {len(groups)} groups, label none in {none_share:.0%}")
    for name, result in results.items():
        print(f"  {name:<14} {result}")
    if args.out:
        args.out.write_text(
            json.dumps(
                {"set": args.set, "groups": len(groups), "results": results}, indent=2
            )
        )


if __name__ == "__main__":
    main()
