#!/usr/bin/env python3
"""Word timing in Hindi, Arabic and Japanese, on more than 100 words each, with exact references.

Each word is synthesized alone and joined with known gaps, so its true start is known to the sample.
Hindi and Arabic use Piper voices, Japanese uses Kokoro, with words split by fugashi.

Words are matched by characters, not whole words, for two reasons. Japanese has no spaces, so Whisper's
words and the script's words split differently. And a lone match beside misheard words is unreliable:
Whisper once heard "हूँ" as "के", and a word-level match paired it with a real "के" two seconds later,
which looked like a 2 second timing error. Only runs of at least MIN_RUN agreeing characters count.

Usage: long_timing.py OUT.json [--runs 3] [--languages hi,ar,ja]   (runs in the benchmark environment)
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import speech_timing as st

from clipforge.analysis import align
from clipforge.analysis.transcribe import Word, sanitize_words

MIN_RUN = 4

TEXTS = {
    "hi": (
        "नमस्ते दोस्तों आज हम लकड़ी की एक छोटी नाव बनाएंगे। बहुत से लोग सोचते हैं कि इसके लिए महंगे औजार चाहिए "
        "लेकिन ऐसा बिल्कुल नहीं है। एक तेज़ चाकू थोड़ा रेगमाल और धैर्य काफी है। सबसे पहले लकड़ी पर नाव का आकार बनाइए। "
        "इस काम में जल्दी मत कीजिए क्योंकि यही सबसे ज़रूरी हिस्सा है। अब मैं आपको एक राज़ बताता हूँ। हमेशा चाकू को "
        "अपने हाथ से दूर चलाइए। अगर आप आज सिर्फ एक बात याद रखें तो यही बात याद रखिए। अब विषय बदलते हैं और आसमान "
        "की बात करते हैं। क्या आपने कभी सोचा है कि तारे कितनी दूर हैं। आज रात जो रोशनी आप देखते हैं वह हज़ारों साल "
        "पहले चली थी। यह सोचकर हैरानी होती है कि हम अतीत को देख रहे हैं। एक छोटी दूरबीन से भी शनि के छल्ले दिखाई "
        "देते हैं। अब रोटी की बात करते हैं क्योंकि ताज़ी रोटी सबको पसंद है। अच्छी रोटी के लिए आटा पानी नमक और "
        "थोड़ा धैर्य चाहिए। सबसे बड़ी गलती है आटे को जल्दी में गूंधना। आटे को आराम करने दीजिए तो स्वाद और गहरा हो "
        "जाता है। तो आज के लिए बस इतना ही। देखने के लिए आपका बहुत धन्यवाद।"
    ),
    "ar": (
        "قبل أن نبدأ أريد أن أشكر كل من شاهد الحلقة الماضية وكتب لنا تعليقا جميلا. "
        "مرحبا بكم مرة أخرى في الورشة. اليوم سنبني قاربا خشبيا صغيرا من بقايا الخشب. يعتقد كثير من الناس أنهم "
        "بحاجة إلى أدوات غالية لكن هذا غير صحيح أبدا. سكين حادة وقليل من ورق الصنفرة والكثير من الصبر تكفي. أولا "
        "ارسم شكل القارب على قطعة الخشب. خذ وقتك في هذه الخطوة لأنها الأهم. الآن سأخبركم بالسر الذي لا يخبركم به "
        "أحد. اقطع دائما بعيدا عن يدك وليس نحوها. إذا تذكرت شيئا واحدا اليوم فتذكر هذا. والآن لنغير الموضوع ونتحدث "
        "عن السماء في الليل. هل تساءلت يوما كم تبعد عنا النجوم. الضوء الذي تراه الليلة خرج من بعض النجوم قبل آلاف "
        "السنين. هذا أمر مدهش حقا لأنك تنظر مباشرة إلى الماضي. حتى منظار صغير في حديقتك يمكنه أن يريك حلقات زحل. "
        "والآن لنتحدث عن الخبز لأن الجميع يحب الخبز الطازج. الخبز الجيد يحتاج فقط إلى الدقيق والماء والملح والخميرة. "
        "أكبر خطأ يقع فيه المبتدئون هو الاستعجال. اترك العجين يرتاح ببطء فيصبح طعمه أعمق وأغنى. هذا كل شيء لهذا "
        "اليوم. شكرا جزيلا على المشاهدة. "
        "في الحلقة القادمة سنصنع كرسيا صغيرا للحديقة ونتعلم كيف نختار الخشب المناسب لكل مشروع."
    ),
    "ja": (
        "皆さん、こんにちは。今日は小さな木の船を作ります。高い道具が必要だと思う人が多いですが、そんなことはありません。"
        "よく切れるナイフと紙やすりと少しの忍耐があれば十分です。まず、木の上に船の形を描きます。ここは一番大切なので、"
        "ゆっくりやりましょう。ここで誰も教えてくれない秘密をお話しします。ナイフはいつも自分の手から離れる方向に動かして"
        "ください。今日一つだけ覚えるなら、これを覚えてください。では、話を変えて夜空について話しましょう。星がどれくらい"
        "遠くにあるか考えたことがありますか。今夜見える光は、何千年も前に星を出発しました。私たちは過去を見ているのです。"
        "小さな望遠鏡でも土星の輪が見えます。次はパンの話です。おいしいパンに必要なのは、小麦粉と水と塩と少しの忍耐です。"
        "一番大きな間違いは、急いで生地をこねることです。生地をゆっくり休ませると、味が深くなります。今日はここまでです。"
        "見てくれて本当にありがとうございました。"
    ),
}


def tokens(language: str) -> list[str]:
    text = TEXTS[language]
    if language == "ja":
        import fugashi

        tagger = fugashi.Tagger()
        return [w.surface for w in tagger(text) if re.search(r"\w", w.surface)]
    return text.split()


def japanese_say():
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="j", repo_id="hexgrad/Kokoro-82M")

    def say(text: str) -> np.ndarray:
        parts = [np.asarray(r.audio) for r in pipeline(text, voice="jf_alpha")]
        return st.to_rate(np.concatenate(parts), 24000) if parts else np.zeros(1600, np.float32)

    return say


def word_by_word(language: str, say, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = random.Random(seed)
    pieces, truth = [np.zeros(st.RATE // 2, np.float32)], []
    position = pieces[0].size
    for token in tokens(language):
        voice = st.trim(say(token))
        truth.append({"text": token, "start": position / st.RATE, "end": (position + voice.size) / st.RATE})
        gap = np.zeros(int(rng.uniform(0.05, 0.15) * st.RATE), np.float32)
        if re.search(r"[.。!?؟।]$", token):
            gap = np.zeros(int(0.5 * st.RATE), np.float32)
        pieces += [voice, gap]
        position += voice.size + gap.size
    return np.concatenate(pieces), truth


def letters(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNM")


def char_pairs(found: list[dict], truth: list[dict]) -> list[tuple[dict, dict]]:
    """(found word, true word) pairs whose first letters line up inside a run of agreeing letters."""

    def flatten(words: list[dict]) -> tuple[str, list[int], set[int]]:
        chars, owner, firsts = [], [], set()
        for index, word in enumerate(words):
            part = letters(word["text"])
            if part:
                firsts.add(len(chars))
            chars += list(part)
            owner += [index] * len(part)
        return "".join(chars), owner, firsts

    f_text, f_owner, f_firsts = flatten(found)
    t_text, t_owner, t_firsts = flatten(truth)
    pairs = []
    for block in difflib.SequenceMatcher(None, f_text, t_text, autojunk=False).get_matching_blocks():
        if block.size < MIN_RUN:
            continue
        for k in range(block.size):
            fi, ti = block.a + k, block.b + k
            if fi in f_firsts and ti in t_firsts:
                pairs.append((found[f_owner[fi]], truth[t_owner[ti]]))
    return pairs


def errors(found: list[dict], truth: list[dict]) -> dict:
    pairs = char_pairs(found, truth)
    if not pairs:
        return {"words_scored": 0, "words_total": len(truth)}
    s = np.array([(f["start"] - t["start"]) * 1000 for f, t in pairs])
    e = np.array([(f["end"] - t["end"]) * 1000 for f, t in pairs])
    return {
        "words_scored": len(pairs),
        "words_total": len(truth),
        "start_median_abs_ms": round(float(np.median(np.abs(s))), 1),
        "start_p90_abs_ms": round(float(np.percentile(np.abs(s), 90)), 1),
        "start_max_abs_ms": round(float(np.abs(s).max()), 1),
        "start_within_100ms": round(float((np.abs(s) <= 100).mean()), 3),
        "end_median_abs_ms": round(float(np.median(np.abs(e))), 1),
        "end_p90_abs_ms": round(float(np.percentile(np.abs(e), 90)), 1),
    }


class GuardCount(logging.Handler):
    """Counts words the guard sent back to Whisper's times, from the aligner's own log."""

    def __init__(self):
        super().__init__()
        self.words = 0

    def emit(self, record):
        if "moved more than" in record.getMessage():
            self.words += int(record.args[0])


def aligned(spans: list[dict], audio: np.ndarray, language: str) -> tuple[list[dict], str]:
    segments = [
        align.Segment(s["start"], s["end"], [Word(w["text"], w["start"], w["end"]) for w in s["words"]])
        for s in spans
    ]
    words, method = align.align_segments(audio, segments, language)
    return [{"text": w.text, "start": w.start, "end": w.end} for w in sanitize_words(words)], method


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--languages", default="hi,ar,ja")
    parser.add_argument("--dump", type=Path, help="also save each sample, to study it without Whisper")
    args = parser.parse_args()
    voices = Path.home() / "clipforge-work" / "bench" / "voices"
    whisper = st.Whisper("large-v3-turbo", "int8_float16", Path.home() / "clipforge-work" / "models")
    results: dict = {}
    for language in args.languages.split(","):
        say = japanese_say() if language == "ja" else st.piper_say(voices, language)
        runs = []
        for run in range(args.runs):
            audio, truth = word_by_word(language, say, seed=3 + run)
            words, spans, _ = whisper.run(audio, language)
            if args.dump:
                import pickle

                args.dump.mkdir(parents=True, exist_ok=True)
                sample = {"audio": audio, "truth": truth, "spans": spans, "words": words, "language": language}
                (args.dump / f"{language}-{run}.pkl").write_bytes(pickle.dumps(sample))
            row: dict = {"whisper": errors(words, truth)}
            counter = GuardCount()
            logging.getLogger("clipforge.align").addHandler(counter)
            fixed, method = aligned(spans, audio, language)
            logging.getLogger("clipforge.align").removeHandler(counter)
            row["words_guarded"] = counter.words
            row["aligned" if method == "aligned" else "no aligner installed"] = errors(fixed, truth)
            moved = [abs(a["start"] - w["start"]) for a, w in zip(fixed, words, strict=False)]
            row["largest_move_ms"] = round(max(moved) * 1000, 1) if moved else None
            runs.append(row)
            print(language, run, json.dumps(row, ensure_ascii=False), flush=True)
        results[language] = {"words_in_script": len(tokens(language)), "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
