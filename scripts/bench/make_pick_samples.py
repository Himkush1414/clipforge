#!/usr/bin/env python3
"""Generate a talk with known good and bad passages, and a picks file saying which are which.

Good passages open with a hook, stand alone and finish their thought. Bad ones are the same length and
just as lively, but open on filler, lean on earlier context, or stop mid-thought, so a scorer has to
judge structure and cannot just count exciting words. The script was written once, before any scorer.

Usage: make_pick_samples.py OUT_FOLDER   (runs in the benchmark environment, which has Piper)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))
import speech_timing as st

GOOD = [
    "Why do cats always land on their feet? It comes down to a reflex that starts within a tenth of a second. A falling cat turns its head first, then its spine follows, and finally its back legs swing around. By the time it reaches the floor, its body is lined up and its legs are ready to absorb the shock. That is why a cat can walk away from a fall that would badly hurt a person.",
    "Nobody expected a broken oven to change the way we eat bread. In the winter of nineteen twenty eight, a baker in Missouri could not get his oven hot enough, so he left the loaves in far longer than usual. The crust came out thick and dark, and customers started asking for it by name. Within a year, every baker in town was copying his mistake. That accident is the reason a dark crust now means good bread.",
    "Here is the strangest thing about honey: it never goes bad. Archaeologists opened tombs in Egypt and found jars of honey that were three thousand years old, and it was still safe to eat. Honey holds so little water that bacteria simply dry out and die inside it. Bees also add an enzyme that makes it slightly acidic. So the jar in your cupboard will outlast you, as long as you keep the lid on.",
    "Most people think lightning never strikes the same place twice, and they are completely wrong. The Empire State Building gets hit about twenty times every single year. Lightning looks for the easiest path to the ground, and a tall metal tower is exactly that. In fact, the first strike does not protect a place at all. So if you are ever caught in a storm, do not shelter under the tree that was already hit.",
    "What would happen if the moon suddenly disappeared? The first thing you would notice is the tides, which would shrink to about a third of their height. Then, over thousands of years, the tilt of the planet would begin to wander, and the seasons would swing wildly. The nights would also be far darker than any human has ever seen. So that quiet rock in the sky turns out to be holding our whole climate steady.",
    "A single decision saved the entire crew of the Endurance. When their ship was crushed by ice in nineteen fifteen, Ernest Shackleton ordered his men to throw away almost everything they owned. Each man could keep only two pounds of belongings. He then threw his own gold watch into the snow to show that he meant it. Every one of his twenty seven men survived nearly two years on the ice.",
    "The best advice I ever got about money fits in one sentence: pay yourself first. On the day your wages arrive, move a tenth of them into savings before you spend anything else. You will not miss it, because you learn to live on what is left. After thirty years, that small habit can grow into more than your whole salary for a decade. That one rule has made more people comfortable than any clever investment.",
    "Your brain is using a fifth of your energy right now, even though it is only two percent of your weight. It burns about as much power as a dim light bulb, all day and all night. Even when you sleep, it barely slows down, because that is when it sorts your memories. So when you feel tired after a hard day of thinking, you are not imagining it. Thinking really is hard physical work.",
]
BAD = [
    "So, um, and then after that we kind of moved on to the other thing, which I mentioned before. Anyway, like I said, it was really not that different from the last one we talked about. He said the same thing about it, and they all more or less agreed with him at the time. That one was a bit better, I suppose, but those were not the main points anyway. So yeah, that is more or less where that part of it stands, and",
    "And that is why it matters, as we were saying earlier, because of what happened with the first one. It was amazing, honestly, and everybody loved it, but it also depended on those other things from before. They kept going back to that same idea, and he never really explained it. This all connects back to what she was describing, which is another story. But to understand why, we first need to go back and look at something else, because",
    "But yeah, right, so the other part of it, I guess, is that one thing they did. You know what I mean, it was incredible, but it only makes sense if you heard the earlier section. Those people were there too, and that was the problem with it, as I said. Then it just kept going like that for a while, and they did it again. Which brings us to the next question, and the answer to that is going to take a while, so",
    "Well, okay, so that is basically the same thing again, as you can probably tell. He went back and did the other one too, and it was a total disaster, as we covered. They said it was fine, and it turned out they were right about that part, more or less. These were all the usual issues that came up last time, and nothing new. So that is the first half of it, and the second half is where it starts to",
    "Yeah, and also, like, the thing is that it was a lot like what I told you about before. She did it that way because of them, and that made it hard for everybody involved. Those two were always going to clash over this, and the rest of that story you already know. It was unbelievable, really, but we already went through all of that. So anyway, moving along, there was another one of those a little later on, which was",
    "Um, so then, right after that, they went ahead and did it again the same way. This was the bit I was talking about, where he changed his mind about that other thing. It was shocking at the time, but it really only matters because of what came before it. Those numbers from earlier still apply here, so I will not go through them all again. And that leads into the next part, which depends on all of this, because they",
    "Okay, so as I was saying, that was the second one of those, and it went about the same. They had already decided by then, because of what he said to them last time. It was a huge fight, and I mean that, but you really had to be following along from the start. Those details are all in that other section, and we will come back to it. So hold that thought, because first I have to explain something else about",
    "And so, you know, it was kind of the same deal with that one too, like I mentioned. He told them about it, and they told her, and then it all came back around again. It was honestly crazy, but it was really just more of what we were just talking about. That is also why those ones did not work out the way they had hoped. Which is sort of the point I was getting to earlier, and we will get there, but",
]


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    say = st.piper_say(Path.home() / "clipforge-work" / "bench" / "voices", "en")
    rng = random.Random(11)
    passages = [("good", t) for t in GOOD] + [("bad", t) for t in BAD]
    rng.shuffle(passages)
    audio: list[np.ndarray] = [np.zeros(st.RATE, np.float32)]
    position = audio[0].size
    picks = []
    for kind, text in passages:
        start = position / st.RATE
        for sentence in st.re.split(r"(?<=[.!?:])\s+", text):
            voice = say(sentence)
            gap = np.zeros(int(rng.uniform(0.25, 0.45) * st.RATE), np.float32)
            audio += [voice, gap]
            position += voice.size + gap.size
        if kind == "good":
            picks.append({"start": round(start, 2), "end": round(position / st.RATE, 2), "note": text[:50]})
        pause = np.zeros(int(0.9 * st.RATE), np.float32)
        audio.append(pause)
        position += pause.size
    wav = out / "generated-talk.wav"
    wavfile.write(wav, st.RATE, (np.concatenate(audio) * 32767).astype(np.int16))
    target = round(sum(p["end"] - p["start"] for p in picks) / len(picks))
    (out / "generated-picks.json").write_text(
        json.dumps(
            {
                "_about": "Generated by make_pick_samples.py. The picks are the passages written to be good.",
                "videos": [
                    {"path": str(wav), "genre": "podcast", "clip_seconds": target, "picks": picks}
                ],
            },
            indent=1,
        )
    )
    print(wav, position / st.RATE, "seconds", target, "s target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
