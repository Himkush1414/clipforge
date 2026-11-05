# Caption presets

A preset is a JSON object. Sizes are relative to the frame, so a preset looks the same at 1080p and 8K. Edit presets on the **Caption presets** page, or through `/api/presets`.

## Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | text | `Custom` | Display name |
| `font` | font name | `Anton` | Main font |
| `emphasis_font` | font name or null | null | Font for big words |
| `uppercase` | boolean | true | Force capitals |
| `max_words` | 1 to 3 | 3 | Words on screen at once |
| `size` | 0.03 to 0.25 | 0.1 | Text size as a share of the frame's short side |
| `emphasis_scale` | 1.0 to 2.0 | 1.3 | Size multiplier for big words |
| `filler_scale` | 0.5 to 1.0 | 0.8 | Size multiplier for filler words |
| `text_color` | `#RRGGBB` | `#FFFFFF` | Normal text |
| `active_color` | `#RRGGBB` | `#FFE600` | The word being spoken |
| `emphasis_color` | `#RRGGBB` or null | null | Big words when not active |
| `outline_color` | `#RRGGBB` | `#000000` | Outline |
| `outline` | 0 to 0.3 | 0.08 | Outline width, share of text size |
| `shadow_color` | `#RRGGBB` | `#000000` | Shadow |
| `shadow` | 0 to 0.3 | 0.04 | Shadow depth, share of text size |
| `glow_color` | `#RRGGBB` or null | null | Glow colour |
| `glow` | 0 to 0.5 | 0 | Glow size, share of text size |
| `active_style` | `color` or `box` | `color` | Colour the active word, or draw a box behind it |
| `box_color` | `#RRGGBB` | `#7C3AED` | Box colour |
| `animation` | `pop`, `bounce`, `fade`, `slide`, `none` | `pop` | How a group appears |
| `active_pop` | 1.0 to 1.5 | 1.12 | Scale of the active word at its peak |
| `tilt` | 0 to 12 | 0 | Alternating word tilt in degrees |
| `position` | `lower`, `center`, `upper` | `lower` | Vertical position |
| `y_offset` | -0.2 to 0.2 | 0 | Nudge, as a share of frame height |
| `word_gap` | 0.1 to 0.8 | 0.28 | Gap between words, share of text size |

## Built-in presets

| ID | Name | Style |
|---|---|---|
| `anime-impact` | Anime Impact | Anton, yellow active word, red big words, heavy outline, pop |
| `manga-punch` | Manga Punch | Bangers, tilted words, bounce |
| `neon-glow` | Neon Glow | Lilita One, cyan active word, magenta glow |
| `karaoke-box` | Karaoke Box | Poppins Black, box behind the active word |
| `one-word-punch` | One Word Punch | Archivo Black, one word at a time, centered |
| `podcast-clean` | Podcast Clean | Poppins ExtraBold, mixed case, gentle fade |
| `cinema-subtle` | Cinema Subtle | Bebas Neue, restrained, slide |

Built-in presets cannot be changed. Save a copy to edit one.

## How words are grouped

- A group holds up to `max_words` words and has a character limit per ratio.
- A group ends at punctuation or a pause of 0.35 seconds or more.
- Groups never overlap. A group stays on screen until the next one starts when the gap is under 0.2 seconds, which prevents flicker. Otherwise it holds for up to 0.3 seconds.
- Each group gets at most one big word.

## Word sizes

| Kind | Rule |
|---|---|
| Emphasis (big) | Numbers, strong words, words ending in `!`, all-caps words, words of 8 or more letters |
| Filler (small) | Common short words such as "the" and "to" |
| Normal | Everything else |

## Safe margins

Text is shrunk, then wrapped to two lines, so it stays inside these margins even at the peak of its pop. A test measures this on rendered pixels for every preset and ratio.

| Ratio | Top | Bottom | Sides | Default line height |
|---|---|---|---|---|
| 9:16 | 12% | 24% | 8% | 66% down the frame |
| 1:1 | 8% | 12% | 7% | 80% |
| 16:9 | 8% | 10% | 6% | 84% |

The large bottom margin at 9:16 keeps text clear of the buttons and captions that Shorts and Reels draw over the video.

## Languages

Captions use the language spoken in the video, and they are not translated. Each word uses a font that really has its letters:

| Language | Font | How it was checked |
|---|---|---|
| Latin scripts | The preset's font | Automated tests on real renders |
| Hindi and other Devanagari | Bundled Hind Bold | Looked at a real render |
| Japanese, Korean, Chinese | Noto Sans CJK Bold, in the correct regional style | Automated tests on real renders |
| Arabic, Persian, Urdu, Hebrew | Bundled Noto Sans Arabic and Hebrew Bold. Words run right to left, in spoken order. | Automated tests on real renders: word order, highlight order, joined letters, width and safe margins |

`scripts/setup.sh` downloads the CJK font. If it is missing, or a language has no font, the job shows a warning instead of silently producing empty boxes.

## Rendering

Captions are burned in with FFmpeg's `ass` filter. Each word is a separate event positioned by hand, using `\t` transforms for colour and scale. Use **Real render** in the editor to see the exact output. The live preview is a close browser approximation.
