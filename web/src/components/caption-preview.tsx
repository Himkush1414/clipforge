"use client";

import { useEffect, useMemo, useState } from "react";
import { apiUrl, type Font, type PresetData } from "@/lib/api";

const SAMPLE = "Nobody expected this to work but it was absolutely incredible!";
const FILLER = new Set(["this", "to", "but", "it", "was", "the", "a", "and"]);
const EMPHASIS = /incredible|absolutely|expected/i;
const FRAME = { "9:16": { w: 270, h: 480, lower: 0.66, top: 0.12 }, "1:1": { w: 360, h: 360, lower: 0.8, top: 0.08 }, "16:9": { w: 480, h: 270, lower: 0.84, top: 0.08 } };
const RATIO_FACTOR = { "9:16": 1, "1:1": 0.9, "16:9": 0.72 };
const WORD_MS = 320;

export type PreviewRatio = keyof typeof FRAME;

function chunk(words: string[], size: number) {
  const out: string[][] = [];
  for (let i = 0; i < words.length; i += size) out.push(words.slice(i, i + size));
  return out;
}

/** A quick browser approximation while editing. "Real render" shows the exact libass output. */
export function CaptionPreview({ preset, ratio, fonts, text }: { preset: PresetData; ratio: PreviewRatio; fonts: Font[]; text?: string }) {
  const words = useMemo(() => (text?.trim() || SAMPLE).split(/\s+/), [text]);
  const groups = useMemo(() => chunk(words, preset.max_words), [words, preset.max_words]);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setTick((t) => t + 1), WORD_MS);
    return () => clearInterval(timer);
  }, []);

  const active = tick % words.length;
  const groupIndex = Math.floor(active / preset.max_words);
  const group = groups[groupIndex] ?? [];
  const activeInGroup = active % preset.max_words;

  const frame = FRAME[ratio];
  const font = fonts.find((f) => f.family === preset.font);
  const em = preset.size * Math.min(frame.w, frame.h) * RATIO_FACTOR[ratio] * (font?.em_ratio ?? 0.75);
  const y = (preset.position === "lower" ? frame.lower : preset.position === "center" ? 0.5 : frame.top + 0.08) + preset.y_offset;

  return (
    <div
      className="relative mx-auto overflow-hidden rounded-lg"
      style={{ width: frame.w, height: frame.h, maxWidth: "100%", background: "linear-gradient(135deg,#1f2937,#4c1d95)" }}
      aria-label="Live caption preview"
      role="img"
    >
      {fonts.map((f) => (
        <style key={f.file}>{`@font-face{font-family:"${f.family}";src:url("${apiUrl(`/api/fonts/${f.file}`)}");font-display:swap}`}</style>
      ))}
      <div
        className="absolute inset-x-[8%] flex flex-wrap items-baseline justify-center"
        style={{ top: `${y * 100}%`, transform: "translateY(-50%)", columnGap: em * preset.word_gap }}
      >
        {group.map((word, index) => {
          const clean = word.replace(/[^\w]/g, "").toLowerCase();
          const kind = EMPHASIS.test(clean) ? "emphasis" : FILLER.has(clean) ? "filler" : "normal";
          const scale = kind === "emphasis" ? preset.emphasis_scale : kind === "filler" ? preset.filler_scale : 1;
          const isActive = index === activeInGroup;
          const size = em * scale;
          const color = isActive && preset.active_style === "color" ? preset.active_color : kind === "emphasis" && preset.emphasis_color ? preset.emphasis_color : preset.text_color;
          const outline = size * preset.outline * 1.3;
          const shadow = size * preset.shadow;
          const shadows = [
            shadow > 0 ? `${shadow}px ${shadow}px 0 ${preset.shadow_color}` : "",
            preset.glow_color && preset.glow > 0 ? `0 0 ${size * preset.glow}px ${preset.glow_color}` : "",
          ].filter(Boolean);
          return (
            <span
              key={`${groupIndex}-${index}`}
              style={{
                fontFamily: `"${preset.font}", sans-serif`,
                fontSize: size,
                lineHeight: 1.1,
                color,
                textTransform: preset.uppercase ? "uppercase" : "none",
                WebkitTextStroke: outline > 0 ? `${outline}px ${preset.outline_color}` : undefined,
                paintOrder: "stroke fill",
                textShadow: shadows.join(",") || undefined,
                background: isActive && preset.active_style === "box" ? preset.box_color : undefined,
                borderRadius: size * 0.18,
                padding: preset.active_style === "box" ? `0 ${size * 0.14}px` : undefined,
                display: "inline-block",
                transform: `scale(${isActive ? preset.active_pop : 1}) rotate(${preset.tilt ? (index % 2 ? -preset.tilt : preset.tilt) : 0}deg)`,
                transition: "transform 80ms ease-out",
              }}
            >
              {word}
            </span>
          );
        })}
      </div>
    </div>
  );
}
