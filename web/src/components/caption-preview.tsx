"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { apiUrl, type Font, type PresetData } from "@/lib/api";

const SAMPLE = "Nobody expected this to work but it was absolutely incredible!";
const FILLER = new Set(["this", "to", "but", "it", "was", "the", "a", "and"]);
const EMPHASIS = /incredible|absolutely|expected/i;
// Same safe areas, anchors and fit rules as the engine (engine/clipforge/captions/layout.py).
const FRAME = {
  "9:16": { w: 270, h: 480, lower: 0.65, top: 0.12, side: 0.08, fill: 1 },
  "1:1": { w: 360, h: 360, lower: 0.7, top: 0.08, side: 0.07, fill: 0.85 },
  "16:9": { w: 480, h: 270, lower: 0.74, top: 0.08, side: 0.06, fill: 0.55 },
};
const REFERENCE_LINE = "THE BEST PART";
const REFERENCE_WORD = "PERFECT";
const SHORT_WORD_GAIN = 2;
const ONE_WORD_GAIN = 1.3;
const BASE_PX = 20; // measuring size; the visible text is scaled from it
const WORD_MS = 320;
const RTL_TEXT = /[֐-ࣿ]/;

export type PreviewRatio = keyof typeof FRAME;

/** Where the centre of the text sits, as a share of the frame height. */
export function centreFraction(preset: Pick<PresetData, "position" | "y_offset">, ratio: PreviewRatio): number {
  const frame = FRAME[ratio];
  const anchor = preset.position === "lower" ? frame.lower : preset.position === "center" ? 0.5 : frame.top + 0.08;
  return anchor + preset.y_offset;
}

function chunk(words: string[], size: number) {
  const out: string[][] = [];
  for (let i = 0; i < words.length; i += size) out.push(words.slice(i, i + size));
  return out;
}

function kindOf(word: string) {
  const clean = word.replace(/[^\w]/g, "").toLowerCase();
  return EMPHASIS.test(clean) ? "emphasis" : FILLER.has(clean) ? "filler" : "normal";
}

/** A quick browser approximation while editing. "Real render" shows the exact libass output. */
export function CaptionPreview({ preset, ratio, fonts, text }: { preset: PresetData; ratio: PreviewRatio; fonts: Font[]; text?: string }) {
  const words = useMemo(() => (text?.trim() || SAMPLE).split(/\s+/), [text]);
  const groups = useMemo(() => chunk(words, preset.max_words), [words, preset.max_words]);
  const [tick, setTick] = useState(0);
  const [fontsReady, setFontsReady] = useState(0);
  const [scale, setScale] = useState(1);
  const groupRef = useRef<HTMLDivElement>(null);
  const referenceRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setInterval(() => setTick((t) => t + 1), WORD_MS);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    void document.fonts.ready.then(() => setFontsReady((n) => n + 1));
  }, [fonts, preset.font]);

  const active = tick % words.length;
  const groupIndex = Math.floor(active / preset.max_words);
  const group = groups[groupIndex] ?? [];
  const activeInGroup = active % preset.max_words;
  const frame = FRAME[ratio];
  const target = frame.w * (1 - 2 * frame.side) * preset.fill * frame.fill;

  // Fit to width: measure at BASE_PX, then scale so the line fills its share of the safe width.
  useLayoutEffect(() => {
    const line = groupRef.current?.scrollWidth ?? 0;
    const reference = referenceRef.current?.scrollWidth ?? 0;
    if (!line || !reference) return;
    const gain = preset.max_words === 1 ? ONE_WORD_GAIN : SHORT_WORD_GAIN;
    // Room for the active word's pop, as the engine reserves it.
    const reserve = 1 + (preset.active_pop - 1);
    setScale(Math.min(target / (line * reserve), (gain * target) / (reference * reserve)));
  }, [group.join(" "), target, preset, fontsReady]); // eslint-disable-line react-hooks/exhaustive-deps

  const wordStyle = (word: string, index: number, px: number): CSSProperties => {
    const kind = kindOf(word);
    const factor = kind === "emphasis" ? preset.emphasis_scale : kind === "filler" ? preset.filler_scale : 1;
    const font = fonts.find((f) => f.family === preset.font);
    const size = px * factor * (font?.em_ratio ?? 0.75);
    const isActive = index === activeInGroup;
    const color = isActive && preset.active_style === "color" ? preset.active_color : kind === "emphasis" && preset.emphasis_color ? preset.emphasis_color : preset.text_color;
    const outline = (size / (font?.em_ratio ?? 0.75)) * preset.outline * 1.3;
    const shadow = (size / (font?.em_ratio ?? 0.75)) * preset.shadow;
    const shadowAlpha = Math.round((preset.shadow_opacity ?? 1) * 255).toString(16).padStart(2, "0");
    const shadows = [
      shadow > 0 ? `${shadow}px ${shadow}px ${(size / (font?.em_ratio ?? 0.75)) * (preset.shadow_blur ?? 0) * 2}px ${preset.shadow_color}${shadowAlpha}` : "",
      preset.glow_color && preset.glow > 0 ? `0 0 ${size * preset.glow}px ${preset.glow_color}` : "",
    ].filter(Boolean);
    return {
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
      whiteSpace: "nowrap",
      transform: `scale(${isActive ? preset.active_pop : 1}) rotate(${preset.tilt ? (index % 2 ? -preset.tilt : preset.tilt) : 0}deg)`,
      transition: "transform 80ms ease-out",
    };
  };
  const gap = (px: number) => px * preset.word_gap;
  const sample = preset.max_words === 1 ? REFERENCE_WORD : REFERENCE_LINE;

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
      {/* Hidden copies at the measuring size. */}
      <div aria-hidden className="pointer-events-none invisible absolute left-0 top-0 flex w-max" style={{ columnGap: gap(BASE_PX) }} ref={groupRef}>
        {group.map((word, index) => (
          <span key={index} style={{ ...wordStyle(word, -1, BASE_PX), transform: undefined }}>
            {word}
          </span>
        ))}
      </div>
      <div aria-hidden className="pointer-events-none invisible absolute left-0 top-0 flex w-max" style={{ columnGap: gap(BASE_PX) }} ref={referenceRef}>
        {sample.split(" ").map((word, index) => (
          <span key={index} style={{ ...wordStyle(word.toLowerCase(), -1, BASE_PX), transform: undefined }}>
            {word}
          </span>
        ))}
      </div>
      <div
        dir={RTL_TEXT.test(words[0] ?? "") ? "rtl" : "ltr"}
        className="absolute inset-x-0 flex flex-nowrap items-baseline justify-center"
        style={{ top: `${centreFraction(preset, ratio) * 100}%`, transform: "translateY(-50%)", columnGap: gap(BASE_PX * scale) }}
      >
        {group.map((word, index) => (
          <span key={`${groupIndex}-${index}`} style={wordStyle(word, index, BASE_PX * scale)}>
            {word}
          </span>
        ))}
      </div>
    </div>
  );
}
