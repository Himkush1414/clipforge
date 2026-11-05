"use client";

import { useEffect, useState } from "react";
import { CaptionPreview, type PreviewRatio } from "@/components/caption-preview";
import { ErrorNote, Field, PageHeader, Panel, Segmented, Toggle } from "@/components/ui";
import { api, apiUrl, usePoll, type Font, type Preset, type PresetData } from "@/lib/api";

function Slider({ label, value, min, max, step, onChange, format }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 flex justify-between font-medium">
        {label}
        <span className="font-normal text-muted">{format ? format(value) : value}</span>
      </span>
      <input type="range" className="w-full accent-[var(--accent)]" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function ColorField({ label, value, onChange, optional }: { label: string; value: string | null; onChange: (v: string | null) => void; optional?: boolean }) {
  return (
    <div className="text-sm">
      <span className="mb-1 block font-medium">{label}</span>
      <div className="flex items-center gap-2">
        <input type="color" aria-label={label} className="h-9 w-12 cursor-pointer rounded border border-line bg-panel-2" value={value ?? "#ffffff"} onChange={(e) => onChange(e.target.value.toUpperCase())} />
        {optional && (
          <button type="button" className="text-xs text-muted underline" onClick={() => onChange(value ? null : "#FFFFFF")}>
            {value ? "Turn off" : "Turn on"}
          </button>
        )}
        {optional && !value && <span className="text-xs text-muted">Off</span>}
      </div>
    </div>
  );
}

export default function PresetsPage() {
  const { data: presets, refresh } = usePoll<Preset[]>("/api/presets", 0);
  const { data: fonts } = usePoll<Font[]>("/api/fonts", 0);
  const [selectedId, setSelectedId] = useState("anime-impact");
  const [draft, setDraft] = useState<PresetData | null>(null);
  const [name, setName] = useState("");
  const [ratio, setRatio] = useState<PreviewRatio>("9:16");
  const [sampleText, setSampleText] = useState("");
  const [real, setReal] = useState<{ url: string; video: boolean } | null>(null);
  const [rendering, setRendering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  const selected = presets?.find((p) => p.id === selectedId) ?? null;

  useEffect(() => {
    if (!selected) return;
    const timer = setTimeout(() => {
      setDraft(selected.data);
      setName(selected.builtin ? `${selected.name} copy` : selected.name);
    }, 0);
    return () => clearTimeout(timer);
  }, [selected]);

  function set<K extends keyof PresetData>(key: K, value: PresetData[K]) {
    setDraft((d) => (d ? { ...d, [key]: value } : d));
    setSaved(null);
  }

  async function renderReal(video: boolean) {
    if (!draft) return;
    setRendering(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/presets/preview"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ preset: draft, ratio, at: video ? null : 0.5, text: sampleText || null }),
      });
      if (!res.ok) throw new Error(`Preview failed (${res.status})`);
      const blob = await res.blob();
      setReal((old) => {
        if (old) URL.revokeObjectURL(old.url);
        return { url: URL.createObjectURL(blob), video };
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRendering(false);
    }
  }

  async function save(asNew: boolean) {
    if (!draft) return;
    setError(null);
    try {
      const body = JSON.stringify({ name, data: draft });
      const row = asNew
        ? await api<Preset>("/api/presets", { method: "POST", body })
        : await api<Preset>(`/api/presets/${selectedId}`, { method: "PUT", body });
      await refresh();
      setSelectedId(row.id);
      setSaved("Saved.");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function remove() {
    setError(null);
    try {
      await api(`/api/presets/${selectedId}`, { method: "DELETE" });
      setSelectedId("anime-impact");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <PageHeader title="Caption presets" subtitle="Edit colors, fonts, sizes and animation. The left preview updates live. Real render shows exactly what an export will look like." />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-4 lg:sticky lg:top-8">
          <Panel>
            <div className="space-y-3">
              {draft && fonts && <CaptionPreview preset={draft} ratio={ratio} fonts={fonts} text={sampleText} />}
              <Segmented
                label="Preview ratio"
                value={ratio}
                onChange={setRatio}
                options={[
                  { value: "9:16", label: "9:16" },
                  { value: "1:1", label: "1:1" },
                  { value: "16:9", label: "16:9" },
                ]}
              />
              <Field label="Sample text">
                <input className="field" maxLength={200} placeholder="Type your own words" value={sampleText} onChange={(e) => setSampleText(e.target.value)} />
              </Field>
              <div className="flex gap-2">
                <button type="button" className="btn flex-1" disabled={rendering || !draft} onClick={() => renderReal(false)}>
                  Real render: frame
                </button>
                <button type="button" className="btn flex-1" disabled={rendering || !draft} onClick={() => renderReal(true)}>
                  Real render: video
                </button>
              </div>
              {rendering && <p className="text-xs text-muted">Rendering with libass…</p>}
              {real && !rendering && (
                <div className="flex justify-center">
                  {real.video ? (
                    <video src={real.url} controls autoPlay loop muted className="max-h-96 rounded-lg" />
                  ) : (
                    // eslint-disable-next-line @next/next/no-img-element -- local blob from the render API
                    <img src={real.url} alt="Exact libass render of this preset" className="max-h-96 rounded-lg" />
                  )}
                </div>
              )}
            </div>
          </Panel>
        </div>

        {draft && (
          <div className="space-y-4">
            <Panel title="Preset">
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Start from">
                  <select className="field" value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
                    {presets?.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                        {p.builtin ? " (built-in)" : ""}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Name">
                  <input className="field" maxLength={80} value={name} onChange={(e) => setName(e.target.value)} />
                </Field>
              </div>
            </Panel>
            <Panel title="Text">
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Font">
                  <select className="field" value={draft.font} onChange={(e) => set("font", e.target.value)}>
                    {fonts?.map((f) => (
                      <option key={f.family} value={f.family}>
                        {f.family}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Emphasis font">
                  <select className="field" value={draft.emphasis_font ?? ""} onChange={(e) => set("emphasis_font", e.target.value || null)}>
                    <option value="">Same as main font</option>
                    {fonts?.map((f) => (
                      <option key={f.family} value={f.family}>
                        {f.family}
                      </option>
                    ))}
                  </select>
                </Field>
                <Slider label="Words on screen" value={draft.max_words} min={1} max={3} step={1} onChange={(v) => set("max_words", v)} />
                <Slider label="Size" value={draft.size} min={0.03} max={0.25} step={0.005} onChange={(v) => set("size", v)} format={(v) => `${Math.round(v * 100)}%`} />
                <Slider label="Big words" value={draft.emphasis_scale} min={1} max={2} step={0.05} onChange={(v) => set("emphasis_scale", v)} format={(v) => `${v.toFixed(2)}×`} />
                <Slider label="Small filler words" value={draft.filler_scale} min={0.5} max={1} step={0.05} onChange={(v) => set("filler_scale", v)} format={(v) => `${v.toFixed(2)}×`} />
                <Slider label="Word gap" value={draft.word_gap} min={0.1} max={0.8} step={0.02} onChange={(v) => set("word_gap", v)} />
                <div className="flex items-end pb-2">
                  <Toggle label="UPPERCASE" checked={draft.uppercase} onChange={(v) => set("uppercase", v)} />
                </div>
              </div>
            </Panel>
            <Panel title="Colors">
              <div className="grid gap-4 sm:grid-cols-3">
                <ColorField label="Text" value={draft.text_color} onChange={(v) => set("text_color", v ?? "#FFFFFF")} />
                <ColorField label="Active word" value={draft.active_color} onChange={(v) => set("active_color", v ?? "#FFE600")} />
                <ColorField label="Big word" value={draft.emphasis_color} optional onChange={(v) => set("emphasis_color", v)} />
                <ColorField label="Outline" value={draft.outline_color} onChange={(v) => set("outline_color", v ?? "#000000")} />
                <ColorField label="Shadow" value={draft.shadow_color} onChange={(v) => set("shadow_color", v ?? "#000000")} />
                <ColorField label="Glow" value={draft.glow_color} optional onChange={(v) => set("glow_color", v)} />
              </div>
              <div className="mt-4 grid gap-4 sm:grid-cols-3">
                <Slider label="Outline width" value={draft.outline} min={0} max={0.3} step={0.01} onChange={(v) => set("outline", v)} />
                <Slider label="Shadow depth" value={draft.shadow} min={0} max={0.3} step={0.01} onChange={(v) => set("shadow", v)} />
                <Slider label="Glow size" value={draft.glow} min={0} max={0.5} step={0.01} onChange={(v) => set("glow", v)} />
              </div>
            </Panel>
            <Panel title="Motion and position">
              <div className="grid gap-4 sm:grid-cols-2">
                <Segmented label="Active word style" value={draft.active_style} onChange={(v) => set("active_style", v)} options={[{ value: "color", label: "Color" }, { value: "box", label: "Box" }]} />
                {draft.active_style === "box" ? (
                  <ColorField label="Box color" value={draft.box_color} onChange={(v) => set("box_color", v ?? "#7C3AED")} />
                ) : (
                  <div />
                )}
                <div className="sm:col-span-2">
                  <Segmented
                    label="Entrance"
                    value={draft.animation}
                    onChange={(v) => set("animation", v)}
                    options={[
                      { value: "pop", label: "Pop" },
                      { value: "bounce", label: "Bounce" },
                      { value: "fade", label: "Fade" },
                      { value: "slide", label: "Slide" },
                      { value: "none", label: "None" },
                    ]}
                  />
                </div>
                <Slider label="Active word pop" value={draft.active_pop} min={1} max={1.5} step={0.01} onChange={(v) => set("active_pop", v)} format={(v) => `${v.toFixed(2)}×`} />
                <Slider label="Tilt" value={draft.tilt} min={0} max={12} step={0.5} onChange={(v) => set("tilt", v)} format={(v) => `${v}°`} />
                <Segmented label="Position" value={draft.position} onChange={(v) => set("position", v)} options={[{ value: "upper", label: "Upper" }, { value: "center", label: "Center" }, { value: "lower", label: "Lower" }]} />
                <Slider label="Nudge up or down" value={draft.y_offset} min={-0.2} max={0.2} step={0.01} onChange={(v) => set("y_offset", v)} format={(v) => `${Math.round(v * 100)}%`} />
              </div>
            </Panel>
            <ErrorNote message={error} />
            <div className="flex flex-wrap items-center justify-end gap-2">
              {saved && <p className="text-sm text-ok">{saved}</p>}
              {selected && !selected.builtin && (
                <button type="button" className="btn btn-danger mr-auto" onClick={remove}>
                  Delete preset
                </button>
              )}
              {selected && !selected.builtin && (
                <button type="button" className="btn" disabled={!name.trim()} onClick={() => save(false)}>
                  Save
                </button>
              )}
              <button type="button" className="btn btn-primary" disabled={!name.trim()} onClick={() => save(true)}>
                Save as new preset
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
