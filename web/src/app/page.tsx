"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { NoChannelDialog } from "@/components/no-channel-dialog";
import { ErrorNote, Field, PageHeader, Panel, ProgressBar, Segmented, Toggle } from "@/components/ui";
import {
  api,
  formatBytes,
  formatDuration,
  RIGHTS_LABELS,
  uploadVideo,
  usePoll,
  type Channel,
  type Hardware,
  type Preset,
  type Source,
} from "@/lib/api";

type Rights = "" | "own" | "public_domain" | "creative_commons" | "written_permission" | "private_test";
type Ratio = "9:16" | "1:1" | "16:9";

const GENRES = ["anime", "movie", "series", "gameplay", "podcast", "other"] as const;
const RESOLUTIONS = ["1080p", "2K", "4K", "8K"];

export default function NewJobPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const { data: sources, refresh: refreshSources } = usePoll<Source[]>("/api/sources", 0);
  const { data: presets } = usePoll<Preset[]>("/api/presets", 0);
  const { data: channels } = usePoll<Channel[]>("/api/channels", 0);
  const { data: hardware } = usePoll<Hardware>("/api/hardware", 0);

  const [sourceId, setSourceId] = useState("");
  const [dragging, setDragging] = useState(false);
  const [upload, setUpload] = useState<number | null>(null);
  const [pathInput, setPathInput] = useState("");

  const [rights, setRights] = useState<Rights>("");
  const [rightsDetail, setRightsDetail] = useState("");

  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const [genre, setGenre] = useState<(typeof GENRES)[number]>("other");
  const [duration, setDuration] = useState(45);
  const [count, setCount] = useState(5);
  const [numbering, setNumbering] = useState(true);
  const [format, setFormat] = useState<"short" | "long">("short");
  const [ratio, setRatio] = useState<Ratio>("9:16");
  const [resolution, setResolution] = useState("1080p");

  const [presetId, setPresetId] = useState("");
  const [captions, setCaptions] = useState(true);
  const [fit, setFit] = useState<"smart" | "crop" | "blur">("smart");
  const [pacing, setPacing] = useState(false);
  const [transition, setTransition] = useState<"cut" | "crossfade">("cut");
  const [channelId, setChannelId] = useState("");
  const [language, setLanguage] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [showNoChannel, setShowNoChannel] = useState(false);

  const source = sources?.find((s) => s.id === sourceId) ?? null;
  const detailRequired = rights === "creative_commons" || rights === "written_permission";
  const rightsValid = rights !== "" && (!detailRequired || rightsDetail.trim().length > 0);
  const blocker = !source ? "Choose a video first." : !rightsValid ? "Fill in the source rights to continue." : null;

  async function handleFile(file: File) {
    setError(null);
    setUpload(0);
    try {
      const created = await uploadVideo(file, setUpload);
      await refreshSources();
      setSourceId(created.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUpload(null);
    }
  }

  async function handleImport() {
    setError(null);
    try {
      const created = await api<Source>("/api/sources/import", { method: "POST", body: JSON.stringify({ path: pathInput }) });
      await refreshSources();
      setSourceId(created.id);
      setPathInput("");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function applyChannel(id: string) {
    setChannelId(id);
    const channel = channels?.find((c) => c.id === id);
    if (!channel) return;
    setFormat(channel.settings.format);
    setRatio(channel.settings.ratio);
    if (channel.settings.preset_id) setPresetId(channel.settings.preset_id);
  }

  /** Warns first when nothing is connected, so you choose where the clips should wait. */
  function requestStart() {
    if (blocker) return;
    if (!channels?.some((c) => c.connected)) {
      setShowNoChannel(true);
      return;
    }
    void submit("unassigned");
  }

  async function submit(destination: "unassigned" | "queued") {
    if (blocker) return;
    setShowNoChannel(false);
    setSubmitting(true);
    setError(null);
    try {
      await api("/api/jobs", {
        method: "POST",
        body: JSON.stringify({
          source_id: sourceId,
          mode,
          destination,
          review: true,
          rights: { type: rights, detail: rightsDetail },
          channel_id: channelId || null,
          manual:
            mode === "manual"
              ? { genre, clip_duration: duration, clip_count: count, part_numbering: numbering, format, ratio, resolution }
              : null,
          edit: { preset_id: presetId || null, captions, fit, remove_pauses: pacing, transition, language: language || null },
        }),
      });
      router.push("/jobs");
    } catch (err) {
      setError((err as Error).message);
      setSubmitting(false);
    }
  }

  return (
    <>
      <PageHeader title="New job" subtitle="Turn a long video you have the rights to into captioned clips." />
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
          <Panel title="1. Video">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                const file = e.dataTransfer.files[0];
                if (file) void handleFile(file);
              }}
              className={`w-full cursor-pointer rounded-lg border-2 border-dashed p-8 text-center text-sm ${
                dragging ? "border-accent bg-accent/10" : "border-line hover:border-muted"
              }`}
            >
              <span className="block font-medium">Drop a video here, or click to choose</span>
              <span className="mt-1 block text-xs text-muted">Any length. It stays on this computer.</span>
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="video/*,.mkv,.ts,.mts"
              aria-label="Choose a video file"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleFile(file);
                e.target.value = "";
              }}
            />
            {upload !== null && (
              <div className="mt-3 space-y-1 text-xs text-muted">
                <ProgressBar value={upload} label="Upload progress" />
                <p>Copying into clipforge: {Math.round(upload * 100)}%</p>
              </div>
            )}
            <div className="mt-3 flex gap-2">
              <input
                className="field"
                aria-label="Path to a video on this computer"
                placeholder="Or paste a path, like D:\videos\episode.mp4"
                value={pathInput}
                onChange={(e) => setPathInput(e.target.value)}
              />
              <button type="button" className="btn shrink-0" disabled={!pathInput.trim()} onClick={handleImport}>
                Use path
              </button>
            </div>
            {!!sources?.length && (
              <div className="mt-3">
                <Field label="Your videos">
                  <select className="field" value={sourceId} onChange={(e) => setSourceId(e.target.value)}>
                    <option value="">Choose a video</option>
                    {sources.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.filename} ({formatDuration(s.probe.duration)})
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
            )}
            {source && (
              <p className="mt-2 text-xs text-muted">
                {source.probe.width}×{source.probe.height}, {source.probe.fps.toFixed(2)} fps, {formatDuration(source.probe.duration)},{" "}
                {formatBytes(source.probe.size_bytes)}
                {!source.probe.has_audio && ", no audio"}
              </p>
            )}
          </Panel>

          <Panel title="2. Source rights" hint="Required. Saved with the job. A job cannot start without it.">
            <div className="space-y-3">
              <Field label="What rights do you have to this footage?">
                <select className="field" value={rights} onChange={(e) => setRights(e.target.value as Rights)}>
                  <option value="">Choose one</option>
                  {Object.entries(RIGHTS_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </Field>
              {rights === "private_test" && (
                <p className="text-xs text-warn">
                  For trying the app on footage you have no rights to share. You can watch and keep the clips, but they can never be
                  queued or uploaded.
                </p>
              )}
              {(rights === "creative_commons" || rights === "written_permission" || rights === "public_domain") && (
                <Field
                  label={rights === "creative_commons" ? "Credit text" : rights === "written_permission" ? "Link or note" : "Source note (optional)"}
                  hint={rights === "creative_commons" ? "Added to each clip's description automatically." : undefined}
                >
                  <textarea className="field" rows={2} value={rightsDetail} onChange={(e) => setRightsDetail(e.target.value)} />
                </Field>
              )}
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="3. Clips">
            <div className="space-y-4">
              <Segmented
                label="Mode"
                value={mode}
                onChange={setMode}
                options={[
                  { value: "auto", label: "Auto: decide from the content" },
                  { value: "manual", label: "Manual: I choose" },
                ]}
              />
              {mode === "auto" ? (
                <p className="text-sm text-muted">
                  clipforge picks the genre, clip length, number of clips, format, ratio and resolution, then shows you why on the Jobs page.
                </p>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  <Field label="Genre">
                    <select className="field capitalize" value={genre} onChange={(e) => setGenre(e.target.value as typeof genre)}>
                      {GENRES.map((g) => (
                        <option key={g} value={g}>
                          {g}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Clip length (seconds)">
                    <input className="field" type="number" min={1} max={1800} value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
                  </Field>
                  <Field label="Number of clips">
                    <input className="field" type="number" min={1} max={50} value={count} onChange={(e) => setCount(Number(e.target.value))} />
                  </Field>
                  <div className="flex items-end pb-2">
                    <Toggle label="Number the parts (Part 1, 2, 3)" checked={numbering} onChange={setNumbering} />
                  </div>
                  <Segmented
                    label="Format"
                    value={format}
                    onChange={(value) => {
                      setFormat(value);
                      setRatio(value === "short" ? "9:16" : "16:9");
                    }}
                    options={[
                      { value: "short", label: "Shorts / Reels" },
                      { value: "long", label: "Long video" },
                    ]}
                  />
                  <Segmented
                    label="Aspect ratio"
                    value={ratio}
                    onChange={setRatio}
                    options={[
                      { value: "9:16", label: "9:16" },
                      { value: "1:1", label: "1:1" },
                      { value: "16:9", label: "16:9" },
                    ]}
                  />
                  <div className="sm:col-span-2">
                    <Segmented
                      label="Resolution"
                      value={resolution}
                      onChange={setResolution}
                      options={RESOLUTIONS.map((r) => ({
                        value: r,
                        label: r,
                        disabled: hardware ? !hardware.allowed_resolutions.includes(r) : false,
                      }))}
                    />
                  </div>
                </div>
              )}
            </div>
          </Panel>

          <Panel title="4. Editing">
            <div className="space-y-4">
              <Toggle label="Burn in word-by-word captions" checked={captions} onChange={setCaptions} />
              <Field label="Spoken language" hint="Captions use the language spoken in the video. They are not translated.">
                <select className="field" value={language} onChange={(e) => setLanguage(e.target.value)}>
                  <option value="">Detect automatically</option>
                  <option value="en">English</option>
                  <option value="hi">Hindi</option>
                  <option value="ja">Japanese</option>
                  <option value="ko">Korean</option>
                  <option value="zh">Chinese</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                  <option value="pt">Portuguese</option>
                </select>
              </Field>
              <Field label="Caption preset">
                <select className="field" value={presetId} disabled={!captions} onChange={(e) => setPresetId(e.target.value)}>
                  <option value="">{mode === "auto" ? "Choose for me, by genre" : "Default (Anime Impact)"}</option>
                  {presets?.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </Field>
              <Segmented
                label="Fitting wide video into a vertical frame"
                value={fit}
                onChange={setFit}
                options={[
                  { value: "smart", label: "Follow the subject" },
                  { value: "crop", label: "Center crop" },
                  { value: "blur", label: "Blurred bars" },
                ]}
              />
              <Segmented
                label="Joins between cuts"
                value={transition}
                onChange={setTransition}
                options={[
                  { value: "cut", label: "Clean cut" },
                  { value: "crossfade", label: "Crossfade" },
                ]}
              />
              <Toggle label="Tighten pacing by removing long pauses" checked={pacing} onChange={setPacing} />
              <Field label="Channel" hint="A channel applies its format, ratio, preset and templates.">
                <select className="field" value={channelId} onChange={(e) => applyChannel(e.target.value)}>
                  <option value="">No channel: just make the clips</option>
                  {channels?.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ({c.platform}){c.connected ? "" : ", not connected"}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          </Panel>

          <ErrorNote message={error} />
          <div className="flex items-center justify-end gap-3">
            {blocker && <p className="text-sm text-muted">{blocker}</p>}
            <button type="button" className="btn btn-primary" disabled={!!blocker || submitting} onClick={requestStart}>
              {submitting ? "Starting…" : "Start job"}
            </button>
          </div>
        </div>
      </div>
      {showNoChannel && (
        <NoChannelDialog
          onJustMake={() => void submit("unassigned")}
          onLater={() => void submit("queued")}
          canQueue={rights !== "private_test"}
          onCancel={() => setShowNoChannel(false)}
        />
      )}
    </>
  );
}
