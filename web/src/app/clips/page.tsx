"use client";

import { useEffect, useState } from "react";
import { Badge, EmptyState, ErrorNote, PageHeader } from "@/components/ui";
import { BulkBar, QueuedBanner } from "@/components/clip-actions";
import { UploadPanel } from "@/components/upload-panel";
import { api, apiUrl, formatBytes, formatDuration, usePoll, type Channel, type Clip, type Upload } from "@/lib/api";

function useNow() {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  return now;
}

function Countdown({ expiresAt }: { expiresAt: number }) {
  const now = useNow();
  const left = Math.max(0, Math.round((expiresAt - now) / 1000));
  const hours = Math.floor(left / 3600);
  const minutes = Math.floor((left % 3600) / 60);
  const seconds = left % 60;
  const text = `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  return <Badge tone={left < 3600 ? "fail" : left < 6 * 3600 ? "warn" : "neutral"}>Deleted in {text}</Badge>;
}

function ClipCard({
  clip,
  channels,
  uploads,
  selected,
  onSelect,
  onChange,
}: {
  clip: Clip;
  channels: Channel[];
  uploads: Upload[];
  selected: boolean;
  onSelect: (on: boolean) => void;
  onChange: () => void;
}) {
  const [title, setTitle] = useState(clip.title);
  const [description, setDescription] = useState(clip.description);
  const [startFrame, setStartFrame] = useState("");
  const [endFrame, setEndFrame] = useState("");
  const [showQc, setShowQc] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expiresAt] = useState(() => (clip.seconds_left === null ? null : Date.now() + clip.seconds_left * 1000));
  const dirty = title !== clip.title || description !== clip.description;
  const qc = clip.qc.status ?? "warn";

  function moveTo(folder: "saved" | "queued") {
    return api(`/api/clips/${clip.id}/move`, { method: "POST", body: JSON.stringify({ folder }) });
  }

  async function run(action: () => Promise<unknown>, done?: string) {
    setError(null);
    setMessage(null);
    try {
      await action();
      if (done) setMessage(done);
      onChange();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <article className={`overflow-hidden rounded-xl border bg-panel ${selected ? "border-accent" : "border-line"}`}>
      <label className="flex cursor-pointer items-center gap-2 border-b border-line px-4 py-2 text-sm">
        <input
          type="checkbox"
          className="size-4 accent-[var(--accent)]"
          aria-label={`Select ${clip.title}`}
          checked={selected}
          onChange={(e) => onSelect(e.target.checked)}
        />
        Select
      </label>
      <div className="flex justify-center bg-black">
        {clip.file_url && (
          <video
            controls
            preload="none"
            poster={clip.thumb_url ? apiUrl(clip.thumb_url) : undefined}
            src={apiUrl(clip.file_url)}
            className="max-h-[420px]"
            style={{ aspectRatio: `${clip.width} / ${clip.height}` }}
          />
        )}
      </div>
      <div className="space-y-3 p-4">
        {clip.reasons?.why && (
          <p className="text-sm">
            <span className="font-medium">Why this clip: </span>
            <span className="text-muted">{clip.reasons.why}</span>
          </p>
        )}
        <div className="flex flex-wrap gap-1.5">
          <Badge>{clip.ratio}</Badge>
          <Badge>{clip.resolution}</Badge>
          <Badge>{formatDuration(clip.duration)}</Badge>
          <Badge>{formatBytes(clip.size_bytes)}</Badge>
          {clip.part && <Badge>Part {clip.part}</Badge>}
          {!clip.uploadable && <Badge tone="warn">Private test: never uploaded</Badge>}
          <button type="button" className="cursor-pointer" aria-expanded={showQc} onClick={() => setShowQc(!showQc)}>
            <Badge tone={qc}>Quality: {qc}</Badge>
          </button>
          {expiresAt !== null ? (
            <Countdown expiresAt={expiresAt} />
          ) : (
            <Badge tone="ok">{clip.folder === "saved" ? "Kept until you delete it" : "Timer paused"}</Badge>
          )}
        </div>
        {showQc && (
          <ul className="space-y-1 rounded-lg bg-panel-2 p-3 text-xs">
            {clip.qc.checks?.map((check) => (
              <li key={check.name} className="flex justify-between gap-3">
                <span className={check.status === "ok" ? "text-ok" : check.status === "warn" ? "text-warn" : "text-bad"}>
                  {check.name.replace("_", " ")}
                </span>
                <span className="text-right text-muted">{check.detail}</span>
              </li>
            ))}
            {clip.qc.encoder && (
              <li className="flex justify-between gap-3">
                <span>encoder</span>
                <span className="text-muted">{clip.qc.encoder}</span>
              </li>
            )}
          </ul>
        )}
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Title</span>
          <input className="field" maxLength={100} value={title} onChange={(e) => setTitle(e.target.value)} />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Description</span>
          <textarea className="field" rows={3} value={description} onChange={(e) => setDescription(e.target.value)} />
        </label>
        <details className="text-sm">
          <summary className="cursor-pointer font-medium">Recut by exact frames</summary>
          <div className="mt-2 flex flex-wrap items-end gap-2">
            <label className="w-28 text-xs">
              First frame
              <input className="field mt-1" type="number" min={0} value={startFrame} onChange={(e) => setStartFrame(e.target.value)} />
            </label>
            <label className="w-28 text-xs">
              Stop before frame
              <input className="field mt-1" type="number" min={1} value={endFrame} onChange={(e) => setEndFrame(e.target.value)} />
            </label>
            <button
              type="button"
              className="btn"
              disabled={!startFrame || !endFrame}
              onClick={() =>
                run(
                  () =>
                    api(`/api/clips/${clip.id}/recut`, {
                      method: "POST",
                      body: JSON.stringify({ start_frame: Number(startFrame), end_frame: Number(endFrame) }),
                    }),
                  "Recut started. Follow it on the Jobs page.",
                )
              }
            >
              Recut
            </button>
          </div>
        </details>
        <UploadPanel clip={clip} channels={channels} uploads={uploads} onChange={onChange} />
        <ErrorNote message={error} />
        {message && <p className="text-sm text-ok">{message}</p>}
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="btn"
            disabled={!dirty}
            onClick={() =>
              run(() => api(`/api/clips/${clip.id}`, { method: "PATCH", body: JSON.stringify({ title, description }) }), "Saved.")
            }
          >
            Save text
          </button>
          {clip.folder !== "saved" && (
            <button type="button" className="btn" onClick={() => run(() => moveTo("saved"), "Moved to Saved. It will not be deleted automatically.")}>
              Move to Saved
            </button>
          )}
          {clip.folder !== "queued" && (
            <button
              type="button"
              className="btn"
              disabled={!clip.uploadable}
              title={clip.uploadable ? undefined : "Private test clips are never uploaded"}
              onClick={() => run(() => moveTo("queued"), "Moved to Queued. The timer is paused.")}
            >
              Move to Queued
            </button>
          )}
          {clip.file_url && (
            <a className="btn" href={apiUrl(`${clip.file_url}?download=true`)}>
              Download
            </a>
          )}
          <button type="button" className="btn btn-danger ml-auto" onClick={() => run(() => api(`/api/clips/${clip.id}`, { method: "DELETE" }))}>
            Delete
          </button>
        </div>
      </div>
    </article>
  );
}

const TABS = [
  { value: "unassigned", label: "Unassigned", help: "Every clip here is deleted 24 hours after it was made." },
  { value: "queued", label: "Queued", help: "The timer is paused until upload. A clip is deleted 24 hours after it uploads." },
  { value: "saved", label: "Saved", help: "Never deleted automatically." },
] as const;

type ChannelGroup = { key: string; label: string; clips: Clip[] };

/** Clips under a heading per channel, "No channel" first, so clips made without one are easy to find. */
function groupByChannel(clips: Clip[], channels: Channel[]): ChannelGroup[] {
  const groups = new Map<string, ChannelGroup>();
  for (const clip of clips) {
    const key = clip.channel_id ?? "";
    if (!groups.has(key)) {
      const label = key ? (channels.find((c) => c.id === key)?.name ?? "Removed channel") : "No channel";
      groups.set(key, { key, label, clips: [] });
    }
    groups.get(key)!.clips.push(clip);
  }
  return [...groups.values()].sort((a, b) => (a.key === "" ? -1 : b.key === "" ? 1 : a.label.localeCompare(b.label)));
}

export default function ClipsPage() {
  const { data: clips, error, refresh } = usePoll<Clip[]>("/api/clips", 5000);
  const { data: channels } = usePoll<Channel[]>("/api/channels", 15000);
  const { data: uploads, refresh: refreshUploads } = usePoll<Upload[]>("/api/uploads", 4000);
  const [tab, setTab] = useState<Clip["folder"]>("unassigned");
  const [selected, setSelected] = useState<string[]>([]);

  const shown = clips?.filter((c) => c.folder === tab) ?? [];
  const queued = clips?.filter((c) => c.folder === "queued") ?? [];
  const help = TABS.find((t) => t.value === tab)?.help;
  const visibleSelected = selected.filter((id) => shown.some((c) => c.id === id));

  function refreshAll() {
    void refresh();
    void refreshUploads();
  }

  return (
    <>
      <PageHeader title="Clips" subtitle="Your exported clips, by where they are kept." />
      <ErrorNote message={error} />
      <QueuedBanner queued={queued} channels={channels ?? []} uploads={uploads ?? []} onChange={refreshAll} />
      <div role="tablist" aria-label="Clip folders" className="mb-2 flex flex-wrap gap-1 border-b border-line">
        {TABS.map((t) => (
          <button
            key={t.value}
            type="button"
            role="tab"
            aria-selected={tab === t.value}
            onClick={() => {
              setTab(t.value);
              setSelected([]);
            }}
            className={`cursor-pointer border-b-2 px-3 py-2 text-sm font-medium ${
              tab === t.value ? "border-accent text-text" : "border-transparent text-muted hover:text-text"
            }`}
          >
            {t.label} ({clips?.filter((c) => c.folder === t.value).length ?? 0})
          </button>
        ))}
      </div>
      <p className="mb-4 text-sm text-muted">{help}</p>
      <BulkBar
        clips={shown}
        selected={visibleSelected}
        channels={channels ?? []}
        onSelectAll={(on) => setSelected(on ? shown.map((c) => c.id) : [])}
        onDone={() => {
          setSelected([]);
          refreshAll();
        }}
      />
      {clips && shown.length === 0 && <EmptyState title={`Nothing in ${TABS.find((t) => t.value === tab)?.label} right now`} />}
      {groupByChannel(shown, channels ?? []).map((group) => (
        <section key={group.key || "none"} aria-label={`Channel: ${group.label}`} className="mb-6">
          <h2 className="mb-2 text-sm font-semibold">
            {group.label} <span className="font-normal text-muted">({group.clips.length})</span>
          </h2>
          <div className="grid items-start gap-4 md:grid-cols-2 xl:grid-cols-3">
            {group.clips.map((clip) => (
              <ClipCard
                key={`${clip.id}-${clip.folder}`}
                clip={clip}
                channels={channels ?? []}
                uploads={uploads?.filter((u) => u.clip_id === clip.id) ?? []}
                selected={selected.includes(clip.id)}
                onSelect={(on) => setSelected((s) => (on ? [...s, clip.id] : s.filter((id) => id !== clip.id)))}
                onChange={refreshAll}
              />
            ))}
          </div>
        </section>
      ))}
    </>
  );
}
