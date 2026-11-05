"use client";

import { useEffect, useState } from "react";
import { Badge, EmptyState, ErrorNote, PageHeader } from "@/components/ui";
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

function ClipCard({ clip, channels, uploads, onChange }: { clip: Clip; channels: Channel[]; uploads: Upload[]; onChange: () => void }) {
  const [title, setTitle] = useState(clip.title);
  const [description, setDescription] = useState(clip.description);
  const [startFrame, setStartFrame] = useState("");
  const [endFrame, setEndFrame] = useState("");
  const [showQc, setShowQc] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expiresAt] = useState(() => Date.now() + clip.seconds_left * 1000);
  const dirty = title !== clip.title || description !== clip.description;
  const qc = clip.qc.status ?? "warn";

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
    <article className="overflow-hidden rounded-xl border border-line bg-panel">
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
        <div className="flex flex-wrap gap-1.5">
          <Badge>{clip.ratio}</Badge>
          <Badge>{clip.resolution}</Badge>
          <Badge>{formatDuration(clip.duration)}</Badge>
          <Badge>{formatBytes(clip.size_bytes)}</Badge>
          {clip.part && <Badge>Part {clip.part}</Badge>}
          <button type="button" className="cursor-pointer" aria-expanded={showQc} onClick={() => setShowQc(!showQc)}>
            <Badge tone={qc}>Quality: {qc}</Badge>
          </button>
          <Countdown expiresAt={expiresAt} />
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

export default function ClipsPage() {
  const { data: clips, error, refresh } = usePoll<Clip[]>("/api/clips", 5000);
  const { data: channels } = usePoll<Channel[]>("/api/channels", 15000);
  const { data: uploads, refresh: refreshUploads } = usePoll<Upload[]>("/api/uploads", 4000);
  return (
    <>
      <PageHeader title="Clips" subtitle="Every exported clip is deleted automatically 24 hours after it is made. Download or upload what you want to keep." />
      <ErrorNote message={error} />
      {clips?.length === 0 && <EmptyState title="No clips right now">Finished clips appear here.</EmptyState>}
      <div className="grid items-start gap-4 md:grid-cols-2 xl:grid-cols-3">
        {clips?.map((clip) => (
          <ClipCard
            key={clip.id}
            clip={clip}
            channels={channels ?? []}
            uploads={uploads?.filter((u) => u.clip_id === clip.id) ?? []}
            onChange={() => {
              void refresh();
              void refreshUploads();
            }}
          />
        ))}
      </div>
    </>
  );
}
