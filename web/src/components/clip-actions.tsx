"use client";

import Link from "next/link";
import { useState } from "react";
import { ErrorNote } from "@/components/ui";
import { api, type BulkResult, type Channel, type Clip, type Upload } from "@/lib/api";

const ACTIVE = new Set(["pending", "uploading", "processing"]);

function describe(result: BulkResult, verb: string): { message: string | null; error: string | null } {
  const message = result.done.length ? `${result.done.length} ${result.done.length === 1 ? "clip" : "clips"} ${verb}.` : null;
  const error = result.failed.length ? `${result.failed.length} could not be done: ${result.failed[0].error}` : null;
  return { message, error };
}

function ChannelPicker({ channels, value, onChange }: { channels: Channel[]; value: string; onChange: (v: string) => void }) {
  return (
    <select className="field w-auto" aria-label="Upload to channel" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Choose a channel</option>
      {channels.map((c) => (
        <option key={c.id} value={c.id}>
          {c.name} ({c.platform})
        </option>
      ))}
    </select>
  );
}

/** Appears when clips are waiting and something is connected to send them to. */
export function QueuedBanner({ queued, channels, uploads, onChange }: { queued: Clip[]; channels: Channel[]; uploads: Upload[]; onChange: () => void }) {
  const connected = channels.filter((c) => c.connected);
  const [choice, setChoice] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = new Set(uploads.filter((u) => ACTIVE.has(u.status)).map((u) => u.clip_id));
  const waiting = queued.filter((c) => !busy.has(c.id));
  if (waiting.length === 0 && !message && !error) return null;
  if (connected.length === 0) {
    return waiting.length ? (
      <p className="mb-4 rounded-lg border border-line bg-panel px-3 py-2 text-sm text-muted">
        {waiting.length} {waiting.length === 1 ? "clip is" : "clips are"} waiting in Queued.{" "}
        <Link className="text-accent underline" href="/channels">
          Connect a channel
        </Link>{" "}
        to upload {waiting.length === 1 ? "it" : "them"}.
      </p>
    ) : null;
  }
  const channelId = connected.length === 1 ? connected[0].id : choice;

  async function uploadAll() {
    setMessage(null);
    setError(null);
    try {
      const result = await api<BulkResult>("/api/uploads/bulk", {
        method: "POST",
        body: JSON.stringify({ clip_ids: waiting.map((c) => c.id), channel_id: channelId }),
      });
      const text = describe(result, "sent to the upload queue");
      setMessage(text.message);
      setError(text.error);
      onChange();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div role="status" className="mb-4 rounded-lg border border-accent/50 bg-accent/10 px-3 py-3 text-sm">
      {waiting.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium">
            {waiting.length} {waiting.length === 1 ? "clip is" : "clips are"} waiting to upload
          </span>
          {connected.length > 1 && <ChannelPicker channels={connected} value={choice} onChange={setChoice} />}
          <button type="button" className="btn btn-primary" disabled={!channelId} onClick={uploadAll}>
            Upload now
          </button>
          <span className="text-xs text-muted">YouTube uploads default to Private. Each clip is deleted 24 hours after it uploads.</span>
        </div>
      )}
      {message && <p className="mt-2 text-ok">{message}</p>}
      {error && (
        <div className="mt-2">
          <ErrorNote message={error} />
        </div>
      )}
    </div>
  );
}

export function BulkBar({
  clips,
  selected,
  channels,
  onSelectAll,
  onDone,
}: {
  clips: Clip[];
  selected: string[];
  channels: Channel[];
  onSelectAll: (on: boolean) => void;
  onDone: () => void;
}) {
  const connected = channels.filter((c) => c.connected);
  const [channelId, setChannelId] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  if (clips.length === 0) return null;
  const count = selected.length;

  async function run(path: string, body: object, verb: string) {
    setMessage(null);
    setError(null);
    setConfirming(false);
    try {
      const result = await api<BulkResult>(path, { method: "POST", body: JSON.stringify(body) });
      const text = describe(result, verb);
      setMessage(text.message);
      setError(text.error);
      onDone();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="mb-4 rounded-lg border border-line bg-panel p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <label className="mr-2 flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            className="size-4 accent-[var(--accent)]"
            aria-label="Select all clips in this folder"
            checked={count === clips.length}
            onChange={(e) => onSelectAll(e.target.checked)}
          />
          {count ? `${count} selected` : "Select all"}
        </label>
        <button type="button" className="btn" disabled={!count} onClick={() => run("/api/clips/bulk", { clip_ids: selected, action: "move_saved" }, "moved to Saved")}>
          Move to Saved
        </button>
        <button type="button" className="btn" disabled={!count} onClick={() => run("/api/clips/bulk", { clip_ids: selected, action: "move_queued" }, "moved to Queued")}>
          Move to Queued
        </button>
        {connected.length > 0 && (
          <>
            <ChannelPicker channels={connected} value={channelId} onChange={setChannelId} />
            <button type="button" className="btn" disabled={!count || !channelId} onClick={() => run("/api/uploads/bulk", { clip_ids: selected, channel_id: channelId }, "sent to the upload queue")}>
              Upload to channel
            </button>
          </>
        )}
        {confirming ? (
          <>
            <button type="button" className="btn btn-danger ml-auto" onClick={() => run("/api/clips/bulk", { clip_ids: selected, action: "delete" }, "deleted")}>
              Delete {count} for good
            </button>
            <button type="button" className="btn" onClick={() => setConfirming(false)}>
              Keep them
            </button>
          </>
        ) : (
          <button type="button" className="btn btn-danger ml-auto" disabled={!count} onClick={() => setConfirming(true)}>
            Delete
          </button>
        )}
      </div>
      {message && <p className="mt-2 text-ok">{message}</p>}
      {error && (
        <div className="mt-2">
          <ErrorNote message={error} />
        </div>
      )}
    </div>
  );
}
