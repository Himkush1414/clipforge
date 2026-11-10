"use client";

import { useRef, useState } from "react";
import { Badge, ErrorNote, Toggle } from "@/components/ui";
import { api, apiUrl, ApiError, type Channel, type Clip, type Upload } from "@/lib/api";

const TONE = { pending: "neutral", uploading: "warn", processing: "warn", published: "ok", failed: "fail", claimed: "fail", stopped: "warn" } as const;
const YOUTUBE_DISPUTE_HELP = "https://support.google.com/youtube/answer/2797454";

function ClaimPanel({ upload, clip, onChange }: { upload: Upload; clip: Clip; onChange: () => void }) {
  const [note, setNote] = useState("");
  const [licence, setLicence] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function resolve(resolution: "trim" | "replace_audio" | "dispute", text = "") {
    await api(`/api/uploads/${upload.id}/resolve`, { method: "POST", body: JSON.stringify({ resolution, note: text }) });
  }

  async function act(action: () => Promise<void>, done: string) {
    setError(null);
    try {
      await action();
      setMessage(done);
      onChange();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function sendAudio(file: File) {
    const params = new URLSearchParams({ filename: file.name, licence });
    const res = await fetch(apiUrl(`/api/clips/${clip.id}/replace-audio?${params}`), { method: "PUT", body: file });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new ApiError(body?.detail ?? "Could not replace the audio.");
    }
    await resolve("replace_audio", licence);
  }

  return (
    <div role="alert" className="mt-2 space-y-3 rounded-lg border border-bad/40 bg-bad/10 p-3 text-xs">
      <div>
        <p className="font-semibold text-bad">Copyright claim: uploads from this video have stopped</p>
        <p className="mt-1 text-muted first-letter:uppercase">
          {upload.claim.reason}
          {upload.claim.note ? `. ${upload.claim.note}` : ""}. Nothing will be re-uploaded automatically.
        </p>
      </div>
      {upload.claim.matches?.map((match, index) => (
        <div key={index} className="rounded bg-panel p-2">
          <p className="font-medium">{match.title ?? "Unnamed match"}</p>
          {match.policy && <p className="text-muted">Owner: {match.policy}</p>}
          {!!match.actions?.length && <p className="text-muted">Action: {match.actions.join(", ")}</p>}
          {match.segments?.map((seg) => (
            <p key={`${seg.start}-${seg.duration}`} className="text-muted">
              {seg.type} from {seg.start}s for {seg.duration}s
            </p>
          ))}
        </div>
      ))}
      {upload.claim_resolution ? (
        <p className="text-ok">You chose: {upload.claim_resolution.replace("_", " ")}.</p>
      ) : (
        <div className="space-y-3">
          <div>
            <p className="font-medium">Option 1: Trim the claimed part</p>
            <p className="text-muted">Use &quot;Recut by exact frames&quot; above to make a version without it.</p>
            <button type="button" className="btn mt-1" onClick={() => act(() => resolve("trim"), "Noted. Recut the clip, then upload the new version.")}>
              I will trim it
            </button>
          </div>
          <div>
            <p className="font-medium">Option 2: Replace the audio</p>
            <p className="text-muted">Use a track you hold a licence for. This makes a new clip.</p>
            <input className="field mt-1" placeholder="What licence do you hold for this track?" value={licence} onChange={(e) => setLicence(e.target.value)} />
            <input
              ref={fileRef}
              type="file"
              accept="audio/*"
              aria-label="Choose a replacement audio file"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void act(() => sendAudio(file), "New clip made with the replacement audio.");
                e.target.value = "";
              }}
            />
            <button type="button" className="btn mt-1" disabled={!licence.trim()} onClick={() => fileRef.current?.click()}>
              Choose audio file
            </button>
          </div>
          <div>
            <p className="font-medium">Option 3: Dispute, if you hold a licence</p>
            <p className="text-muted">
              Disputes are filed with the platform, not through clipforge.{" "}
              {upload.platform === "youtube" ? (
                <a className="text-accent underline" href={YOUTUBE_DISPUTE_HELP} target="_blank" rel="noreferrer">
                  How to dispute on YouTube
                </a>
              ) : (
                "Follow the appeal option in the notice Meta sends you."
              )}
            </p>
            <input className="field mt-1" placeholder="Note about the licence you hold" value={note} onChange={(e) => setNote(e.target.value)} />
            <button type="button" className="btn mt-1" disabled={!note.trim()} onClick={() => act(() => resolve("dispute", note), "Dispute noted. File it on the platform.")}>
              I will dispute it
            </button>
          </div>
        </div>
      )}
      <ErrorNote message={error} />
      {message && <p className="text-ok">{message}</p>}
    </div>
  );
}

function UploadRow({ upload, clip, onChange }: { upload: Upload; clip: Clip; onChange: () => void }) {
  const [reporting, setReporting] = useState(false);
  const [details, setDetails] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function run(action: () => Promise<unknown>) {
    setError(null);
    try {
      await action();
      onChange();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <li className="rounded-lg bg-panel-2 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{upload.channel_name ?? "Removed channel"}</span>
        <Badge tone={TONE[upload.status]}>{upload.status}</Badge>
        {upload.platform === "youtube" && <Badge>{upload.privacy}</Badge>}
        {upload.url && (
          <a className="text-accent underline" href={upload.url} target="_blank" rel="noreferrer">
            Open
          </a>
        )}
        {upload.status === "processing" && (
          <button type="button" className="ml-auto underline" onClick={() => run(() => api(`/api/uploads/${upload.id}/refresh`, { method: "POST" }))}>
            Check now
          </button>
        )}
        {(upload.status === "processing" || upload.status === "published") && (
          <button type="button" className="underline" onClick={() => setReporting(!reporting)}>
            Report a claim
          </button>
        )}
      </div>
      {upload.error && <p className="mt-1 text-bad">{upload.error}</p>}
      {reporting && (
        <div className="mt-2 space-y-1">
          <p className="text-muted">Platform APIs do not report every claim. If you see one in the platform&apos;s app, record it here to stop further uploads.</p>
          <textarea className="field" rows={2} placeholder="What does the claim say?" value={details} onChange={(e) => setDetails(e.target.value)} />
          <button
            type="button"
            className="btn"
            disabled={!details.trim()}
            onClick={() => run(() => api(`/api/uploads/${upload.id}/report-claim`, { method: "POST", body: JSON.stringify({ details }) }))}
          >
            Record claim
          </button>
        </div>
      )}
      <ErrorNote message={error} />
      {upload.status === "claimed" && <ClaimPanel upload={upload} clip={clip} onChange={onChange} />}
    </li>
  );
}

export function UploadPanel({ clip, channels, uploads, onChange }: { clip: Clip; channels: Channel[]; uploads: Upload[]; onChange: () => void }) {
  const connected = channels.filter((c) => c.connected);
  const groups = [...new Set(connected.map((c) => c.link_group).filter(Boolean))] as string[];
  const [target, setTarget] = useState("");
  const [privacy, setPrivacy] = useState("private");
  const [kids, setKids] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isGroup = target.startsWith("group:");
  const channel = connected.find((c) => c.id === target);
  const showsPrivacy = isGroup || channel?.platform === "youtube";

  // Nothing to upload to and nothing to report yet, so there is no control to show.
  if (connected.length === 0 && uploads.length === 0) return null;

  async function send() {
    setError(null);
    try {
      await api(`/api/clips/${clip.id}/upload`, {
        method: "POST",
        body: JSON.stringify({
          ...(isGroup ? { link_group: target.slice(6) } : { channel_id: target }),
          privacy,
          made_for_kids: kids,
        }),
      });
      setTarget("");
      onChange();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <details className="text-sm" open={uploads.length > 0}>
      <summary className="cursor-pointer font-medium">Upload to channel{uploads.length ? ` (${uploads.length})` : ""}</summary>
      <div className="mt-2 space-y-2">
        {!clip.uploadable ? (
          <>
            <p className="text-xs text-warn">From a private test video: this clip is never uploaded.</p>
            <button type="button" className="btn" disabled>
              Upload this clip
            </button>
          </>
        ) : connected.length === 0 ? (
          <p className="text-xs text-muted">Connect a channel on the Channels page first.</p>
        ) : (
          <>
            <select className="field" aria-label="Upload to" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Choose where to upload</option>
              {connected.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.platform})
                </option>
              ))}
              {groups.map((g) => (
                <option key={g} value={`group:${g}`}>
                  All linked channels ({g.slice(0, 6)})
                </option>
              ))}
            </select>
            {showsPrivacy && (
              <div className="space-y-2">
                <select className="field" aria-label="YouTube privacy" value={privacy} onChange={(e) => setPrivacy(e.target.value)}>
                  <option value="private">YouTube: Private</option>
                  <option value="unlisted">YouTube: Unlisted</option>
                  <option value="public">YouTube: Public</option>
                </select>
                <Toggle label="This video is made for kids (YouTube)" checked={kids} onChange={setKids} />
              </div>
            )}
            {target && !showsPrivacy && <p className="text-xs text-muted">Reels on this platform are published publicly.</p>}
            <button type="button" className="btn" disabled={!target} onClick={send}>
              Upload this clip
            </button>
          </>
        )}
        <ErrorNote message={error} />
        {uploads.length > 0 && (
          <ul className="space-y-2">
            {uploads.map((upload) => (
              <UploadRow key={upload.id} upload={upload} clip={clip} onChange={onChange} />
            ))}
          </ul>
        )}
      </div>
    </details>
  );
}
