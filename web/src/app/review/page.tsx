"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { Badge, ErrorNote, PageHeader, Panel } from "@/components/ui";
import { api, apiUrl, type ReviewClip, type ReviewPlan } from "@/lib/api";

function clock(frame: number, fps: number) {
  const total = frame / fps;
  const minutes = Math.floor(total / 60);
  return `${minutes}:${(total - minutes * 60).toFixed(3).padStart(6, "0")}`;
}

/** One edge of a clip: its frame number, with steps of a frame or a second either way. */
function FrameControl({
  label,
  frame,
  fps,
  onChange,
  onShow,
}: {
  label: string;
  frame: number;
  fps: number;
  onChange: (frame: number) => Promise<void>;
  onShow: () => void;
}) {
  const second = Math.round(fps);

  return (
    <fieldset className="min-w-0">
      <legend className="mb-1 text-sm font-medium">{label}</legend>
      <div className="flex flex-wrap items-center gap-1.5">
        <button type="button" className="btn" aria-label={`${label}: back one second`} onClick={() => onChange(frame - second)}>
          −1 s
        </button>
        <button type="button" className="btn" aria-label={`${label}: back one frame`} onClick={() => onChange(frame - 1)}>
          −1 f
        </button>
        <input
          className="field w-24"
          type="number"
          aria-label={`${label} frame`}
          key={frame}
          defaultValue={frame}
          onBlur={async (e) => {
            const input = e.currentTarget;
            const value = Number(input.value);
            if (input.value !== "" && Number.isInteger(value) && value !== frame) await onChange(value);
            // A new frame replaces this input. Otherwise the edit was refused, so show what still applies.
            input.value = String(frame);
          }}
        />
        <button type="button" className="btn" aria-label={`${label}: forward one frame`} onClick={() => onChange(frame + 1)}>
          +1 f
        </button>
        <button type="button" className="btn" aria-label={`${label}: forward one second`} onClick={() => onChange(frame + second)}>
          +1 s
        </button>
        <button type="button" className="btn" aria-label={`Show ${label.toLowerCase()} frame`} onClick={onShow}>
          Show
        </button>
      </div>
      <p className="mt-1 text-xs text-muted">{clock(frame, fps)}</p>
    </fieldset>
  );
}

function ClipCard({
  clip,
  number,
  plan,
  onChange,
  onShow,
  onPlay,
}: {
  clip: ReviewClip;
  number: number;
  plan: ReviewPlan;
  onChange: (plan: ReviewPlan) => void;
  onShow: (frame: number) => void;
  onPlay: (clip: ReviewClip) => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const url = `/api/jobs/${plan.job_id}/review/${clip.key}`;

  async function send(path: string, init: RequestInit) {
    setError(null);
    try {
      onChange(await api<ReviewPlan>(path, init));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function setFrames(start_frame: number, end_frame: number) {
    return send(url, { method: "PATCH", body: JSON.stringify({ start_frame, end_frame }) });
  }

  const seconds = (clip.end_frame - clip.start_frame) / plan.fps;

  return (
    <Panel>
      <article aria-label={`Clip ${number}`} className={clip.rejected ? "opacity-60" : ""}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-semibold">Clip {number}</h2>
          <div className="flex items-center gap-1.5">
            {clip.rejected ? <Badge tone="fail">Rejected</Badge> : <Badge tone="ok">Will export</Badge>}
            <Badge>{seconds.toFixed(1)} s</Badge>
            <Badge>{clip.end_frame - clip.start_frame} frames</Badge>
          </div>
        </div>
        <p className="mt-2 text-sm">
          <span className="font-medium">Why this clip: </span>
          <span className="text-muted">{clip.why || "No reason recorded."}</span>
        </p>
        {clip.text && <p className="mt-2 rounded-lg bg-panel-2 p-3 text-sm">{clip.text}</p>}
        {clip.warnings.map((warning) => (
          <p key={warning} className="mt-2 text-sm text-warn" role="status">
            {warning}
          </p>
        ))}
        <ErrorNote message={error} />
        {!clip.rejected && (
          <div className="mt-3 grid gap-3 lg:grid-cols-2">
            <FrameControl
              label="Start"
              frame={clip.start_frame}
              fps={plan.fps}
              onChange={(frame) => setFrames(frame, clip.end_frame)}
              onShow={() => onShow(clip.start_frame)}
            />
            <FrameControl
              label="End"
              frame={clip.end_frame}
              fps={plan.fps}
              onChange={(frame) => setFrames(clip.start_frame, frame)}
              onShow={() => onShow(Math.max(0, clip.end_frame - 1))}
            />
          </div>
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          {!clip.rejected && (
            <>
              <button type="button" className="btn" onClick={() => onPlay(clip)}>
                Play this clip
              </button>
              <button
                type="button"
                className="btn"
                disabled={plan.alternates_left === 0}
                onClick={() => send(`${url}/swap`, { method: "POST" })}
              >
                Swap for next best
              </button>
            </>
          )}
          <button
            type="button"
            className={clip.rejected ? "btn" : "btn btn-danger"}
            onClick={() => send(`${url}/reject`, { method: "POST", body: JSON.stringify({ rejected: !clip.rejected }) })}
          >
            {clip.rejected ? "Restore" : "Reject"}
          </button>
        </div>
      </article>
    </Panel>
  );
}

function Review() {
  const jobId = useSearchParams().get("job");
  const router = useRouter();
  const [plan, setPlan] = useState<ReviewPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const stopAt = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;
    api<ReviewPlan>(`/api/jobs/${jobId}/review`).then(setPlan, (err) => setError((err as Error).message));
  }, [jobId]);

  const show = useCallback(
    (frame: number) => {
      if (!video.current || !plan) return;
      stopAt.current = null;
      video.current.pause();
      // The middle of the frame, so rounding in the browser cannot land on its neighbour.
      video.current.currentTime = (frame + 0.5) / plan.fps;
    },
    [plan],
  );

  const play = useCallback(
    (clip: ReviewClip) => {
      if (!video.current || !plan) return;
      stopAt.current = clip.end_frame / plan.fps;
      video.current.currentTime = (clip.start_frame + 0.5) / plan.fps;
      void video.current.play();
    },
    [plan],
  );

  async function finish(path: string, destination: string) {
    setBusy(true);
    setError(null);
    try {
      await api(path, { method: "POST" });
      router.push(destination);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  if (!jobId) return <ErrorNote message="No job was chosen. Open a job waiting for review from the Jobs page." />;
  if (!plan) {
    return (
      <>
        <PageHeader title="Review clips" subtitle="Loading the clips." />
        <ErrorNote message={error} />
        {error && (
          <Link className="text-accent underline" href="/jobs">
            Back to Jobs
          </Link>
        )}
      </>
    );
  }

  const kept = plan.clips.filter((clip) => !clip.rejected).length;

  return (
    <>
      <PageHeader
        title="Review clips"
        subtitle="Nothing is rendered until you export. Adjust the edges to the exact frame, swap a clip for the next best, or reject it."
      />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="lg:sticky lg:top-4 lg:self-start">
          {plan.source_url && (
            <video
              ref={video}
              controls
              preload="metadata"
              aria-label="Source video preview"
              src={apiUrl(plan.source_url)}
              className="w-full rounded-xl bg-black"
              onTimeUpdate={(e) => {
                if (stopAt.current !== null && e.currentTarget.currentTime >= stopAt.current) {
                  e.currentTarget.pause();
                  stopAt.current = null;
                }
              }}
            />
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || kept === 0}
              onClick={() => finish(`/api/jobs/${plan.job_id}/review/approve`, "/jobs")}
            >
              Export {kept} {kept === 1 ? "clip" : "clips"}
            </button>
            <button
              type="button"
              className="btn btn-danger"
              disabled={busy}
              onClick={() => finish(`/api/jobs/${plan.job_id}/cancel`, "/jobs")}
            >
              Cancel job
            </button>
          </div>
          <ErrorNote message={error} />
          <p className="mt-2 text-xs text-muted">
            {plan.fps.toFixed(3).replace(/\.?0+$/, "")} frames per second · {plan.alternates_left} other candidates
            available
          </p>
        </div>
        <div className="space-y-3">
          {plan.clips.map((clip, index) => (
            <ClipCard key={clip.key} clip={clip} number={index + 1} plan={plan} onChange={setPlan} onShow={show} onPlay={play} />
          ))}
        </div>
      </div>
    </>
  );
}

export default function ReviewPage() {
  return (
    <Suspense>
      <Review />
    </Suspense>
  );
}
