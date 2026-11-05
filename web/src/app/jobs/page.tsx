"use client";

import Link from "next/link";
import { Badge, EmptyState, ErrorNote, PageHeader, Panel, ProgressBar } from "@/components/ui";
import { api, RIGHTS_LABELS, usePoll, type Job } from "@/lib/api";

const STATUS_TONE = { queued: "neutral", running: "warn", done: "ok", failed: "fail", cancelled: "neutral" } as const;

function Decisions({ job }: { job: Job }) {
  const reasons = job.decisions.reasons as Record<string, string> | undefined;
  const labels: Record<string, string> = { clip_duration: "Clip length", clip_count: "Clips" };
  const notes = (job.decisions.notes as string[] | undefined) ?? [];
  if (!reasons && !notes.length) return null;
  return (
    <div className="mt-3 border-t border-line pt-3 text-sm">
      {reasons && (
        <>
          <h3 className="mb-2 text-xs font-semibold text-muted">What auto mode decided, and why</h3>
          <dl className="grid gap-x-4 gap-y-1 sm:grid-cols-[13rem_1fr]">
            {Object.entries(reasons).map(([key, why]) => (
              <div key={key} className="contents">
                <dt className="font-medium">
                  <span className="capitalize">{labels[key] ?? key}</span>: {String(job.decisions[key] ?? "")}
                  {key === "clip_duration" ? " s" : ""}
                </dt>
                <dd className="text-muted">{why}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
      {notes.map((note) => (
        <p key={note} className="mt-2 text-warn">
          {note}
        </p>
      ))}
    </div>
  );
}

export default function JobsPage() {
  const { data: jobs, error, refresh } = usePoll<Job[]>("/api/jobs", 1500);

  async function cancel(id: string) {
    await api(`/api/jobs/${id}/cancel`, { method: "POST" });
    await refresh();
  }

  return (
    <>
      <PageHeader title="Jobs" subtitle="Progress updates live. Closing the app restarts a running job when it opens again." />
      <ErrorNote message={error} />
      {jobs?.length === 0 && (
        <EmptyState title="No jobs yet">
          <Link className="text-accent underline" href="/">
            Start your first job
          </Link>
        </EmptyState>
      )}
      <div className="space-y-3">
        {jobs?.map((job) => (
          <Panel key={job.id}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate font-medium">{job.source_filename ?? "Deleted video"}</p>
                <p className="text-xs text-muted">
                  {job.mode === "auto" ? "Auto" : "Manual"} · Rights: {RIGHTS_LABELS[job.rights_type] ?? "Missing"}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge tone={STATUS_TONE[job.status]}>{job.status}</Badge>
                {(job.status === "queued" || job.status === "running") && (
                  <button type="button" className="btn btn-danger" onClick={() => cancel(job.id)}>
                    Cancel
                  </button>
                )}
              </div>
            </div>
            {(job.status === "running" || job.status === "queued") && (
              <div className="mt-3 space-y-1" aria-live="polite">
                <ProgressBar value={job.progress} label={`Progress for ${job.source_filename ?? "job"}`} />
                <p className="text-xs text-muted">
                  {job.stage} · {Math.round(job.progress * 100)}%
                </p>
              </div>
            )}
            {job.status === "failed" && <div className="mt-3"><ErrorNote message={job.error} /></div>}
            {job.status === "done" && (
              <p className="mt-2 text-sm text-muted">
                {job.clip_count} {job.clip_count === 1 ? "clip" : "clips"} made.{" "}
                <Link className="text-accent underline" href="/clips">
                  Open clips
                </Link>
              </p>
            )}
            <Decisions job={job} />
          </Panel>
        ))}
      </div>
    </>
  );
}
