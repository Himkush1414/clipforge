"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

export function NoChannelDialog({ onJustMake, onLater, onCancel }: { onJustMake: () => void; onLater: () => void; onCancel: () => void }) {
  const first = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    first.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div role="alertdialog" aria-modal="true" aria-labelledby="no-channel-title" aria-describedby="no-channel-text" className="w-full max-w-lg rounded-xl border border-warn/50 bg-panel p-5">
        <h2 id="no-channel-title" className="text-lg font-semibold">
          No channel is connected.
        </h2>
        <p id="no-channel-text" className="mt-1 text-sm text-muted">
          You can still make the clips. Choose where they should wait.
        </p>
        <div className="mt-4 space-y-3">
          <div>
            <button ref={first} type="button" className="btn btn-primary w-full" onClick={onJustMake}>
              Just make the clips
            </button>
            <p className="mt-1 text-xs text-muted">They go to Unassigned and are deleted 24 hours after they are made.</p>
          </div>
          <div>
            <button type="button" className="btn w-full" onClick={onLater}>
              Make now, upload later
            </button>
            <p className="mt-1 text-xs text-muted">They go to Queued, with the timer paused, until a channel is connected.</p>
          </div>
          <div>
            <Link className="btn w-full" href="/channels">
              Go to Channels
            </Link>
            <p className="mt-1 text-xs text-muted">Connect a channel first. This job does not start.</p>
          </div>
        </div>
        <button type="button" className="mt-4 text-sm text-muted underline" onClick={onCancel}>
          Back to the job
        </button>
      </div>
    </div>
  );
}
