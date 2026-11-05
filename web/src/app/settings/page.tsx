"use client";

import { useEffect, useState } from "react";
import { ErrorNote, Field, PageHeader, Panel } from "@/components/ui";
import { api, formatBytes, usePoll, type AppSettings, type StorageReport } from "@/lib/api";

export default function SettingsPage() {
  const { data: settings, refresh } = usePoll<AppSettings>("/api/settings", 0);
  const { data: storage, refresh: refreshStorage } = usePoll<StorageReport>("/api/storage", 15000);
  const [savedDir, setSavedDir] = useState("");
  const [queuedLimit, setQueuedLimit] = useState(50);
  const [savedLimit, setSavedLimit] = useState(200);
  const [lowDisk, setLowDisk] = useState(5);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!settings) return;
    const timer = setTimeout(() => {
      setSavedDir(settings.saved_dir ?? "");
      setQueuedLimit(settings.queued_limit_gb);
      setSavedLimit(settings.saved_limit_gb);
      setLowDisk(settings.low_disk_gb);
    }, 0);
    return () => clearTimeout(timer);
  }, [settings]);

  async function save() {
    setMessage(null);
    setError(null);
    try {
      await api("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ saved_dir: savedDir.trim() || null, queued_limit_gb: queuedLimit, saved_limit_gb: savedLimit, low_disk_gb: lowDisk }),
      });
      await refresh();
      await refreshStorage();
      setMessage("Saved.");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <PageHeader title="Settings" subtitle="Where clips are kept, and when clipforge warns you about disk space." />
      <div className="max-w-2xl space-y-4">
        <Panel title="Saved folder" hint="Clips you move to Saved are never deleted automatically.">
          <Field label="Folder path" hint="Leave empty for the default. A Windows path works, such as D:\clipforge-saved. Clips already in Saved stay where they are.">
            <input className="field" placeholder={settings?.saved_path} value={savedDir} onChange={(e) => setSavedDir(e.target.value)} />
          </Field>
          {settings && <p className="mt-2 break-all text-xs text-muted">Now using: {settings.saved_path}</p>}
        </Panel>
        <Panel title="Disk space limits">
          <div className="grid gap-4 sm:grid-cols-3">
            <Field label="Warn when Queued passes (GB)">
              <input className="field" type="number" min={1} value={queuedLimit} onChange={(e) => setQueuedLimit(Number(e.target.value))} />
            </Field>
            <Field label="Warn when Saved passes (GB)">
              <input className="field" type="number" min={1} value={savedLimit} onChange={(e) => setSavedLimit(Number(e.target.value))} />
            </Field>
            <Field label="Stop new jobs below (GB free)">
              <input className="field" type="number" min={0} value={lowDisk} onChange={(e) => setLowDisk(Number(e.target.value))} />
            </Field>
          </div>
        </Panel>
        {storage && (
          <Panel title="Space used">
            <dl className="grid grid-cols-2 gap-y-1 text-sm sm:grid-cols-[12rem_1fr]">
              <dt className="text-muted">Unassigned</dt>
              <dd>{formatBytes(storage.folders.unassigned)}</dd>
              <dt className="text-muted">Queued</dt>
              <dd>{formatBytes(storage.folders.queued)}</dd>
              <dt className="text-muted">Saved</dt>
              <dd>{formatBytes(storage.folders.saved)}</dd>
              <dt className="text-muted">Free on the work drive</dt>
              <dd>{formatBytes(storage.free_bytes)}</dd>
              <dt className="text-muted">Free on the Saved drive</dt>
              <dd>{formatBytes(storage.saved_free_bytes)}</dd>
            </dl>
          </Panel>
        )}
        <ErrorNote message={error} />
        <div className="flex items-center gap-3">
          <button type="button" className="btn btn-primary" onClick={save}>
            Save settings
          </button>
          {message && <span className="text-sm text-ok">{message}</span>}
        </div>
      </div>
    </>
  );
}
