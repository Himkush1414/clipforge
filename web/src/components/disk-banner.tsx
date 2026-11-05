"use client";

import { usePoll, type StorageReport } from "@/lib/api";

export function DiskBanner() {
  const { data } = usePoll<StorageReport>("/api/storage", 15000);
  if (!data?.warnings.length) return null;
  return (
    <div role="alert" className="mb-4 space-y-1 rounded-lg border border-warn/50 bg-warn/10 px-3 py-2 text-sm text-warn">
      {data.warnings.map((warning) => (
        <p key={warning}>{warning}</p>
      ))}
    </div>
  );
}
