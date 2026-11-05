"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { formatBytes, usePoll, type Hardware } from "@/lib/api";

const LINKS = [
  { href: "/", label: "New job" },
  { href: "/jobs", label: "Jobs" },
  { href: "/clips", label: "Clips" },
  { href: "/presets", label: "Caption presets" },
  { href: "/channels", label: "Channels" },
];

export function Nav() {
  const pathname = usePathname();
  const { data: hardware, error } = usePoll<Hardware>("/api/hardware", 0);
  const { data: storage } = usePoll<{ free_bytes: number; renders_bytes: number; media_bytes: number }>("/api/storage", 15000);

  return (
    <aside className="border-b border-line bg-panel md:sticky md:top-0 md:flex md:h-screen md:w-56 md:shrink-0 md:flex-col md:border-b-0 md:border-r">
      <div className="px-4 pt-4 text-lg font-bold tracking-tight md:px-5 md:pt-5">
        clip<span className="text-accent">forge</span>
      </div>
      <nav aria-label="Main" className="flex gap-1 overflow-x-auto px-2 py-3 md:flex-col md:px-3 md:py-5">
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              aria-current={active ? "page" : undefined}
              className={`whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium ${
                active ? "bg-panel-2 text-text" : "text-muted hover:text-text"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto hidden space-y-1 border-t border-line p-4 text-xs text-muted md:block">
        {error && <p className="text-bad">Engine offline</p>}
        {hardware && (
          <>
            <p className="text-text">{hardware.gpu_name ?? `${hardware.cpu_threads} CPU threads`}</p>
            <p>Encoder: {hardware.h264_encoder}</p>
            <p>Speech: {hardware.whisper_device === "cuda" ? "GPU" : "CPU"}</p>
          </>
        )}
        {storage && <p>{formatBytes(storage.free_bytes)} free</p>}
      </div>
    </aside>
  );
}
