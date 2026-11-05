"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type Hardware = {
  gpu_name: string | null;
  gpu_memory_mb: number | null;
  cpu_threads: number;
  ram_gb: number;
  disk_free_gb: number;
  h264_encoder: string;
  hevc_encoder: string;
  whisper_device: string;
  allowed_resolutions: string[];
};

export type Probe = {
  duration: number;
  width: number;
  height: number;
  fps: number;
  has_audio: boolean;
  frame_count: number;
  size_bytes: number;
};

export type Source = {
  id: string;
  filename: string;
  owned_copy: boolean;
  probe: Probe;
  created_at: string;
};

export type QcCheck = { name: string; status: "ok" | "warn" | "fail"; detail: string };

export type Clip = {
  id: string;
  job_id: string;
  part: number | null;
  start: number;
  end: number;
  score: number;
  title: string;
  description: string;
  format: string;
  ratio: string;
  resolution: string;
  width: number;
  height: number;
  duration: number;
  frames: number;
  size_bytes: number;
  status: string;
  qc: { status?: "ok" | "warn" | "fail"; checks?: QcCheck[]; encoder?: string };
  seconds_left: number;
  file_url: string | null;
  thumb_url: string | null;
};

export type Job = {
  id: string;
  source_id: string;
  source_filename?: string | null;
  mode: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string;
  progress: number;
  message: string;
  rights_type: string;
  rights_detail: string;
  decisions: Record<string, unknown>;
  error: string | null;
  clip_count: number;
  created_at: string;
};

export type PresetData = {
  name: string;
  font: string;
  emphasis_font: string | null;
  uppercase: boolean;
  max_words: number;
  size: number;
  emphasis_scale: number;
  filler_scale: number;
  text_color: string;
  active_color: string;
  emphasis_color: string | null;
  outline_color: string;
  outline: number;
  shadow_color: string;
  shadow: number;
  glow_color: string | null;
  glow: number;
  active_style: "color" | "box";
  box_color: string;
  animation: "pop" | "bounce" | "fade" | "slide" | "none";
  active_pop: number;
  tilt: number;
  position: "lower" | "center" | "upper";
  y_offset: number;
  word_gap: number;
};

export type Preset = { id: string; name: string; builtin: boolean; data: PresetData };

export type Font = { family: string; file: string; em_ratio: number };

export type ChannelSettings = {
  format: "short" | "long";
  ratio: "9:16" | "1:1" | "16:9";
  preset_id: string | null;
  title_template: string;
  description_template: string;
};

export type Channel = {
  id: string;
  name: string;
  platform: "youtube" | "instagram" | "facebook";
  link_group: string | null;
  settings: ChannelSettings;
  connected: boolean;
  account_name: string | null;
};

export function apiBase(): string {
  if (typeof window === "undefined") return "http://127.0.0.1:8765";
  return `${window.location.protocol}//${window.location.hostname}:8765`;
}

export function apiUrl(path: string): string {
  return `${apiBase()}${path}`;
}

export class ApiError extends Error {}

function readDetail(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => (d.msg ?? "").replace(/^Value error, /, "")).join(" ");
  }
  return fallback;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: init?.body ? { "content-type": "application/json", ...init.headers } : init?.headers,
    });
  } catch {
    throw new ApiError("The clipforge engine is not reachable. Start it with scripts/dev.sh.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(readDetail(body, `Request failed (${res.status})`));
  }
  return (await res.json()) as T;
}

export function uploadVideo(file: File, onProgress: (share: number) => void): Promise<Source> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", apiUrl(`/api/sources/upload?filename=${encodeURIComponent(file.name)}`));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // Handled below: a non-JSON failure falls back to the status code.
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as Source);
      else reject(new ApiError(readDetail(body, `Upload failed (${xhr.status})`)));
    };
    xhr.onerror = () => reject(new ApiError("The clipforge engine is not reachable."));
    xhr.send(file);
  });
}

/** Load data now and keep it fresh. Polling is enough: one user, on one machine. */
export function usePoll<T>(path: string, intervalMs = 2000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const value = await api<T>(path);
      if (alive.current) {
        setData(value);
        setError(null);
      }
    } catch (err) {
      if (alive.current) setError((err as Error).message);
    }
  }, [path]);

  useEffect(() => {
    alive.current = true;
    const first = setTimeout(refresh, 0);
    const timer = intervalMs > 0 ? setInterval(refresh, intervalMs) : undefined;
    return () => {
      alive.current = false;
      clearTimeout(first);
      if (timer) clearInterval(timer);
    };
  }, [refresh, intervalMs]);

  return { data, error, refresh };
}

export function formatDuration(seconds: number): string {
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export const RIGHTS_LABELS: Record<string, string> = {
  own: "My own footage",
  public_domain: "Public domain",
  creative_commons: "Creative Commons",
  written_permission: "Written permission",
};

export type ClaimMatch = {
  title?: string | null;
  policy?: string | null;
  actions?: string[];
  segments?: { start: number; duration: number; type: string }[];
};

export type Upload = {
  id: string;
  clip_id: string;
  channel_id: string;
  channel_name: string | null;
  platform: Channel["platform"] | null;
  status: "pending" | "uploading" | "processing" | "published" | "failed" | "claimed" | "stopped";
  privacy: string;
  url: string | null;
  error: string | null;
  claim: { platform?: string; reason?: string; note?: string; matches?: ClaimMatch[] };
  claim_resolution: "trim" | "replace_audio" | "dispute" | null;
  claim_note: string;
};

export type PlatformInfo = { group: "google" | "meta"; configured: boolean; redirect_uri: string };
