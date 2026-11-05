"use client";

import { useState } from "react";
import { Badge, EmptyState, ErrorNote, Field, PageHeader, Panel, Segmented } from "@/components/ui";
import { api, usePoll, type Channel, type ChannelSettings, type Preset } from "@/lib/api";

const PLATFORMS = { youtube: "YouTube", instagram: "Instagram", facebook: "Facebook" } as const;

function ChannelCard({ channel, presets, selected, onSelect, onChange }: { channel: Channel; presets: Preset[]; selected: boolean; onSelect: (v: boolean) => void; onChange: () => void }) {
  const [settings, setSettings] = useState<ChannelSettings>(channel.settings);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  function update<K extends keyof ChannelSettings>(key: K, value: ChannelSettings[K]) {
    setSettings((s) => ({ ...s, [key]: value }));
    setSaved(false);
  }

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
    <Panel>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label className="flex cursor-pointer items-center gap-2 font-medium">
          <input type="checkbox" aria-label={`Select ${channel.name} for linking`} className="size-4 accent-[var(--accent)]" checked={selected} onChange={(e) => onSelect(e.target.checked)} />
          {channel.name}
        </label>
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge>{PLATFORMS[channel.platform]}</Badge>
          {channel.link_group && <Badge tone="warn">Linked: {channel.link_group.slice(0, 6)}</Badge>}
          <Badge tone={channel.connected ? "ok" : "neutral"}>{channel.connected ? `Connected${channel.account_name ? `: ${channel.account_name}` : ""}` : "Not connected"}</Badge>
        </div>
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Segmented label="Default format" value={settings.format} onChange={(v) => update("format", v)} options={[{ value: "short", label: "Shorts / Reels" }, { value: "long", label: "Long video" }]} />
        <Segmented label="Default ratio" value={settings.ratio} onChange={(v) => update("ratio", v)} options={[{ value: "9:16", label: "9:16" }, { value: "1:1", label: "1:1" }, { value: "16:9", label: "16:9" }]} />
        <Field label="Caption preset">
          <select className="field" value={settings.preset_id ?? ""} onChange={(e) => update("preset_id", e.target.value || null)}>
            <option value="">Use the job&apos;s preset</option>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Title template" hint="Placeholders: {title} {part} {genre}">
          <input className="field" value={settings.title_template} onChange={(e) => update("title_template", e.target.value)} />
        </Field>
        <div className="sm:col-span-2">
          <Field label="Description template" hint="Placeholders: {summary} {hashtags} {credit}. Required credits are always added.">
            <textarea className="field" rows={3} value={settings.description_template} onChange={(e) => update("description_template", e.target.value)} />
          </Field>
        </div>
      </div>
      <div className="mt-3"><ErrorNote message={error} /></div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn"
          onClick={() =>
            run(async () => {
              await api(`/api/channels/${channel.id}/settings`, { method: "PUT", body: JSON.stringify(settings) });
              setSaved(true);
            })
          }
        >
          Save settings
        </button>
        {saved && <span className="text-sm text-ok">Saved.</span>}
        <button type="button" className="btn btn-danger ml-auto" onClick={() => run(() => api(`/api/channels/${channel.id}`, { method: "DELETE" }))}>
          Remove
        </button>
      </div>
    </Panel>
  );
}

export default function ChannelsPage() {
  const { data: channels, error, refresh } = usePoll<Channel[]>("/api/channels", 0);
  const { data: presets } = usePoll<Preset[]>("/api/presets", 0);
  const [name, setName] = useState("");
  const [platform, setPlatform] = useState<Channel["platform"]>("youtube");
  const [selected, setSelected] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  async function add() {
    setFormError(null);
    try {
      await api("/api/channels", { method: "POST", body: JSON.stringify({ name, platform }) });
      setName("");
      await refresh();
    } catch (err) {
      setFormError((err as Error).message);
    }
  }

  async function link() {
    setFormError(null);
    try {
      await api("/api/channels/link", { method: "POST", body: JSON.stringify({ channel_ids: selected }) });
      setSelected([]);
      await refresh();
    } catch (err) {
      setFormError((err as Error).message);
    }
  }

  return (
    <>
      <PageHeader title="Channels" subtitle="Each channel keeps its own defaults. Linked channels can receive the same clip." />
      <div className="space-y-4">
        <Panel title="Add a channel">
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-48 flex-1">
              <Field label="Name">
                <input className="field" maxLength={120} value={name} onChange={(e) => setName(e.target.value)} />
              </Field>
            </div>
            <Segmented label="Platform" value={platform} onChange={setPlatform} options={Object.entries(PLATFORMS).map(([value, label]) => ({ value: value as Channel["platform"], label }))} />
            <button type="button" className="btn btn-primary" disabled={!name.trim()} onClick={add}>
              Add
            </button>
          </div>
        </Panel>
        <ErrorNote message={error ?? formError} />
        {channels?.length === 0 && <EmptyState title="No channels yet">Add one above to set its defaults.</EmptyState>}
        {selected.length > 1 && (
          <button type="button" className="btn" onClick={link}>
            Link {selected.length} selected channels
          </button>
        )}
        {channels?.map((channel) => (
          <ChannelCard
            key={channel.id}
            channel={channel}
            presets={presets ?? []}
            selected={selected.includes(channel.id)}
            onSelect={(on) => setSelected((s) => (on ? [...s, channel.id] : s.filter((id) => id !== channel.id)))}
            onChange={refresh}
          />
        ))}
      </div>
    </>
  );
}
