"use client";

import { useState } from "react";
import { Badge, ErrorNote, Field, Panel } from "@/components/ui";
import { api, usePoll, type PlatformInfo } from "@/lib/api";

const TEXT = {
  google: {
    title: "Google (YouTube)",
    id: "OAuth client ID",
    secret: "OAuth client secret",
    steps: [
      "Create a project in Google Cloud Console and enable YouTube Data API v3.",
      "Configure the OAuth consent screen and add yourself as a test user.",
      "Create a Web application OAuth client with the redirect address below.",
    ],
    limit: "Default limit: 100 uploads a day per API project.",
  },
  meta: {
    title: "Meta (Facebook and Instagram)",
    id: "App ID",
    secret: "App secret",
    steps: [
      "Create an app at developers.facebook.com using Facebook Login for Business.",
      "Add the redirect address below as a valid OAuth redirect.",
      "Instagram needs a professional account linked to a Facebook Page.",
    ],
    limit: "Facebook: 30 reels a day, 3 to 90 seconds. Instagram: see content_publishing_limit.",
  },
};

function Group({ info, onChange }: { info: PlatformInfo; onChange: () => void }) {
  const text = TEXT[info.group];
  const [id, setId] = useState("");
  const [secret, setSecret] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function save() {
    setError(null);
    try {
      await api(`/api/platforms/${info.group}/credentials`, { method: "PUT", body: JSON.stringify({ client_id: id, client_secret: secret }) });
      setId("");
      setSecret("");
      onChange();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="rounded-lg border border-line p-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">{text.title}</h3>
        <Badge tone={info.configured ? "ok" : "neutral"}>{info.configured ? "App added" : "Needs your app"}</Badge>
      </div>
      <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-muted">
        {text.steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      <p className="mt-2 break-all text-xs">
        Redirect address: <code className="text-accent">{info.redirect_uri}</code>
      </p>
      <p className="mt-1 text-xs text-muted">{text.limit}</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Field label={text.id}>
          <input className="field" autoComplete="off" value={id} onChange={(e) => setId(e.target.value)} />
        </Field>
        <Field label={text.secret} hint="Stored only on this computer and never shown again.">
          <input className="field" type="password" autoComplete="off" value={secret} onChange={(e) => setSecret(e.target.value)} />
        </Field>
      </div>
      <div className="mt-3 space-y-2">
        <ErrorNote message={error} />
        <button type="button" className="btn" disabled={!id.trim() || !secret.trim()} onClick={save}>
          {info.configured ? "Replace app details" : "Save app details"}
        </button>
      </div>
    </div>
  );
}

export function PlatformSetup() {
  const { data, refresh } = usePoll<PlatformInfo[]>("/api/platforms", 0);
  return (
    <Panel title="Platform apps" hint="Uploads use each platform's official API, through an app you own. Setup is a one-time job.">
      <div className="grid gap-3 lg:grid-cols-2">
        {data?.map((info) => <Group key={info.group} info={info} onChange={refresh} />)}
      </div>
    </Panel>
  );
}
