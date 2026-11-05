"""Facebook Page Reels and Instagram Reels through the official Graph API.

Checked against Meta's docs on 2026-11-05 (Reels Publishing, Content Publishing, Resumable Uploads).
Facebook exposes copyright_check_information, so a matched reel is held before it is published.
Instagram documents no copyright signal, so that case relies on the manual "report a claim".
"""

from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx

from clipforge.uploads.base import (
    NotConfigured,
    RemoteState,
    UploadError,
    UploadRequest,
    load_tokens,
    platform_credentials,
    redirect_uri,
    save_tokens,
)

GRAPH_VERSION = "v27.0"
GRAPH = f"https://graph.facebook.com/{GRAPH_VERSION}"
RUPLOAD = "https://rupload.facebook.com"
AUTH_URL = f"https://www.facebook.com/{GRAPH_VERSION}/dialog/oauth"
SCOPES = {
    "facebook": "pages_show_list,pages_read_engagement,pages_manage_posts",
    "instagram": "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement",
}


def _credentials() -> dict:
    creds = platform_credentials("meta")
    if not creds:
        raise NotConfigured("Add your Meta app ID and secret on the Channels page first.")
    return creds


def begin_auth(platform: str, state: str) -> tuple[str, str]:
    query = urlencode(
        {
            "client_id": _credentials()["client_id"],
            "redirect_uri": redirect_uri(),
            "response_type": "code",
            "scope": SCOPES[platform],
            "state": state,
        }
    )
    return f"{AUTH_URL}?{query}", secrets.token_urlsafe(8)


def _raise_for(res: httpx.Response, prefix: str) -> None:
    if res.is_success:
        return
    message = res.text[:200]
    try:
        body = res.json()
        message = (
            (body.get("error") or {}).get("message")
            or (body.get("debug_info") or {}).get("message")
            or message
        )
    except (ValueError, AttributeError):
        pass
    raise UploadError(f"{prefix} ({res.status_code}): {message}")


def _auth(token: str) -> dict:
    return {"Authorization": f"OAuth {token}"}


def finish_auth(http: httpx.Client, platform: str, channel_id: str, code: str) -> str:
    """Exchange the code, then pick the first usable Page or Instagram account."""
    creds = _credentials()
    res = http.post(
        f"{GRAPH}/oauth/access_token",
        data={
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "redirect_uri": redirect_uri(),
            "code": code,
        },
    )
    _raise_for(res, "Meta sign-in failed")
    short = res.json()["access_token"]
    res = http.post(
        f"{GRAPH}/oauth/access_token",
        data={
            "grant_type": "fb_exchange_token",
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "fb_exchange_token": short,
        },
    )
    _raise_for(res, "Meta sign-in failed")
    user_token = res.json()["access_token"]
    res = http.get(
        f"{GRAPH}/me/accounts",
        params={"fields": "id,name,access_token,instagram_business_account{id,username}"},
        headers=_auth(user_token),
    )
    _raise_for(res, "Could not read your Facebook Pages")
    targets = _targets(res.json().get("data") or [], platform)
    if not targets:
        need = (
            "an Instagram professional account linked to a Facebook Page"
            if platform == "instagram"
            else "a Facebook Page"
        )
        raise UploadError(f"This account has no usable target. You need {need}.")
    save_tokens(channel_id, {"targets": targets, "target": targets[0]["id"]})
    return targets[0]["name"]


def _targets(pages: list[dict], platform: str) -> list[dict]:
    targets = []
    for page in pages:
        if platform == "facebook":
            targets.append({"id": page["id"], "name": page["name"], "token": page["access_token"]})
        elif page.get("instagram_business_account"):
            account = page["instagram_business_account"]
            targets.append(
                {
                    "id": account["id"],
                    "name": account.get("username", page["name"]),
                    "token": page["access_token"],
                }
            )
    return targets


def _target(channel_id: str) -> dict:
    tokens = load_tokens(channel_id)
    if not tokens:
        raise UploadError("This channel is not connected.")
    return next(t for t in tokens["targets"] if t["id"] == tokens["target"])


def validate_facebook_reel(request: UploadRequest) -> None:
    """Meta's documented limits, checked first so the failure message says exactly what to change."""
    if not 3 <= request.duration <= 90:
        raise UploadError(
            f"Facebook Reels must be 3 to 90 seconds. This clip is {request.duration:.0f} seconds."
        )
    if min(request.width, request.height) < 540:
        raise UploadError("Facebook Reels need at least 540x960.")
    if not 24 <= request.fps <= 60:
        raise UploadError(f"Facebook Reels need 24 to 60 fps. This clip is {request.fps:.0f} fps.")
    if request.width > request.height:
        raise UploadError("Facebook Reels must be vertical. Export this clip at 9:16.")


def upload_facebook(http: httpx.Client, channel_id: str, request: UploadRequest) -> RemoteState:
    validate_facebook_reel(request)
    target = _target(channel_id)
    res = http.post(
        f"{GRAPH}/{target['id']}/video_reels",
        json={"upload_phase": "start", "access_token": target["token"]},
    )
    _raise_for(res, "Facebook refused the upload")
    video_id = res.json()["video_id"]
    with request.path.open("rb") as handle:
        res = http.post(
            f"{RUPLOAD}/video-upload/{GRAPH_VERSION}/{video_id}",
            headers={
                **_auth(target["token"]),
                "offset": "0",
                "file_size": str(request.size_bytes),
                "Content-Type": "application/octet-stream",
            },
            content=handle.read(),
        )
    _raise_for(res, "Facebook failed during upload")
    # Not published yet. check_facebook publishes only after the copyright check comes back clean.
    return RemoteState(
        "processing",
        remote_id=video_id,
        detail={"title": request.title, "description": request.description, "finished": False},
    )


def _facebook_claim(info: dict) -> dict:
    matches = []
    for match in info.get("copyright_matches") or []:
        policy = match.get("owner_copyright_policy") or {}
        matches.append(
            {
                "title": match.get("content_title"),
                "policy": policy.get("name"),
                "actions": [a.get("action") for a in policy.get("actions") or []],
                "segments": [
                    {
                        "start": s.get("start_time_in_seconds"),
                        "duration": s.get("duration_in_seconds"),
                        "type": s.get("segment_type"),
                    }
                    for s in match.get("matched_segments") or []
                ],
            }
        )
    return {"platform": "facebook", "reason": "copyright match", "matches": matches}


def check_facebook(http: httpx.Client, channel_id: str, remote_id: str, detail: dict) -> RemoteState:
    target = _target(channel_id)
    url = f"https://www.facebook.com/reel/{remote_id}"
    if not detail.get("finished"):
        res = http.get(
            f"{GRAPH}/{remote_id}",
            params={"fields": "copyright_check_information"},
            headers=_auth(target["token"]),
        )
        _raise_for(res, "Could not read Facebook's copyright check")
        info = res.json().get("copyright_check_information") or {}
        status = info.get("status") or {}
        if status.get("matches_found"):
            return RemoteState("claimed", remote_id, url, claim=_facebook_claim(info), detail=detail)
        if status.get("status") != "complete":
            return RemoteState("processing", remote_id, detail=detail)
        res = http.post(
            f"{GRAPH}/{target['id']}/video_reels",
            data={
                "access_token": target["token"],
                "video_id": remote_id,
                "upload_phase": "finish",
                "video_state": "PUBLISHED",
                "title": detail.get("title", ""),
                "description": detail.get("description", ""),
            },
        )
        _raise_for(res, "Facebook refused to publish")
        detail = {**detail, "finished": True}

    res = http.get(f"{GRAPH}/{remote_id}", params={"fields": "status"}, headers=_auth(target["token"]))
    _raise_for(res, "Could not read the reel's status")
    status = res.json().get("status") or {}
    if status.get("video_status") in ("error", "upload_failed", "expired"):
        return RemoteState(
            "failed", remote_id, url, error=f"Facebook: {status['video_status']}.", detail=detail
        )
    published = (status.get("publishing_phase") or {}).get("status") == "complete"
    return RemoteState("published" if published else "processing", remote_id, url, detail=detail)


def upload_instagram(http: httpx.Client, channel_id: str, request: UploadRequest) -> RemoteState:
    target = _target(channel_id)
    res = http.post(
        f"{GRAPH}/{target['id']}/media",
        data={
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": request.description[:2200],
            "access_token": target["token"],
        },
    )
    _raise_for(res, "Instagram refused the upload")
    container = res.json()["id"]
    with request.path.open("rb") as handle:
        res = http.post(
            f"{RUPLOAD}/ig-api-upload/{GRAPH_VERSION}/{container}",
            headers={**_auth(target["token"]), "offset": "0", "file_size": str(request.size_bytes)},
            content=handle.read(),
        )
    _raise_for(res, "Instagram failed during upload")
    return RemoteState("processing", remote_id=container, detail={"container": container})


def check_instagram(http: httpx.Client, channel_id: str, remote_id: str, detail: dict) -> RemoteState:
    if detail.get("media_id"):
        return RemoteState("published", remote_id, detail.get("permalink"), detail=detail)
    target = _target(channel_id)
    res = http.get(
        f"{GRAPH}/{remote_id}", params={"fields": "status_code,status"}, headers=_auth(target["token"])
    )
    _raise_for(res, "Could not read Instagram's status")
    body = res.json()
    code = body.get("status_code")
    if code == "ERROR":
        return RemoteState(
            "failed",
            remote_id,
            error=f"Instagram: {body.get('status') or 'processing failed'}.",
            detail=detail,
        )
    if code == "EXPIRED":
        return RemoteState(
            "failed", remote_id, error="Instagram: the upload expired before publishing.", detail=detail
        )
    if code != "FINISHED":
        return RemoteState("processing", remote_id, detail=detail)
    res = http.post(
        f"{GRAPH}/{target['id']}/media_publish",
        data={"creation_id": remote_id, "access_token": target["token"]},
    )
    _raise_for(res, "Instagram refused to publish")
    media_id = res.json()["id"]
    res = http.get(f"{GRAPH}/{media_id}", params={"fields": "permalink"}, headers=_auth(target["token"]))
    permalink = res.json().get("permalink") if res.is_success else None
    return RemoteState(
        "published", remote_id, permalink, detail={**detail, "media_id": media_id, "permalink": permalink}
    )
