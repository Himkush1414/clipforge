"""YouTube Data API v3: OAuth, resumable upload and status.

Checked against Google's docs on 2026-11-05: videos.insert has its own bucket of 100 calls a day.
Claims are only visible when status.rejectionReason says so; a Content ID claim that leaves the
video live is not exposed to non-partners, so that case relies on the manual "report a claim".
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import urlencode

import httpx

from clipforge.uploads.base import (
    CHUNK_BYTES,
    NotConfigured,
    RemoteState,
    UploadError,
    UploadRequest,
    load_tokens,
    platform_credentials,
    redirect_uri,
    save_tokens,
)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
API_URL = "https://www.googleapis.com/youtube/v3"
SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
CLAIM_REASONS = {"copyright", "claim", "trademark"}
MAX_CHUNK_RETRIES = 3


def _credentials() -> dict:
    creds = platform_credentials("google")
    if not creds:
        raise NotConfigured("Add your Google OAuth client ID and secret on the Channels page first.")
    return creds


def begin_auth(state: str) -> tuple[str, str]:
    """Return (consent URL, PKCE verifier)."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    query = urlencode(
        {
            "client_id": _credentials()["client_id"],
            "redirect_uri": redirect_uri(),
            "response_type": "code",
            "scope": SCOPES,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{AUTH_URL}?{query}", verifier


def finish_auth(http: httpx.Client, channel_id: str, code: str, verifier: str) -> str:
    """Exchange the code for tokens, save them, and return the channel's name."""
    creds = _credentials()
    res = http.post(
        TOKEN_URL,
        data={
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri(),
        },
    )
    _raise_for(res, "Google sign-in failed")
    body = res.json()
    tokens = {
        "access_token": body["access_token"],
        "refresh_token": body.get("refresh_token"),
        "expires_at": time.time() + body.get("expires_in", 3600),
    }
    save_tokens(channel_id, tokens)
    me = http.get(
        f"{API_URL}/channels",
        params={"part": "snippet", "mine": "true"},
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    _raise_for(me, "Could not read your YouTube channel")
    items = me.json().get("items") or []
    if not items:
        raise UploadError("This Google account has no YouTube channel.")
    return items[0]["snippet"]["title"]


def _access_token(http: httpx.Client, channel_id: str) -> str:
    tokens = load_tokens(channel_id)
    if not tokens:
        raise UploadError("This channel is not connected.")
    if tokens["expires_at"] - time.time() > 60:
        return tokens["access_token"]
    if not tokens.get("refresh_token"):
        raise UploadError("The YouTube sign-in has expired. Connect the channel again.")
    creds = _credentials()
    res = http.post(
        TOKEN_URL,
        data={
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": tokens["refresh_token"],
            "grant_type": "refresh_token",
        },
    )
    _raise_for(res, "The YouTube sign-in has expired. Connect the channel again")
    body = res.json()
    tokens.update(access_token=body["access_token"], expires_at=time.time() + body.get("expires_in", 3600))
    save_tokens(channel_id, tokens)
    return tokens["access_token"]


def _raise_for(res: httpx.Response, prefix: str) -> None:
    if res.is_success:
        return
    reason, message = "", ""
    try:
        error = res.json().get("error")
        if isinstance(error, dict):
            message = error.get("message", "")
            reason = (error.get("errors") or [{}])[0].get("reason", "")
        elif isinstance(error, str):
            message = error
    except (ValueError, AttributeError):
        message = res.text[:200]
    if reason in ("quotaExceeded", "uploadLimitExceeded", "rateLimitExceeded"):
        raise UploadError(
            "YouTube's daily upload limit was reached for this API project "
            "(100 uploads a day by default). Try again after it resets, or request more quota."
        )
    raise UploadError(f"{prefix} ({res.status_code}): {message or reason or 'no details given'}")


def upload(http: httpx.Client, channel_id: str, request: UploadRequest) -> RemoteState:
    token = _access_token(http, channel_id)
    headers = {"Authorization": f"Bearer {token}"}
    metadata = {
        "snippet": {
            # YouTube rejects angle brackets in both fields.
            "title": request.title.replace("<", "").replace(">", "")[:100],
            "description": request.description.replace("<", "").replace(">", "")[:5000],
        },
        "status": {"privacyStatus": request.privacy, "selfDeclaredMadeForKids": request.made_for_kids},
    }
    start = http.post(
        UPLOAD_URL,
        params={"uploadType": "resumable", "part": "snippet,status"},
        headers={
            **headers,
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(request.size_bytes),
            "X-Upload-Content-Type": "video/mp4",
        },
        content=json.dumps(metadata),
    )
    _raise_for(start, "YouTube refused the upload")
    session_url = start.headers.get("Location")
    if not session_url:
        raise UploadError("YouTube did not return an upload address.")

    body = None
    position = 0
    retries = 0
    with request.path.open("rb") as handle:
        while position < request.size_bytes:
            handle.seek(position)
            chunk = handle.read(CHUNK_BYTES)
            end = position + len(chunk) - 1
            try:
                res = http.put(
                    session_url,
                    headers={**headers, "Content-Range": f"bytes {position}-{end}/{request.size_bytes}"},
                    content=chunk,
                )
            except httpx.TransportError as exc:
                retries += 1
                if retries > MAX_CHUNK_RETRIES:
                    raise UploadError(f"The connection kept failing during upload: {exc}") from exc
                time.sleep(2**retries)
                continue
            if res.status_code == 308:
                sent = res.headers.get("Range")
                position = int(sent.rsplit("-", 1)[1]) + 1 if sent else position
                retries = 0
            elif res.status_code >= 500:
                retries += 1
                if retries > MAX_CHUNK_RETRIES:
                    _raise_for(res, "YouTube failed during upload")
                time.sleep(2**retries)
            else:
                _raise_for(res, "YouTube refused the upload")
                body = res.json()
                break
    if not body:
        raise UploadError("YouTube finished the upload without confirming it.")
    video_id = body["id"]
    return RemoteState("processing", remote_id=video_id, url=f"https://youtu.be/{video_id}")


def check(http: httpx.Client, channel_id: str, remote_id: str) -> RemoteState:
    token = _access_token(http, channel_id)
    res = http.get(
        f"{API_URL}/videos",
        params={"part": "status,processingDetails", "id": remote_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    _raise_for(res, "Could not read the video's status")
    items = res.json().get("items") or []
    url = f"https://youtu.be/{remote_id}"
    if not items:
        return RemoteState("failed", remote_id, url, error="YouTube no longer lists this video.")
    status = items[0].get("status", {})
    upload_status = status.get("uploadStatus")
    if upload_status == "rejected":
        reason = status.get("rejectionReason", "unknown")
        if reason in CLAIM_REASONS:
            claim = {
                "platform": "youtube",
                "reason": reason,
                "matches": [],
                "note": "YouTube's API does not give match details. See this video in YouTube Studio.",
            }
            return RemoteState("claimed", remote_id, url, claim=claim)
        return RemoteState("failed", remote_id, url, error=f"YouTube rejected this video: {reason}.")
    if upload_status == "failed":
        return RemoteState(
            "failed", remote_id, url, error=f"YouTube failed: {status.get('failureReason', 'unknown')}."
        )
    if upload_status == "processed":
        return RemoteState("published", remote_id, url)
    return RemoteState("processing", remote_id, url)
