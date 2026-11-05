import json
import logging
import stat
import subprocess
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from clipforge import cleanup
from clipforge.api.app import create_app
from clipforge.config import get_settings
from clipforge.db import session_scope
from clipforge.media.ffmpeg import find_tool
from clipforge.media.probe import probe
from clipforge.models import Channel, Clip, Job, Source, Upload, UploadStatus, utcnow
from clipforge.uploads import base, meta, service, youtube
from clipforge.uploads.base import UploadError, UploadRequest, save_tokens

TOKEN = "SECRET-ACCESS-TOKEN"
PAGE_TOKEN = "SECRET-PAGE-TOKEN"


class Fake:
    """A scripted platform. Records every request so tests can inspect what was really sent."""

    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))

    def urls(self):
        return [str(r.url) for r in self.requests]


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def use_platform(monkeypatch, handler):
    fake = Fake(handler)
    monkeypatch.setattr(service, "http_client", fake.client)
    return fake


@pytest.fixture
def clip_file(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=108x192:r=30:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=f=300:d=4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def make_clip(
    clip_file,
    source_id=None,
    rights="own",
    detail="",
    duration=20.0,
    width=1080,
    height=1920,
    qc="ok",
    description="Words.",
):
    with session_scope() as s:
        if source_id is None:
            source = Source(filename="a.mp4", path="/x/a.mp4")
            s.add(source)
            s.commit()
            source_id = source.id
        job = Job(source_id=source_id, rights_type=rights, rights_detail=detail)
        s.add(job)
        s.commit()
        clip = Clip(
            job_id=job.id,
            start=0,
            end=duration,
            title="A title",
            description=description,
            path=str(clip_file),
            width=width,
            height=height,
            duration=duration,
            frames=int(duration * 30),
            qc={"status": qc},
            expires_at=utcnow() + timedelta(hours=24),
        )
        s.add(clip)
        s.commit()
        return clip.id, source_id


def make_channel(platform, name="Chan", group=None):
    with session_scope() as s:
        channel = Channel(name=name, platform=platform, connected=True, link_group=group)
        s.add(channel)
        s.commit()
        if platform == "youtube":
            save_tokens(channel.id, {"access_token": TOKEN, "refresh_token": "R", "expires_at": 9e12})
        else:
            save_tokens(
                channel.id, {"target": "T1", "targets": [{"id": "T1", "name": "Target", "token": PAGE_TOKEN}]}
            )
        return channel.id


def upload_row(upload_id):
    with session_scope() as s:
        return s.get(Upload, upload_id)


def youtube_ok(status=None):
    status = status or {"uploadStatus": "processed", "privacyStatus": "private"}

    def handler(request):
        if request.url.path.startswith("/upload/youtube") and request.method == "POST":
            return httpx.Response(200, headers={"Location": "https://www.googleapis.com/upload/session/1"})
        if request.url.path == "/upload/session/1":
            return httpx.Response(200, json={"id": "VID123"})
        if request.url.path == "/youtube/v3/videos":
            return httpx.Response(200, json={"items": [{"status": status}]})
        return httpx.Response(404, json={"error": {"message": "unexpected " + str(request.url)}})

    return handler


# ---------- secrets ----------
def test_secret_files_are_private_and_never_in_api_or_repo(client):
    res = client.put(
        "/api/platforms/google/credentials", json={"client_id": "id", "client_secret": "TOP-SECRET"}
    )
    assert res.status_code == 200 and "TOP-SECRET" not in res.text
    listed = client.get("/api/platforms").text
    assert "TOP-SECRET" not in listed and '"configured":true' in listed
    path = get_settings().secrets_dir / "app-google.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_drive_that_ignores_chmod_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(base.stat, "S_IMODE", lambda _mode: 0o777)
    with pytest.raises(UploadError, match="cannot protect secret files"):
        base._write_secret(tmp_path / "x.json", {"a": 1})
    assert not (tmp_path / "x.json").exists()


def test_tokens_never_appear_in_any_url_or_channel_json(client, clip_file, monkeypatch):
    fake = use_platform(monkeypatch, youtube_ok())
    clip_id, _ = make_clip(clip_file)
    channel_id = make_channel("youtube")
    upload_id = service.queue_upload(clip_id, [channel_id], "private", False)[0].id
    service.run_upload(upload_id)
    assert upload_row(upload_id).remote_id == "VID123"
    assert all(TOKEN not in url for url in fake.urls())
    assert TOKEN not in client.get("/api/channels").text
    assert TOKEN not in client.get("/api/uploads").text
    assert TOKEN not in client.get(f"/api/channels/{channel_id}/targets").text


def test_meta_tokens_never_in_urls_and_targets_hide_them(client, clip_file, monkeypatch):
    fake = use_platform(monkeypatch, _facebook_handler(matches=False))
    clip_id, _ = make_clip(clip_file)
    channel_id = make_channel("facebook")
    upload_id = service.queue_upload(clip_id, [channel_id], "private", False)[0].id
    service.run_upload(upload_id)
    service.refresh_upload(upload_id)
    assert all(PAGE_TOKEN not in url for url in fake.urls())
    assert PAGE_TOKEN not in client.get(f"/api/channels/{channel_id}/targets").text


def test_httpx_url_logging_is_silenced():
    from clipforge.api.app import setup_logging

    setup_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


# ---------- oauth ----------
def test_connect_needs_app_credentials_first(client):
    channel = client.post("/api/channels", json={"name": "Y", "platform": "youtube"}).json()
    res = client.post(f"/api/channels/{channel['id']}/connect")
    assert res.status_code == 409
    assert "Channels page" in res.json()["detail"]


def test_youtube_oauth_flow_with_state_and_pkce(client, monkeypatch):
    client.put("/api/platforms/google/credentials", json={"client_id": "cid", "client_secret": "sec"})
    channel = client.post("/api/channels", json={"name": "Y", "platform": "youtube"}).json()
    url = client.post(f"/api/channels/{channel['id']}/connect").json()["auth_url"]
    query = parse_qs(urlparse(url).query)
    assert query["client_id"] == ["cid"]
    assert query["code_challenge_method"] == ["S256"]
    assert "sec" not in url
    assert "youtube.upload" in query["scope"][0]

    def handler(request):
        if request.url.host == "oauth2.googleapis.com":
            assert b"code_verifier=" in request.content
            return httpx.Response(200, json={"access_token": "AT", "refresh_token": "RT", "expires_in": 3600})
        return httpx.Response(200, json={"items": [{"snippet": {"title": "My Channel"}}]})

    use_platform(monkeypatch, handler)
    assert client.get("/api/oauth/callback", params={"state": "forged", "code": "c"}).status_code == 400
    res = client.get("/api/oauth/callback", params={"state": query["state"][0], "code": "c"})
    assert res.status_code == 200 and "My Channel" in res.text
    row = client.get("/api/channels").json()[0]
    assert row["connected"] is True and row["account_name"] == "My Channel"
    # A state can only be used once.
    assert (
        client.get("/api/oauth/callback", params={"state": query["state"][0], "code": "c"}).status_code == 400
    )
    assert client.post(f"/api/channels/{channel['id']}/disconnect").json()["connected"] is False
    assert base.load_tokens(channel["id"]) is None


def test_oauth_error_page_escapes_platform_text(client):
    client.put("/api/platforms/google/credentials", json={"client_id": "c", "client_secret": "s"})
    channel = client.post("/api/channels", json={"name": "Y", "platform": "youtube"}).json()
    url = client.post(f"/api/channels/{channel['id']}/connect").json()["auth_url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    res = client.get("/api/oauth/callback", params={"state": state, "error": "<script>x</script>"})
    assert res.status_code == 400 and "<script>x" not in res.text


def test_meta_connect_lists_targets_and_lets_user_choose(client, monkeypatch):
    client.put("/api/platforms/meta/credentials", json={"client_id": "cid", "client_secret": "sec"})
    channel = client.post("/api/channels", json={"name": "I", "platform": "instagram"}).json()
    url = client.post(f"/api/channels/{channel['id']}/connect").json()["auth_url"]
    query = parse_qs(urlparse(url).query)
    assert "instagram_content_publish" in query["scope"][0]

    def handler(request):
        if request.url.path.endswith("/oauth/access_token"):
            return httpx.Response(200, json={"access_token": "U"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "P0", "name": "No IG", "access_token": "X"},
                    {
                        "id": "P1",
                        "name": "A",
                        "access_token": "X",
                        "instagram_business_account": {"id": "IG1", "username": "one"},
                    },
                    {
                        "id": "P2",
                        "name": "B",
                        "access_token": "X",
                        "instagram_business_account": {"id": "IG2", "username": "two"},
                    },
                ]
            },
        )

    use_platform(monkeypatch, handler)
    assert (
        client.get("/api/oauth/callback", params={"state": query["state"][0], "code": "c"}).status_code == 200
    )
    targets = client.get(f"/api/channels/{channel['id']}/targets").json()
    assert targets["selected"] == "IG1"
    assert {t["id"] for t in targets["targets"]} == {"IG1", "IG2"}
    assert (
        client.post(f"/api/channels/{channel['id']}/target", json={"target_id": "IG2"}).json()["account_name"]
        == "two"
    )
    assert client.post(f"/api/channels/{channel['id']}/target", json={"target_id": "nope"}).status_code == 404


# ---------- youtube ----------
def test_youtube_upload_sends_metadata_and_defaults_private(clip_file, monkeypatch):
    fake = use_platform(monkeypatch, youtube_ok())
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0].id
    service.run_upload(upload_id)
    start = fake.requests[0]
    body = json.loads(start.content)
    assert start.url.params["uploadType"] == "resumable"
    assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
    assert body["snippet"]["title"] == "A title"
    assert upload_row(upload_id).status == UploadStatus.processing.value
    assert service.refresh_upload(upload_id).status == UploadStatus.published.value


def test_youtube_resumes_after_308_from_the_reported_byte(tmp_path, monkeypatch):
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 100)
    monkeypatch.setattr(youtube, "CHUNK_BYTES", 40)
    ranges = []

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, headers={"Location": "https://www.googleapis.com/upload/session/1"})
        ranges.append(request.headers["Content-Range"])
        if len(ranges) == 1:
            return httpx.Response(308, headers={"Range": "bytes=0-29"})  # Only 30 bytes were stored.
        if len(ranges) == 2:
            return httpx.Response(308, headers={"Range": "bytes=0-69"})
        return httpx.Response(200, json={"id": "V"})

    fake = Fake(handler)
    channel_id = make_channel("youtube")
    req = UploadRequest(path, "t", "d", "private", False, 1080, 1920, 20, 30, 100)
    assert youtube.upload(fake.client(), channel_id, req).remote_id == "V"
    assert ranges == ["bytes 0-39/100", "bytes 30-69/100", "bytes 70-99/100"]


def test_youtube_refreshes_an_expired_token_and_saves_it(monkeypatch):
    channel_id = make_channel("youtube")
    save_tokens(channel_id, {"access_token": "OLD", "refresh_token": "R", "expires_at": 0})
    base.save_platform_credentials("google", "c", "s")

    def handler(request):
        return httpx.Response(200, json={"access_token": "NEW", "expires_in": 3600})

    assert youtube._access_token(Fake(handler).client(), channel_id) == "NEW"
    assert base.load_tokens(channel_id)["access_token"] == "NEW"


def test_youtube_quota_error_is_plain_and_not_retried(clip_file, monkeypatch):
    def handler(request):
        return httpx.Response(403, json={"error": {"message": "x", "errors": [{"reason": "quotaExceeded"}]}})

    fake = use_platform(monkeypatch, handler)
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0].id
    service.run_upload(upload_id)
    row = upload_row(upload_id)
    assert row.status == UploadStatus.failed.value
    assert "daily upload limit" in row.error
    assert len(fake.requests) == 1


@pytest.mark.parametrize("reason", ["copyright", "claim", "trademark"])
def test_youtube_copyright_rejection_becomes_a_claim(clip_file, monkeypatch, reason):
    use_platform(monkeypatch, youtube_ok({"uploadStatus": "rejected", "rejectionReason": reason}))
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0].id
    service.run_upload(upload_id)
    row = service.refresh_upload(upload_id)
    assert row.status == UploadStatus.claimed.value
    assert row.claim["reason"] == reason


def test_youtube_other_rejection_is_a_failure_not_a_claim(clip_file, monkeypatch):
    use_platform(monkeypatch, youtube_ok({"uploadStatus": "rejected", "rejectionReason": "duplicate"}))
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0].id
    service.run_upload(upload_id)
    row = service.refresh_upload(upload_id)
    assert row.status == UploadStatus.failed.value and "duplicate" in row.error


# ---------- claim rules ----------
def _claimed_upload(clip_file, monkeypatch, **clip_facts):
    use_platform(monkeypatch, youtube_ok({"uploadStatus": "rejected", "rejectionReason": "copyright"}))
    clip_id, source_id = make_clip(clip_file, **clip_facts)
    channel_id = make_channel("youtube")
    upload_id = service.queue_upload(clip_id, [channel_id], "private", False)[0].id
    service.run_upload(upload_id)
    service.refresh_upload(upload_id)
    return clip_id, source_id, channel_id, upload_id


def test_claim_stops_waiting_uploads_from_the_same_source_only(clip_file, monkeypatch):
    use_platform(monkeypatch, youtube_ok({"uploadStatus": "rejected", "rejectionReason": "copyright"}))
    clip_id, source_id = make_clip(clip_file)
    sibling, _ = make_clip(clip_file, source_id)
    unrelated, _ = make_clip(clip_file)
    channel_id = make_channel("youtube")
    first = service.queue_upload(clip_id, [channel_id], "private", False)[0].id
    waiting = service.queue_upload(sibling, [channel_id], "private", False)[0].id
    other = service.queue_upload(unrelated, [channel_id], "private", False)[0].id
    service.run_upload(first)
    service.refresh_upload(first)
    assert upload_row(waiting).status == UploadStatus.stopped.value
    assert upload_row(other).status == UploadStatus.pending.value


def test_claim_blocks_new_uploads_until_the_user_decides(clip_file, monkeypatch):
    clip_id, source_id, channel_id, upload_id = _claimed_upload(clip_file, monkeypatch)
    sibling, _ = make_clip(clip_file, source_id)
    with pytest.raises(service.UploadBlocked, match="stopped by a copyright claim"):
        service.queue_upload(sibling, [channel_id], "private", False)
    service.resolve_claim(upload_id, "trim", "")
    assert service.queue_upload(sibling, [channel_id], "private", False)


def test_the_claimed_clip_itself_can_never_be_sent_again(clip_file, monkeypatch):
    clip_id, _, channel_id, upload_id = _claimed_upload(clip_file, monkeypatch)
    service.resolve_claim(upload_id, "trim", "")
    with pytest.raises(service.UploadBlocked, match="will not be uploaded again"):
        service.queue_upload(clip_id, [channel_id], "private", False)


def test_nothing_re_uploads_by_itself_after_a_claim(clip_file, monkeypatch):
    """The worker runs repeatedly after a claim. It must send nothing and re-edit nothing."""
    _, _, _, upload_id = _claimed_upload(clip_file, monkeypatch)
    service.resolve_claim(upload_id, "replace_audio", "")
    fake = use_platform(monkeypatch, youtube_ok())
    worker = service.UploadWorker(status_seconds=0)
    for _ in range(5):
        worker.step()
    assert fake.requests == []
    assert upload_row(upload_id).status == UploadStatus.claimed.value


def test_dispute_needs_a_licence_note(clip_file, monkeypatch):
    _, _, _, upload_id = _claimed_upload(clip_file, monkeypatch)
    with pytest.raises(service.UploadBlocked, match="licence"):
        service.resolve_claim(upload_id, "dispute", "  ")
    assert service.resolve_claim(upload_id, "dispute", "Licence 42").claim_note == "Licence 42"


def test_manual_claim_report_uses_the_same_stop(client, clip_file, monkeypatch):
    use_platform(monkeypatch, youtube_ok())
    clip_id, source_id = make_clip(clip_file)
    channel_id = make_channel("youtube")
    upload_id = service.queue_upload(clip_id, [channel_id], "private", False)[0].id
    service.run_upload(upload_id)
    res = client.post(
        f"/api/uploads/{upload_id}/report-claim", json={"details": "Studio shows a music claim"}
    )
    assert res.status_code == 200 and res.json()["status"] == "claimed"
    sibling, _ = make_clip(clip_file, source_id)
    res = client.post(f"/api/clips/{sibling}/upload", json={"channel_id": channel_id})
    assert res.status_code == 409 and "copyright claim" in res.json()["detail"]


def test_replace_audio_only_after_a_claim_and_with_a_licence(client, clip_file, monkeypatch, tmp_path):
    clean, _ = make_clip(clip_file)
    wav = tmp_path / "t.wav"
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=f=500:d=2",
            str(wav),
        ],
        check=True,
    )
    data = wav.read_bytes()
    params = {"filename": "t.wav", "licence": "Bought from Example Music"}
    assert client.put(f"/api/clips/{clean}/replace-audio", params=params, content=data).status_code == 409

    # The real facts of this file: the new clip goes through real quality checks.
    clip_id, source_id, channel_id, upload_id = _claimed_upload(
        clip_file, monkeypatch, width=108, height=192, duration=4.0
    )
    blank = {"filename": "t.wav", "licence": " "}
    assert client.put(f"/api/clips/{clip_id}/replace-audio", params=blank, content=data).status_code == 422
    assert (
        client.put(
            f"/api/clips/{clip_id}/replace-audio", params={**params, "filename": "t.exe"}, content=data
        ).status_code
        == 415
    )
    res = client.put(f"/api/clips/{clip_id}/replace-audio", params=params, content=data)
    assert res.status_code == 201, res.text
    fresh = res.json()
    assert fresh["id"] != clip_id
    assert "Bought from Example Music" in fresh["description"]
    with session_scope() as s:
        path = s.get(Clip, fresh["id"]).path
    info = probe(path)
    assert info.has_audio
    assert abs(info.duration - probe(clip_file).duration) < 0.2
    assert fresh["qc"]["checks"]
    assert fresh["qc"]["encoder"] == "video copied"
    assert fresh["qc"]["status"] != "fail", fresh["qc"]
    assert fresh["thumb_url"] is None  # The test clip has no thumbnail to copy.
    with pytest.raises(service.UploadBlocked, match="stopped by a copyright claim"):
        service.queue_upload(fresh["id"], [channel_id], "private", False)
    service.resolve_claim(upload_id, "replace_audio", "")
    assert service.queue_upload(fresh["id"], [channel_id], "private", False)


# ---------- rules before sending ----------
def test_upload_refused_without_rights_or_with_failed_qc_or_expired(clip_file):
    channel_id = make_channel("youtube")
    no_rights, _ = make_clip(clip_file, rights="")
    with pytest.raises(service.UploadBlocked, match="rights"):
        service.queue_upload(no_rights, [channel_id], "private", False)
    bad_qc, _ = make_clip(clip_file, qc="fail")
    with pytest.raises(service.UploadBlocked, match="quality"):
        service.queue_upload(bad_qc, [channel_id], "private", False)
    gone, _ = make_clip(clip_file)
    with session_scope() as s:
        row = s.get(Clip, gone)
        row.path = None
        s.add(row)
        s.commit()
    with pytest.raises(service.UploadBlocked, match="expired"):
        service.queue_upload(gone, [channel_id], "private", False)


def test_disconnected_channel_and_duplicates_are_refused(clip_file):
    clip_id, _ = make_clip(clip_file)
    channel_id = make_channel("youtube")
    with session_scope() as s:
        row = s.get(Channel, channel_id)
        row.connected = False
        s.add(row)
        s.commit()
    with pytest.raises(service.UploadBlocked, match="not connected"):
        service.queue_upload(clip_id, [channel_id], "private", False)
    other = make_channel("youtube", "Second")
    service.queue_upload(clip_id, [other], "private", False)
    with pytest.raises(service.UploadBlocked, match="already on"):
        service.queue_upload(clip_id, [other], "private", False)


def test_required_credit_survives_a_user_editing_the_description(clip_file):
    clip_id, _ = make_clip(
        clip_file, rights="creative_commons", detail="By A, CC BY", description="I removed it"
    )
    row = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0]
    assert "By A, CC BY" in row.description


def test_upload_to_a_linked_group_makes_one_upload_per_channel(client, clip_file):
    make_channel("youtube", "A", group="g1")
    make_channel("instagram", "B", group="g1")
    clip_id, _ = make_clip(clip_file)
    res = client.post(f"/api/clips/{clip_id}/upload", json={"link_group": "g1"})
    assert res.status_code == 201 and len(res.json()) == 2
    assert client.post(f"/api/clips/{clip_id}/upload", json={}).status_code == 422
    assert client.post(f"/api/clips/{clip_id}/upload", json={"link_group": "none"}).status_code == 404


def test_only_youtube_gets_a_privacy_choice(clip_file):
    clip_id, _ = make_clip(clip_file)
    assert service.queue_upload(clip_id, [make_channel("instagram")], "private", False)[0].privacy == "public"


def test_clip_is_not_deleted_while_uploading_even_when_expired(clip_file):
    clip_id, _ = make_clip(clip_file)
    channel_id = make_channel("youtube")
    upload = service.queue_upload(clip_id, [channel_id], "private", False)[0]
    with session_scope() as s:
        row = s.get(Upload, upload.id)
        row.status = UploadStatus.uploading.value
        s.add(row)
        clip = s.get(Clip, clip_id)
        clip.expires_at = utcnow() - timedelta(seconds=1)
        s.add(clip)
        s.commit()
    assert cleanup.cleanup_expired_clips() == []
    assert clip_file.exists()


# ---------- facebook ----------
def _facebook_handler(matches, complete=True, match_list=None):
    def handler(request):
        path = request.url.path
        if path.endswith("/video_reels"):
            if request.headers.get("content-type", "").startswith("application/json"):
                return httpx.Response(200, json={"video_id": "V1", "upload_url": "x"})
            return httpx.Response(200, json={"success": True})
        if request.url.host == "rupload.facebook.com":
            assert request.headers["offset"] == "0"
            return httpx.Response(200, json={"success": True})
        fields = request.url.params.get("fields")
        if fields == "copyright_check_information":
            status = {"status": "complete" if complete else "in_progress"}
            if complete:
                status["matches_found"] = matches
            return httpx.Response(
                200,
                json={
                    "copyright_check_information": {"status": status, "copyright_matches": match_list or []}
                },
            )
        return httpx.Response(
            200, json={"status": {"video_status": "ready", "publishing_phase": {"status": "complete"}}}
        )

    return handler


def _finish_calls(fake):
    return [r for r in fake.requests if b"upload_phase=finish" in r.content]


def test_facebook_clean_check_publishes(clip_file, monkeypatch):
    fake = use_platform(monkeypatch, _facebook_handler(matches=False))
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("facebook")], "public", False)[0].id
    service.run_upload(upload_id)
    assert _finish_calls(fake) == []  # Not published before the copyright check.
    assert service.refresh_upload(upload_id).status == UploadStatus.published.value
    assert len(_finish_calls(fake)) == 1


def test_facebook_waits_while_the_check_is_running(clip_file, monkeypatch):
    fake = use_platform(monkeypatch, _facebook_handler(matches=False, complete=False))
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("facebook")], "public", False)[0].id
    service.run_upload(upload_id)
    assert service.refresh_upload(upload_id).status == UploadStatus.processing.value
    assert _finish_calls(fake) == []


def test_facebook_match_stops_before_publishing_and_keeps_details(clip_file, monkeypatch):
    found = [
        {
            "content_title": "Song",
            "owner_copyright_policy": {"name": "Label", "actions": [{"action": "BLOCK"}]},
            "matched_segments": [
                {"start_time_in_seconds": 3, "duration_in_seconds": 4, "segment_type": "AUDIO"}
            ],
        }
    ]
    fake = use_platform(monkeypatch, _facebook_handler(matches=True, match_list=found))
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("facebook")], "public", False)[0].id
    service.run_upload(upload_id)
    row = service.refresh_upload(upload_id)
    assert row.status == UploadStatus.claimed.value
    assert _finish_calls(fake) == []
    match = row.claim["matches"][0]
    assert match["title"] == "Song" and match["actions"] == ["BLOCK"]
    assert match["segments"][0] == {"start": 3, "duration": 4, "type": "AUDIO"}


@pytest.mark.parametrize(
    "kw,message",
    [
        (dict(duration=2.0), "3 to 90"),
        (dict(duration=91.0), "3 to 90"),
        (dict(width=300, height=500), "540"),
        (dict(width=1920, height=1080), "vertical"),
    ],
)
def test_facebook_limits_are_checked_before_sending(clip_file, monkeypatch, kw, message):
    fake = use_platform(monkeypatch, _facebook_handler(matches=False))
    clip_id, _ = make_clip(clip_file, **kw)
    upload_id = service.queue_upload(clip_id, [make_channel("facebook")], "public", False)[0].id
    service.run_upload(upload_id)
    row = upload_row(upload_id)
    assert row.status == UploadStatus.failed.value and message in row.error
    assert fake.requests == []


# ---------- instagram ----------
def _instagram_handler(code="FINISHED"):
    def handler(request):
        path = request.url.path
        if path.endswith("/media"):
            assert b"upload_type=resumable" in request.content
            assert b"media_type=REELS" in request.content
            return httpx.Response(200, json={"id": "C1"})
        if request.url.host == "rupload.facebook.com":
            return httpx.Response(200, json={"success": True})
        if path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "M1"})
        if request.url.params.get("fields") == "permalink":
            return httpx.Response(200, json={"permalink": "https://www.instagram.com/reel/x/"})
        return httpx.Response(200, json={"status_code": code, "status": "detail text"})

    return handler


def test_instagram_flow_publishes_once_when_finished(clip_file, monkeypatch):
    fake = use_platform(monkeypatch, _instagram_handler())
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("instagram")], "public", False)[0].id
    service.run_upload(upload_id)
    row = service.refresh_upload(upload_id)
    assert row.status == UploadStatus.published.value
    assert row.url == "https://www.instagram.com/reel/x/"
    assert sum(r.url.path.endswith("/media_publish") for r in fake.requests) == 1


@pytest.mark.parametrize(
    "code,state", [("IN_PROGRESS", "processing"), ("ERROR", "failed"), ("EXPIRED", "failed")]
)
def test_instagram_other_states(clip_file, monkeypatch, code, state):
    fake = use_platform(monkeypatch, _instagram_handler(code))
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("instagram")], "public", False)[0].id
    service.run_upload(upload_id)
    assert service.refresh_upload(upload_id).status == state
    assert not any(r.url.path.endswith("/media_publish") for r in fake.requests)


def test_network_failure_marks_failed_and_does_not_crash(clip_file, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no route")

    use_platform(monkeypatch, handler)
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("instagram")], "public", False)[0].id
    service.run_upload(upload_id)
    assert upload_row(upload_id).status == UploadStatus.failed.value


def test_upload_interrupted_by_a_crash_is_marked_failed_and_stops_blocking_cleanup(clip_file):
    clip_id, _ = make_clip(clip_file)
    upload = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0]
    with session_scope() as s:
        row = s.get(Upload, upload.id)
        row.status = UploadStatus.uploading.value
        s.add(row)
        clip = s.get(Clip, clip_id)
        clip.expires_at = utcnow() - timedelta(seconds=1)
        s.add(clip)
        s.commit()
    assert cleanup.cleanup_expired_clips() == []
    assert service.recover_interrupted_uploads() == 1
    row = upload_row(upload.id)
    assert row.status == UploadStatus.failed.value and "closed during" in row.error
    assert cleanup.cleanup_expired_clips() == [clip_id]


def test_clip_cannot_be_deleted_mid_upload(client, clip_file):
    clip_id, _ = make_clip(clip_file)
    upload = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0]
    with session_scope() as s:
        row = s.get(Upload, upload.id)
        row.status = UploadStatus.uploading.value
        s.add(row)
        s.commit()
    assert client.delete(f"/api/clips/{clip_id}").status_code == 409
    assert clip_file.exists()


def test_meta_upload_streams_the_file(clip_file, monkeypatch):
    seen = []

    def handler(request):
        if request.url.host == "rupload.facebook.com":
            seen.append(len(request.read()))
            return httpx.Response(200, json={"success": True})
        return _instagram_handler()(request)

    use_platform(monkeypatch, handler)
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("instagram")], "public", False)[0].id
    service.run_upload(upload_id)
    assert upload_row(upload_id).status == UploadStatus.processing.value
    assert seen == [clip_file.stat().st_size]


def test_youtube_strips_characters_it_refuses(clip_file, monkeypatch):
    fake = use_platform(monkeypatch, youtube_ok())
    clip_id, _ = make_clip(clip_file, description="a <b> c")
    with session_scope() as s:
        row = s.get(Clip, clip_id)
        row.title = "Title <x>"
        s.add(row)
        s.commit()
    service.run_upload(service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0].id)
    snippet = json.loads(fake.requests[0].content)["snippet"]
    assert snippet == {"title": "Title x", "description": "a b c"}


def test_refresh_network_failure_is_a_clear_message(client, clip_file, monkeypatch):
    use_platform(monkeypatch, youtube_ok())
    clip_id, _ = make_clip(clip_file)
    upload_id = service.queue_upload(clip_id, [make_channel("youtube")], "private", False)[0].id
    service.run_upload(upload_id)

    def handler(request):
        raise httpx.ConnectError("no route")

    use_platform(monkeypatch, handler)
    res = client.post(f"/api/uploads/{upload_id}/refresh")
    assert res.status_code == 409 and "Could not reach youtube" in res.json()["detail"]


def test_upload_error_never_contains_a_token():
    assert TOKEN not in str(UploadError("YouTube refused"))
    assert meta.GRAPH_VERSION.startswith("v")
