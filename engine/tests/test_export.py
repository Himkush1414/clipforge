import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from clipforge.analysis.transcribe import Word
from clipforge.captions.presets import BUILTIN_PRESETS
from clipforge.export.plan import Segment, Timeline, output_size, to_frame_segments
from clipforge.export.reframe import CameraKey, crop_x_expression, plan_camera, sample_centers
from clipforge.export.render import RenderSpec, encoder_args, make_thumbnail, pick_encoder, render_clip
from clipforge.media.ffmpeg import find_tool
from clipforge.media.probe import probe
from clipforge.qc import run_qc


def count_frames(path):
    out = subprocess.run(
        [
            find_tool("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(json.loads(out.stdout)["streams"][0]["nb_read_frames"])


def grab(path, frame, tmp_path):
    out = tmp_path / f"g{frame}.png"
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )
    return np.asarray(Image.open(out).convert("RGB")).astype(int)


def spec_for(scene_video, tmp_path, segments, **kw):
    info = probe(scene_video)
    defaults = dict(
        source=scene_video,
        info=info,
        segments=segments,
        ratio="16:9",
        resolution="1080p",
        output=tmp_path / "out.mp4",
        fit="crop",
    )
    defaults.update(kw)
    return RenderSpec(**defaults)


@pytest.mark.parametrize(
    "ratio,resolution,size",
    [
        ("9:16", "1080p", (1080, 1920)),
        ("1:1", "2K", (1440, 1440)),
        ("16:9", "4K", (3840, 2160)),
        ("9:16", "8K", (4320, 7680)),
    ],
)
def test_output_sizes(ratio, resolution, size):
    assert output_size(ratio, resolution) == size


def test_frame_segments_merge_clamp_and_keep_one_frame():
    segs = to_frame_segments([(0, 1), (0.5, 2), (5, 5), (9.9, 20)], fps=30, max_frame=300)
    assert [(s.start_frame, s.end_frame) for s in segs] == [(0, 60), (150, 151), (297, 300)]


def test_timeline_remaps_words_across_a_removed_gap():
    tl = Timeline([Segment(0, 30), Segment(90, 120)], fps=30)
    assert tl.total_frames == 60
    words = [Word("a", 0.1, 0.5), Word("gone", 1.5, 2.0), Word("b", 3.1, 3.5)]
    mapped = tl.remap_words(words)
    assert [w.text for w in mapped] == ["a", "b"]
    assert mapped[1].start == pytest.approx(1.1)


def test_timeline_crossfade_shortens_output():
    tl = Timeline([Segment(0, 30), Segment(60, 90)], fps=30, transition_frames=6)
    assert tl.total_frames == 54
    assert tl.output_starts[1] == pytest.approx(24 / 30)


def test_three_frame_cut_is_exact(scene_video, tmp_path):
    """Frames 58..60 of the source: two red frames, then the first blue frame."""
    result = render_clip(spec_for(scene_video, tmp_path, [Segment(58, 61)]))
    assert count_frames(result.path) == 3
    first, last = grab(result.path, 0, tmp_path), grab(result.path, 2, tmp_path)
    assert first[:, :, 0].mean() > 200 and first[:, :, 2].mean() < 60
    assert last[:, :, 2].mean() > 200 and last[:, :, 0].mean() < 60


def test_single_frame_cut(scene_video, tmp_path):
    result = render_clip(spec_for(scene_video, tmp_path, [Segment(60, 61)]))
    assert count_frames(result.path) == 1
    assert grab(result.path, 0, tmp_path)[:, :, 2].mean() > 200


def test_seek_path_is_exact_far_into_the_file(scene_video, tmp_path):
    """Start past the seek margin, so the input -ss path is used, and check the frame after a cut."""
    result = render_clip(spec_for(scene_video, tmp_path, [Segment(119, 125)]))
    assert count_frames(result.path) == 6
    assert grab(result.path, 0, tmp_path)[:, :, 2].mean() > 200  # last blue frame
    assert grab(result.path, 1, tmp_path)[:, :, 2].mean() < 200  # first testsrc frame


def test_two_segments_join_with_exact_frames_and_audio_length(scene_video, tmp_path):
    result = render_clip(spec_for(scene_video, tmp_path, [Segment(0, 30), Segment(90, 120)]))
    info = probe(result.path)
    assert count_frames(result.path) == 60
    assert info.has_audio
    assert abs(info.duration - 2.0) < 0.1
    assert grab(result.path, 29, tmp_path)[:, :, 0].mean() > 200  # red
    assert grab(result.path, 30, tmp_path)[:, :, 2].mean() > 200  # blue


def test_crossfade_blends_and_shortens(scene_video, tmp_path):
    spec = spec_for(
        scene_video,
        tmp_path,
        [Segment(0, 30), Segment(90, 120)],
        transition="crossfade",
        transition_seconds=0.2,
    )
    result = render_clip(spec)
    assert count_frames(result.path) == 54
    middle = grab(result.path, 27, tmp_path)
    assert middle[:, :, 0].mean() > 50 and middle[:, :, 2].mean() > 50  # mixed red and blue


@pytest.mark.parametrize("fit", ["crop", "blur", "smart"])
@pytest.mark.parametrize("ratio", ["9:16", "1:1"])
def test_reframe_modes_produce_exact_size(scene_video, tmp_path, fit, ratio):
    result = render_clip(spec_for(scene_video, tmp_path, [Segment(0, 45)], fit=fit, ratio=ratio))
    info = probe(result.path)
    assert (info.width, info.height) == output_size(ratio, "1080p")
    assert count_frames(result.path) == 45


def test_captions_are_burned_in(scene_video, tmp_path):
    words = [Word("HELLO", 0.0, 1.0)]
    plain = render_clip(spec_for(scene_video, tmp_path, [Segment(0, 30)], output=tmp_path / "a.mp4"))
    burned = render_clip(
        spec_for(
            scene_video,
            tmp_path,
            [Segment(0, 30)],
            output=tmp_path / "b.mp4",
            words=words,
            preset=BUILTIN_PRESETS["anime-impact"],
        )
    )
    difference = np.abs(grab(plain.path, 15, tmp_path) - grab(burned.path, 15, tmp_path)).max(axis=2)
    assert (difference > 60).sum() > 1000


def test_loudness_normalized_near_target(scene_video, tmp_path):
    result = render_clip(spec_for(scene_video, tmp_path, [Segment(0, 180)]))
    qc = run_qc(result.path, result.width, result.height, result.duration, 30, True)
    checks = {c["name"]: c for c in qc["checks"]}
    assert checks["loudness"]["status"] == "ok", checks["loudness"]
    assert checks["true_peak"]["status"] == "ok", checks["true_peak"]


def test_loud_audio_ends_under_the_true_peak_ceiling_after_encoding(tmp_path):
    """Audio that forces the limiter. The encoded file must still measure under -1.0 dBFS."""
    from clipforge.media.ffmpeg import find_tool

    source = tmp_path / "loud.mp4"
    subprocess.run(
        [
            find_tool("ffmpeg"), "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=6",
            # A quiet tone with short full-scale bursts: reaching -14 LUFS needs more gain than the
            # bursts have headroom for, so the limiter must work, as on dialogue with sudden hits.
            "-f", "lavfi", "-i",
            "aevalsrc=0.02*sin(2*PI*440*t)+if(lt(mod(t\\,0.4)\\,0.02)\\,0.95*sin(2*PI*2500*t)\\,0):d=6:s=48000",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
        ],
        check=True,
    )  # fmt: skip
    result = render_clip(spec_for(source, tmp_path, [Segment(0, 150)]))
    out = subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-nostats", "-i", str(result.path), "-filter_complex",
         "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr  # fmt: skip
    peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", out.split("Summary:")[-1])[-1])
    assert peak <= -1.0, peak
    qc = run_qc(result.path, result.width, result.height, result.duration, 30, True)
    assert {c["name"]: c for c in qc["checks"]}["true_peak"]["status"] == "ok"


def test_temp_files_are_removed_after_render(scene_video, tmp_path):
    from clipforge.config import get_settings

    render_clip(
        spec_for(
            scene_video,
            tmp_path,
            [Segment(0, 10)],
            words=[Word("x", 0, 0.2)],
            preset=BUILTIN_PRESETS["anime-impact"],
        )
    )
    assert list(get_settings().tmp_dir.iterdir()) == []
    assert [p.name for p in tmp_path.glob(".*part*")] == []


def test_failed_render_leaves_no_partial_output(scene_video, tmp_path):
    from clipforge.config import get_settings
    from clipforge.media.ffmpeg import FFmpegError

    spec = spec_for(scene_video, tmp_path, [Segment(0, 10)])
    spec.source = tmp_path / "missing.mp4"
    with pytest.raises(FFmpegError):
        render_clip(spec)
    assert not spec.output.exists()
    assert list(get_settings().tmp_dir.iterdir()) == []


def test_encoder_args_never_pick_h264_above_its_limit():
    assert "hevc" in pick_encoder(7680, 4320) or "265" in pick_encoder(7680, 4320)
    assert "hvc1" in encoder_args("hevc_nvenc")
    assert "hvc1" not in encoder_args("libx264")


def test_qc_fails_on_wrong_size_and_duration(scene_video):
    qc = run_qc(scene_video, 1920, 1080, 10.0, 30, True)
    assert qc["status"] == "fail"
    bad = {c["name"] for c in qc["checks"] if c["status"] == "fail"}
    assert bad == {"resolution", "duration"}


def test_qc_warns_on_upscale_and_black(tmp_path):
    black = tmp_path / "black.mp4"
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
            "color=c=black:s=320x240:r=30:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(black),
        ],
        check=True,
    )
    qc = run_qc(black, 320, 240, 2.0, 30, False, source_height=100)
    states = {c["name"]: c["status"] for c in qc["checks"]}
    assert states["black_frames"] == "fail"
    assert states["upscale"] == "warn"
    assert states["audio"] == "warn"


def test_thumbnail(scene_video, tmp_path):
    out = make_thumbnail(scene_video, tmp_path / "t.jpg", 1.0)
    assert Image.open(out).size[0] == 480


def test_camera_holds_inside_dead_zone():
    samples = [(i / 2, 0.5 + (0.03 if i % 2 else -0.03)) for i in range(20)]
    assert len(plan_camera(samples, [])) == 1


def test_camera_follows_a_move_and_jumps_on_a_cut():
    samples = [(i / 2, 0.2) for i in range(10)] + [(5 + i / 2, 0.8) for i in range(10)]
    moved = plan_camera(samples, [])
    assert moved[0].center == pytest.approx(0.2) or moved[0].center == pytest.approx(0.5)
    assert moved[-1].center == pytest.approx(0.8)
    assert moved[-1].time - moved[-2].time == pytest.approx(0.5)
    cut = plan_camera(samples, [5.0])
    assert any(k.center == pytest.approx(0.8) and k.time == pytest.approx(5.0) for k in cut)


def test_crop_expression_is_clamped_to_the_frame():
    assert crop_x_expression([CameraKey(0, 0.0)], 1920, 608) == "0.0"
    assert crop_x_expression([CameraKey(0, 1.0)], 1920, 608) == "1312.0"
    moving = crop_x_expression([CameraKey(0, 0.2), CameraKey(1, 0.8)], 1920, 608)
    assert moving.startswith("if(lt(t,1.000)")


def test_subject_tracking_follows_a_moving_object(tmp_path):
    """A shaking white box sits left, then right: the sampled centers must follow it."""
    video = tmp_path / "box.mp4"
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
            "color=c=black:s=1280x720:r=30:d=6",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=120x120:r=30:d=6",
            "-filter_complex",
            "[0][1]overlay=x='if(lt(t,3),100+20*mod(n,7),1000+20*mod(n,7))':y=300",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    samples = [(t, c) for t, c in sample_centers(video, 0, 6) if c is not None]
    early = [c for t, c in samples if t < 2.5]
    late = [c for t, c in samples if t > 3.5]
    assert early and late
    assert max(early) < 0.3
    assert min(late) > 0.7


def _flash_and_beep(tmp_path):
    """10 s at 30 fps: black with a white flash from 5.0 s and a tone that starts at exactly 5.0 s."""
    video = tmp_path / "sync.mp4"
    graph = (
        "color=c=black:s=320x240:r=30:d=10,drawbox=c=white:t=fill:enable='gte(t,5)'[v];"
        "sine=f=1000:r=48000:d=10,volume='if(gte(t,5),1,0)':eval=frame[a]"
    )
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "15",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    return video


def _sync_offset(path, tmp_path):
    """Seconds between the first white frame and the first loud audio in a rendered file."""
    wav = tmp_path / "o.wav"
    subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "8000",
            str(wav),
        ],
        check=True,
    )
    rate, samples = wavfile.read(wav)
    window = rate // 100
    rms = np.array([np.abs(samples[i : i + window]).mean() for i in range(0, len(samples) - window, window)])
    audio_at = np.argmax(rms > rms.max() * 0.3) * window / rate
    out = subprocess.run(
        [
            find_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=8:8,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(out, np.uint8).reshape(-1, 64).mean(axis=1)
    return (np.argmax(frames > 128) / 30) - audio_at


@pytest.mark.parametrize(
    "cuts",
    [
        [(0, 60), (90, 120), (140, 170)],
        [(0, 60), (90, 120), (130, 140), (142, 175), (200, 230), (240, 260), (270, 280)],
    ],
    ids=["direct", "one-by-one"],
)
def test_audio_stays_in_sync_across_cuts(tmp_path, cuts):
    video = _flash_and_beep(tmp_path)
    segments = [Segment(a, b) for a, b in cuts]
    spec = RenderSpec(video, probe(video), segments, "16:9", "1080p", tmp_path / "o.mp4", fit="crop")
    result = render_clip(spec)
    assert count_frames(result.path) == sum(s.frames for s in segments)
    assert abs(_sync_offset(result.path, tmp_path)) <= 2 / 30


def test_many_cuts_are_rendered_one_at_a_time_with_exact_frames_and_captions(scene_video, tmp_path):
    from clipforge.config import get_settings

    cuts = [Segment(a, a + 10) for a in (0, 20, 40, 55, 80, 100, 115, 140)]
    words = [Word("HELLO", 0.1, 0.5), Word("THERE", 3.4, 3.8)]
    spec = spec_for(scene_video, tmp_path, cuts, words=words, preset=BUILTIN_PRESETS["anime-impact"])
    result = render_clip(spec)
    assert count_frames(result.path) == 80
    assert result.frames == 80
    # Output frame 34 is source frame 59, the last red one. Output frame 35 is source 60, the first blue.
    assert grab(result.path, 34, tmp_path)[:, :, 0].mean() > 150
    assert grab(result.path, 35, tmp_path)[:, :, 2].mean() > 150
    assert list(get_settings().tmp_dir.iterdir()) == []


def test_crossfade_with_many_cuts_falls_back_and_says_so(scene_video, tmp_path):
    cuts = [Segment(a, a + 10) for a in (0, 20, 40, 60, 80, 100)]
    result = render_clip(spec_for(scene_video, tmp_path, cuts, transition="crossfade"))
    assert count_frames(result.path) == 60
    assert any("clean cuts" in note for note in result.notes)


@pytest.mark.parametrize("folder", ["it's here", "a [b], c; d", "x:y", "100% = done"])
def test_captions_render_when_the_work_folder_has_awkward_characters(tmp_path, monkeypatch, folder):
    from clipforge import config
    from clipforge.captions.preview import render_preview, sample_words

    monkeypatch.setenv("CLIPFORGE_WORK_DIR", str(tmp_path / folder))
    config.get_settings.cache_clear()
    out = render_preview(
        BUILTIN_PRESETS["anime-impact"], tmp_path / "p.png", "9:16", words=sample_words("hello there")
    )
    assert Image.open(out).size == (540, 960)


def test_plain_gain_when_peaks_allow_and_limiter_when_they_do_not():
    from clipforge.export.render import loudnorm_filter

    stats = {
        "input_i": "-24.0",
        "input_tp": "-12.0",
        "input_lra": "5",
        "input_thresh": "-34",
        "target_offset": "0",
    }
    assert loudnorm_filter(stats) == "volume=10.00dB"
    assert loudnorm_filter({**stats, "input_tp": "-3.0"}).startswith("loudnorm=")
    assert loudnorm_filter(None) is None


def test_loudness_is_measured_after_converting_to_the_final_format():
    from clipforge.export.render import finish_audio

    chain = finish_audio("volume=1dB")
    assert chain.index("aformat=channel_layouts=stereo") < chain.index("volume=1dB")
    assert chain.endswith("aformat=channel_layouts=stereo")
    assert finish_audio(None) == "aresample=48000,aformat=channel_layouts=stereo"


YUNET_SOURCE = os.environ.get("CLIPFORGE_TEST_YUNET") or str(
    Path.home() / "clipforge-work" / "models" / "face" / "face_detection_yunet_2023mar.onnx"
)


def _face_picture():
    """A blank frame. Real faces are measured by benchmarks/faces.py, since photos stay out of the repo."""
    return np.full((180, 320, 3), 40, np.uint8)


@pytest.mark.skipif(not Path(YUNET_SOURCE).exists(), reason="YuNet model not downloaded")
def test_yunet_is_used_when_installed_and_finds_no_face_in_a_blank_frame(monkeypatch):
    from clipforge.config import get_settings
    from clipforge.export import reframe

    target = get_settings().models_dir / reframe.YUNET_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(YUNET_SOURCE)
    reframe._yunet.cache_clear()
    monkeypatch.setattr(reframe, "_haar", lambda: pytest.fail("Haar ran although YuNet is installed"))
    assert reframe.yunet_available()
    assert reframe.detect_faces(_face_picture()) == []


def test_a_damaged_yunet_file_falls_back_to_haar_instead_of_failing_the_render(monkeypatch):
    from clipforge.config import get_settings
    from clipforge.export import reframe

    target = get_settings().models_dir / reframe.YUNET_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"this is not an onnx model")
    reframe._yunet.cache_clear()
    used_haar = []
    real_haar = reframe._haar
    monkeypatch.setattr(reframe, "_haar", lambda: used_haar.append(True) or real_haar())
    assert reframe.detect_faces(_face_picture()) == []
    assert used_haar


def test_haar_fallback_runs_when_the_model_is_missing(monkeypatch):
    from clipforge.export import reframe

    used_haar = []
    real_haar = reframe._haar
    monkeypatch.setattr(reframe, "_haar", lambda: used_haar.append(True) or real_haar())
    assert not reframe.yunet_available()
    assert reframe.detect_faces(_face_picture()) == []
    assert used_haar


def test_motion_is_the_fallback_when_there_is_no_face():
    from clipforge.export import reframe

    previous = _face_picture()
    frame = previous.copy()
    frame[60:120, 240:300] = 255
    center, confidence = reframe.find_subject(frame, previous)
    assert confidence == 0.5
    assert abs(center - 270 / 320) < 0.02


def test_second_loudness_pass_limits_peaks_at_minus_1_5():
    from clipforge.export.render import loudnorm_filter

    stats = {
        "input_i": "-30",
        "input_tp": "-3",
        "input_lra": "5",
        "input_thresh": "-40",
        "target_offset": "0",
    }
    assert "TP=-1.5" in loudnorm_filter(stats)  # +16 dB would push peaks past the limit
    quiet = {**stats, "input_tp": "-20"}
    assert loudnorm_filter(quiet) == "volume=16.00dB"  # +16 dB leaves the peak at -4: plain gain
