import json
import subprocess

import numpy as np
import pytest
from PIL import Image

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
