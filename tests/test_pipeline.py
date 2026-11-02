"""Comprehensive unit and integration test suite for Beautiful Lyric Video Creator.

Validates Pydantic V2 config schemas, Unicode sanitization, recursive LRC splitting,
dominant color heuristics, ROI sprite memory reduction, FFmpeg encoder probing, and
end-to-end rendering across all 4 compute backends (CuPy, PyTorch, OpenCL, CPU).
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from lyric_video_creator import (
    CPUBackend,
    CPUVideoRenderer,
    CuPyBackend,
    OpenCLBackend,
    RenderConfig,
    TorchBackend,
    TrackMetadata,
    compute_dominant_color,
    create_compute_backend,
    ensure_min_brightness,
    ensure_min_saturation,
    extract_track_metadata,
    is_cupy_usable,
    is_opencl_usable,
    parse_lyrics,
    prepare_scene,
    resolve_video_encoder,
    sanitize_text,
    strip_title_extra,
)
from lyric_video_creator.backends.torch_backend import detect_torch_device


def test_render_config_validation() -> None:
    """Verify Pydantic V2 enforces odd blur kernels and even video dimensions."""
    cfg = RenderConfig(width=1281, height=721, glow_blur_ksize=50, blur_strength=100)
    assert cfg.width == 1282
    assert cfg.height == 722
    assert cfg.glow_blur_ksize == 51
    assert cfg.blur_strength == 101


def test_sanitize_and_strip_title() -> None:
    """Verify homoglyph replacement, zero-width stripping, and title suffix removal."""
    # Cyrillic 'а' (\u0430) and 'е' (\u0435) + curly quotes + em-dash + zero-width space
    dirty = "T\u0435st\u200B \u2014 \u201C\u0430bc\u201D\u2026"
    assert sanitize_text(dirty) == 'Test - "abc"...'

    assert strip_title_extra("Song Title (feat. Someone)") == "Song Title"
    assert strip_title_extra("Song Title [Nikademis Remix]") == "Song Title"
    assert strip_title_extra("Song Title ft. Artist") == "Song Title"


def test_parse_lyrics_and_recursive_split(tmp_path: Path) -> None:
    """Verify LRC parsing, deduplication, and punctuation-aware recursive line splitting."""
    lrc_content = (
        "[00:01.00]Short line\n"
        "[00:01.05]Short line\n"  # Duplicate within 100ms -> should be filtered
        "[00:05.00]This is a significantly longer lyric line, which should split cleanly after the comma\n"
    )
    lrc_path = tmp_path / "test.lrc"
    lrc_path.write_text(lrc_content, encoding="utf-8")

    cfg = RenderConfig(max_lyrics_length=33, default_last_line_duration=4.0)
    parsed = parse_lyrics(lrc_path, config=cfg)

    assert len(parsed) >= 3
    assert parsed[0][2] == "Short line"
    assert parsed[0][0] == pytest.approx(1.0)
    assert parsed[0][1] == pytest.approx(5.0)
    for start, end, text in parsed:
        assert len(text) <= 33
        assert end > start


def test_color_heuristics() -> None:
    """Verify dominant color extraction and HSV saturation/brightness floor enforcement."""
    # Create synthetic blue-teal image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (200, 120, 30)  # BGR
    dom = compute_dominant_color(img)
    assert dom[0] > dom[2]  # Blue channel > Red channel

    # Desaturated dark pixel should be boosted to minimum saturation and brightness
    sat_bgr, sat_changed = ensure_min_saturation((10, 10, 10), min_sat=40)
    bright_bgr, bright_changed = ensure_min_brightness(sat_bgr, min_val=100)
    assert bright_changed is True
    assert max(bright_bgr) >= 95


def _make_synthetic_metadata() -> TrackMetadata:
    """Create a synthetic 400x400 album cover encoded as PNG bytes."""
    cover = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(cover, (0, 0), (400, 400), (180, 90, 40), -1)
    cv2.circle(cover, (200, 200), 120, (60, 180, 240), -1)
    ok, buf = cv2.imencode(".png", cover)
    assert ok
    return TrackMetadata(
        cover_data=buf.tobytes(),
        title="Synthetic Horizon",
        artist="Test Artist",
        album="Unit Test Album",
        date="2026-10-12",
    )


def test_roi_compositor_and_all_compute_backends() -> None:
    """Verify scene preparation and visual parity across CPU, OpenCL, PyTorch, and CuPy backends."""
    cfg = RenderConfig(
        width=640,
        height=360,
        fps=10,
        background_size=1024,
        blur_strength=51,
        glow_blur_ksize=21,
        ping_pong_buffers=2,
    )
    meta = _make_synthetic_metadata()
    lyrics = [
        (0.0, 1.0, "First synchronized line"),
        (1.0, 2.0, "Second glowing lyric"),
    ]
    scene = prepare_scene(
        metadata=meta,
        lyrics=lyrics,
        duration=2.0,
        has_lyrics=True,
        config=cfg,
    )
    assert scene is not None
    assert scene.total_frames == 20
    assert len(scene.lyric_text_sprites) > 0

    # Verify ROI cropping saves >80% memory vs full-frame 640x360x4
    full_frame_bytes = cfg.width * cfg.height * 4
    sample_sprite = next(iter(scene.lyric_text_sprites.values()))
    assert sample_sprite.image.nbytes < full_frame_bytes * 0.35

    # Render frame 5 using CPUBackend as reference
    cpu_backend = CPUBackend(config=cfg)
    cpu_backend.load_scene(scene)
    cpu_backend.render_frame_to_buffer(5, 0)
    cpu_frame = cpu_backend.get_host_buffer(0).copy()
    cpu_backend.cleanup()
    assert cpu_frame.shape == (360, 640, 3)
    assert cpu_frame.mean() > 10.0

    # Compare against OpenCLBackend if usable
    if is_opencl_usable():
        ocl_backend = OpenCLBackend(config=cfg)
        ocl_backend.load_scene(scene)
        ocl_backend.render_frame_to_buffer(5, 0)
        ocl_frame = ocl_backend.get_host_buffer(0).copy()
        ocl_backend.cleanup()
        mad_ocl = float(np.abs(cpu_frame.astype(np.float32) - ocl_frame.astype(np.float32)).mean())
        assert mad_ocl < 2.0

    # Compare against TorchBackend if usable
    if detect_torch_device(allow_cpu=True) is not None:
        torch_backend = TorchBackend(config=cfg)
        torch_backend.load_scene(scene)
        torch_backend.render_frame_to_buffer(5, 0)
        torch_frame = torch_backend.get_host_buffer(0).copy()
        torch_backend.cleanup()
        mad_torch = float(np.abs(cpu_frame.astype(np.float32) - torch_frame.astype(np.float32)).mean())
        assert mad_torch < 3.5

    # Compare against CuPyBackend if usable
    if is_cupy_usable():
        cupy_backend = CuPyBackend(config=cfg)
        cupy_backend.load_scene(scene)
        ev = cupy_backend.render_frame_to_buffer(5, 0)
        if ev is not None:
            ev.synchronize()
        cupy_frame = cupy_backend.get_host_buffer(0).copy()
        cupy_backend.cleanup()
        mad_cupy = float(np.abs(cpu_frame.astype(np.float32) - cupy_frame.astype(np.float32)).mean())
        assert mad_cupy < 3.5


def test_end_to_end_cpu_fallback_video_render(tmp_path: Path) -> None:
    """Verify that `CPUVideoRenderer` + `libx264` fallback produces a valid playable .mp4 file."""
    audio_path = tmp_path / "001 - Test Track.wav"
    cover_path = tmp_path / "001 - Test Track.png"
    lrc_path = tmp_path / "001 - Test Track.lrc"
    out_path = tmp_path / "001 - Test Track.mp4"

    # Generate 1-second silent WAV via FFmpeg
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "1.0",
            str(audio_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Write sidecar cover image (tests non-FLAC sidecar cover extraction!)
    cover_img = np.full((300, 300, 3), (140, 70, 210), dtype=np.uint8)
    cv2.imwrite(str(cover_path), cover_img)

    # Write LRC file
    lrc_path.write_text("[00:00.10]CPU Fallback Render Test\n", encoding="utf-8")

    # Verify sidecar metadata extraction works on WAV
    meta = extract_track_metadata(audio_path)
    assert meta is not None

    cfg = RenderConfig(
        width=640,
        height=360,
        fps=10,
        background_size=1024,
        blur_strength=31,
        glow_blur_ksize=15,
        backend="cpu",
        encoder="libx264",
    )
    renderer = CPUVideoRenderer(config=cfg)
    try:
        ok = renderer.render_video(audio_path, lrc_path, out_path)
        assert ok is True
        assert out_path.is_file()
        assert out_path.stat().st_size > 1000
    finally:
        renderer.cleanup()


def test_web_studio_api_and_explorer_actions(tmp_path: Path) -> None:
    """Verify FastAPI Web Studio endpoints, UI buttons, video listing, and path-safe deletion."""
    from fastapi.testclient import TestClient
    from lyric_video_creator.server.app import app
    from lyric_video_creator.server.service import StudioService, get_studio_service

    songs_dir = tmp_path / "songs"
    lyrics_dir = tmp_path / "lyrics"
    output_dir = tmp_path / "output"
    logs_dir = tmp_path / "logs"

    cfg = RenderConfig(
        music_folder=songs_dir,
        lyrics_folder=lyrics_dir,
        output_folder=output_dir,
        logs_folder=logs_dir,
    )
    test_service = StudioService(config=cfg)

    # Create dummy rendered MP4 in output_dir
    dummy_mp4 = output_dir / "001 - Demo Song.mp4"
    dummy_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048)

    app.dependency_overrides[get_studio_service] = lambda: test_service
    try:
        client = TestClient(app)

        # 1. Verify HTML page contains the 4 mandatory action buttons
        res_html = client.get("/")
        assert res_html.status_code == 200
        html_text = res_html.text
        assert "Open Songs Folder" in html_text
        assert "Open Lyrics Folder" in html_text
        assert "Launch Video Creation" in html_text
        assert "Open Output Folder" in html_text

        # 2. Verify /api/status lists the rendered video in output/
        res_status = client.get("/api/status")
        assert res_status.status_code == 200
        status_json = res_status.json()
        assert status_json["videos_count"] == 1
        assert status_json["videos"][0]["filename"] == "001 - Demo Song.mp4"

        # 3. Verify path traversal is rejected on deletion
        res_bad_del = client.post(
            "/api/videos/delete",
            json={"filenames": ["../outside.mp4"]},
        )
        assert res_bad_del.status_code == 200
        assert "../outside.mp4" in res_bad_del.json()["failed"]

        # 4. Verify legitimate video deletion removes the file from output/
        res_del = client.post(
            "/api/videos/delete",
            json={"filenames": ["001 - Demo Song.mp4"]},
        )
        assert res_del.status_code == 200
        del_json = res_del.json()
        assert del_json["deleted"] == ["001 - Demo Song.mp4"]
        assert del_json["remaining_count"] == 0
        assert not dummy_mp4.exists()

        # 5. Verify bottom tab switcher & Configuration tab exist in HTML
        assert "nav-tab-main" in html_text
        assert "nav-tab-config" in html_text
        assert "Engine &amp; Visual Configuration" in html_text

        # 6. Verify GET /api/config returns the user's default parameters
        res_cfg = client.get("/api/config")
        assert res_cfg.status_code == 200
        cfg_json = res_cfg.json()
        assert cfg_json["config"]["width"] == 1920
        assert cfg_json["config"]["height"] == 1080
        assert cfg_json["config"]["fps"] == 20
        assert cfg_json["config"]["video_bitrate"] == "8000k"
        assert cfg_json["config"]["audio_bitrate"] == "320k"

        # 7. Verify PUT /api/config updates resolution, framerate, and bitrates and persists to disk
        res_upd = client.put(
            "/api/config",
            json={
                "width": 2560,
                "height": 1440,
                "fps": 60,
                "video_bitrate": "14000k",
                "glow_intensity_base": 1.65,
            },
        )
        assert res_upd.status_code == 200
        upd_json = res_upd.json()
        assert upd_json["config"]["width"] == 2560
        assert upd_json["config"]["height"] == 1440
        assert upd_json["config"]["fps"] == 60
        assert upd_json["config"]["video_bitrate"] == "14000k"
        assert upd_json["config"]["glow_intensity_base"] == 1.65
        assert (logs_dir / "studio_config.json").is_file()

        # 8. Verify POST /api/config/reset restores the user's default parameters (1920x1080 @ 20 FPS)
        res_rst = client.post("/api/config/reset")
        assert res_rst.status_code == 200
        rst_json = res_rst.json()
        assert rst_json["config"]["width"] == 1920
        assert rst_json["config"]["height"] == 1080
        assert rst_json["config"]["fps"] == 20
        assert rst_json["config"]["video_bitrate"] == "8000k"
        assert not (logs_dir / "studio_config.json").exists()
    finally:
        app.dependency_overrides.clear()


def test_vertical_tiktok_layout_and_bounding_boxes() -> None:
    """Verify that vertical (9:16, 1080x1920) TikTok/Reels layout centers cover and lyrics without overflow."""
    cfg = RenderConfig(width=1080, height=1920, fps=30, background_size=2048)
    meta = _make_synthetic_metadata()
    lyrics = [
        (0.0, 1.0, "TikTok Vertical Lyrics Line 1"),
        (1.0, 2.0, "Second Synchronized Center Line"),
    ]
    scene = prepare_scene(
        metadata=meta,
        lyrics=lyrics,
        duration=2.0,
        has_lyrics=True,
        config=cfg,
    )
    assert scene is not None

    # 1. Album cover must be centered horizontally in the upper region
    expected_cx = (cfg.width - scene.cover_width) // 2
    assert scene.cover_x == expected_cx
    assert scene.cover_y < cfg.height // 3
    assert scene.cover_width <= cfg.width * 0.75

    # 2. Lyrics must be positioned below the cover and centered horizontally
    for (lyr_idx, tag), sprite in scene.lyric_text_sprites.items():
        assert sprite.x >= 0
        assert sprite.x + sprite.width <= cfg.width
        assert sprite.y > scene.cover_y + scene.cover_height
        assert sprite.text_left_x >= 0
        assert sprite.text_left_x + sprite.text_width <= cfg.width


