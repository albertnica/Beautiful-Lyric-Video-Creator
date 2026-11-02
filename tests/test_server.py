"""Unit tests for Beautiful Lyric Video Creator Web Studio API endpoints and service methods.

Tests standard presets, track discovery, folder management, and single-track rendering.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import pytest

from lyric_video_creator.server.app import app
from lyric_video_creator.server.presets import PRESETS_CATALOG, get_preset_by_id
from lyric_video_creator.server.service import get_studio_service


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient instance bound to the FastAPI application."""
    return TestClient(app)


def test_get_presets_catalog(client: TestClient) -> None:
    """Verify standard presets endpoint returns all pre-calibrated profiles."""
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert len(data["presets"]) >= 5
    ids = [p["id"] for p in data["presets"]]
    assert "standard_1080p" in ids
    assert "reels_vertical" in ids
    assert "cinema_4k" in ids
    assert "fast_preview" in ids


def test_apply_preset(client: TestClient) -> None:
    """Verify applying a preset updates the active configuration."""
    res = client.post("/api/presets/apply", json={"preset_id": "fast_preview"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["config"]["width"] == 1280
    assert data["config"]["height"] == 720
    assert data["config"]["fps"] == 15

    # Reset back to default
    reset_res = client.post("/api/config/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["config"]["width"] == 1920


def test_get_status(client: TestClient) -> None:
    """Verify status endpoint returns required folder and hardware telemetry."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "songs_count" in data
    assert "lyrics_count" in data
    assert "videos_count" in data
    assert "active_backend" in data
    assert "active_encoder" in data
    assert "render_progress" in data


def test_folder_set_and_reset(client: TestClient, tmp_path: Path) -> None:
    """Verify updating and resetting an active workspace media directory."""
    custom_songs = tmp_path / "custom_songs"
    custom_songs.mkdir()

    # Apply valid custom directory
    res = client.post("/api/folders/set", json={"target": "songs", "path": str(custom_songs)})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["path"] == str(custom_songs.resolve())
    assert data["is_default"] is False

    # Attempt setting non-existent directory -> should return error status
    bad_res = client.post("/api/folders/set", json={"target": "songs", "path": str(tmp_path / "does_not_exist")})
    assert bad_res.status_code == 200
    assert bad_res.json()["status"] == "error"

    # Reset back to default
    reset_res = client.post("/api/folders/set", json={"target": "songs", "path": "__default__"})
    assert reset_res.status_code == 200
    assert reset_res.json()["is_default"] is True


def test_list_tracks(client: TestClient, tmp_path: Path) -> None:
    """Verify track discovery and LRC auto-matching."""
    # Create synthetic songs and lyrics folders
    music_dir = tmp_path / "music_test"
    lyrics_dir = tmp_path / "lyrics_test"
    music_dir.mkdir()
    lyrics_dir.mkdir()

    # Create dummy audio and matching LRC
    (music_dir / "01 - Test Song.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    (lyrics_dir / "01 - Test Song.lrc").write_text("[00:01.00]Hello world\n", encoding="utf-8")

    client.post("/api/folders/set", json={"target": "songs", "path": str(music_dir)})
    client.post("/api/folders/set", json={"target": "lyrics", "path": str(lyrics_dir)})

    try:
        res = client.get("/api/tracks")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert len(data["tracks"]) >= 1
        track = data["tracks"][0]
        assert track["filename"] == "01 - Test Song.wav"
        assert track["matched_lrc"] == "01 - Test Song.lrc"
        assert track["lrc_status"] == "auto_matched"
        assert "01 - Test Song.lrc" in data["available_lrcs"]
    finally:
        # Restore defaults
        client.post("/api/folders/set", json={"target": "songs", "path": "__default__"})
        client.post("/api/folders/set", json={"target": "lyrics", "path": "__default__"})


def test_single_render_endpoint_and_apostrophe_handling(client: TestClient, tmp_path: Path) -> None:
    """Verify single track rendering endpoint handles filenames with apostrophes and starts cleanly."""
    music_dir = tmp_path / "music_apostrophe"
    lyrics_dir = tmp_path / "lyrics_apostrophe"
    output_dir = tmp_path / "output_apostrophe"
    music_dir.mkdir()
    lyrics_dir.mkdir()
    output_dir.mkdir()

    # Create dummy audio with apostrophe in filename (reproducing 001 - I'll Come Runnin)
    dummy_wav = music_dir / "001 - I'll Come Runnin.wav"
    dummy_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    dummy_lrc = lyrics_dir / "001 - I'll Come Runnin.lrc"
    dummy_lrc.write_text("[00:01.00]I'll come runnin\n", encoding="utf-8")

    client.post("/api/folders/set", json={"target": "songs", "path": str(music_dir)})
    client.post("/api/folders/set", json={"target": "lyrics", "path": str(lyrics_dir)})
    client.post("/api/folders/set", json={"target": "output", "path": str(output_dir)})

    try:
        # Verify tracks list returns the track with apostrophe
        res_tracks = client.get("/api/tracks")
        assert res_tracks.status_code == 200
        data = res_tracks.json()
        assert len(data["tracks"]) == 1
        assert data["tracks"][0]["filename"] == "001 - I'll Come Runnin.wav"

        # Verify POST /api/render/single starts without needing batch render
        res_single = client.post(
            "/api/render/single",
            json={
                "audio_filename": "001 - I'll Come Runnin.wav",
                "lrc_filename": "001 - I'll Come Runnin.lrc",
            },
        )
        assert res_single.status_code == 200
        single_data = res_single.json()
        assert single_data["status"] == "started"
        assert single_data["track"] == "001 - I'll Come Runnin.wav"
    finally:
        client.post("/api/folders/set", json={"target": "songs", "path": "__default__"})
        client.post("/api/folders/set", json={"target": "lyrics", "path": "__default__"})
        client.post("/api/folders/set", json={"target": "output", "path": "__default__"})


def test_open_folder_and_video_external_endpoints(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify open-folder and play-external endpoints trigger native OS process launches without exceptions."""
    import subprocess

    launched_cmds: list[list[str] | str] = []

    def mock_popen(cmd: list[str] | str, **kwargs: object) -> object:
        launched_cmds.append(cmd)
        class DummyProc:
            pid = 99999
        return DummyProc()

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    # 1. Test opening output folder
    res_out = client.post("/api/open-folder", json={"target": "output"})
    assert res_out.status_code == 200
    out_data = res_out.json()
    assert out_data["status"] == "ok"
    assert out_data["target"] == "output"
    assert len(launched_cmds) >= 1
    if sys.platform == "win32":
        assert any("explorer.exe" in str(c) for c in launched_cmds)

    # 2. Test opening songs folder
    res_songs = client.post("/api/open-folder", json={"target": "songs"})
    assert res_songs.status_code == 200
    assert res_songs.json()["target"] == "songs"

    # 3. Test invalid folder target
    res_bad = client.post("/api/open-folder", json={"target": "invalid_folder"})
    assert res_bad.status_code == 422


