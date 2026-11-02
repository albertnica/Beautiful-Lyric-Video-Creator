"""Pydantic V2 Request and Response schemas for the Web Studio API.

Enforces strict input validation (preventing path traversal on file deletion/folder opening)
and guarantees every endpoint returns a deterministic, documented JSON schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lyric_video_creator.config import BackendType, EncoderType, RenderConfig

type FolderTarget = Literal["songs", "lyrics", "output", "logs"]
type LRCBindingStatus = Literal["auto_matched", "missing", "custom"]


class ApiErrorResponse(BaseModel):
    """Standardized JSON error envelope returned by global exception handlers."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    code: int = Field(description="HTTP status code.")
    message: str = Field(description="Human-readable error description.")
    detail: str | None = Field(default=None, description="Optional technical context.")


class OpenFolderRequest(BaseModel):
    """Request payload for opening a project folder in the OS file explorer."""

    model_config = ConfigDict(extra="forbid")

    target: FolderTarget = Field(description="Target project folder to open ('songs', 'lyrics', 'output', 'logs').")


class OpenFolderResponse(BaseModel):
    """Response confirming the target directory was opened in Windows Explorer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    target: FolderTarget
    path: str


class SetFolderRequest(BaseModel):
    """Request payload for updating an active media directory path."""

    model_config = ConfigDict(extra="forbid")

    target: Literal["songs", "lyrics", "output"] = Field(description="Target folder key ('songs', 'lyrics', 'output').")
    path: str = Field(min_length=1, description="Absolute folder path, or '__default__' to restore the factory default.")


class SetFolderResponse(BaseModel):
    """Response returned when an active media directory path is changed."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"] = "ok"
    target: str
    path: str
    is_default: bool
    message: str


class VideoItemModel(BaseModel):
    """Metadata representation of a rendered `.mp4` video inside `output/`."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(description="Basename of the rendered .mp4 file.")
    title: str = Field(description="Display title derived from filename.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    size_human: str = Field(description="Formatted size string (e.g., '24.5 MB').")
    modified_iso: str = Field(description="ISO-8601 last modified timestamp.")
    modified_ts: float = Field(description="POSIX timestamp for client-side sorting.")
    url: str = Field(description="Relative HTTP streaming URL (`/output/...`).")


class TrackItemModel(BaseModel):
    """Metadata and LRC status for an audio track in the active music folder."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(description="Basename of the audio file.")
    stem: str = Field(description="Filename without extension.")
    format: str = Field(description="File extension (e.g., .flac, .mp3).")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    size_human: str = Field(description="Human-readable file size.")
    duration: float = Field(ge=0.0, description="Track duration in seconds.")
    duration_human: str = Field(description="Formatted duration string (MM:SS).")
    title: str = Field(description="Track title.")
    artist: str = Field(description="Artist name.")
    album: str = Field(description="Album name.")
    has_cover: bool = Field(description="Whether embedded or sidecar artwork was found.")
    matched_lrc: str | None = Field(default=None, description="Basename of matched .lrc in lyrics folder.")
    lrc_status: LRCBindingStatus = Field(description="LRC binding status ('auto_matched', 'missing', 'custom').")


class TrackListResponse(BaseModel):
    """Discovered audio tracks and companion .lrc candidates in active directories."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    music_folder: str = Field(description="Active music directory.")
    lyrics_folder: str = Field(description="Active lyrics directory.")
    tracks: list[TrackItemModel]
    available_lrcs: list[str] = Field(description="All .lrc filenames discovered in the active lyrics folder.")


class RenderProgressModel(BaseModel):
    """Live batch or single rendering telemetry exposed to the Web Studio UI."""

    model_config = ConfigDict(extra="forbid")

    is_rendering: bool = False
    mode: Literal["batch", "single", "idle"] = "idle"
    current_index: int = 0
    total_files: int = 0
    current_track: str = ""
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    percent: float = 0.0
    message: str = "Idle"


class StudioStatusResponse(BaseModel):
    """Complete Web Studio state including folder counts, hardware backends, config, and rendered videos."""

    model_config = ConfigDict(extra="forbid")

    songs_count: int = Field(ge=0)
    lyrics_count: int = Field(ge=0)
    videos_count: int = Field(ge=0)
    songs_dir: str
    lyrics_dir: str
    output_dir: str
    active_backend: str
    active_encoder: str
    render_progress: RenderProgressModel
    videos: list[VideoItemModel]
    config: RenderConfig


class StartRenderRequest(BaseModel):
    """Optional configuration overrides when triggering batch video creation from the UI."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendType | None = Field(default=None)
    encoder: EncoderType | None = Field(default=None)
    fps: int | None = Field(default=None, ge=1, le=240)


class SingleRenderRequest(BaseModel):
    """Payload for launching isolated video rendering for a single audio track."""

    model_config = ConfigDict(extra="forbid")

    audio_filename: str = Field(min_length=1, description="Basename of the audio file in the active music folder.")
    lrc_filename: str | None = Field(
        default=None,
        description="Basename of .lrc in lyrics folder, '__none__' for cover-only mode, or None for auto-matching.",
    )
    backend: BackendType | None = Field(default=None)
    encoder: EncoderType | None = Field(default=None)
    fps: int | None = Field(default=None, ge=1, le=240)
    preset: str | None = Field(default=None, description="Optional preset ID to apply for this render.")


class SingleRenderResponse(BaseModel):
    """Response returned when initiating single-track video creation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["started", "busy", "not_found", "error"]
    message: str
    track: str
    has_lyrics: bool = False


class PresetDefinitionModel(BaseModel):
    """Structured standard rendering configuration preset."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    width: int
    height: int
    fps: int
    video_bitrate: str
    audio_bitrate: str
    left_margin: int
    glow_blur_ksize: int
    blur_strength: int
    glow_intensity_base: float
    glow_boost: float
    rotation_speed: float


class PresetListResponse(BaseModel):
    """Catalog of standard configuration presets."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    active_preset: str | None = None
    presets: list[PresetDefinitionModel]


class ApplyPresetRequest(BaseModel):
    """Payload for activating a standard preset."""

    model_config = ConfigDict(extra="forbid")

    preset_id: str = Field(min_length=1, description="Preset identifier to activate.")


class ConfigUpdateRequest(BaseModel):
    """Pydantic V2 request payload for updating tunable rendering configuration parameters."""

    model_config = ConfigDict(extra="ignore")

    music_folder: str | None = None
    lyrics_folder: str | None = None
    output_folder: str | None = None
    width: int = Field(default=1920, ge=320, le=7680)
    height: int = Field(default=1080, ge=240, le=4320)
    fps: int = Field(default=20, ge=1, le=240)
    video_bitrate: str = Field(default="8000k", pattern=r"^\d+[kKmM]$")
    audio_bitrate: str = Field(default="320k", pattern=r"^\d+[kKmM]$")
    left_margin: int = Field(default=50, ge=0, le=500)
    block_size: int = Field(default=32, ge=8, le=32)
    ping_pong_buffers: int = Field(default=12, ge=2, le=64)
    queue_max: int = Field(default=400, ge=4, le=2000)
    max_lyrics_length: int = Field(default=33, ge=10, le=120)
    glow_blur_ksize: int = Field(default=111, ge=3, le=501)
    glow_intensity_base: float = Field(default=0.5, ge=0.0, le=2.0)
    blur_strength: int = Field(default=501, ge=3, le=1001)
    fixed_text_glow_intensity: float = Field(default=0.5, ge=0.0, le=2.0)
    assets_glow_intensity: float = Field(default=0.5, ge=0.0, le=2.0)
    glow_pulse_amp: float = Field(default=0.04, ge=0.0, le=1.0)
    glow_pulse_freq: float = Field(default=0.5, ge=0.0, le=10.0)
    glow_edge_px: float = Field(default=24.0, ge=1.0, le=200.0)
    glow_boost: float = Field(default=6.0, ge=0.1, le=20.0)
    alpha_min: float = Field(default=0.32, ge=0.0, le=1.0)
    default_last_line_duration: float = Field(default=4.0, gt=0.1, le=30.0)
    background_size: int = Field(default=3840, ge=1024, le=7680)
    rotation_speed: float = Field(default=0.005, ge=0.0, le=0.2)
    min_glow_sat: int = Field(default=20, ge=0, le=255)
    min_glow_val: int = Field(default=80, ge=0, le=255)
    backend: BackendType = Field(default="auto")
    encoder: EncoderType = Field(default="auto")
    debug: bool = Field(default=False)


class ConfigResponse(BaseModel):
    """Response returning active configuration alongside defaults and matched preset."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    message: str
    config: RenderConfig
    defaults: RenderConfig
    matched_preset: str | None = None


class StartRenderResponse(BaseModel):
    """Response returned when batch rendering is launched or rejected if already busy."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["started", "busy", "empty"]
    message: str
    total_files: int = 0


class StopRenderResponse(BaseModel):
    """Response returned when requesting cancellation of an active batch render."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["stopping", "idle"]
    message: str


class DeleteVideosRequest(BaseModel):
    """Request payload for deleting one or more rendered videos from `output/`."""

    model_config = ConfigDict(extra="forbid")

    filenames: list[str] = Field(min_length=1, description="List of .mp4 basenames inside output/ to delete.")


class DeleteVideosResponse(BaseModel):
    """Response summarizing deleted and failed filenames."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    deleted: list[str]
    failed: list[str]
    remaining_count: int


class PlayExternalRequest(BaseModel):
    """Request payload to open a rendered video in the OS default media player."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, description="Basename of the .mp4 file inside output/.")


class PlayExternalResponse(BaseModel):
    """Response confirming the video was launched in the external OS player."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    filename: str


class ShutdownResponse(BaseModel):
    """Response returned when shutting down the background Web Studio server."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["shutting_down"] = "shutting_down"
    message: str
