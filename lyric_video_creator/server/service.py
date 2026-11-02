"""Business logic and state management service for the Beautiful Lyric Video Creator Web Studio.

Encapsulates OS folder launching, directory inspection, path-traversal-safe video
deletion, preset application, and background rendering thread lifecycle (batch and single track).
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock, Thread
from typing import Any, Literal
from urllib.parse import quote

from lyric_video_creator.backends import (
    detect_torch_device,
    is_cupy_usable,
    is_opencl_usable,
)
from lyric_video_creator.config import DEFAULT_CONFIG, BackendType, EncoderType, RenderConfig
from lyric_video_creator.encoder import resolve_video_encoder
from lyric_video_creator.media import (
    extract_track_metadata,
    find_lyrics_file,
    find_music_files,
    get_audio_duration,
)
from lyric_video_creator.pipeline import get_renderer, run_batch_pipeline
from lyric_video_creator.server.models import (
    ConfigResponse,
    ConfigUpdateRequest,
    DeleteVideosResponse,
    FolderTarget,
    PresetListResponse,
    RenderProgressModel,
    SetFolderResponse,
    SingleRenderRequest,
    SingleRenderResponse,
    StartRenderResponse,
    StopRenderResponse,
    StudioStatusResponse,
    TrackItemModel,
    TrackListResponse,
    VideoItemModel,
)
from lyric_video_creator.server.presets import (
    PRESETS_CATALOG,
    detect_matching_preset,
    get_preset_by_id,
)
from lyric_video_creator.utils import get_logger, setup_directories


def _format_size(size_bytes: int) -> str:
    """Convert byte count into human-readable KB/MB/GB string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    kb = size_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


def _detect_hardware_summary() -> tuple[str, str]:
    """Return a concise description of the active compute accelerator and FFmpeg encoder."""
    if is_cupy_usable():
        backend_desc = "NVIDIA CUDA (CuPy RawKernels)"
    else:
        torch_dev = detect_torch_device(allow_cpu=False)
        if torch_dev is not None:
            backend_desc = f"PyTorch GPU ({torch_dev.upper()})"
        elif is_opencl_usable():
            backend_desc = "OpenCV OpenCL (UMat GPU)"
        else:
            backend_desc = "CPU (OpenCV C++ SIMD + NumPy)"

    try:
        codec, _ = resolve_video_encoder("auto")
    except Exception:
        codec = "libx264"
    return backend_desc, codec


class StudioService:
    """Stateful singleton service coordinating Web Studio actions, presets, and background rendering."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        self._defaults = config.model_copy()
        self.config = config.model_copy()
        setup_directories(self.config)
        self._config_file = Path(self.config.logs_folder).resolve() / "studio_config.json"
        self._load_persisted_config()
        self._lock = Lock()
        self._stop_event = Event()
        self._render_thread: Thread | None = None
        self._progress = RenderProgressModel()
        self._backend_summary, self._encoder_summary = _detect_hardware_summary()

    def _load_persisted_config(self) -> None:
        """Load user-saved studio configuration from `logs/studio_config.json` if present."""
        if not self._config_file.is_file():
            return
        try:
            raw = json.loads(self._config_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged = self.config.model_dump()
                for key, val in raw.items():
                    if key in ("music_folder", "lyrics_folder", "output_folder"):
                        # Restore custom folder only if the directory exists on disk
                        p = Path(val).resolve()
                        if p.exists() and p.is_dir():
                            merged[key] = p
                    elif key not in ("assets_folder", "logs_folder"):
                        merged[key] = val
                self.config = RenderConfig.model_validate(merged)
        except Exception:
            pass

    def _save_persisted_config(self) -> None:
        """Persist current tunable parameters and custom directories to `logs/studio_config.json`."""
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            self._config_file.write_text(
                self.config.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_config_response(self, message: str = "Configuration loaded.") -> ConfigResponse:
        """Return current active `RenderConfig` alongside factory defaults and detected preset."""
        with self._lock:
            matched = detect_matching_preset(self.config.model_dump())
            return ConfigResponse(
                message=message,
                config=self.config.model_copy(),
                defaults=self._defaults.model_copy(),
                matched_preset=matched,
            )

    def update_config(self, req: ConfigUpdateRequest) -> ConfigResponse:
        """Validate, apply, and persist updated configuration parameters."""
        raw_updates = req.model_dump(exclude_unset=True)
        updates: dict[str, Any] = {}

        for k, v in raw_updates.items():
            if v is not None:
                if k in ("music_folder", "lyrics_folder", "output_folder"):
                    p = Path(v).resolve()
                    if p.exists() and p.is_dir():
                        updates[k] = p
                else:
                    updates[k] = v

        with self._lock:
            self.config = self.config.model_copy(update=updates)
            self._save_persisted_config()
        return self.get_config_response(
            message=f"Saved configuration ({self.config.width}x{self.config.height} @ {self.config.fps} FPS)."
        )

    def reset_config(self) -> ConfigResponse:
        """Restore all configuration parameters back to the user's original factory defaults."""
        with self._lock:
            self.config = self._defaults.model_copy()
            try:
                if self._config_file.is_file():
                    self._config_file.unlink()
            except OSError:
                pass
        return self.get_config_response(message="Configuration restored to default values.")

    def get_presets(self) -> PresetListResponse:
        """Return all standard configuration presets alongside active matching preset."""
        with self._lock:
            active = detect_matching_preset(self.config.model_dump())
            return PresetListResponse(
                status="ok",
                active_preset=active,
                presets=list(PRESETS_CATALOG.values()),
            )

    def apply_preset(self, preset_id: str) -> ConfigResponse:
        """Apply a standard preset configuration and persist."""
        preset = get_preset_by_id(preset_id)
        if not preset:
            return self.get_config_response(message=f"Preset '{preset_id}' not found.")

        with self._lock:
            updates = preset.model_dump(exclude={"id", "name", "description"})
            self.config = self.config.model_copy(update=updates)
            self._save_persisted_config()

        return self.get_config_response(
            message=f"Applied preset '{preset.name}' ({self.config.width}x{self.config.height} @ {self.config.fps} FPS)."
        )

    def resolve_folder_path(self, target: FolderTarget) -> Path:
        """Map a logical folder key to its resolved absolute directory path."""
        mapping: dict[FolderTarget, Path] = {
            "songs": Path(self.config.music_folder).resolve(),
            "lyrics": Path(self.config.lyrics_folder).resolve(),
            "output": Path(self.config.output_folder).resolve(),
            "logs": Path(self.config.logs_folder).resolve(),
        }
        folder = mapping[target]
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def set_folder(self, target: Literal["songs", "lyrics", "output"], path: str) -> SetFolderResponse:
        """Update active workspace directory to a custom path or restore default."""
        with self._lock:
            if path == "__default__":
                default_map = {
                    "songs": self._defaults.music_folder,
                    "lyrics": self._defaults.lyrics_folder,
                    "output": self._defaults.output_folder,
                }
                resolved_default = Path(default_map[target]).resolve()
                if target == "songs":
                    self.config.music_folder = resolved_default
                elif target == "lyrics":
                    self.config.lyrics_folder = resolved_default
                elif target == "output":
                    self.config.output_folder = resolved_default

                self._save_persisted_config()
                return SetFolderResponse(
                    status="ok",
                    target=target,
                    path=str(resolved_default),
                    is_default=True,
                    message=f"Restored default folder for '{target}'.",
                )

            candidate = Path(path).resolve()
            if not candidate.exists() or not candidate.is_dir():
                return SetFolderResponse(
                    status="error",
                    target=target,
                    path=path,
                    is_default=False,
                    message=f"Directory does not exist: '{path}'.",
                )

            if target == "songs":
                self.config.music_folder = candidate
            elif target == "lyrics":
                self.config.lyrics_folder = candidate
            elif target == "output":
                self.config.output_folder = candidate

            self._save_persisted_config()
            return SetFolderResponse(
                status="ok",
                target=target,
                path=str(candidate),
                is_default=False,
                message=f"Updated '{target}' directory to '{candidate}'.",
            )

    def open_folder_in_explorer(self, target: FolderTarget) -> Path:
        """Launch the native OS file explorer focused on `target`.

        Why direct subprocess explorer.exe over os.startfile on Windows:
        When running as a background Uvicorn daemon or windowless process (pythonw.exe),
        os.startfile relies on ShellExecuteW from worker threads without an STA COM apartment
        or interactive window handle. Windows Explorer silently ignores such requests due to
        foreground lock restrictions. Spawning 'explorer.exe <path>' explicitly invokes the
        native Windows shell process, ensuring the folder window is reliably created or focused.
        """
        folder = self.resolve_folder_path(target)
        norm_path = os.path.normpath(str(folder.resolve()))
        logger = get_logger()
        logger.info("Opening '%s' folder in file explorer: %s", target, norm_path)
        if sys.platform == "win32":
            try:
                subprocess.Popen(["explorer.exe", norm_path])
            except Exception as exc:
                logger.warning("subprocess explorer.exe failed (%s); falling back to os.startfile", exc)
                if hasattr(os, "startfile"):
                    os.startfile(norm_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", norm_path])
        else:
            subprocess.Popen(["xdg-open", norm_path])
        return folder

    def _validate_output_video_path(self, filename: str) -> Path:
        """Validate that `filename` is a strict `.mp4` child of `output_folder` (prevents path traversal)."""
        clean_name = Path(filename).name
        if clean_name != filename or ".." in filename or not clean_name.lower().endswith(".mp4"):
            raise ValueError(f"Invalid video filename: '{filename}'")
        out_dir = Path(self.config.output_folder).resolve()
        candidate = (out_dir / clean_name).resolve()
        if candidate.parent != out_dir:
            raise ValueError(f"Path traversal blocked for '{filename}'")
        return candidate

    def open_video_external(self, filename: str) -> Path:
        """Open a rendered `.mp4` video in the OS default media player.

        Why cmd start over os.startfile:
        Spawning 'start \"\" \"<path>\"' via cmd.exe activates Windows registered file
        associations with an interactive process token, bypassing windowless thread restrictions.
        """
        video_path = self._validate_output_video_path(filename)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {filename}")
        norm_path = os.path.normpath(str(video_path.resolve()))
        logger = get_logger()
        logger.info("Opening video in external player: %s", norm_path)
        if sys.platform == "win32":
            try:
                subprocess.Popen(f'start "" "{norm_path}"', shell=True)
            except Exception as exc:
                logger.warning("cmd start failed (%s); falling back to os.startfile", exc)
                if hasattr(os, "startfile"):
                    os.startfile(norm_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", norm_path])
        else:
            subprocess.Popen(["xdg-open", norm_path])
        return video_path

    def list_rendered_videos(self) -> list[VideoItemModel]:
        """Scan `output_folder` and return metadata for all rendered `.mp4` files."""
        out_dir = Path(self.config.output_folder).resolve()
        if not out_dir.exists():
            return []

        items: list[VideoItemModel] = []
        for entry in out_dir.iterdir():
            if not entry.is_file() or entry.suffix.lower() != ".mp4":
                continue
            try:
                stat = entry.stat()
                mtime = stat.st_mtime
                dt_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                encoded_name = quote(entry.name)
                items.append(
                    VideoItemModel(
                        filename=entry.name,
                        title=entry.stem,
                        size_bytes=stat.st_size,
                        size_human=_format_size(stat.st_size),
                        modified_iso=dt_str,
                        modified_ts=mtime,
                        url=f"/output/{encoded_name}",
                    )
                )
            except OSError:
                continue

        items.sort(key=lambda v: (-v.modified_ts, v.filename.lower()))
        return items

    def list_available_tracks(self) -> TrackListResponse:
        """Scan the active music folder and discover tracks with matched LRCs."""
        music_dir = self.resolve_folder_path("songs")
        lyrics_dir = self.resolve_folder_path("lyrics")

        available_lrcs: list[str] = []
        if lyrics_dir.exists():
            available_lrcs = sorted(
                [p.name for p in lyrics_dir.iterdir() if p.is_file() and p.suffix.lower() == ".lrc"],
                key=lambda s: s.lower(),
            )

        audio_paths = find_music_files(music_dir)
        tracks: list[TrackItemModel] = []

        for path_str in audio_paths:
            p = Path(path_str)
            try:
                stat = p.stat()
                size_bytes = stat.st_size
                size_human = _format_size(size_bytes)
            except OSError:
                size_bytes = 0
                size_human = "0 B"

            meta = extract_track_metadata(p)
            has_cover = meta is not None and meta.cover_data is not None
            title = meta.title if meta else p.stem
            artist = meta.artist if meta else "Unknown"
            album = meta.album if meta else "Unknown"

            dur = get_audio_duration(p)
            mins = int(dur // 60)
            secs = int(dur % 60)
            duration_human = f"{mins:02d}:{secs:02d}"

            matched_lrc_path = find_lyrics_file(p, lyrics_dir)
            matched_lrc_name = Path(matched_lrc_path).name if matched_lrc_path else None
            lrc_status = "auto_matched" if matched_lrc_name else "missing"

            tracks.append(
                TrackItemModel(
                    filename=p.name,
                    stem=p.stem,
                    format=p.suffix.lower(),
                    size_bytes=size_bytes,
                    size_human=size_human,
                    duration=dur,
                    duration_human=duration_human,
                    title=title,
                    artist=artist,
                    album=album,
                    has_cover=has_cover,
                    matched_lrc=matched_lrc_name,
                    lrc_status=lrc_status,
                )
            )

        return TrackListResponse(
            status="ok",
            music_folder=str(music_dir),
            lyrics_folder=str(lyrics_dir),
            tracks=tracks,
            available_lrcs=available_lrcs,
        )

    def delete_videos(self, filenames: list[str]) -> DeleteVideosResponse:
        """Delete one or more `.mp4` files from `output_folder` safely."""
        logger = get_logger()
        deleted: list[str] = []
        failed: list[str] = []

        for name in filenames:
            try:
                target = self._validate_output_video_path(name)
                if target.is_file():
                    target.unlink()
                    deleted.append(name)
                    logger.info("Deleted rendered video: %s", name)
                else:
                    failed.append(name)
            except Exception as exc:
                logger.warning("Failed to delete video '%s': %s", name, exc)
                failed.append(name)

        remaining = len(self.list_rendered_videos())
        return DeleteVideosResponse(
            deleted=deleted,
            failed=failed,
            remaining_count=remaining,
        )

    def get_status(self) -> StudioStatusResponse:
        """Aggregate folder counts, active hardware backends, render progress, and output videos."""
        songs_dir = self.resolve_folder_path("songs")
        lyrics_dir = self.resolve_folder_path("lyrics")
        output_dir = self.resolve_folder_path("output")

        songs_list = find_music_files(songs_dir)
        lyrics_count = sum(
            1 for p in lyrics_dir.iterdir() if p.is_file() and p.suffix.lower() == ".lrc"
        )
        videos = self.list_rendered_videos()

        with self._lock:
            progress_copy = self._progress.model_copy()
            cfg_copy = self.config.model_copy()

        return StudioStatusResponse(
            songs_count=len(songs_list),
            lyrics_count=lyrics_count,
            videos_count=len(videos),
            songs_dir=str(songs_dir),
            lyrics_dir=str(lyrics_dir),
            output_dir=str(output_dir),
            active_backend=self._backend_summary,
            active_encoder=self._encoder_summary,
            render_progress=progress_copy,
            videos=videos,
            config=cfg_copy,
        )

    def start_batch_render(
        self,
        backend: BackendType | None = None,
        encoder: EncoderType | None = None,
        fps: int | None = None,
    ) -> StartRenderResponse:
        """Launch the batch video creation pipeline in a background daemon thread."""
        with self._lock:
            if self._progress.is_rendering and self._render_thread and self._render_thread.is_alive():
                return StartRenderResponse(
                    status="busy",
                    message="A video rendering job is already running in the background.",
                    total_files=self._progress.total_files,
                )

            songs = find_music_files(self.config.music_folder)
            if not songs:
                return StartRenderResponse(
                    status="empty",
                    message=f"No audio files found in '{self.config.music_folder}'. Add songs first.",
                    total_files=0,
                )

            self._stop_event.clear()
            self._progress = RenderProgressModel(
                is_rendering=True,
                mode="batch",
                current_index=0,
                total_files=len(songs),
                current_track="Initializing renderer...",
                processed=0,
                skipped=0,
                failed=0,
                percent=0.0,
                message=f"Starting batch render of {len(songs)} track(s)...",
            )
            overrides: dict[str, Any] = {}
            if backend is not None:
                overrides["backend"] = backend
            if encoder is not None:
                overrides["encoder"] = encoder
            if fps is not None:
                overrides["fps"] = fps
            run_cfg = self.config.model_copy(update=overrides) if overrides else self.config.model_copy()

        def _on_progress(info: dict[str, Any]) -> None:
            cur = int(info.get("current_index", 0))
            tot = max(1, int(info.get("total_files", 1)))
            pct = round((cur / float(tot)) * 100.0, 1)
            with self._lock:
                if info.get("backend"):
                    self._backend_summary = str(info["backend"])
                self._progress = RenderProgressModel(
                    is_rendering=True,
                    mode="batch",
                    current_index=cur,
                    total_files=tot,
                    current_track=str(info.get("current_track", "")),
                    processed=int(info.get("processed", 0)),
                    skipped=int(info.get("skipped", 0)),
                    failed=int(info.get("failed", 0)),
                    percent=min(100.0, pct),
                    message=str(info.get("message", "Rendering...")),
                )

        def _worker() -> None:
            try:
                run_batch_pipeline(
                    config=run_cfg,
                    progress_callback=_on_progress,
                    stop_event=self._stop_event,
                )
            finally:
                with self._lock:
                    self._progress.is_rendering = False
                    if self._stop_event.is_set():
                        self._progress.message = "Batch rendering stopped by user."
                    else:
                        self._progress.percent = 100.0
                        self._progress.message = (
                            f"Completed: {self._progress.processed} rendered, "
                            f"{self._progress.skipped} skipped, {self._progress.failed} failed."
                        )
                        if self._progress.processed > 0:
                            try:
                                self.open_folder_in_explorer("output")
                            except Exception as exc:
                                logger = get_logger()
                                logger.warning("Could not auto-open output directory: %s", exc)

        self._render_thread = Thread(target=_worker, daemon=True, name="BatchVideoRenderWorker")
        self._render_thread.start()

        return StartRenderResponse(
            status="started",
            message=f"Launched batch video creation for {len(songs)} track(s) ({run_cfg.width}x{run_cfg.height} @ {run_cfg.fps} FPS).",
            total_files=len(songs),
        )

    def start_single_render(self, req: SingleRenderRequest) -> SingleRenderResponse:
        """Launch isolated video rendering for a single audio track in a background thread."""
        with self._lock:
            if self._progress.is_rendering and self._render_thread and self._render_thread.is_alive():
                return SingleRenderResponse(
                    status="busy",
                    message="A rendering job is already active in the background.",
                    track=req.audio_filename,
                )

            music_dir = self.resolve_folder_path("songs")
            audio_path = (music_dir / Path(req.audio_filename).name).resolve()
            if not audio_path.is_file():
                return SingleRenderResponse(
                    status="not_found",
                    message=f"Audio file '{req.audio_filename}' not found in '{music_dir}'.",
                    track=req.audio_filename,
                )

            lyrics_dir = self.resolve_folder_path("lyrics")
            lrc_path: Path | None = None
            has_lyrics = False

            if req.lrc_filename == "__none__":
                lrc_path = None
                has_lyrics = False
            elif req.lrc_filename:
                candidate = (lyrics_dir / Path(req.lrc_filename).name).resolve()
                if candidate.is_file():
                    lrc_path = candidate
                    has_lyrics = True
                else:
                    return SingleRenderResponse(
                        status="not_found",
                        message=f"Selected lyrics file '{req.lrc_filename}' not found in '{lyrics_dir}'.",
                        track=req.audio_filename,
                    )
            else:
                auto_lrc = find_lyrics_file(audio_path, lyrics_dir)
                if auto_lrc and os.path.exists(auto_lrc):
                    lrc_path = Path(auto_lrc)
                    has_lyrics = True

            out_dir = self.resolve_folder_path("output")
            output_path = out_dir / f"{audio_path.stem}.mp4"

            overrides: dict[str, Any] = {}
            if req.preset:
                preset = get_preset_by_id(req.preset)
                if preset:
                    overrides.update(preset.model_dump(exclude={"id", "name", "description"}))
            if req.backend:
                overrides["backend"] = req.backend
            if req.encoder:
                overrides["encoder"] = req.encoder
            if req.fps:
                overrides["fps"] = req.fps

            run_cfg = self.config.model_copy(update=overrides) if overrides else self.config.model_copy()

            self._stop_event.clear()
            self._progress = RenderProgressModel(
                is_rendering=True,
                mode="single",
                current_index=1,
                total_files=1,
                current_track=audio_path.name,
                processed=0,
                skipped=0,
                failed=0,
                percent=10.0,
                message=f"Rendering '{audio_path.stem}' ({'with lyrics' if has_lyrics else 'cover art only'})...",
            )

        def _on_single_frame_progress(cur_f: int, tot_f: int, pct: float) -> None:
            with self._lock:
                self._progress.percent = pct
                self._progress.message = f"Rendering '{audio_path.stem}': {cur_f}/{tot_f} frames ({pct:.0f}%)..."

        def _single_worker() -> None:
            renderer = get_renderer(run_cfg)
            success = False
            logger = get_logger()
            try:
                success = renderer.render_video(
                    audio_file=audio_path,
                    lrc_file=lrc_path,
                    output_file=output_path,
                    progress_callback=_on_single_frame_progress,
                )
            except Exception as exc:
                logger.error("Single render failed for '%s': %s", audio_path.name, exc)
                success = False
            finally:
                renderer.cleanup()
                with self._lock:
                    self._progress.is_rendering = False
                    if success:
                        self._progress.processed = 1
                        self._progress.percent = 100.0
                        self._progress.message = f"Completed: '{output_path.name}' saved to output/."
                        try:
                            self.open_folder_in_explorer("output")
                        except Exception as exc:
                            logger.warning("Could not auto-open output directory: %s", exc)
                    else:
                        self._progress.failed = 1
                        self._progress.message = f"Rendering failed for '{audio_path.name}'. Check logs."

        self._render_thread = Thread(target=_single_worker, daemon=True, name="SingleVideoRenderWorker")
        self._render_thread.start()

        return SingleRenderResponse(
            status="started",
            message=f"Rendering '{audio_path.name}' ({run_cfg.width}x{run_cfg.height} @ {run_cfg.fps} FPS).",
            track=audio_path.name,
            has_lyrics=has_lyrics,
        )

    def stop_batch_render(self) -> StopRenderResponse:
        """Signal the background renderer to halt after the current track."""
        with self._lock:
            if not self._progress.is_rendering:
                return StopRenderResponse(status="idle", message="No active render job is running.")
            self._stop_event.set()
            self._progress.message = "Stopping after current track completes..."
            return StopRenderResponse(
                status="stopping",
                message="Stop signal sent. Pipeline will halt cleanly.",
            )


_studio_service_instance: StudioService | None = None


def get_studio_service() -> StudioService:
    """FastAPI Dependency provider returning the singleton `StudioService` instance."""
    global _studio_service_instance
    if _studio_service_instance is None:
        _studio_service_instance = StudioService(config=DEFAULT_CONFIG)
    return _studio_service_instance
