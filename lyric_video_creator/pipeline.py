"""High-level video renderer and batch processing pipeline orchestration.

Decouples track iteration, execution logging, and error recovery from low-level pixel
compositing and hardware compute backends.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import time

from lyric_video_creator.backends import (
    CPUBackend,
    ComputeBackend,
    CuPyBackend,
    create_compute_backend,
)
from lyric_video_creator.color import (
    adjust_color_brightness,
    compute_dominant_color,
    ensure_min_brightness,
    ensure_min_saturation,
)
from lyric_video_creator.compositor import lyric_vertical_multiplier, precompute_rotation_angles, prepare_scene
from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.encoder import stream_frames_to_ffmpeg
from lyric_video_creator.media import (
    extract_track_metadata,
    find_lyrics_file,
    find_music_files,
    get_audio_duration,
    parse_lyrics,
)
from lyric_video_creator.models import BGRColor
from lyric_video_creator.utils import get_logger, setup_directories


def probe_video_resolution(video_path: str | Path) -> tuple[int, int] | None:
    """Probe the width and height of an existing .mp4 video using ffprobe."""
    try:
        res = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.5,
        ).strip()
        parts = res.split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


class VideoRenderer:
    """Unified video renderer delegating frame computation to any `ComputeBackend`.

    Why unified renderer instead of separate `GPUVideoRenderer` / `CPUVideoRenderer` silos:
    Previously, all metadata extraction, color clustering, and OpenCV text layout logic lived
    exclusively inside `GPUVideoRenderer`, leaving `CPUVideoRenderer` as an empty stub.
    By injecting a `ComputeBackend` strategy, GPU (CUDA, ROCm, MPS, XPU, OpenCL) and CPU
    share 100% of the visual preparation and FFmpeg encoding pipeline.
    """

    def __init__(
        self,
        config: RenderConfig = DEFAULT_CONFIG,
        backend: ComputeBackend | None = None,
    ) -> None:
        self.config = config
        self.backend: ComputeBackend = backend or create_compute_backend(config)
        self.glow_color: BGRColor = (60, 160, 255)
        self.zoom_cache: dict[object, object] = {}

    # Legacy color helper delegations preserved for backward compatibility
    def _ensure_min_saturation(
        self, bgr: BGRColor, min_sat: int | None = None
    ) -> tuple[BGRColor, bool]:
        return ensure_min_saturation(
            bgr, min_sat=self.config.min_glow_sat if min_sat is None else min_sat
        )

    def _ensure_min_brightness(
        self, bgr: BGRColor, min_val: int | None = None
    ) -> tuple[BGRColor, bool]:
        return ensure_min_brightness(
            bgr, min_val=self.config.min_glow_val if min_val is None else min_val
        )

    def _compute_dominant_color(
        self, img_bgr: object, k: int = 3, sample: int | None = None
    ) -> BGRColor:
        cfg = (
            self.config
            if sample is None
            else self.config.model_copy(update={"sample_grid": sample})
        )
        return compute_dominant_color(img_bgr, k=k, config=cfg)  # type: ignore[arg-type]

    def _adjust_color_brightness(
        self, bgr: BGRColor, boost_v: float = 1.12, boost_s: float = 1.04
    ) -> BGRColor:
        return adjust_color_brightness(bgr, boost_v=boost_v, boost_s=boost_s)

    def _lyric_multiplier(self, offset: int) -> float:
        return lyric_vertical_multiplier(offset)

    def _precompute_rotation_angles(self, total_frames: int) -> list[float]:
        return precompute_rotation_angles(total_frames, speed=self.config.rotation_speed)

    @property
    def memory_pool(self) -> object | None:
        """Expose underlying GPU memory pool if present on the active backend."""
        return getattr(self.backend, "memory_pool", None)

    def render_video(
        self,
        audio_file: str | Path,
        lrc_file: str | Path | None,
        output_file: str | Path,
        progress_callback: object | None = None,
    ) -> bool:
        """Render a single synchronized lyric (or cover-only) music video."""
        logger = get_logger()
        audio_path = Path(audio_file)
        logger.info("Processing: %s [Backend: %s]", audio_path.name, self.backend.name)

        self.backend.release_scene()

        metadata = extract_track_metadata(audio_path)
        if metadata is None:
            logger.warning("Cover art or metadata not found for '%s'; skipping.", audio_path.name)
            return False

        has_lyrics = bool(lrc_file and os.path.exists(lrc_file))
        lyrics = parse_lyrics(lrc_file, config=self.config) if (has_lyrics and lrc_file) else []
        if not lyrics:
            has_lyrics = False

        duration = get_audio_duration(audio_path)
        if duration <= 0.0:
            logger.error("Could not determine positive duration for '%s'.", audio_path.name)
            return False

        total_frames = max(1, int(round(duration * self.config.fps)))
        logger.info("Duration: %.3fs (%d frames @ %d FPS)", duration, total_frames, self.config.fps)
        logger.info(
            "Lyrics: %s (%d segments)",
            "available" if has_lyrics else "not available",
            len(lyrics),
        )

        scene = prepare_scene(
            metadata=metadata,
            lyrics=lyrics,
            duration=duration,
            has_lyrics=has_lyrics,
            config=self.config,
        )
        if scene is None:
            return False

        self.glow_color = scene.glow_color

        try:
            self.backend.load_scene(scene)
            return stream_frames_to_ffmpeg(
                backend=self.backend,
                audio_file=audio_path,
                output_file=output_file,
                total_frames=scene.total_frames,
                config=self.config,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.error("Rendering error on '%s': %s", audio_path.name, exc)
            return False
        finally:
            self.backend.release_scene()

    def cleanup(self) -> None:
        """Release all backend device and host resources."""
        self.zoom_cache.clear()
        self.backend.cleanup()


class GPUVideoRenderer(VideoRenderer):
    """Backward-compatible GPU renderer facade that prefers CuPy, falling back to Torch/OpenCL."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        gpu_cfg = (
            config
            if config.backend in ("cupy", "torch", "opencl")
            else config.model_copy(update={"backend": "auto"})
        )
        super().__init__(config=gpu_cfg)


class CPUVideoRenderer(VideoRenderer):
    """Production CPU renderer facade using vectorized OpenCV C++ SIMD and NumPy ROI blending."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        cpu_cfg = config.model_copy(update={"backend": "cpu"})
        super().__init__(config=cpu_cfg, backend=CPUBackend(config=cpu_cfg))


def get_renderer(config: RenderConfig = DEFAULT_CONFIG) -> VideoRenderer:
    """Factory returning a configured `VideoRenderer` with automatic hardware fallback."""
    return VideoRenderer(config=config)


def run_batch_pipeline(
    config: RenderConfig = DEFAULT_CONFIG,
    progress_callback: object | None = None,
    stop_event: object | None = None,
) -> int:
    """Execute batch music video rendering across all tracks in `config.music_folder`.

    Why `progress_callback` and `stop_event`:
    Allows the FastAPI Web Studio service to observe real-time track progress and request
    clean cancellation between tracks without duplicating batch orchestration logic.
    """
    logger = get_logger()
    logger.info("Starting batch music video processing")
    setup_directories(config)

    music_files = find_music_files(config.music_folder)
    if not music_files:
        logger.error(
            "No music files found in '%s'. Supported formats: .flac, .mp3, .m4a, .ogg, .wav",
            config.music_folder,
        )
        return 1

    logger.info("Found %d music file(s):", len(music_files))
    for i, path_str in enumerate(music_files, 1):
        logger.info("  %2d. %s", i, os.path.basename(path_str))

    renderer = get_renderer(config)
    total_start = time.time()
    processed = failed = skipped = 0

    log_name = f"000 output {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logs_dir = Path(config.logs_folder)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_filename = logs_dir / log_name

    def _notify(current_idx: int, current_track: str, status_msg: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(
                    {
                        "current_index": current_idx,
                        "total_files": len(music_files),
                        "current_track": current_track,
                        "processed": processed,
                        "skipped": skipped,
                        "failed": failed,
                        "backend": renderer.backend.name,
                        "message": status_msg,
                    }
                )
            except Exception:
                pass

    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"Execution Log - {datetime.now()}\n")
        log_file.write(f"Compute Backend: {renderer.backend.name}\n")
        log_file.write("========================================\n")

        try:
            for i, audio_file in enumerate(music_files, 1):
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    logger.warning("Batch rendering stopped by user request.")
                    _notify(i, os.path.basename(audio_file), "Batch rendering cancelled by user.")
                    break

                track_name = os.path.basename(audio_file)
                logger.info("=" * 60)
                logger.info("Processing file %d/%d: %s", i, len(music_files), track_name)
                _notify(i, track_name, f"Rendering ({i}/{len(music_files)}): {track_name}")

                lrc_file = find_lyrics_file(audio_file, lyrics_folder=config.lyrics_folder)
                base = os.path.splitext(track_name)[0]
                out_path = os.path.join(str(config.output_folder), f"{base}.mp4")

                # Skip if video already exists in output folder, regardless of format, container, or resolution
                existing_video_path: str | None = None
                for ext in (".mp4", ".mkv", ".mov", ".webm", ".avi"):
                    candidate = os.path.join(str(config.output_folder), f"{base}{ext}")
                    if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                        existing_video_path = candidate
                        break

                if existing_video_path is not None:
                    logger.info("Output video already exists: %s; skipping.", existing_video_path)
                    skipped += 1
                    log_file.write(f"[ALREADY PROCESSED] {track_name}\n")
                    log_file.flush()
                    _notify(i, track_name, f"Skipped existing: {track_name}")
                    continue

                success = renderer.render_video(audio_file, lrc_file, out_path)
                if success:
                    processed += 1
                    logger.info("Video generated: %s", out_path)
                    log_file.write(f"[PROPERLY PROCESSED] {track_name}\n")
                    _notify(i, track_name, f"Completed: {track_name}")
                else:
                    failed += 1
                    logger.error("Failed to render: %s", track_name)
                    log_file.write(f"[FAILED] {track_name}\n")
                    _notify(i, track_name, f"Failed: {track_name}")

                log_file.flush()

                elapsed_total = time.time() - total_start
                avg_time = elapsed_total / i if i > 0 else 0.0
                remaining = len(music_files) - i
                eta_total = remaining * avg_time

                logger.info(
                    "Overall: %d/%d processed - success %d, skipped %d, failed %d",
                    i,
                    len(music_files),
                    processed,
                    skipped,
                    failed,
                )
                if remaining > 0:
                    logger.info("ETA remaining: %.1f minutes", eta_total / 60.0)

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user.")
        except Exception as exc:
            logger.error("Critical error during batch processing: %s", exc)
        finally:
            renderer.cleanup()

            total_elapsed = time.time() - total_start
            logger.info("=" * 60)
            logger.info("FINAL SUMMARY")
            logger.info("=" * 60)
            logger.info("Active Backend: %s", renderer.backend.name)
            logger.info("Total time: %.1f minutes", total_elapsed / 60.0)
            logger.info("Files found: %d", len(music_files))
            logger.info("Videos generated: %d", processed)
            logger.info("Already existed: %d", skipped)
            logger.info("Failed: %d", failed)
            if processed > 0:
                logger.info("Average time per video: %.1fs", total_elapsed / processed)
                logger.info("Videos written to: %s", os.path.abspath(str(config.output_folder)))
            logger.info("Processing finished.")
            _notify(len(music_files), "", "Batch processing finished.")

    return 0
