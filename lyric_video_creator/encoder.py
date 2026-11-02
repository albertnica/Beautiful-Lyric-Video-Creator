"""FFmpeg hardware encoder auto-detection and asynchronous frame-piping writer.

Solves two critical flaws in the legacy renderer:
  1. Hardcoded `h264_nvenc`: Previously crashed immediately with `BrokenPipeError` on AMD, Intel,
     Apple Silicon, or CPU-only machines. Now dynamically probes hardware encoders via a 1-frame
     test encode and falls back automatically to `libx264`.
  2. Frame-dropping race condition: Previously, a `queue.Empty` timeout on `available_buffers.get()`
     executed `continue` inside the `for frame_index in range(total_frames)` loop, permanently
     skipping that frame and failing the entire render at completion. Now uses a retry loop tied
     to writer-thread liveness so no frame index is ever skipped.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
import time
from typing import Any

from lyric_video_creator.backends.base import ComputeBackend
from lyric_video_creator.config import DEFAULT_CONFIG, EncoderType, RenderConfig
from lyric_video_creator.utils import get_logger

# Ordered candidate list: (short_key, ffmpeg_codec_name, extra_codec_args)
_ENCODER_CANDIDATES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("nvenc", "h264_nvenc", ("-preset", "p7")),
    ("amf", "h264_amf", ("-quality", "quality")),
    ("qsv", "h264_qsv", ("-preset", "veryslow")),
    ("videotoolbox", "h264_videotoolbox", ()),
    ("vaapi", "h264_vaapi", ()),
    ("libx264", "libx264", ("-preset", "fast")),
)


@lru_cache(maxsize=16)
def _probe_ffmpeg_encoder(codec_name: str) -> bool:
    """Test whether FFmpeg can actually initialize `codec_name` on the current hardware.

    Why live 1-frame probe instead of `ffmpeg -encoders`:
    Standard FFmpeg builds list `h264_nvenc`, `h264_amf`, and `h264_qsv` in `-encoders` output
    even when the underlying GPU or driver DLL is absent. Encoding a single 256x256 black frame
    to the null muxer is the only reliable check that the hardware encoder session opens cleanly.
    Note: Frame dimensions must be >= 128x128 (using 256x256) because hardware ASIC encoders like
    NVIDIA NVENC reject sub-128p initialization with `invalid param (8): Frame Dimension less than minimum`.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:r=1",
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        codec_name,
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def resolve_video_encoder(preference: EncoderType = "auto") -> tuple[str, list[str]]:
    """Resolve the best working FFmpeg H.264 encoder and its preset arguments."""
    logger = get_logger()

    if preference != "auto":
        for key, codec, extra in _ENCODER_CANDIDATES:
            if key == preference:
                if _probe_ffmpeg_encoder(codec):
                    logger.info("Using requested FFmpeg video encoder: %s", codec)
                    return codec, list(extra)
                logger.warning(
                    "Requested FFmpeg encoder '%s' (%s) failed hardware probe; falling back to auto.",
                    preference,
                    codec,
                )
                break

    for _, codec, extra in _ENCODER_CANDIDATES:
        if _probe_ffmpeg_encoder(codec):
            logger.info("Auto-selected FFmpeg video encoder: %s", codec)
            return codec, list(extra)

    logger.warning("All encoder probes failed; defaulting to 'libx264'.")
    return "libx264", ["-preset", "fast"]


def build_ffmpeg_command(
    audio_file: str | Path,
    output_file: str | Path,
    config: RenderConfig = DEFAULT_CONFIG,
    codec_override: tuple[str, list[str]] | None = None,
) -> list[str]:
    """Construct the complete FFmpeg command line for raw BGR24 stdin piping."""
    codec, extra_args = codec_override or resolve_video_encoder(config.encoder)
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{config.width}x{config.height}",
        "-r",
        str(config.fps),
        "-i",
        "-",
        "-i",
        str(audio_file),
        "-c:v",
        codec,
        *extra_args,
        "-b:v",
        config.video_bitrate,
        "-c:a",
        "aac",
        "-b:a",
        config.audio_bitrate,
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(output_file),
    ]


def stream_frames_to_ffmpeg(
    backend: ComputeBackend,
    audio_file: str | Path,
    output_file: str | Path,
    total_frames: int,
    config: RenderConfig = DEFAULT_CONFIG,
    progress_callback: object | None = None,
) -> bool:
    """Render `total_frames` via `backend` and stream them asynchronously into FFmpeg.

    Implements double/multi-buffered producer-consumer pipelining:
    - The main thread dispatches frame rendering onto `backend` and pushes `(buf_idx, sync_event)`
      into `frame_queue`.
    - The daemon writer thread waits on `sync_event` (for async CUDA DMA) and writes the raw
      host buffer memoryview into `ffmpeg.stdin`, returning `buf_idx` to `available_buffers`.
    """
    logger = get_logger()
    ffmpeg_cmd = build_ffmpeg_command(audio_file=audio_file, output_file=output_file, config=config)

    stdout_target = None if config.debug else subprocess.DEVNULL
    stderr_target = None if config.debug else subprocess.DEVNULL

    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            bufsize=10**7,
            stdout=stdout_target,
            stderr=stderr_target,
        )
    except Exception as exc:
        logger.error("Failed to launch FFmpeg process: %s", exc)
        return False

    num_buffers = len(backend.host_buffers)
    available_buffers: Queue[int] = Queue(maxsize=num_buffers)
    for idx in range(num_buffers):
        available_buffers.put(idx)

    frame_queue: Queue[tuple[int, Any] | None] = Queue(maxsize=config.queue_max)

    def _writer_loop() -> None:
        try:
            while True:
                item = frame_queue.get()
                if item is None:
                    frame_queue.task_done()
                    break

                buf_idx, sync_event = item
                if sync_event is not None:
                    try:
                        sync_event.synchronize()
                    except Exception:
                        pass

                host_arr = backend.get_host_buffer(buf_idx)
                try:
                    if ffmpeg_process.stdin is not None:
                        ffmpeg_process.stdin.write(memoryview(host_arr))
                except BrokenPipeError:
                    logger.error("FFmpeg stdin pipe broken; encoder process exited unexpectedly.")
                    frame_queue.task_done()
                    break
                finally:
                    available_buffers.put(buf_idx)

                frame_queue.task_done()
        except Exception as exc:
            logger.error("Writer thread encountered an unhandled error: %s", exc)

    writer_thread = Thread(target=_writer_loop, daemon=True, name="FFmpegFrameWriter")
    writer_thread.start()

    start_time = time.time()
    processed_count = 0

    try:
        for frame_index in range(total_frames):
            # Acquire a free host buffer without EVER skipping frame_index on timeout
            buf_idx: int | None = None
            while buf_idx is None:
                if not writer_thread.is_alive():
                    logger.error("Writer thread terminated early at frame %d.", frame_index)
                    break
                try:
                    buf_idx = available_buffers.get(timeout=1.0)
                except Empty:
                    continue

            if buf_idx is None:
                break

            sync_event = backend.render_frame_to_buffer(frame_index, buf_idx)

            # Enqueue rendered frame for FFmpeg writer thread, checking writer liveness on backpressure
            enqueued = False
            while not enqueued:
                if not writer_thread.is_alive():
                    available_buffers.put(buf_idx)
                    break
                try:
                    frame_queue.put((buf_idx, sync_event), timeout=1.0)
                    enqueued = True
                except Exception:
                    continue

            if not enqueued:
                break

            processed_count += 1

            if progress_callback is not None and (frame_index % 30 == 0 or frame_index == total_frames - 1):
                try:
                    pct = round((processed_count / float(total_frames)) * 100.0, 1)
                    progress_callback(processed_count, total_frames, pct)
                except Exception:
                    pass

            if frame_index > 0 and frame_index % 2500 == 0:
                elapsed = max(1e-6, time.time() - start_time)
                fps_rate = processed_count / elapsed
                remaining = total_frames - processed_count
                eta = remaining / fps_rate if fps_rate > 0 else 0.0
                logger.info(
                    "Progress: %d/%d frames (%.1f%%) - %.1f FPS - ETA: %.0fs",
                    processed_count,
                    total_frames,
                    (processed_count / float(total_frames)) * 100.0,
                    fps_rate,
                    eta,
                )
    finally:
        if writer_thread.is_alive():
            try:
                frame_queue.put(None, timeout=5.0)
            except Exception:
                pass
            writer_thread.join(timeout=60.0)

        try:
            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.close()
        except Exception:
            pass

        return_code = ffmpeg_process.wait()

    elapsed = max(1e-6, time.time() - start_time)
    logger.info(
        "Completed in %.2fs -> %.2f FPS",
        elapsed,
        processed_count / elapsed if processed_count > 0 else 0.0,
    )

    if processed_count < total_frames or return_code != 0:
        logger.warning(
            "Render incomplete or FFmpeg exited with non-zero status (frames=%d/%d, rc=%d).",
            processed_count,
            total_frames,
            return_code,
        )
        return False

    return True
