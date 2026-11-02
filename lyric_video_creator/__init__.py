"""Beautiful Lyric Video Creator — Modular Multi-Backend Rendering Package.

Provides hardware-accelerated music video generation with synchronized lyrics across
NVIDIA CUDA (CuPy), multi-vendor GPUs (PyTorch CUDA/ROCm/MPS/XPU, OpenCV OpenCL),
and vectorized CPU execution.
"""

from __future__ import annotations

from lyric_video_creator.backends import (
    CUPY_AVAILABLE,
    TORCH_AVAILABLE,
    CPUBackend,
    ComputeBackend,
    CuPyBackend,
    OpenCLBackend,
    TorchBackend,
    create_compute_backend,
    detect_torch_device,
    is_cupy_usable,
    is_opencl_usable,
)
from lyric_video_creator.color import (
    adjust_color_brightness,
    compute_dominant_color,
    ensure_min_brightness,
    ensure_min_saturation,
    select_glow_color,
)
from lyric_video_creator.compositor import (
    create_background_texture,
    create_fixed_overlays,
    lyric_vertical_multiplier,
    precompute_rotation_angles,
    prepare_lyric_sprites,
    prepare_scene,
)
from lyric_video_creator.config import DEFAULT_CONFIG, BackendType, EncoderType, RenderConfig
from lyric_video_creator.encoder import (
    build_ffmpeg_command,
    resolve_video_encoder,
    stream_frames_to_ffmpeg,
)
from lyric_video_creator.media import (
    extract_metadata,
    extract_track_metadata,
    find_lyrics_file,
    find_music_files,
    get_audio_duration,
    parse_lyrics,
    quantize_lyrics_to_frames,
)
from lyric_video_creator.models import (
    BGRColor,
    LyricSegment,
    LyricTuple,
    PreparedScene,
    ROISprite,
    TrackMetadata,
)
from lyric_video_creator.pipeline import (
    CPUVideoRenderer,
    GPUVideoRenderer,
    VideoRenderer,
    get_renderer,
    run_batch_pipeline,
)
from lyric_video_creator.utils import (
    configure_logging,
    debug_print,
    get_logger,
    sanitize_text,
    setup_directories,
    strip_title_extra,
    validate_environment,
)

__version__ = "2.0.0"

__all__ = [
    "BGRColor",
    "BackendType",
    "CPUBackend",
    "CPUVideoRenderer",
    "CUPY_AVAILABLE",
    "ComputeBackend",
    "CuPyBackend",
    "DEFAULT_CONFIG",
    "EncoderType",
    "GPUVideoRenderer",
    "LyricSegment",
    "LyricTuple",
    "OpenCLBackend",
    "PreparedScene",
    "ROISprite",
    "RenderConfig",
    "TORCH_AVAILABLE",
    "TorchBackend",
    "TrackMetadata",
    "VideoRenderer",
    "__version__",
    "adjust_color_brightness",
    "build_ffmpeg_command",
    "compute_dominant_color",
    "configure_logging",
    "create_background_texture",
    "create_compute_backend",
    "create_fixed_overlays",
    "debug_print",
    "ensure_min_brightness",
    "ensure_min_saturation",
    "extract_metadata",
    "extract_track_metadata",
    "find_lyrics_file",
    "find_music_files",
    "get_audio_duration",
    "get_logger",
    "get_renderer",
    "lyric_vertical_multiplier",
    "parse_lyrics",
    "precompute_rotation_angles",
    "prepare_lyric_sprites",
    "prepare_scene",
    "quantize_lyrics_to_frames",
    "resolve_video_encoder",
    "run_batch_pipeline",
    "sanitize_text",
    "select_glow_color",
    "setup_directories",
    "stream_frames_to_ffmpeg",
    "strip_title_extra",
    "validate_environment",
]
