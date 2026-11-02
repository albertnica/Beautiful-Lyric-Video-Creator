"""Configuration management and Pydantic V2 validation schemas.

Centralizes all visual, performance, and hardware parameters to prevent magic numbers
across rendering backends and ensure invalid kernel sizes or dimensions fail fast at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

type BackendType = Literal["auto", "cupy", "torch", "opencl", "cpu"]
type EncoderType = Literal["auto", "nvenc", "amf", "qsv", "videotoolbox", "vaapi", "libx264"]


class RenderConfig(BaseModel):
    """Validated configuration schema controlling video rendering, visual styling, and hardware selection.

    Why Pydantic V2:
    OpenCV Gaussian blur kernels strictly require positive odd integers, and FFmpeg requires positive
    even frame dimensions for YUV420p encoding. Validating these constraints upfront prevents obscure
    mid-render C++/CUDA crashes.
    """

    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # Directory paths
    music_folder: Path = Field(default=Path("songs"), description="Directory containing input audio files.")
    lyrics_folder: Path = Field(default=Path("lyrics"), description="Directory containing synchronized .lrc files.")
    output_folder: Path = Field(default=Path("output"), description="Directory where rendered .mp4 files are written.")
    assets_folder: Path = Field(default=Path("assets"), description="Directory containing icon PNG assets.")
    logs_folder: Path = Field(default=Path("logs"), description="Directory storing execution and Web Studio logs.")

    # Video & encoding parameters
    width: int = Field(default=1920, ge=320, le=7680, description="Output video width in pixels.")
    height: int = Field(default=1080, ge=240, le=4320, description="Output video height in pixels.")
    fps: int = Field(default=20, ge=1, le=240, description="Target frames per second.")
    video_bitrate: str = Field(default="8000k", pattern=r"^\d+[kKmM]$", description="Target FFmpeg video bitrate.")
    audio_bitrate: str = Field(default="320k", pattern=r"^\d+[kKmM]$", description="Target FFmpeg audio bitrate.")

    # Layout & pipeline tuning
    left_margin: int = Field(default=50, ge=0, description="Base horizontal margin in pixels.")
    block_size: int = Field(default=32, ge=8, le=32, description="CUDA 2D thread block dimension (block_size x block_size).")
    ping_pong_buffers: int = Field(default=12, ge=2, le=64, description="Number of host buffers for async GPU-to-CPU DMA.")
    queue_max: int = Field(default=400, ge=4, le=2000, description="Maximum queued frames before applying backpressure.")
    max_lyrics_length: int = Field(default=33, ge=10, le=120, description="Maximum characters per lyric line before splitting.")

    # Glow & visual aesthetics
    glow_blur_ksize: int = Field(default=111, ge=3, description="Gaussian kernel size for text/icon glow maps (must be odd).")
    glow_intensity_base: float = Field(default=0.5, ge=0.0, le=2.0, description="Base alpha multiplier for active lyric glow.")
    blur_strength: int = Field(default=501, ge=3, description="Gaussian kernel size for abstract background blur (must be odd).")
    fixed_text_glow_intensity: float = Field(default=0.5, ge=0.0, le=2.0, description="Glow intensity for static metadata and border.")
    assets_glow_intensity: float = Field(default=0.5, ge=0.0, le=2.0, description="Glow intensity for static PNG icons.")
    glow_pulse_amp: float = Field(default=0.04, ge=0.0, le=1.0, description="Amplitude of sinusoidal glow pulsation.")
    glow_pulse_freq: float = Field(default=0.5, ge=0.0, le=10.0, description="Frequency (Hz) of sinusoidal glow pulsation.")
    glow_edge_px: float = Field(default=24.0, ge=1.0, description="Feather width in pixels for horizontal glow wipe.")
    glow_boost: float = Field(default=6.0, ge=0.1, le=20.0, description="Brightness multiplier applied when colorizing glow masks.")
    alpha_min: float = Field(default=0.32, ge=0.0, le=1.0, description="Minimum glow alpha floor to preserve visibility.")
    default_last_line_duration: float = Field(default=4.0, gt=0.1, description="Fallback duration in seconds for the final lyric line.")
    background_size: int = Field(default=3840, ge=1024, description="Square resolution of the pre-blurred rotating background texture.")
    rotation_speed: float = Field(default=0.005, description="Background rotation increment in radians per frame.")

    # Dominant color extraction heuristics
    p_low_pct: float = Field(default=10.0, ge=0.0, le=49.0, description="Lower brightness percentile cutoff to exclude shadows.")
    p_high_pct: float = Field(default=90.0, ge=51.0, le=100.0, description="Upper brightness percentile cutoff to exclude highlights.")
    min_sat: int = Field(default=30, ge=0, le=255, description="Minimum HSV saturation required for candidate pixels.")
    random_sample_tries: int = Field(default=1600, ge=50, description="Maximum Monte Carlo sampling iterations.")
    random_candidates_min: int = Field(default=40, ge=5, description="Minimum valid samples before falling back to K-Means.")
    sample_grid: int = Field(default=120, ge=16, le=512, description="Downsampled cover art dimension for fast color analysis.")
    min_glow_sat: int = Field(default=20, ge=0, le=255, description="Minimum enforced HSV saturation for the final glow color.")
    min_glow_val: int = Field(default=80, ge=0, le=255, description="Minimum enforced HSV brightness (Value) for the final glow color.")

    # Hardware & diagnostic controls
    backend: BackendType = Field(default="auto", description="Compute backend selection ('auto' probes CuPy -> Torch -> OpenCL -> CPU).")
    encoder: EncoderType = Field(default="auto", description="FFmpeg H.264 encoder selection ('auto' probes hardware encoders -> libx264).")
    debug: bool = Field(default=False, description="Enable verbose diagnostic logging and FFmpeg stderr output.")

    @field_validator("glow_blur_ksize", "blur_strength")
    @classmethod
    def _ensure_odd_kernel(cls, value: int) -> int:
        """Ensure Gaussian blur kernel sizes are always odd integers as required by OpenCV."""
        return value if value % 2 == 1 else value + 1

    @field_validator("width", "height")
    @classmethod
    def _ensure_even_dimension(cls, value: int) -> int:
        """Ensure frame dimensions are divisible by 2 as required by YUV420p H.264 encoders."""
        return value if value % 2 == 0 else value + 1


# Default singleton configuration instance for module-level compatibility
DEFAULT_CONFIG = RenderConfig()
