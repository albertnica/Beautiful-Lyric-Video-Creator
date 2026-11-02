"""Standard configuration presets for Beautiful Lyric Video Creator Web Studio.

Provides pre-tested profiles balancing resolution, framerate, bitrates, and glow heuristics
for YouTube, Mobile Shorts/Reels, 4K Cinema, and Fast Low-Res Previews.
"""

from __future__ import annotations

from typing import Any

from lyric_video_creator.server.models import PresetDefinitionModel

PRESETS_CATALOG: dict[str, PresetDefinitionModel] = {
    "standard_1080p": PresetDefinitionModel(
        id="standard_1080p",
        name="YouTube Standard (1080p @ 20 FPS)",
        description="Standard balanced profile. High stability, smooth glow pulsation, and standard bitrates.",
        width=1920,
        height=1080,
        fps=20,
        video_bitrate="8000k",
        audio_bitrate="320k",
        left_margin=50,
        glow_blur_ksize=111,
        blur_strength=501,
        glow_intensity_base=0.5,
        glow_boost=6.0,
        rotation_speed=0.005,
    ),
    "smooth_60fps": PresetDefinitionModel(
        id="smooth_60fps",
        name="Full HD 60 FPS (Fluid Motion)",
        description="Ultra-smooth 60 frames per second with increased bitrate for high-speed dynamic lyrics.",
        width=1920,
        height=1080,
        fps=60,
        video_bitrate="14000k",
        audio_bitrate="320k",
        left_margin=50,
        glow_blur_ksize=111,
        blur_strength=501,
        glow_intensity_base=0.5,
        glow_boost=6.0,
        rotation_speed=0.002,
    ),
    "cinema_4k": PresetDefinitionModel(
        id="cinema_4k",
        name="4K Cinema Master (3840×2160 @ 30 FPS)",
        description="Ultra-high definition 4K UHD rendering with expanded text margins and high-radius Gaussian blur.",
        width=3840,
        height=2160,
        fps=30,
        video_bitrate="24000k",
        audio_bitrate="320k",
        left_margin=100,
        glow_blur_ksize=171,
        blur_strength=701,
        glow_intensity_base=0.6,
        glow_boost=7.0,
        rotation_speed=0.003,
    ),
    "reels_vertical": PresetDefinitionModel(
        id="reels_vertical",
        name="TikTok / Reels / Shorts (Vertical 9:16)",
        description="Vertical 1080×1920 layout tailored for smartphone screens and mobile social media feeds.",
        width=1080,
        height=1920,
        fps=30,
        video_bitrate="8000k",
        audio_bitrate="320k",
        left_margin=35,
        glow_blur_ksize=111,
        blur_strength=501,
        glow_intensity_base=0.55,
        glow_boost=6.0,
        rotation_speed=0.004,
    ),
    "fast_preview": PresetDefinitionModel(
        id="fast_preview",
        name="Fast Preview (720p @ 15 FPS)",
        description="Lightweight 1280×720 draft profile for instant lyric timing checks and quick verification.",
        width=1280,
        height=720,
        fps=15,
        video_bitrate="3500k",
        audio_bitrate="192k",
        left_margin=35,
        glow_blur_ksize=81,
        blur_strength=301,
        glow_intensity_base=0.45,
        glow_boost=5.0,
        rotation_speed=0.006,
    ),
    "vibrant_glow": PresetDefinitionModel(
        id="vibrant_glow",
        name="Vibrant Neon Aesthetic (1080p @ 30 FPS)",
        description="Intensified glow brightness, boosted saturation multipliers, and vivid aesthetic styling.",
        width=1920,
        height=1080,
        fps=30,
        video_bitrate="10000k",
        audio_bitrate="320k",
        left_margin=50,
        glow_blur_ksize=131,
        blur_strength=501,
        glow_intensity_base=1.2,
        glow_boost=8.5,
        rotation_speed=0.005,
    ),
}


def get_preset_by_id(preset_id: str) -> PresetDefinitionModel | None:
    """Retrieve a preset definition by its unique identifier."""
    return PRESETS_CATALOG.get(preset_id)


def detect_matching_preset(config_dict: dict[str, Any]) -> str | None:
    """Check if the given configuration values exactly match any standard preset."""
    for pid, p in PRESETS_CATALOG.items():
        if (
            config_dict.get("width") == p.width
            and config_dict.get("height") == p.height
            and config_dict.get("fps") == p.fps
            and config_dict.get("video_bitrate") == p.video_bitrate
        ):
            return pid
    return None

