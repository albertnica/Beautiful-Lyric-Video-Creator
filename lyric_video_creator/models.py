"""Strongly-typed domain models for audio metadata, synchronized lyrics, and visual sprites.

Separating data structures from rendering logic allows CPU pre-compositing and GPU/CPU
compute backends to communicate through immutable, self-documenting contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

type BGRColor = tuple[int, int, int]
type LyricTuple = tuple[float, float, str]


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Extracted metadata and raw cover artwork for an audio file.

    Why frozen & slots:
    Metadata is read-only once extracted and passed across pipeline stages;
    slots minimize memory overhead during large batch runs.
    """

    cover_data: bytes
    title: str
    artist: str
    album: str
    date: str


@dataclass(frozen=True, slots=True)
class LyricSegment:
    """Time-aligned lyric segment in both seconds and discrete frame indices."""

    start_time: float
    end_time: float
    text: str
    start_frame: int = 0
    end_frame: int = 0


@dataclass(frozen=True, slots=True)
class ROISprite:
    """Cropped Region-of-Interest (ROI) image buffer with global frame coordinates.

    Why ROI cropping instead of full-frame (1920x1080) overlays:
    A single line of lyric text or its blurred glow occupies less than 8% of a 1080p frame.
    Storing and blending only the non-zero bounding box reduces VRAM/RAM consumption by >92%
    and accelerates per-frame pixel blending by an order of magnitude across both GPU and CPU backends.
    """

    image: NDArray[np.uint8]  # Shape: (h, w, 4) for RGBA alpha blend or (h, w, 3) for BGR additive glow
    x: int
    y: int
    width: int
    height: int
    # Horizontal text bounds in full-frame coordinates (used for karaoke glow wipe calculation)
    text_left_x: int = 0
    text_width: int = 0


@dataclass(slots=True)
class PreparedScene:
    """Complete pre-computed visual state for a single song prior to frame-by-frame rendering.

    Why pre-composite static layers:
    The album cover, white frame, static metadata text (Title/Artist/Album/Year), PNG icons,
    and their associated static glows never change positions or opacity during a song.
    Combining them into unified static layers once per song avoids executing 4 redundant
    full-screen blending passes on every rendered frame.
    """

    background_large: NDArray[np.uint8]  # Shape: (bg_h, bg_w, 3) BGR
    cover_image: NDArray[np.uint8]  # Shape: (cover_h, cover_w, 3) BGR
    cover_x: int
    cover_y: int
    cover_width: int
    cover_height: int

    # Full-frame static overlays (retained for backward compatibility & direct kernel execution)
    fixed_text_rgba: NDArray[np.uint8]  # Shape: (H, W, 4) straight RGBA
    fixed_text_glow_bgr: NDArray[np.uint8]  # Shape: (H, W, 3) BGR
    assets_glow_bgr: NDArray[np.uint8]  # Shape: (H, W, 3) BGR

    # Pre-calculated bounding box covering all static elements (cover + metadata + static glows)
    static_roi_x: int
    static_roi_y: int
    static_roi_w: int
    static_roi_h: int

    glow_color: BGRColor
    has_lyrics: bool
    total_frames: int
    duration: float
    rotation_angles: list[float]

    # Dynamic lyric caches keyed by (lyric_index, role_tag) e.g. (0, 'active') or (1, 'near_1')
    lyric_text_sprites: dict[tuple[int, str], ROISprite] = field(default_factory=dict)
    lyric_glow_sprites: dict[tuple[int, str], ROISprite] = field(default_factory=dict)
    frame_to_overlay_keys: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    key_frame_map: dict[tuple[int, str], tuple[int, int, int, int]] = field(default_factory=dict)
