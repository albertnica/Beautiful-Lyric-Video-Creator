"""Vectorized CPU fallback backend powered by multi-threaded OpenCV C++ SIMD and NumPy.

Replaces the legacy non-functional `CPUVideoRenderer` stub with a real, production-ready
CPU renderer that uses C++ SIMD `cv2.warpAffine` and ROI-cropped fixed-point blending.
"""

from __future__ import annotations

from typing import Any

import cv2

from lyric_video_creator.backends.base import (
    ComputeBackend,
    blend_roi_add_cpu,
    blend_roi_alpha_cpu,
    build_combined_static_roi,
)
from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.models import PreparedScene, ROISprite


class CPUBackend(ComputeBackend):
    """Multi-threaded OpenCV C++ / NumPy SIMD compute backend requiring zero GPU hardware."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config=config)
        self.static_rgba_sprite: ROISprite | None = None
        self.static_glow_sprite: ROISprite | None = None

    @property
    def name(self) -> str:
        return "CPU (OpenCV C++ SIMD + NumPy ROI)"

    def load_scene(self, scene: PreparedScene) -> None:
        """Pre-merge static cover, text, and glow ROI layers for fast CPU compositing."""
        self.release_scene()
        self.scene = scene
        self.static_rgba_sprite, self.static_glow_sprite = build_combined_static_roi(
            scene, config=self.config
        )

    def render_frame_to_buffer(self, frame_index: int, buf_idx: int) -> Any:
        """Render `frame_index` directly into `host_buffers[buf_idx]` using OpenCV C++ and ROI slices."""
        if (
            self.scene is None
            or self.static_rgba_sprite is None
            or self.static_glow_sprite is None
        ):
            raise RuntimeError("Scene must be loaded before rendering frames.")

        scene = self.scene
        angle = scene.rotation_angles[frame_index]
        bg_h, bg_w = scene.background_large.shape[:2]
        inv_matrix = self.compute_inverse_affine_matrix(angle, bg_w=bg_w, bg_h=bg_h)

        frame = self.host_buffers[buf_idx]

        # 1. C++ SIMD bilinear inverse affine warp directly into the pre-allocated host buffer
        cv2.warpAffine(
            scene.background_large,
            inv_matrix,
            (self.config.width, self.config.height),
            dst=frame,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 2. Blend static cover + metadata text + icons within static ROI
        blend_roi_alpha_cpu(frame, self.static_rgba_sprite)

        # 3. Blend static glow within static ROI
        blend_roi_add_cpu(
            frame,
            self.static_glow_sprite,
            alpha=1.0,
            reveal=1.0,
            edge=self.config.glow_edge_px,
        )

        # 4. Blend active & neighbor lyric ROI sprites
        if scene.has_lyrics:
            keys = scene.frame_to_overlay_keys.get(frame_index, [])
            if keys:
                pulse_alpha = self.compute_lyric_pulse_alpha(frame_index)
                for key in keys:
                    text_sprite = scene.lyric_text_sprites.get(key)
                    if text_sprite is not None:
                        blend_roi_alpha_cpu(frame, text_sprite)

                    glow_sprite = scene.lyric_glow_sprites.get(key)
                    if glow_sprite is not None:
                        frame_pair = scene.key_frame_map.get(key)
                        if frame_pair is None or (frame_pair[0] <= frame_index <= frame_pair[1]):
                            blend_roi_add_cpu(
                                frame,
                                glow_sprite,
                                alpha=pulse_alpha,
                                reveal=1.0,
                                edge=self.config.glow_edge_px,
                            )

        return None

    def release_scene(self) -> None:
        """Clear references to per-song scene arrays."""
        self.scene = None
        self.static_rgba_sprite = None
        self.static_glow_sprite = None

    def cleanup(self) -> None:
        """Release all host buffers."""
        self.release_scene()
        self.host_buffers.clear()
