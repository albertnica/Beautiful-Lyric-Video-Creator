"""Abstract base class and shared math helpers for video rendering compute backends.

Defines a hardware-agnostic contract so the rendering pipeline can switch seamlessly between
NVIDIA CUDA (CuPy), multi-vendor GPU tensors (PyTorch), cross-vendor OpenCL (OpenCV UMat),
and vectorized CPU execution without altering orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import cos, pi, sin
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.models import PreparedScene, ROISprite


class ComputeBackend(ABC):
    """Contract for hardware-specific frame rendering backends."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.scene: PreparedScene | None = None
        self.host_buffers: list[NDArray[np.uint8]] = [
            np.empty((config.height, config.width, 3), dtype=np.uint8)
            for _ in range(config.ping_pong_buffers)
        ]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the active compute backend and device."""

    @abstractmethod
    def load_scene(self, scene: PreparedScene) -> None:
        """Upload or prepare static and dynamic visual assets for a single track."""

    @abstractmethod
    def render_frame_to_buffer(self, frame_index: int, buf_idx: int) -> Any:
        """Render `frame_index` into `host_buffers[buf_idx]`, returning an optional sync event."""

    def get_host_buffer(self, buf_idx: int) -> NDArray[np.uint8]:
        """Return the host-accessible BGR24 frame buffer at `buf_idx`."""
        return self.host_buffers[buf_idx]

    @abstractmethod
    def release_scene(self) -> None:
        """Release per-song device buffers and caches to prevent memory leaks across batch runs."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release all device contexts, memory pools, and host buffers on shutdown."""

    def compute_lyric_pulse_alpha(self, frame_index: int) -> float:
        """Calculate the sinusoidal pulsating glow alpha for `frame_index`."""
        t = frame_index / float(self.config.fps)
        sin_val = sin(2.0 * pi * self.config.glow_pulse_freq * t)
        alpha = self.config.glow_intensity_base + (self.config.glow_pulse_amp * (sin_val - 1.0) / 2.0)
        return float(max(alpha, self.config.alpha_min))

    def compute_inverse_affine_matrix(self, angle: float, bg_w: int, bg_h: int) -> NDArray[np.float32]:
        """Compute the 2x3 inverse affine matrix matching the CUDA `sample_background` kernel.

        Why `WARP_INVERSE_MAP`:
        The CUDA kernel rotates around the output frame's bottom-right corner `(W, H)` and maps
        into the center of the blurred background `(bg_w / 2, bg_h / 2)`:
          sx = (x - W) * cos(angle) - (y - H) * sin(angle) + bg_w * 0.5
          sy = (x - W) * sin(angle) + (y - H) * cos(angle) + bg_h * 0.5
        Passing this exact 2x3 matrix to `cv2.warpAffine(..., flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)`
        produces pixel-identical bilinear background sampling on CPU and OpenCL.
        """
        cx = float(self.config.width)
        cy = float(self.config.height)
        bg_cx = float(bg_w) * 0.5
        bg_cy = float(bg_h) * 0.5
        ca = cos(angle)
        sa = sin(angle)

        m02 = bg_cx - cx * ca + cy * sa
        m12 = bg_cy - cx * sa - cy * ca
        return np.array(
            [
                [ca, -sa, m02],
                [sa, ca, m12],
            ],
            dtype=np.float32,
        )


def build_combined_static_roi(
    scene: PreparedScene,
    config: RenderConfig = DEFAULT_CONFIG,
) -> tuple[ROISprite, ROISprite]:
    """Pre-merge static cover art + fixed text RGBA and static glow layers into tight ROI sprites.

    Why pre-merge on scene load:
    1. Placing the opaque album cover into the static RGBA canvas underneath the white frame
       allows a single alpha-blend operation to render cover, border, metadata text, and icons.
    2. Because saturated uint8 addition is associative (`min(255, min(255, F + G1) + G2) == min(255, F + min(255, G1 + G2))`),
       pre-summing `fixed_text_glow * alpha_fixed` and `assets_glow * alpha_assets` once per song
       replaces two full-screen additive passes per frame with a single ROI `cv2.add`.
    """
    rx, ry, rw, rh = scene.static_roi_x, scene.static_roi_y, scene.static_roi_w, scene.static_roi_h

    # Build combined RGBA canvas where cover art sits behind fixed_text_rgba
    combined_rgba_full = np.zeros_like(scene.fixed_text_rgba)
    frame_h, frame_w = combined_rgba_full.shape[:2]
    cx, cy, cw, ch = scene.cover_x, scene.cover_y, scene.cover_width, scene.cover_height

    dst_x1 = max(0, cx)
    dst_y1 = max(0, cy)
    dst_x2 = min(frame_w, cx + cw)
    dst_y2 = min(frame_h, cy + ch)
    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        src_x1 = dst_x1 - cx
        src_y1 = dst_y1 - cy
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        combined_rgba_full[dst_y1:dst_y2, dst_x1:dst_x2, :3] = scene.cover_image[
            src_y1:src_y2, src_x1:src_x2
        ]
        combined_rgba_full[dst_y1:dst_y2, dst_x1:dst_x2, 3] = 255

    # Alpha-blend fixed_text_rgba over cover_image in the overlap region
    fg = scene.fixed_text_rgba[ry : ry + rh, rx : rx + rw]
    bg = combined_rgba_full[ry : ry + rh, rx : rx + rw]

    fg_a = fg[..., 3:4].astype(np.float32) / 255.0
    bg_a = bg[..., 3:4].astype(np.float32) / 255.0
    out_a = fg_a + bg_a * (1.0 - fg_a)
    safe_out_a = np.where(out_a > 1e-5, out_a, 1.0)
    out_rgb = (
        fg[..., :3].astype(np.float32) * fg_a
        + bg[..., :3].astype(np.float32) * bg_a * (1.0 - fg_a)
    ) / safe_out_a

    merged_roi_rgba = np.empty((rh, rw, 4), dtype=np.uint8)
    merged_roi_rgba[..., :3] = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)
    merged_roi_rgba[..., 3:4] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)

    # Pre-scale and sum fixed text glow + assets glow within static ROI
    alpha_fixed = max(config.fixed_text_glow_intensity, config.alpha_min)
    alpha_assets = max(config.assets_glow_intensity, config.alpha_min)

    g_fixed_roi = (
        scene.fixed_text_glow_bgr[ry : ry + rh, rx : rx + rw].astype(np.float32) * alpha_fixed
    ).astype(np.uint8)
    g_assets_roi = (
        scene.assets_glow_bgr[ry : ry + rh, rx : rx + rw].astype(np.float32) * alpha_assets
    ).astype(np.uint8)
    merged_glow_bgr = cv2.add(g_fixed_roi, g_assets_roi)

    static_rgba_sprite = ROISprite(
        image=np.ascontiguousarray(merged_roi_rgba),
        x=rx,
        y=ry,
        width=rw,
        height=rh,
    )
    static_glow_sprite = ROISprite(
        image=np.ascontiguousarray(merged_glow_bgr),
        x=rx,
        y=ry,
        width=rw,
        height=rh,
        text_left_x=0,
        text_width=config.width,
    )
    return static_rgba_sprite, static_glow_sprite


def blend_roi_alpha_cpu(frame: NDArray[np.uint8], sprite: ROISprite) -> None:
    """In-place alpha blend an RGBA `ROISprite` onto a BGR24 `frame` using vectorized uint16 math.

    Why uint16 fixed-point arithmetic instead of float32:
    Converting `(rh, rw, 3)` slices to float32 allocates temporary 4-byte buffers and performs
    floating-point division. Fixed-point uint16 arithmetic `(src * a + dst * (255 - a) + 127) >> 8`
    or `// 255` uses half the memory bandwidth and vectorizes cleanly in NumPy SIMD.
    """
    x, y, w, h = sprite.x, sprite.y, sprite.width, sprite.height
    dst = frame[y : y + h, x : x + w]
    rgba = sprite.image
    alpha = rgba[..., 3:4].astype(np.uint16)
    inv_alpha = (255 - alpha).astype(np.uint16)
    src_rgb = rgba[..., :3].astype(np.uint16)
    dst_rgb = dst.astype(np.uint16)
    blended = (src_rgb * alpha + dst_rgb * inv_alpha) // 255
    dst[:] = blended.astype(np.uint8)


def blend_roi_add_cpu(
    frame: NDArray[np.uint8],
    sprite: ROISprite,
    alpha: float,
    reveal: float = 1.0,
    edge: float = 24.0,
) -> None:
    """In-place additive glow blend with optional horizontal karaoke wipe using OpenCV C++ `cv2.add`."""
    x, y, w, h = sprite.x, sprite.y, sprite.width, sprite.height
    dst = frame[y : y + h, x : x + w]

    if reveal >= 0.999:
        scaled = cv2.convertScaleAbs(sprite.image, alpha=float(alpha), beta=0.0)
        cv2.add(dst, scaled, dst=dst)
        return

    # Horizontal karaoke reveal wipe across [sprite.text_left_x, sprite.text_left_x + sprite.text_width]
    cutoff = float(sprite.text_left_x) + float(reveal) * float(sprite.text_width)
    half = max(0.5, float(edge) * 0.5)
    xs = np.arange(x, x + w, dtype=np.float32)
    weights = np.clip((cutoff + half - xs) / max(1e-5, float(edge)), 0.0, 1.0)
    combined_scale = (weights * float(alpha)).reshape((1, w, 1))
    add_bgr = (sprite.image.astype(np.float32) * combined_scale).astype(np.uint8)
    cv2.add(dst, add_bgr, dst=dst)
