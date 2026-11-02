"""Cross-vendor OpenCL GPU backend via OpenCV's Transparent API (`cv2.UMat`).

Enables hardware-accelerated affine background warping on AMD Radeon, Intel Arc/Iris, and NVIDIA
GPUs without requiring CuPy or PyTorch installation.
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


def is_opencl_usable() -> bool:
    """Return True if OpenCV OpenCL runtime is available and can allocate a `cv2.UMat`."""
    try:
        if not cv2.ocl.haveOpenCL():
            return False
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return False
        dev = cv2.ocl.Device.getDefault()
        return bool(dev is not None and dev.available())
    except Exception:
        return False


class OpenCLBackend(ComputeBackend):
    """OpenCL GPU backend using `cv2.UMat` for rotating background warps and ROI compositing."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        if not is_opencl_usable():
            raise RuntimeError("OpenCV OpenCL is not available on this system.")
        super().__init__(config=config)
        cv2.ocl.setUseOpenCL(True)
        dev_name = "OpenCL Device"
        try:
            dev = cv2.ocl.Device.getDefault()
            if dev is not None and dev.name():
                dev_name = dev.name().strip()
        except Exception:
            pass
        self._device_label = f"OpenCV OpenCL ({dev_name})"
        self.bg_umat: Any = None
        self.static_rgba_sprite: ROISprite | None = None
        self.static_glow_sprite: ROISprite | None = None

    @property
    def name(self) -> str:
        return self._device_label

    def load_scene(self, scene: PreparedScene) -> None:
        """Upload the 3840x3840 background texture to an OpenCL `cv2.UMat` and build static ROI sprites."""
        self.release_scene()
        self.scene = scene
        self.bg_umat = cv2.UMat(scene.background_large)
        self.static_rgba_sprite, self.static_glow_sprite = build_combined_static_roi(
            scene, config=self.config
        )

    def render_frame_to_buffer(self, frame_index: int, buf_idx: int) -> Any:
        """Dispatch OpenCL affine background warp and composite ROI sprites in-place."""
        if (
            self.scene is None
            or self.bg_umat is None
            or self.static_rgba_sprite is None
            or self.static_glow_sprite is None
        ):
            raise RuntimeError("Scene must be loaded before rendering frames.")

        scene = self.scene
        angle = scene.rotation_angles[frame_index]
        bg_h, bg_w = scene.background_large.shape[:2]
        inv_matrix = self.compute_inverse_affine_matrix(angle, bg_w=bg_w, bg_h=bg_h)

        # 1. Hardware-accelerated OpenCL bilinear affine warp
        warped_umat = cv2.warpAffine(
            self.bg_umat,
            inv_matrix,
            (self.config.width, self.config.height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        frame = self.host_buffers[buf_idx]
        frame[:] = warped_umat.get()

        # 2. Blend static cover + metadata text + icons ROI
        blend_roi_alpha_cpu(frame, self.static_rgba_sprite)

        # 3. Blend static glow ROI
        blend_roi_add_cpu(
            frame,
            self.static_glow_sprite,
            alpha=1.0,
            reveal=1.0,
            edge=self.config.glow_edge_px,
        )

        # 4. Blend dynamic lyric ROI sprites
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
        """Release per-song OpenCL UMat buffers."""
        self.scene = None
        self.bg_umat = None
        self.static_rgba_sprite = None
        self.static_glow_sprite = None

    def cleanup(self) -> None:
        """Release all OpenCL buffers and host arrays."""
        self.release_scene()
        self.host_buffers.clear()
