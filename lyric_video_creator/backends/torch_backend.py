"""Multi-vendor GPU compute backend powered by PyTorch.

Supports NVIDIA CUDA (when CuPy is not installed), AMD ROCm (HIP), Apple Silicon Metal (MPS),
Intel Arc/Data Center GPUs (XPU), and DirectML devices using hardware-accelerated tensor operations.
"""

from __future__ import annotations

from math import cos, sin
from typing import Any

import numpy as np

from lyric_video_creator.backends.base import ComputeBackend, build_combined_static_roi
from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.models import PreparedScene

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def detect_torch_device(allow_cpu: bool = False) -> str | None:
    """Detect the best available PyTorch accelerator device (`cuda`, `mps`, `xpu`, or `cpu`)."""
    if not TORCH_AVAILABLE or torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if allow_cpu:
            return "cpu"
    except Exception:
        pass
    return None


class TorchBackend(ComputeBackend):
    """Hardware-accelerated tensor backend for CUDA, AMD ROCm, Apple MPS, and Intel XPU GPUs."""

    def __init__(
        self,
        config: RenderConfig = DEFAULT_CONFIG,
        device_str: str | None = None,
    ) -> None:
        if not TORCH_AVAILABLE or torch is None:
            raise RuntimeError("PyTorch is required for TorchBackend.")
        resolved = device_str or detect_torch_device(allow_cpu=True)
        if resolved is None:
            raise RuntimeError("No usable PyTorch device detected.")
        super().__init__(config=config)
        self.device = torch.device(resolved)
        self._device_label = f"PyTorch ({self.device.type.upper()})"
        try:
            if self.device.type == "cuda":
                dev_name = torch.cuda.get_device_name(0)
                self._device_label = f"PyTorch CUDA/ROCm ({dev_name})"
        except Exception:
            pass

        # Precompute 1D coordinate offsets relative to bottom-right rotation center (W, H)
        w_frame, h_frame = config.width, config.height
        dx = torch.arange(w_frame, dtype=torch.float32, device=self.device) - float(w_frame)
        dy = torch.arange(h_frame, dtype=torch.float32, device=self.device) - float(h_frame)
        self.dx_row = dx.view(1, w_frame)  # Shape: (1, W)
        self.dy_col = dy.view(h_frame, 1)  # Shape: (H, 1)

        self.bg_nchw: Any = None
        self.frame_gpu: Any = None
        self.static_rgb_gpu: Any = None
        self.static_alpha_gpu: Any = None
        self.static_glow_gpu: Any = None
        self.static_roi_bounds: tuple[int, int, int, int] = (0, 0, w_frame, h_frame)
        self.lyric_text_gpu: dict[tuple[int, str], tuple[Any, Any, int, int, int, int]] = {}
        self.lyric_glow_gpu: dict[tuple[int, str], tuple[Any, int, int, int, int]] = {}

    @property
    def name(self) -> str:
        return self._device_label

    def load_scene(self, scene: PreparedScene) -> None:
        """Upload background texture, static ROI layers, and lyric ROI sprites as GPU tensors."""
        self.release_scene()
        self.scene = scene

        with torch.no_grad():
            # Upload background as NCHW float32 tensor for hardware bilinear grid_sample
            bg_t = torch.from_numpy(scene.background_large).to(self.device, non_blocking=True)
            self.bg_nchw = bg_t.permute(2, 0, 1).unsqueeze(0).to(torch.float32)

            static_rgba_sprite, static_glow_sprite = build_combined_static_roi(
                scene, config=self.config
            )
            self.static_roi_bounds = (
                static_rgba_sprite.x,
                static_rgba_sprite.y,
                static_rgba_sprite.width,
                static_rgba_sprite.height,
            )

            s_rgba = torch.from_numpy(static_rgba_sprite.image).to(self.device, dtype=torch.float32)
            self.static_rgb_gpu = s_rgba[..., :3]
            self.static_alpha_gpu = s_rgba[..., 3:4] / 255.0
            self.static_glow_gpu = torch.from_numpy(static_glow_sprite.image).to(
                self.device, dtype=torch.float32
            )

            for key, sprite in scene.lyric_text_sprites.items():
                t_rgba = torch.from_numpy(sprite.image).to(self.device, dtype=torch.float32)
                self.lyric_text_gpu[key] = (
                    t_rgba[..., :3],
                    t_rgba[..., 3:4] / 255.0,
                    sprite.x,
                    sprite.y,
                    sprite.width,
                    sprite.height,
                )

            for key, sprite in scene.lyric_glow_sprites.items():
                t_glow = torch.from_numpy(sprite.image).to(self.device, dtype=torch.float32)
                self.lyric_glow_gpu[key] = (
                    t_glow,
                    sprite.x,
                    sprite.y,
                    sprite.width,
                    sprite.height,
                )

    def render_frame_to_buffer(self, frame_index: int, buf_idx: int) -> Any:
        """Render `frame_index` on the PyTorch device and copy into `host_buffers[buf_idx]`."""
        if self.scene is None:
            raise RuntimeError("Scene must be loaded before rendering frames.")

        scene = self.scene
        angle = scene.rotation_angles[frame_index]
        bg_h, bg_w = scene.background_large.shape[:2]

        with torch.no_grad():
            ca = float(cos(angle))
            sa = float(sin(angle))
            bg_cx = float(bg_w) * 0.5
            bg_cy = float(bg_h) * 0.5

            # Broadcast 1D vectors (1, W) and (H, 1) to form (H, W) normalized sample grid [-1, 1]
            sx = (self.dx_row * ca - self.dy_col * sa) + bg_cx
            sy = (self.dx_row * sa + self.dy_col * ca) + bg_cy
            u = sx * (2.0 / float(bg_w - 1)) - 1.0
            v = sy * (2.0 / float(bg_h - 1)) - 1.0
            grid = torch.stack((u, v), dim=-1).unsqueeze(0)

            sampled = F.grid_sample(
                self.bg_nchw,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            # Convert (1, 3, H, W) -> (H, W, 3) float32 in [0, 255]
            frame_f = sampled.squeeze(0).permute(1, 2, 0).contiguous()

            # 2. Blend static cover + fixed text ROI
            sx0, sy0, sw, sh = self.static_roi_bounds
            roi_slice = frame_f[sy0 : sy0 + sh, sx0 : sx0 + sw]
            roi_slice.mul_(1.0 - self.static_alpha_gpu).add_(
                self.static_rgb_gpu * self.static_alpha_gpu
            )

            # 3. Additive blend static glow ROI
            roi_slice.add_(self.static_glow_gpu).clamp_(0.0, 255.0)

            # 4. Blend dynamic lyric ROI sprites
            if scene.has_lyrics:
                keys = scene.frame_to_overlay_keys.get(frame_index, [])
                if keys:
                    pulse_alpha = self.compute_lyric_pulse_alpha(frame_index)
                    for key in keys:
                        t_entry = self.lyric_text_gpu.get(key)
                        if t_entry is not None:
                            t_rgb, t_a, rx, ry, rw, rh = t_entry
                            sub = frame_f[ry : ry + rh, rx : rx + rw]
                            sub.mul_(1.0 - t_a).add_(t_rgb * t_a)

                        g_entry = self.lyric_glow_gpu.get(key)
                        if g_entry is not None:
                            g_rgb, gx, gy, gw, gh = g_entry
                            frame_pair = scene.key_frame_map.get(key)
                            if frame_pair is None or (frame_pair[0] <= frame_index <= frame_pair[1]):
                                sub_g = frame_f[gy : gy + gh, gx : gx + gw]
                                sub_g.add_(g_rgb * pulse_alpha).clamp_(0.0, 255.0)

            frame_u8 = frame_f.clamp_(0.0, 255.0).to(torch.uint8)
            self.host_buffers[buf_idx][:] = frame_u8.cpu().numpy()

        return None

    def release_scene(self) -> None:
        """Release per-song PyTorch GPU tensors and empty allocator cache."""
        self.scene = None
        self.bg_nchw = None
        self.static_rgb_gpu = None
        self.static_alpha_gpu = None
        self.static_glow_gpu = None
        self.lyric_text_gpu.clear()
        self.lyric_glow_gpu.clear()
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Release all PyTorch tensors and host buffers."""
        self.release_scene()
        self.dx_row = None
        self.dy_col = None
        self.host_buffers.clear()
