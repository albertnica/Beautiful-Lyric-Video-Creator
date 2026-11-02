"""NVIDIA CUDA compute backend powered by CuPy C++ RawKernels and pinned DMA transfers.

Extends the original CUDA implementation with ROI-scoped blending kernels (`blend_roi_alpha`,
`blend_roi_add`) that launch threads strictly within non-empty sprite bounding boxes, cutting
per-sprite GPU thread dispatches by >90% while preserving full-frame kernels for compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyric_video_creator.backends.base import ComputeBackend, build_combined_static_roi
from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.models import PreparedScene
from lyric_video_creator.utils import get_logger

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


_CUDA_KERNELS_SOURCE = r"""
extern "C" __global__
void sample_background(unsigned char* frame, const unsigned char* bg,
                       int bg_h, int bg_w,
                       float angle, int frame_w, int frame_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= frame_w || y >= frame_h) return;

    float cx = (float)frame_w;
    float cy = (float)frame_h;
    float dx = (float)x - cx;
    float dy = (float)y - cy;

    float ca = cosf(angle);
    float sa = sinf(angle);

    float rx = dx * ca - dy * sa;
    float ry = dx * sa + dy * ca;

    float bg_cx = (float)bg_w * 0.5f;
    float bg_cy = (float)bg_h * 0.5f;

    float sx = rx + bg_cx;
    float sy = ry + bg_cy;

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    if (x0 < 0) x0 = 0; if (x0 >= bg_w) x0 = bg_w - 1;
    if (x1 < 0) x1 = 0; if (x1 >= bg_w) x1 = bg_w - 1;
    if (y0 < 0) y0 = 0; if (y0 >= bg_h) y0 = bg_h - 1;
    if (y1 < 0) y1 = 0; if (y1 >= bg_h) y1 = bg_h - 1;

    int idx00 = (y0 * bg_w + x0) * 3;
    int idx10 = (y0 * bg_w + x1) * 3;
    int idx01 = (y1 * bg_w + x0) * 3;
    int idx11 = (y1 * bg_w + x1) * 3;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    int dst_idx = (y * frame_w + x) * 3;
    float r = w00 * (float)bg[idx00]   + w10 * (float)bg[idx10]   + w01 * (float)bg[idx01]   + w11 * (float)bg[idx11];
    float g = w00 * (float)bg[idx00+1] + w10 * (float)bg[idx10+1] + w01 * (float)bg[idx01+1] + w11 * (float)bg[idx11+1];
    float b = w00 * (float)bg[idx00+2] + w10 * (float)bg[idx10+2] + w01 * (float)bg[idx01+2] + w11 * (float)bg[idx11+2];

    frame[dst_idx]   = (unsigned char)(r + 0.5f);
    frame[dst_idx+1] = (unsigned char)(g + 0.5f);
    frame[dst_idx+2] = (unsigned char)(b + 0.5f);
}

extern "C" __global__
void render_cover(unsigned char* frame, const unsigned char* cover,
                  int cover_x, int cover_y, int cover_w, int cover_h,
                  int frame_w, int frame_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= frame_w || y >= frame_h) return;

    int dx = x - cover_x;
    int dy = y - cover_y;
    int dst_idx = (y * frame_w + x) * 3;
    if (dx >= 0 && dx < cover_w && dy >= 0 && dy < cover_h) {
        int src_idx = (dy * cover_w + dx) * 3;
        frame[dst_idx]     = cover[src_idx];
        frame[dst_idx + 1] = cover[src_idx + 1];
        frame[dst_idx + 2] = cover[src_idx + 2];
    }
}

extern "C" __global__
void blend_overlay_alpha(unsigned char* frame, const unsigned char* overlay,
                         int frame_w, int frame_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= frame_w || y >= frame_h) return;

    int idx = (y * frame_w + x) * 3;
    int o_idx = (y * frame_w + x) * 4;
    float alpha = (float)overlay[o_idx + 3] / 255.0f;
    if (alpha <= 0.0f) return;

    float r = (float)overlay[o_idx];
    float g = (float)overlay[o_idx + 1];
    float b = (float)overlay[o_idx + 2];

    float fr = (float)frame[idx];
    float fg = (float)frame[idx + 1];
    float fb = (float)frame[idx + 2];

    frame[idx]     = (unsigned char)(r * alpha + fr * (1.0f - alpha));
    frame[idx + 1] = (unsigned char)(g * alpha + fg * (1.0f - alpha));
    frame[idx + 2] = (unsigned char)(b * alpha + fb * (1.0f - alpha));
}

extern "C" __global__
void blend_overlay_add(unsigned char* frame, const unsigned char* overlay,
                       float alpha, float reveal, float edge,
                       int left_x, int text_w, int frame_w, int frame_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= frame_w || y >= frame_h) return;

    int idx = (y * frame_w + x) * 3;
    unsigned char orv = overlay[idx];
    unsigned char og  = overlay[idx + 1];
    unsigned char ob  = overlay[idx + 2];
    if (orv == 0 && og == 0 && ob == 0) return;

    float cutoff = (float)left_x + reveal * (float)text_w;
    float half = edge * 0.5f;
    float wgt;
    if ((float)x <= cutoff - half) wgt = 1.0f;
    else if ((float)x >= cutoff + half) wgt = 0.0f;
    else wgt = (cutoff + half - (float)x) / edge;

    int r = (int)frame[idx]     + (int)((float)orv * alpha * wgt);
    int g = (int)frame[idx + 1] + (int)((float)og  * alpha * wgt);
    int b = (int)frame[idx + 2] + (int)((float)ob  * alpha * wgt);

    if (r > 255) r = 255; if (g > 255) g = 255; if (b > 255) b = 255;
    frame[idx]     = (unsigned char)r;
    frame[idx + 1] = (unsigned char)g;
    frame[idx + 2] = (unsigned char)b;
}

extern "C" __global__
void blend_roi_alpha(unsigned char* frame, const unsigned char* roi_overlay,
                     int roi_x, int roi_y, int roi_w, int roi_h,
                     int frame_w, int frame_h) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= roi_w || ly >= roi_h) return;

    int fx = roi_x + lx;
    int fy = roi_y + ly;
    if (fx < 0 || fx >= frame_w || fy < 0 || fy >= frame_h) return;

    int o_idx = (ly * roi_w + lx) * 4;
    float alpha = (float)roi_overlay[o_idx + 3] / 255.0f;
    if (alpha <= 0.0f) return;

    int f_idx = (fy * frame_w + fx) * 3;
    float r = (float)roi_overlay[o_idx];
    float g = (float)roi_overlay[o_idx + 1];
    float b = (float)roi_overlay[o_idx + 2];

    float fr = (float)frame[f_idx];
    float fg = (float)frame[f_idx + 1];
    float fb = (float)frame[f_idx + 2];

    frame[f_idx]     = (unsigned char)(r * alpha + fr * (1.0f - alpha));
    frame[f_idx + 1] = (unsigned char)(g * alpha + fg * (1.0f - alpha));
    frame[f_idx + 2] = (unsigned char)(b * alpha + fb * (1.0f - alpha));
}

extern "C" __global__
void blend_roi_add(unsigned char* frame, const unsigned char* roi_glow,
                   int roi_x, int roi_y, int roi_w, int roi_h,
                   float alpha, float reveal, float edge,
                   int left_x, int text_w, int frame_w, int frame_h) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= roi_w || ly >= roi_h) return;

    int fx = roi_x + lx;
    int fy = roi_y + ly;
    if (fx < 0 || fx >= frame_w || fy < 0 || fy >= frame_h) return;

    int g_idx = (ly * roi_w + lx) * 3;
    unsigned char orv = roi_glow[g_idx];
    unsigned char og  = roi_glow[g_idx + 1];
    unsigned char ob  = roi_glow[g_idx + 2];
    if (orv == 0 && og == 0 && ob == 0) return;

    float cutoff = (float)left_x + reveal * (float)text_w;
    float half = edge * 0.5f;
    float wgt;
    if ((float)fx <= cutoff - half) wgt = 1.0f;
    else if ((float)fx >= cutoff + half) wgt = 0.0f;
    else wgt = (cutoff + half - (float)fx) / edge;

    int f_idx = (fy * frame_w + fx) * 3;
    int r = (int)frame[f_idx]     + (int)((float)orv * alpha * wgt);
    int g = (int)frame[f_idx + 1] + (int)((float)og  * alpha * wgt);
    int b = (int)frame[f_idx + 2] + (int)((float)ob  * alpha * wgt);

    if (r > 255) r = 255; if (g > 255) g = 255; if (b > 255) b = 255;
    frame[f_idx]     = (unsigned char)r;
    frame[f_idx + 1] = (unsigned char)g;
    frame[f_idx + 2] = (unsigned char)b;
}
"""


def is_cupy_usable() -> bool:
    """Return True if CuPy is installed and at least one CUDA device can execute kernels."""
    if not CUPY_AVAILABLE or cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return False
        _ = cp.zeros((1,), dtype=cp.uint8)
        return True
    except Exception:
        return False


class CuPyBackend(ComputeBackend):
    """High-throughput NVIDIA CUDA backend using CuPy RawKernels and pinned DMA memory."""

    def __init__(self, config: RenderConfig = DEFAULT_CONFIG) -> None:
        if not is_cupy_usable():
            raise RuntimeError("CuPy with an active CUDA device is required for CuPyBackend.")
        super().__init__(config=config)
        self._device_name = "NVIDIA CUDA (CuPy)"
        self._setup_gpu_env()
        self._compile_kernels()
        self._setup_buffers()

        # Device caches for current scene
        self.background_large_gpu: Any = None
        self.static_rgba_gpu: Any = None
        self.static_glow_gpu: Any = None
        self.static_roi_bounds: tuple[int, int, int, int] = (0, 0, config.width, config.height)
        self.overlay_gpu_cache: dict[tuple[int, str], tuple[Any, int, int, int, int]] = {}
        self.overlay_glow_gpu_cache: dict[tuple[int, str], tuple[Any, int, int, int, int]] = {}

    @property
    def name(self) -> str:
        return self._device_name

    def _setup_gpu_env(self) -> None:
        """Select CUDA device 0 and bind CuPy default device/pinned memory pools."""
        try:
            dev = cp.cuda.Device(0)
            dev.use()
            props = cp.cuda.runtime.getDeviceProperties(0)
            dev_name = props.get("name", b"CUDA GPU")
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode("utf-8", errors="replace")
            self._device_name = f"CUDA CuPy ({dev_name})"
        except Exception:
            pass

        try:
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
            self.memory_pool = cp.get_default_memory_pool()
            self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        except Exception:
            self.memory_pool = None
            self.pinned_memory_pool = None

    def _compile_kernels(self) -> None:
        """Compile CUDA C++ kernels for background rotation and ROI/full-frame blending."""
        opts = ("--std=c++14",)
        self.sample_background = cp.RawKernel(_CUDA_KERNELS_SOURCE, "sample_background", options=opts)
        self.render_cover = cp.RawKernel(_CUDA_KERNELS_SOURCE, "render_cover", options=opts)
        self.blend_overlay_alpha = cp.RawKernel(_CUDA_KERNELS_SOURCE, "blend_overlay_alpha", options=opts)
        self.blend_overlay_add = cp.RawKernel(_CUDA_KERNELS_SOURCE, "blend_overlay_add", options=opts)
        self.blend_roi_alpha = cp.RawKernel(_CUDA_KERNELS_SOURCE, "blend_roi_alpha", options=opts)
        self.blend_roi_add = cp.RawKernel(_CUDA_KERNELS_SOURCE, "blend_roi_add", options=opts)

    def _setup_buffers(self) -> None:
        """Allocate page-locked (pinned) host memory buffers for asynchronous Device-to-Host DMA."""
        self.ping_pong_frames: list[Any] = []
        self.host_buffers = []
        self.ping_pong_mem_ptrs: list[int] = []
        self.main_stream = cp.cuda.Stream()
        size = self.config.height * self.config.width * 3

        for _ in range(self.config.ping_pong_buffers):
            mem = cp.cuda.alloc_pinned_memory(size)
            arr = np.frombuffer(mem, dtype=np.uint8, count=size).reshape(
                self.config.height, self.config.width, 3
            )
            self.ping_pong_frames.append(mem)
            self.host_buffers.append(arr)
            try:
                ptr = int(mem.ptr)
            except Exception:
                ptr = int(arr.__array_interface__["data"][0])
            self.ping_pong_mem_ptrs.append(ptr)

        self.frame_gpu = cp.empty((self.config.height, self.config.width, 3), dtype=cp.uint8)

    def _grid_for_dims(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        bs = self.config.block_size
        bx = (width + bs - 1) // bs
        by = (height + bs - 1) // bs
        return (bx, by), (bs, bs)

    def load_scene(self, scene: PreparedScene) -> None:
        """Upload pre-composited static ROI layers and dynamic lyric ROI sprites to VRAM."""
        self.release_scene()
        self.scene = scene

        static_rgba_sprite, static_glow_sprite = build_combined_static_roi(scene, config=self.config)
        self.static_roi_bounds = (
            static_rgba_sprite.x,
            static_rgba_sprite.y,
            static_rgba_sprite.width,
            static_rgba_sprite.height,
        )

        self.background_large_gpu = cp.ascontiguousarray(cp.asarray(scene.background_large))
        self.static_rgba_gpu = cp.ascontiguousarray(cp.asarray(static_rgba_sprite.image))
        self.static_glow_gpu = cp.ascontiguousarray(cp.asarray(static_glow_sprite.image))

        for key, sprite in scene.lyric_text_sprites.items():
            self.overlay_gpu_cache[key] = (
                cp.ascontiguousarray(cp.asarray(sprite.image)),
                sprite.x,
                sprite.y,
                sprite.width,
                sprite.height,
            )

        for key, sprite in scene.lyric_glow_sprites.items():
            self.overlay_glow_gpu_cache[key] = (
                cp.ascontiguousarray(cp.asarray(sprite.image)),
                sprite.x,
                sprite.y,
                sprite.width,
                sprite.height,
            )

    def render_frame_to_buffer(self, frame_index: int, buf_idx: int) -> Any:
        """Execute CUDA kernels for `frame_index` and queue an async DMA copy into `buf_idx`."""
        if self.scene is None:
            raise RuntimeError("Scene must be loaded before rendering frames.")

        scene = self.scene
        w_frame, h_frame = self.config.width, self.config.height
        full_grid, block = self._grid_for_dims(w_frame, h_frame)
        angle = scene.rotation_angles[frame_index]
        bg_h, bg_w = scene.background_large.shape[:2]

        with self.main_stream:
            # 1. Sample rotating blurred background across full frame
            self.sample_background(
                full_grid,
                block,
                (
                    self.frame_gpu,
                    self.background_large_gpu,
                    np.int32(bg_h),
                    np.int32(bg_w),
                    np.float32(angle),
                    np.int32(w_frame),
                    np.int32(h_frame),
                ),
            )

            # 2. Blend pre-merged static cover + metadata text + icons within static ROI
            sx, sy, sw, sh = self.static_roi_bounds
            static_grid, _ = self._grid_for_dims(sw, sh)
            self.blend_roi_alpha(
                static_grid,
                block,
                (
                    self.frame_gpu,
                    self.static_rgba_gpu,
                    np.int32(sx),
                    np.int32(sy),
                    np.int32(sw),
                    np.int32(sh),
                    np.int32(w_frame),
                    np.int32(h_frame),
                ),
            )

            # 3. Blend pre-merged static glow (text + border + icons) within static ROI
            self.blend_roi_add(
                static_grid,
                block,
                (
                    self.frame_gpu,
                    self.static_glow_gpu,
                    np.int32(sx),
                    np.int32(sy),
                    np.int32(sw),
                    np.int32(sh),
                    np.float32(1.0),
                    np.float32(1.0),
                    np.float32(self.config.glow_edge_px),
                    np.int32(0),
                    np.int32(w_frame),
                    np.int32(w_frame),
                    np.int32(h_frame),
                ),
            )

            # 4. Blend active & neighbor lyric ROI sprites
            if scene.has_lyrics:
                keys = scene.frame_to_overlay_keys.get(frame_index, [])
                if keys:
                    alpha32 = np.float32(self.compute_lyric_pulse_alpha(frame_index))
                    edge32 = np.float32(self.config.glow_edge_px)
                    reveal32 = np.float32(1.0)

                    for key in keys:
                        text_entry = self.overlay_gpu_cache.get(key)
                        if text_entry is not None:
                            gpu_img, rx, ry, rw, rh = text_entry
                            roi_grid, _ = self._grid_for_dims(rw, rh)
                            self.blend_roi_alpha(
                                roi_grid,
                                block,
                                (
                                    self.frame_gpu,
                                    gpu_img,
                                    np.int32(rx),
                                    np.int32(ry),
                                    np.int32(rw),
                                    np.int32(rh),
                                    np.int32(w_frame),
                                    np.int32(h_frame),
                                ),
                            )

                        glow_entry = self.overlay_glow_gpu_cache.get(key)
                        if glow_entry is not None:
                            glow_img, gx, gy, gw, gh = glow_entry
                            roi_grid, _ = self._grid_for_dims(gw, gh)
                            frame_pair = scene.key_frame_map.get(key)
                            if frame_pair is None:
                                self.blend_roi_add(
                                    roi_grid,
                                    block,
                                    (
                                        self.frame_gpu,
                                        glow_img,
                                        np.int32(gx),
                                        np.int32(gy),
                                        np.int32(gw),
                                        np.int32(gh),
                                        alpha32,
                                        reveal32,
                                        edge32,
                                        np.int32(0),
                                        np.int32(w_frame),
                                        np.int32(w_frame),
                                        np.int32(h_frame),
                                    ),
                                )
                            else:
                                s_f, e_f, left_x, text_w = frame_pair
                                if s_f <= frame_index <= e_f:
                                    self.blend_roi_add(
                                        roi_grid,
                                        block,
                                        (
                                            self.frame_gpu,
                                            glow_img,
                                            np.int32(gx),
                                            np.int32(gy),
                                            np.int32(gw),
                                            np.int32(gh),
                                            alpha32,
                                            reveal32,
                                            edge32,
                                            np.int32(left_x),
                                            np.int32(text_w),
                                            np.int32(w_frame),
                                            np.int32(h_frame),
                                        ),
                                    )

            # 5. Asynchronous DMA copy from VRAM to pinned host buffer
            size_bytes = h_frame * w_frame * 3
            src_ptr = int(self.frame_gpu.data.ptr)
            dst_ptr = int(self.ping_pong_mem_ptrs[buf_idx])
            try:
                cp.cuda.runtime.memcpyAsync(
                    dst_ptr,
                    src_ptr,
                    size_bytes,
                    cp.cuda.runtime.cudaMemcpyDeviceToHost,
                    self.main_stream.ptr,
                )
                event = cp.cuda.Event()
                self.main_stream.record_event(event)
                return event
            except Exception:
                cp.asnumpy(self.frame_gpu, out=self.host_buffers[buf_idx])
                return None

    def release_scene(self) -> None:
        """Free per-song GPU arrays and return blocks to the CuPy memory pool."""
        self.scene = None
        self.overlay_gpu_cache.clear()
        self.overlay_glow_gpu_cache.clear()
        self.background_large_gpu = None
        self.static_rgba_gpu = None
        self.static_glow_gpu = None
        if getattr(self, "memory_pool", None) is not None:
            try:
                self.memory_pool.free_all_blocks()
            except Exception:
                pass

    def cleanup(self) -> None:
        """Release all CUDA arrays, compiled kernels, streams, and pinned memory pools."""
        self.release_scene()
        self.ping_pong_frames.clear()
        self.host_buffers.clear()
        self.ping_pong_mem_ptrs.clear()
        self.frame_gpu = None
        self.sample_background = None
        self.render_cover = None
        self.blend_overlay_alpha = None
        self.blend_overlay_add = None
        self.blend_roi_alpha = None
        self.blend_roi_add = None
        try:
            if getattr(self, "memory_pool", None) is not None:
                self.memory_pool.free_all_blocks()
            if getattr(self, "pinned_memory_pool", None) is not None:
                self.pinned_memory_pool.free_all_blocks()
        except Exception:
            pass
