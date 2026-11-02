"""Compute backend registry, hardware capability probing, and automatic fallback factory.

Implements a resilient 4-tier hardware fallback chain:
  1. NVIDIA CUDA via CuPy RawKernels (`cupy`)
  2. Multi-vendor GPU via PyTorch (`torch`: CUDA, AMD ROCm, Apple MPS, Intel XPU)
  3. Cross-vendor GPU via OpenCV OpenCL (`opencl`: AMD Radeon, Intel Iris/Arc, NVIDIA)
  4. Vectorized CPU via OpenCV C++ SIMD + NumPy (`cpu`)
"""

from __future__ import annotations

from lyric_video_creator.backends.base import ComputeBackend
from lyric_video_creator.backends.cpu_backend import CPUBackend
from lyric_video_creator.backends.cupy_backend import CUPY_AVAILABLE, CuPyBackend, is_cupy_usable
from lyric_video_creator.backends.opencl_backend import OpenCLBackend, is_opencl_usable
from lyric_video_creator.backends.torch_backend import (
    TORCH_AVAILABLE,
    TorchBackend,
    detect_torch_device,
)
from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.utils import get_logger

__all__ = [
    "CUPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "CPUBackend",
    "ComputeBackend",
    "CuPyBackend",
    "OpenCLBackend",
    "TorchBackend",
    "create_compute_backend",
    "detect_torch_device",
    "is_cupy_usable",
    "is_opencl_usable",
]


def create_compute_backend(config: RenderConfig = DEFAULT_CONFIG) -> ComputeBackend:
    """Instantiate the requested `ComputeBackend`, automatically falling back if unavailable.

    Why automatic fallback chain:
    Hard-failing when CuPy or an NVIDIA GPU is absent locks out systems with AMD/Intel/Apple GPUs
    or CPU-only environments. Probing each accelerator tier in order of throughput guarantees the
    fastest available hardware is selected while always succeeding via `CPUBackend`.
    """
    logger = get_logger()
    preference = config.backend

    if preference == "cupy":
        if is_cupy_usable():
            backend = CuPyBackend(config=config)
            logger.info("Initialized requested compute backend: %s", backend.name)
            return backend
        logger.warning("Requested 'cupy' backend is unavailable; falling back to auto-detection.")

    elif preference == "torch":
        dev = detect_torch_device(allow_cpu=True)
        if dev is not None:
            backend = TorchBackend(config=config, device_str=dev)
            logger.info("Initialized requested compute backend: %s", backend.name)
            return backend
        logger.warning("Requested 'torch' backend is unavailable; falling back to auto-detection.")

    elif preference == "opencl":
        if is_opencl_usable():
            backend = OpenCLBackend(config=config)
            logger.info("Initialized requested compute backend: %s", backend.name)
            return backend
        logger.warning("Requested 'opencl' backend is unavailable; falling back to auto-detection.")

    elif preference == "cpu":
        backend = CPUBackend(config=config)
        logger.info("Initialized requested compute backend: %s", backend.name)
        return backend

    # Auto-detection cascade: CuPy CUDA -> PyTorch GPU -> OpenCV OpenCL -> CPU
    if is_cupy_usable():
        try:
            backend = CuPyBackend(config=config)
            logger.info("Auto-selected compute backend: %s", backend.name)
            return backend
        except Exception as exc:
            logger.warning("CuPy backend initialization failed (%s); trying next backend.", exc)

    torch_gpu = detect_torch_device(allow_cpu=False)
    if torch_gpu is not None:
        try:
            backend = TorchBackend(config=config, device_str=torch_gpu)
            logger.info("Auto-selected compute backend: %s", backend.name)
            return backend
        except Exception as exc:
            logger.warning("PyTorch GPU backend initialization failed (%s); trying next backend.", exc)

    if is_opencl_usable():
        try:
            backend = OpenCLBackend(config=config)
            logger.info("Auto-selected compute backend: %s", backend.name)
            return backend
        except Exception as exc:
            logger.warning("OpenCL backend initialization failed (%s); falling back to CPU.", exc)

    backend = CPUBackend(config=config)
    logger.info("Auto-selected compute backend: %s", backend.name)
    return backend
