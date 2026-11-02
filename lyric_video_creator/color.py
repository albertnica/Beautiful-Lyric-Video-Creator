"""Dominant color extraction and HSV color-space adjustments for dynamic glow effects.

Isolates perceptual color heuristics from GPU/CPU rendering backends so every backend
produces identical color palettes for a given album cover.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.models import BGRColor
from lyric_video_creator.utils import debug_print


def ensure_min_saturation(
    bgr: BGRColor,
    min_sat: int = DEFAULT_CONFIG.min_glow_sat,
) -> tuple[BGRColor, bool]:
    """Clamp the HSV saturation channel to at least `min_sat`.

    Why necessary:
    Monochrome or desaturated album covers yield gray/white dominant colors whose additive
    glow looks muddy rather than vibrant. Enforcing a saturation floor preserves hue while
    injecting subtle chromatic tint.
    """
    try:
        arr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.int32)
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        changed = False
        if s < min_sat:
            s = min_sat
            changed = True
        new_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return (int(new_bgr[0]), int(new_bgr[1]), int(new_bgr[2])), changed
    except Exception:
        return (int(bgr[0]), int(bgr[1]), int(bgr[2])), False


def ensure_min_brightness(
    bgr: BGRColor,
    min_val: int = DEFAULT_CONFIG.min_glow_val,
) -> tuple[BGRColor, bool]:
    """Clamp the HSV brightness (Value) channel to at least `min_val`.

    Why necessary:
    Glow layers are composited via additive blending (`frame + glow * alpha`). A near-black
    dominant color contributes almost zero luminance under additive blending, making the glow
    effect invisible against dark backgrounds.
    """
    try:
        arr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.int32)
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        changed = False
        if v < min_val:
            v = min_val
            changed = True
        new_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return (int(new_bgr[0]), int(new_bgr[1]), int(new_bgr[2])), changed
    except Exception:
        return (int(bgr[0]), int(bgr[1]), int(bgr[2])), False


def adjust_color_brightness(
    bgr: BGRColor,
    boost_v: float = 1.12,
    boost_s: float = 1.04,
) -> BGRColor:
    """Apply multiplicative boosts to HSV Value and Saturation to make glow colors pop."""
    try:
        arr = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).astype(np.float32)[0, 0]
        hsv[1] = min(255.0, hsv[1] * boost_s)
        hsv[2] = min(255.0, hsv[2] * boost_v)
        new_bgr = cv2.cvtColor(np.uint8([[hsv.astype(np.uint8)]]), cv2.COLOR_HSV2BGR)[0, 0]
        return (int(new_bgr[0]), int(new_bgr[1]), int(new_bgr[2]))
    except Exception:
        return bgr


def compute_dominant_color(
    img_bgr: NDArray[np.uint8],
    k: int = 3,
    config: RenderConfig = DEFAULT_CONFIG,
) -> BGRColor:
    """Compute a representative vibrant color from an album cover image.

    Why hybrid Monte Carlo median + K-Means fallback:
    1. Downsampling to `sample_grid x sample_grid` bounds complexity to O(1) with respect to
       input image resolution (some FLAC files embed 4000x4000 artwork).
    2. Filtering out extreme luminance percentiles (`p_low_pct` to `p_high_pct`) and low
       saturation (`min_sat`) rejects black backgrounds and white typography.
    3. Vectorized random sampling followed by channel-wise median is resistant to outliers
       and significantly faster than iterative K-Means, while K-Means serves as a deterministic
       fallback when candidate pixels are sparse.
    """
    try:
        sample = config.sample_grid
        small = cv2.resize(img_bgr, (sample, sample), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].reshape(-1).astype(np.float32)
        s = hsv[..., 1].reshape(-1).astype(np.float32)

        p_low = np.percentile(v, config.p_low_pct)
        p_high = np.percentile(v, config.p_high_pct)
        mask = (v >= p_low) & (v <= p_high) & (s >= config.min_sat)

        bgr_flat = small.reshape(-1, 3)
        indices = np.nonzero(mask)[0]

        if indices.size >= config.random_candidates_min:
            rng = np.random.default_rng(seed=42)
            sample_count = min(config.random_candidates_min, indices.size)
            chosen_indices = rng.choice(indices, size=sample_count, replace=False)
            candidates = bgr_flat[chosen_indices].astype(np.float32)
            med = np.median(candidates, axis=0)
            return (int(med[0]), int(med[1]), int(med[2]))

        filtered = bgr_flat[mask]
        if filtered.shape[0] >= max(30, config.random_candidates_min // 2):
            data = filtered.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_PP_CENTERS
            try:
                _, labels, centers = cv2.kmeans(data, k, None, criteria, 8, flags)
                counts = np.bincount(labels.flatten())
                dominant = centers[int(np.argmax(counts))]
                return (int(dominant[0]), int(dominant[1]), int(dominant[2]))
            except Exception:
                mean_col = data.mean(axis=0)
                return (int(mean_col[0]), int(mean_col[1]), int(mean_col[2]))

        mean_col = bgr_flat.mean(axis=0)
        return (int(mean_col[0]), int(mean_col[1]), int(mean_col[2]))
    except Exception:
        mean_col = img_bgr.mean(axis=(0, 1))
        return (int(mean_col[0]), int(mean_col[1]), int(mean_col[2]))


def select_glow_color(
    img_bgr: NDArray[np.uint8],
    config: RenderConfig = DEFAULT_CONFIG,
) -> BGRColor:
    """Compute and normalize the final BGR glow color for an album cover."""
    dom = compute_dominant_color(img_bgr, k=3, config=config)
    dom_boosted = adjust_color_brightness(dom, boost_v=1.12, boost_s=1.04)
    sat_checked, sat_changed = ensure_min_saturation(dom_boosted, min_sat=config.min_glow_sat)
    bright_checked, bright_changed = ensure_min_brightness(sat_checked, min_val=config.min_glow_val)

    if sat_changed or bright_changed:
        debug_print(
            "Adjusted glow color for S/V minimum -> (B,G,R):",
            bright_checked,
            debug_enabled=config.debug,
        )
    else:
        debug_print("Selected glow color (B,G,R):", bright_checked, debug_enabled=config.debug)

    return bright_checked
