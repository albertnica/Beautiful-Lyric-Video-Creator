"""Utility functions for text normalization, logging configuration, and system diagnostics.

Centralizes Unicode homoglyph sanitization and title cleanup to guarantee consistent
font rendering in OpenCV Hershey fonts, which lack full Unicode glyph coverage.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig

LOGGER_NAME = "lyric_video_creator"

# Compiled regex patterns to avoid repeated compilation overhead in tight loops
_INVISIBLE_CHARS_RE = re.compile(r"[\u200B-\u200F\uFEFF\u2060]")
_TITLE_EXTRA_RE = re.compile(r"\(|\[|\bfeat\b|\bft\.?", flags=re.IGNORECASE)

# Mapping of Cyrillic and special Latin characters/punctuation to ASCII equivalents
# supported by OpenCV's built-in Hershey vector fonts.
_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",
    "\u0410": "A",
    "\u0435": "e",
    "\u0415": "E",
    "\u0454": "e",
    "\u043E": "o",
    "\u041E": "O",
    "\u0441": "c",
    "\u0421": "C",
    "\u0440": "p",
    "\u0420": "P",
    "\u0443": "y",
    "\u0423": "Y",
    "\u0445": "x",
    "\u0425": "X",
    "\u0456": "i",
    "\u0401": "E",
    "\u00F8": "o",
    "\u00D8": "O",
    "\u00E6": "ae",
    "\u00C6": "AE",
    "\u00E5": "a",
    "\u00C5": "A",
    "\u2014": "-",
    "\u2013": "-",
    "\u201C": '"',
    "\u201D": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2026": "...",
    "\u00AB": '"',
    "\u00BB": '"',
    "\u2039": "'",
    "\u203A": "'",
    "\u00D7": "x",
    "\u2022": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2015": "-",
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)


def configure_logging(debug: bool = False) -> logging.Logger:
    """Configure and return the package-wide structured logger.

    Why structured logging over raw print():
    Allows runtime filtering by severity (DEBUG vs INFO/WARNING/ERROR) and consistent
    timestamp/level formatting across CLI execution and automated test suites.
    """
    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        stream = sys.stdout if sys.stdout is not None else sys.stderr
        if stream is not None:
            handler = logging.StreamHandler(stream)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            try:
                logs_dir = Path(DEFAULT_CONFIG.logs_folder)
                logs_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(logs_dir / "studio_server.log", encoding="utf-8")
                file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                logger.addHandler(file_handler)
            except Exception:
                logger.addHandler(logging.NullHandler())
    else:
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger


def get_logger() -> logging.Logger:
    """Return the active package logger, initializing default handlers if needed."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return configure_logging(debug=DEFAULT_CONFIG.debug)
    return logger


def debug_print(*args: object, debug_enabled: bool | None = None) -> None:
    """Emit a debug message when debug mode is active.

    Maintained for backward compatibility with legacy callers while delegating to logging.
    """
    is_debug = DEFAULT_CONFIG.debug if debug_enabled is None else debug_enabled
    if is_debug:
        msg = " ".join(str(a) for a in args)
        get_logger().debug(msg)


def sanitize_text(text: str | None) -> str | None:
    """Normalize and sanitize text for deterministic OpenCV Hershey font rendering.

    Why necessary:
    OpenCV's `cv2.putText` only supports a subset of ASCII characters and renders unsupported
    codepoints (such as curly quotes, em-dashes, zero-width joiners, or Cyrillic lookalikes)
    as question marks (`?`) or corrupted glyphs. Using `str.translate` performs the replacement
    in a single C-level pass rather than iterating over keys in Python.
    """
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = _INVISIBLE_CHARS_RE.sub("", normalized)
    return cleaned.translate(_HOMOGLYPH_TABLE)


def strip_title_extra(title: str | None) -> str | None:
    """Remove trailing parenthetical or featuring credits from a track or album title.

    Why necessary:
    Streaming metadata frequently appends verbose suffixes like `(feat. Artist Name) [Club Remix]`
    that overflow the fixed horizontal layout beneath the album cover. Truncating at the first
    qualifier keeps the visual hierarchy clean and centered.
    """
    if not title:
        return title
    parts = _TITLE_EXTRA_RE.split(title)
    cleaned = parts[0].strip()
    return cleaned if cleaned else title.strip()


def setup_directories(config: RenderConfig = DEFAULT_CONFIG) -> None:
    """Ensure required input, output, asset, and log directories exist on disk."""
    logger = get_logger()
    for folder in (
        config.music_folder,
        config.output_folder,
        config.assets_folder,
        config.lyrics_folder,
        config.logs_folder,
    ):
        path = Path(folder)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            if path == Path(config.music_folder):
                logger.info("Created input folder: %s", path)
                logger.info("Place your audio files (.flac, .mp3, .m4a, .ogg, .wav) inside '%s'.", path)


def validate_environment() -> bool:
    """Verify that external binaries (FFmpeg/FFprobe) and core Python dependencies are installed.

    Why CPU/Multi-GPU tolerant:
    Unlike the legacy validator that treated missing CuPy as a degraded error state, this check
    requires only FFmpeg, FFprobe, Mutagen, and OpenCV, logging available GPU accelerators
    informatively since automatic fallback to PyTorch, OpenCL, or CPU is fully supported.
    """
    logger = get_logger()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        try:
            subprocess.check_call(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.check_call(
                ["ffprobe", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.error("FFmpeg/FFprobe not found. Please install FFmpeg and add it to your system PATH.")
            return False

    try:
        importlib.import_module("mutagen")
    except ImportError:
        logger.error("Mutagen is not installed. Run: pip install mutagen")
        return False

    try:
        importlib.import_module("cv2")
    except ImportError:
        logger.error("OpenCV is not installed. Run: pip install opencv-python")
        return False

    return True
