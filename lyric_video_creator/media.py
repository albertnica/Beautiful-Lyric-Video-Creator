"""Audio discovery, multi-format metadata/artwork extraction, and LRC lyric parsing.

Fixes the legacy limitation where non-FLAC files (.mp3, .m4a, .ogg, .wav) were discovered
by `find_music_files()` but immediately rejected by `extract_metadata()`.
"""

from __future__ import annotations

import base64
import glob
from math import isinf
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import mutagen
from mutagen.flac import FLAC, Picture
from mutagen.id3 import ID3
from mutagen.mp4 import MP4

from lyric_video_creator.config import DEFAULT_CONFIG, RenderConfig
from lyric_video_creator.models import LyricSegment, LyricTuple, TrackMetadata
from lyric_video_creator.utils import get_logger, sanitize_text, strip_title_extra

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".flac", ".mp3", ".wav", ".m4a", ".ogg"})
_LEADING_NUMBER_RE = re.compile(r"^(\d+)")
_LEADING_THREE_DIGITS_RE = re.compile(r"^(\d{3})")
_LRC_TIMESTAMP_RE = re.compile(r"\[(\d+):(\d+\.?\d*)\]")
_SPLIT_PUNCTUATION = ",.!?;:¡¿"


def find_music_files(music_folder: str | Path = DEFAULT_CONFIG.music_folder) -> list[str]:
    """Scan the target directory for supported audio files and sort them deterministically.

    Why custom sort key:
    Users frequently number playlists (e.g., `001 - Intro.flac`, `2 - Verse.flac`, `10 - Finale.flac`).
    Extracting the leading integer ensures numerical ordering (`2` before `10`) rather than
    lexicographical ASCII sorting (`10` before `2`), while placing unnumbered tracks afterward.
    """
    folder = Path(music_folder)
    if not folder.exists():
        return []

    audio_files = [
        str(entry)
        for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    def _sort_key(path_str: str) -> tuple[int, int, str]:
        base = os.path.basename(path_str)
        match = _LEADING_NUMBER_RE.match(base)
        if match:
            return (0, int(match.group(1)), base.lower())
        return (1, 0, base.lower())

    audio_files.sort(key=_sort_key)
    return audio_files


def find_lyrics_file(
    audio_file: str | Path,
    lyrics_folder: str | Path = DEFAULT_CONFIG.lyrics_folder,
) -> str | None:
    """Locate the synchronized `.lrc` file matching an audio track.

    Why two-stage lookup:
    First checks for an exact basename match. If a track title was slightly renamed after
    downloading lyrics, falls back to matching the 3-digit index prefix (e.g., `001*.lrc`).
    """
    base = Path(audio_file).stem
    lyrics_dir = Path(lyrics_folder)
    exact_match = lyrics_dir / f"{base}.lrc"
    if exact_match.is_file():
        return str(exact_match)

    match = _LEADING_THREE_DIGITS_RE.match(base)
    if match:
        prefix = match.group(1)
        pattern = str(lyrics_dir / f"{prefix}*.lrc")
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _extract_cover_bytes(audio_path: Path, parsed_audio: Any) -> bytes | None:
    """Extract embedded cover art bytes across FLAC, ID3 (MP3/WAV), MP4/M4A, Ogg, or sidecar files.

    Why multi-container inspection:
    Different audio containers store binary artwork in incompatible tag structures (`pictures` list
    in FLAC, `APIC:` frames in ID3, `covr` atoms in MP4, and base64 `metadata_block_picture` in Ogg).
    Supporting all of them—plus sidecar image files—eliminates silent failures on non-FLAC libraries.
    """
    # 1. FLAC pictures
    if isinstance(parsed_audio, FLAC) or getattr(parsed_audio, "pictures", None):
        pictures = getattr(parsed_audio, "pictures", None)
        if pictures:
            return bytes(pictures[0].data)

    # 2. MP4 / M4A 'covr' atom
    tags = getattr(parsed_audio, "tags", None)
    if isinstance(parsed_audio, MP4) and tags:
        covr = tags.get("covr")
        if covr:
            return bytes(covr[0])

    # 3. ID3 APIC frames (MP3, WAV with ID3)
    if tags is not None:
        if isinstance(tags, ID3) or hasattr(tags, "getall"):
            try:
                apic_frames = tags.getall("APIC")
                if apic_frames:
                    return bytes(apic_frames[0].data)
            except Exception:
                pass
        for key, value in tags.items():
            if str(key).startswith("APIC") and hasattr(value, "data"):
                return bytes(value.data)

        # 4. Ogg Vorbis / Opus base64-encoded FLAC Picture block
        b64_pics = tags.get("metadata_block_picture") if hasattr(tags, "get") else None
        if b64_pics:
            try:
                raw_pic = base64.b64decode(b64_pics[0])
                pic = Picture(raw_pic)
                return bytes(pic.data)
            except Exception:
                pass

    # 5. Sidecar image fallback (<track_stem>.jpg/png or cover.jpg/png in the same directory)
    parent = audio_path.parent
    stem = audio_path.stem
    for candidate_name in (f"{stem}.jpg", f"{stem}.png", f"{stem}.jpeg", "cover.jpg", "cover.png", "folder.jpg"):
        candidate = parent / candidate_name
        if candidate.is_file():
            try:
                return candidate.read_bytes()
            except OSError:
                continue

    return None


def _get_tag_first(tags: Any, keys: tuple[str, ...], default: str) -> str:
    """Retrieve the first non-empty string value from a sequence of container-specific tag keys."""
    if not tags:
        return default
    for key in keys:
        try:
            if key in tags:
                val = tags[key]
                if isinstance(val, (list, tuple)) and val:
                    return str(val[0])
                if hasattr(val, "text") and val.text:
                    return str(val.text[0])
                return str(val)
        except Exception:
            continue
    return default


def extract_track_metadata(audio_file: str | Path) -> TrackMetadata | None:
    """Extract normalized track metadata and cover artwork from any supported audio file.

    Returns `None` if the file cannot be read or contains neither embedded nor sidecar artwork.
    """
    logger = get_logger()
    path = Path(audio_file)
    try:
        parsed = mutagen.File(str(path))
        if parsed is None:
            logger.warning("Unsupported or corrupted audio container: %s", path.name)
            return None

        cover_data = _extract_cover_bytes(path, parsed)
        if not cover_data:
            return None

        tags = getattr(parsed, "tags", None) or {}
        raw_title = _get_tag_first(tags, ("title", "TITLE", "TIT2", "\xa9nam"), path.stem)
        title = strip_title_extra(raw_title) or path.stem

        artist_raw = _get_tag_first(tags, ("artist", "ARTIST", "TPE1", "\xa9ART"), "Unknown")
        artist_parts = [part.strip() for part in artist_raw.split(",") if part.strip()]
        artist = ", ".join(artist_parts[:2]) if len(artist_parts) > 2 else artist_raw.strip()

        album_raw = _get_tag_first(tags, ("album", "ALBUM", "TALB", "\xa9alb"), "Unknown")
        album = strip_title_extra(album_raw) or "Unknown"

        date = _get_tag_first(tags, ("date", "DATE", "year", "YEAR", "TDRC", "TYER", "\xa9day"), "Unknown")

        return TrackMetadata(
            cover_data=cover_data,
            title=title,
            artist=artist,
            album=album,
            date=date,
        )
    except Exception as exc:
        logger.debug("Failed to extract metadata from '%s': %s", path.name, exc)
        return None


def extract_metadata(
    audio_file: str | Path,
) -> tuple[bytes | None, str | None, str | None, str | None, str | None]:
    """Legacy tuple wrapper around `extract_track_metadata` for backward compatibility."""
    meta = extract_track_metadata(audio_file)
    if meta is None:
        return None, None, None, None, None
    return meta.cover_data, meta.title, meta.artist, meta.album, meta.date


def _recursive_split_line(
    start: float,
    end: float,
    text: str,
    max_length: int,
    default_last_duration: float,
) -> list[LyricTuple]:
    """Recursively split an overlong lyric line into balanced segments with proportional timing.

    Why punctuation-weighted center splitting:
    Splitting strictly at `len(text) // 2` often breaks mid-phrase when a natural comma or semicolon
    sits just a few characters away. Granting a 6-character score bonus to spaces immediately following
    punctuation preserves grammatical phrasing while keeping both halves visually balanced.
    """
    if len(text) <= max_length:
        return [(start, end, text)]

    length = len(text)
    target = length // 2
    best_split = -1
    best_score = float("inf")
    punctuation_bonus = 6

    for idx, ch in enumerate(text):
        if ch == " ":
            score = float(abs(idx - target))
            if idx > 0 and text[idx - 1] in _SPLIT_PUNCTUATION:
                score -= punctuation_bonus
            if score < best_score:
                best_score = score
                best_split = idx

    if best_split == -1:
        best_split = target

    part1 = text[:best_split].rstrip()
    part2 = text[best_split:].lstrip()

    if not part1 or not part2:
        best_split = target
        part1 = text[:best_split].rstrip()
        part2 = text[best_split:].lstrip()

    total_duration = default_last_duration if isinf(end) else max(0.001, end - start)
    len1 = max(1, len(part1))
    len2 = max(1, len(part2))
    midpoint = start + (len1 / float(len1 + len2)) * total_duration

    return _recursive_split_line(
        start, midpoint, part1, max_length, default_last_duration
    ) + _recursive_split_line(midpoint, end, part2, max_length, default_last_duration)


def parse_lyrics(
    lrc_file: str | Path,
    config: RenderConfig = DEFAULT_CONFIG,
) -> list[LyricTuple]:
    """Parse an `.lrc` file into chronologically sorted, length-bounded `(start, end, text)` tuples.

    Handles multi-timestamp chorus lines (e.g., `[01:10.00][02:15.00] Lyric`), filters duplicate
    timestamps within 100ms, normalizes Unicode homoglyphs, and splits lines exceeding
    `config.max_lyrics_length`.
    """
    raw_entries: list[tuple[float, str]] = []
    with open(lrc_file, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            matches = _LRC_TIMESTAMP_RE.findall(line)
            if not matches:
                continue
            text = _LRC_TIMESTAMP_RE.sub("", line).strip()
            if not text:
                continue
            cleaned_text = sanitize_text(text) or ""
            if not cleaned_text:
                continue
            for minutes, seconds in matches:
                timestamp = int(minutes) * 60.0 + float(seconds)
                raw_entries.append((timestamp, cleaned_text))

    raw_entries.sort(key=lambda item: item[0])

    filtered: list[tuple[float, str]] = []
    for idx, (timestamp, txt) in enumerate(raw_entries):
        if idx == 0 or txt != raw_entries[idx - 1][1] or (timestamp - raw_entries[idx - 1][0]) > 0.1:
            filtered.append((timestamp, txt))

    output: list[LyricTuple] = []
    for idx, (start, text) in enumerate(filtered):
        end = filtered[idx + 1][0] if (idx + 1) < len(filtered) else (start + config.default_last_line_duration)
        output.extend(
            _recursive_split_line(
                start=start,
                end=end,
                text=text,
                max_length=config.max_lyrics_length,
                default_last_duration=config.default_last_line_duration,
            )
        )

    return output


def quantize_lyrics_to_frames(
    lyrics: list[LyricTuple],
    fps: int,
    total_frames: int,
) -> list[LyricSegment]:
    """Convert floating-point lyric intervals into clamped discrete frame index ranges."""
    segments: list[LyricSegment] = []
    max_frame = max(0, total_frames - 1)
    for start_sec, end_sec, text in lyrics:
        s_frame = max(0, int(round(start_sec * fps)))
        if total_frames > 0:
            s_frame = min(s_frame, max_frame)

        if isinf(end_sec):
            e_frame = max_frame if total_frames > 0 else s_frame
        else:
            e_frame = max(0, int(round(end_sec * fps)))
            if total_frames > 0:
                e_frame = min(e_frame, max_frame)

        if e_frame < s_frame:
            e_frame = s_frame

        cleaned = sanitize_text(text) or ""
        segments.append(
            LyricSegment(
                start_time=start_sec,
                end_time=end_sec,
                text=cleaned,
                start_frame=s_frame,
                end_frame=e_frame,
            )
        )
    return segments


def get_audio_duration(audio_file: str | Path) -> float:
    """Query audio duration in seconds via Mutagen stream info, falling back to ffprobe."""
    try:
        parsed = mutagen.File(str(audio_file))
        if parsed is not None and getattr(parsed, "info", None) is not None:
            length = getattr(parsed.info, "length", 0.0)
            if length > 0.0:
                return float(length)
    except Exception:
        pass

    try:
        no_win = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_file),
            ],
            stderr=subprocess.DEVNULL,
            creationflags=no_win,
        )
        return max(0.0, float(out.strip()))
    except Exception:
        return 0.0
