import glob
import os
import re
import shutil
import subprocess
import time
from itertools import cycle
from math import sin, pi, ceil, isinf
from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np
from mutagen.flac import FLAC
import unicodedata
from datetime import datetime

# ------------------------- Configuration (tunable constants) -------------------------
# All configurable parameters below — change these to adjust behavior/visuals.

MUSIC_FOLDER = "songs"                 # Directory with audio files (filenames often start with 001...)
LYRICS_FOLDER = "lyrics"               # Directory with .lrc lyric files matching audio basenames
OUTPUT_FOLDER = "output"               # Output directory for generated .mp4 videos

W, H = 1920, 1080                      # Output video resolution (width, height)
FPS = 20                               # Frames per second for rendering. Lower FPS = faster render, but less smooth.
BITRATE_V = "8000k"                    # Video bitrate passed to ffmpeg (controls video quality/size)
BITRATE_A = "320k"                     # Audio bitrate passed to ffmpeg (controls audio quality)
LEFT_MARGIN = 50                       # Base left margin used for cover/text (will be doubled where requested)

BLOCK_SIZE = 32                        # CUDA/OpenCL block size for kernel launches (32x32 threads)
PING_PONG_BUFFERS = 12                 # Number of host pinned buffers for async transfers (CPU <-> GPU)
QUEUE_MAX = 400                        # Max queued frames for writer thread (prevents memory overflow)
MAX_LYRICS_LENGHT = 33                 # Max length of lyrics line before forcing a split

# Glow / visual tuning
GLOW_BLUR_KSIZE = 111                  # Kernel size for Gaussian blur used in glow maps (must be odd). Larger = softer glow.
GLOW_INTENSITY_BASE = 0.5              # Base glow intensity for lyric overlays (pulsating adds to this)
BLUR_STRENGTH = 501                    # Background blur strength (kernel size). Very high for abstract background.
FIXED_TEXT_GLOW_INTENSITY = 0.5        # Alpha/brightness for fixed elements: title, artist, album/year, dot and frame
ASSETS_GLOW_INTENSITY = 0.5            # Glow intensity for assets (icons)
GLOW_PULSE_AMP = 0.04                  # Amplitude of glow pulsation for lyric highlights (how much it throbs)
GLOW_PULSE_FREQ = 0.5                  # Frequency (Hz) of glow pulsation (speed of throb)
GLOW_EDGE_PX = 24.0                    # Feather radius (px) for reveal edge on lyric glow (softness of karaoke wipe)
GLOW_BOOST = 6.0                       # Multiplier when colorizing blurred masks to produce glow (brightness boost)
ALPHA_MIN = 0.32                       # Minimum alpha used when blending glow (prevents invisibility)
DEFAULT_LAST_LINE_DURATION = 4.0       # Fallback duration (seconds) for last lyric line when no end timestamp

# Color sampling / dominant color heuristics
P_LOW_PCT = 10.0                       # Percentile low bound for brightness sampling (ignore too dark pixels)
P_HIGH_PCT = 90.0                      # Percentile high bound for brightness sampling (ignore too bright pixels)
MIN_SAT = 30                           # Minimum saturation threshold for candidate pixels (avoid grays)
RANDOM_SAMPLE_TRIES = 1600             # Attempts when sampling candidate pixels randomly (Monte Carlo sampling)
RANDOM_CANDIDATES_MIN = 40             # Minimum candidates required from random sampling before falling back
SAMPLE_GRID = 120                      # Downsample size used when computing dominant color (speed optimization)

MIN_GLOW_SAT = 20                      # Ensure glow color has at least this saturation (avoid white/gray glow)
MIN_GLOW_VAL = 80                      # Ensure glow color has at least this brightness/value (avoid dark glow)

DEBUG_TRIGGER = False                  # Debug toggle (when True prints debug lines prefixed with [DEBUG])

# -------------------------------------------------------------------------------------





































# -------------------------
# CuPy import
# -------------------------
CUPY_AVAILABLE = True
cp = None
try:
    import cupy as cp
except Exception:
    CUPY_AVAILABLE = False
    cp = None

# -------------------------
# UTIL: Unicode sanitization helpers
# -------------------------
def sanitize_text(s: str) -> str:
    """
    Cleans and normalizes text to ensure consistent rendering.
    - Normalizes Unicode (NFKC) to handle accents and special chars.
    - Removes invisible characters (zero-width spaces, etc.).
    - Replaces Cyrillic homoglyphs with Latin equivalents to ensure font compatibility.
    """
    if s is None:
        return s
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\u200B-\u200F\uFEFF\u2060]', '', s)
    # Mapping of Cyrillic characters that look like Latin ones to their Latin counterparts
    homoglyphs = {
        '\u0430': 'a', '\u0410': 'A',
        '\u0435': 'e', '\u0415': 'E',
        '\u0454': 'e', '\u043E': 'o',
        '\u041E': 'O', '\u0441': 'c',
        '\u0421': 'C', '\u0440': 'p',
        '\u0420': 'P', '\u0443': 'y',
        '\u0423': 'Y', '\u0445': 'x',
        '\u0425': 'X', '\u0456': 'i',
        '\u0401': 'E',
        '\u00F8': 'o', '\u00D8': 'O',
        '\u00E6': 'ae', '\u00C6': 'AE',
        '\u00E5': 'a', '\u00C5': 'A',
        # Punctuation normalization for Hershey font compatibility
        '\u2014': '-',
        '\u2013': '-',
        '\u201C': '"',
        '\u201D': '"',
        '\u2018': "'",
        '\u2019': "'",
        '\u2026': '...',
        '\u00AB': '"',
        '\u00BB': '"',
        '\u2039': "'",
        '\u203A': "'",
        '\u00D7': 'x',
        '\u2022': '-',
        '\u2010': '-',
        '\u2011': '-',
        '\u2012': '-',
        '\u2015': '-',
    }
    for k, v in homoglyphs.items():
        if k in s:
            s = s.replace(k, v)
    return s


def strip_title_extra(title: str) -> str:
    """
    Remove trailing parenthetical/featuring parts from a title.
    Cuts the title at the first occurrence of: '(' '[' 'feat' 'ft' (case-insensitive).
    Examples:
      "Song Name (with Artist)" -> "Song Name"
      "Song Name [Remix]" -> "Song Name"
      "Song Name feat. Artist" -> "Song Name"
      "Song Name ft Artist" -> "Song Name"
    """
    if not title:
        return title
    try:
        # Split on '(', '[' or 'feat' or 'ft' (with optional dot), case-insensitive.
        parts = re.split(r'\(|\[|\bfeat\b|\bft\.?', title, flags=re.IGNORECASE)
        cleaned = parts[0].strip()
        return cleaned
    except Exception:
        return title


def debug_print(*args, **kwargs):
    """Print only when DEBUG_TRIGGER is True, prefix with [DEBUG]."""
    try:
        if DEBUG_TRIGGER:
            print("[DEBUG]", *args, **kwargs)
    except Exception:
        # Avoid raising during debug printing
        pass

def find_music_files():
    """
    Scans the MUSIC_FOLDER for audio files starting with 3 digits (e.g., 001...).
    Returns a sorted list of absolute paths.
    """
    pattern = os.path.join(MUSIC_FOLDER, "[0-9][0-9][0-9]*")
    entries = glob.glob(pattern)
    audio_exts = {'.flac', '.mp3', '.wav', '.m4a', '.ogg'}
    audio_files = [p for p in entries if os.path.splitext(p.lower())[1] in audio_exts]
    def _extract_number(path):
        # Extract the leading number for sorting
        base = os.path.basename(path)
        m = re.match(r'^(\d{3})', base)
        return int(m.group(1)) if m else 999
    audio_files.sort(key=_extract_number)
    return audio_files

def find_lyrics_file(audio_file):
    """
    Finds the corresponding .lrc file for a given audio file.
    First checks for exact name match, then checks for matching leading number.
    """
    base = os.path.splitext(os.path.basename(audio_file))[0]
    exact = os.path.join(LYRICS_FOLDER, base + ".lrc")
    if os.path.exists(exact):
        return exact
    # Fallback: match by number prefix (e.g., "001" in "001 - Song.flac")
    m = re.match(r'^(\d{3})', base)
    if m:
        num = m.group(1)
        pattern = os.path.join(LYRICS_FOLDER, f"{num}*.lrc")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def extract_metadata(audio_file):
    """
    Extracts metadata (cover, title, artist, album, date) from FLAC files.
    Returns the cover art as bytes (or None).
    """
    try:
        audio = FLAC(audio_file)
        if not getattr(audio, 'pictures', None):
            return None, None, None, None, None
        
        # Return raw image data directly
        cover_data = audio.pictures[0].data
        
        tags = audio.tags or {}
        # Get first item or fallback to filename if title tag is missing
        title = (tags.get("title", [os.path.splitext(os.path.basename(audio_file))[0]]))[0]
        # Strip parenthetical/featuring info from title (e.g., "(with ...)", "[Remix]", "feat.", "ft.")
        title = strip_title_extra(title)
        artist_raw = (tags.get("artist", ["Unknown"]))[0]
        # Simplify artist list if too long (take first 2)
        artist_parts = [p.strip() for p in artist_raw.split(',')]
        artist = ', '.join(artist_parts[:2]) if len(artist_parts) > 2 else artist_raw.strip()
        album = (tags.get("album", ["Unknown"]))[0]
        album = strip_title_extra(album)
        date = (tags.get("date", tags.get("year", ["Unknown"])))[0]
        return cover_data, title, artist, album, date
    except Exception:
        return None, None, None, None, None

def parse_lyrics(lrc_file):
    """
    Parses an LRC file into a list of (start_time, end_time, text) tuples.
    Handles timestamp parsing, text sanitization, and splitting long lines.
    """
    timestamp_regex = re.compile(r'\[(\d+):(\d+\.?\d*)\]')
    raw_entries = []
    with open(lrc_file, encoding="utf-8") as f:
        for line in f:
            matches = timestamp_regex.findall(line)
            if not matches:
                continue
            text = timestamp_regex.sub("", line).strip()
            if not text:
                continue
            text = sanitize_text(text)
            for mm, ss in matches:
                raw_entries.append((int(mm) * 60 + float(ss), text))
    raw_entries.sort(key=lambda x: x[0])

    # Filter duplicates or extremely close timestamps
    filtered = []
    for i, (t, txt) in enumerate(raw_entries):
        if i == 0 or txt != raw_entries[i - 1][1] or t - raw_entries[i - 1][0] > 0.1:
            filtered.append((t, txt))

    # Convert to intervals (start, end, text)
    intervals = []
    for i in range(len(filtered)):
        start = filtered[i][0]
        end = filtered[i + 1][0] if i + 1 < len(filtered) else float('inf')
        intervals.append((start, end, filtered[i][1]))

    out = []
    def recursive_split(start, end, text):
        """
        Recursively splits a lyric line if it exceeds MAX_LYRICS_LENGHT.
        Tries to split at punctuation or spaces near the center.
        Adjusts timing proportionally.
        """
        if len(text) <= MAX_LYRICS_LENGHT:
            # Debug print for keeping line
            if DEBUG_TRIGGER:
                print(f"[DEBUG] Keeping line '{text}' (len={len(text)})")
            return [(start, end, text)]
        
        if DEBUG_TRIGGER:
            print(f"[DEBUG] Splitting line '{text}' (len={len(text)})")
        
        L = len(text)
        target = L // 2
        best_split = -1
        best_score = float('inf')
        
        # Punctuation characters to prioritize (split AFTER these)
        puncts = ",.!?;:¡¿"
        bonus = 6  # Willing to deviate up to 6 chars for punctuation
        
        for i in range(L):
            if text[i] == ' ':
                # Score based on distance from center
                dist = abs(i - target)
                score = dist
                
                # Apply bonus if preceded by punctuation
                if i > 0 and text[i-1] in puncts:
                    score -= bonus
                
                if score < best_score:
                    best_score = score
                    best_split = i
        
        # Fallback if no space found (force split at center)
        if best_split == -1:
            best_split = target
            
        part1 = text[:best_split].rstrip()
        part2 = text[best_split:].lstrip()
        
        # Handle edge case where split results in empty string (e.g. split at very start/end)
        if not part1 or not part2:
             # Fallback to hard split at target
             best_split = target
             part1 = text[:best_split].rstrip()
             part2 = text[best_split:].lstrip()

        # Time calculation
        if isinf(end):
             # This should be handled by caller, but just in case
             total = DEFAULT_LAST_LINE_DURATION
        else:
             total = max(0.001, end - start)

        len1 = max(1, len(part1))
        len2 = max(1, len(part2))
        p1 = len1 / float(len1 + len2)
        mid = start + p1 * total
        
        return recursive_split(start, mid, part1) + recursive_split(mid, end, part2)

    out = []
    for start, end, text in intervals:
        # Handle infinite end time once at the top level
        if isinf(end):
            duration = DEFAULT_LAST_LINE_DURATION
            end = start + duration
        
        out.extend(recursive_split(start, end, text))

    return out

def get_audio_duration(audio_file):
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
        ])
        return float(out)
    except Exception:
        return 0.0

# -------------------------
# GPU Renderer
# -------------------------
class GPUVideoRenderer:
    def __init__(self):
        """
        Initializes the GPU renderer.
        - Checks for CuPy availability.
        - Sets up CUDA device and memory pools.
        - Compiles custom CUDA kernels for image processing.
        - Allocates pinned memory buffers for efficient host-device transfer.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for GPU renderer.")
        self._setup_gpu_env()
        self._compile_kernels()
        self._setup_buffers()
        # Caches for overlay images (lyrics, glows) to avoid re-uploading every frame
        self.overlay_gpu_cache = {}
        self.overlay_glow_gpu_cache = {}
        # Map frame index to active overlays
        self.frame_overlay_indices = {}
        # Map overlay key to timing info for glow effects
        self.key_frame_map = {}
        self.zoom_cache = {}
        self.glow_color = (60, 160, 255) # Default glow color (blue-ish)

    def _setup_gpu_env(self):
        """
        Configures the CUDA environment.
        - Selects device 0.
        - Sets up memory allocators to use CuPy's memory pool (reduces allocation overhead).
        """
        try:
            cp.cuda.Device(0).use()
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

    def _compile_kernels(self):
        """
        Compiles custom CUDA C++ kernels for high-performance image manipulation.
        These kernels run in parallel on the GPU for every pixel.
        """
        kernel_code = r'''
        extern "C" __global__
        void sample_background(unsigned char* frame, const unsigned char* bg,
                               int bg_h, int bg_w,
                               float angle, int frame_w, int frame_h) {
            // Kernel to sample the background image with rotation and scaling.
            // Each thread computes one pixel of the output frame.
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= frame_w || y >= frame_h) return;

            // Center of rotation: Bottom-Right of the frame (or center, depending on effect)
            // Here we use the frame bottom-right for rotation as requested
            float cx = (float)frame_w;
            float cy = (float)frame_h;

            // Vector from center to pixel
            float dx = (float)x - cx;
            float dy = (float)y - cy;

            // Inverse rotation to find source pixel in the background image
            float ca = cosf(angle);
            float sa = sinf(angle);
            
            // rx, ry relative to background center
            float rx = dx * ca - dy * sa;
            float ry = dx * sa + dy * ca;

            // Map back to background image coordinates (centered)
            float bg_cx = (float)bg_w * 0.5f;
            float bg_cy = (float)bg_h * 0.5f;

            float sx = rx + bg_cx;
            float sy = ry + bg_cy;

            // Bilinear interpolation for smooth sampling
            int x0 = (int)floorf(sx);
            int y0 = (int)floorf(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            // Clamp/Edge handling (clamp to edge)
            if (x0 < 0) x0 = 0; if (x0 >= bg_w) x0 = bg_w - 1;
            if (x1 < 0) x1 = 0; if (x1 >= bg_w) x1 = bg_w - 1;
            if (y0 < 0) y0 = 0; if (y0 >= bg_h) y0 = bg_h - 1;
            if (y1 < 0) y1 = 0; if (y1 >= bg_h) y1 = bg_h - 1;

            int idx00 = (y0 * bg_w + x0) * 3;
            int idx10 = (y0 * bg_w + x1) * 3;
            int idx01 = (y1 * bg_w + x0) * 3;
            int idx11 = (y1 * bg_w + x1) * 3;

            // Interpolate weights
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            // Compute final RGB values
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
            // Simple kernel to copy the album cover onto the frame at a specific position.
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= frame_w || y >= frame_h) return;

            // Check if current pixel is within the cover's bounding box
            int dx = x - cover_x;
            int dy = y - cover_y;
            
            int dst_idx = (y * frame_w + x) * 3;
            if (dx >= 0 && dx < cover_w && dy >= 0 && dy < cover_h) {
                int src_idx = (dy * cover_w + dx) * 3;
                frame[dst_idx] = cover[src_idx];
                frame[dst_idx + 1] = cover[src_idx + 1];
                frame[dst_idx + 2] = cover[src_idx + 2];
            }
        }

        extern "C" __global__
        void blend_overlay(unsigned char* frame, const unsigned char* overlay,
                           int frame_w, int frame_h) {
            // Basic additive/replacement blend (not used much, legacy?)
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= frame_w || y >= frame_h) return;
            int idx = (y * frame_w + x) * 3;
            unsigned char r = overlay[idx], g = overlay[idx+1], b = overlay[idx+2];
            if (r != 0 || g != 0 || b != 0) {
                frame[idx] = r;
                frame[idx+1] = g;
                frame[idx+2] = b;
            }
        }

        extern "C" __global__
        void blend_overlay_add(unsigned char* frame, const unsigned char* overlay,
                               float alpha, float reveal, float edge, int left_x, int text_w, int frame_w, int frame_h) {
            // Additive blending kernel for GLOW effects.
            // Supports "reveal" animation (karaoke style) where the glow wipes across.
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= frame_w || y >= frame_h) return;
            
            int idx = (y * frame_w + x) * 3;
            unsigned char orv = overlay[idx];
            unsigned char og = overlay[idx+1];
            unsigned char ob = overlay[idx+2];
            
            if (orv == 0 && og == 0 && ob == 0) return;

            // Calculate horizontal wipe mask
            float cutoff = (float)left_x + reveal * (float)text_w;
            float half = edge * 0.5f;
            float wgt; 
            
            if ((float)x <= cutoff - half) wgt = 1.0f; // Fully visible
            else if ((float)x >= cutoff + half) wgt = 0.0f; // Fully hidden
            else wgt = (cutoff + half - (float)x) / edge; // Smooth transition

            // Add glow color * alpha * wipe_weight
            float add_r = (float)orv * alpha * wgt;
            float add_g = (float)og * alpha * wgt;
            float add_b = (float)ob * alpha * wgt;
            
            // Saturated addition
            int r = (int)frame[idx] + (int)(add_r);
            int g = (int)frame[idx+1] + (int)(add_g);
            int b = (int)frame[idx+2] + (int)(add_b);
            
            if (r > 255) r = 255; if (g > 255) g = 255; if (b > 255) b = 255;
            frame[idx]   = (unsigned char)r;
            frame[idx+1] = (unsigned char)g;
            frame[idx+2] = (unsigned char)b;
        }

        extern "C" __global__
        void blend_overlay_alpha(unsigned char* frame, const unsigned char* overlay,
                                 int frame_w, int frame_h) {
            // Standard Alpha Blending kernel.
            // Expects overlay to be RGBA (4 channels).
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= frame_w || y >= frame_h) return;
            
            int idx = (y * frame_w + x) * 3; // Frame is RGB
            int o_idx = (y * frame_w + x) * 4; // Overlay is RGBA
            
            float alpha = (float)overlay[o_idx+3] / 255.0f;
            
            // Optimization: skip transparent pixels
            if (alpha <= 0.0f) return;

            float r = (float)overlay[o_idx];
            float g = (float)overlay[o_idx+1];
            float b = (float)overlay[o_idx+2];

            float fr = (float)frame[idx];
            float fg = (float)frame[idx+1];
            float fb = (float)frame[idx+2];

            // Standard alpha blending formula: out = src * alpha + dst * (1 - alpha)
            frame[idx]   = (unsigned char)(r * alpha + fr * (1.0f - alpha));
            frame[idx+1] = (unsigned char)(g * alpha + fg * (1.0f - alpha));
            frame[idx+2] = (unsigned char)(b * alpha + fb * (1.0f - alpha));
        }
        '''
        self.sample_background = cp.RawKernel(kernel_code, 'sample_background', options=('--std=c++14',))
        self.render_cover = cp.RawKernel(kernel_code, 'render_cover', options=('--std=c++14',))
        self.blend_overlay = cp.RawKernel(kernel_code, 'blend_overlay', options=('--std=c++14',))
        self.blend_overlay_add = cp.RawKernel(kernel_code, 'blend_overlay_add', options=('--std=c++14',))
        self.blend_overlay_alpha = cp.RawKernel(kernel_code, 'blend_overlay_alpha', options=('--std=c++14',))

    def _setup_buffers(self):
        """
        Allocates pinned memory buffers for the "Ping-Pong" transfer strategy.
        Pinned memory allows for faster, asynchronous DMA transfers between CPU and GPU.
        We use multiple buffers to overlap rendering (GPU) with encoding (CPU/FFmpeg).
        """
        self.ping_pong_frames = []
        self.ping_pong_arrays = []
        self.ping_pong_mem_ptrs = []
        self.streams = []
        self.events = [None] * PING_PONG_BUFFERS
        size = H * W * 3
        for _ in range(PING_PONG_BUFFERS):
            mem = cp.cuda.alloc_pinned_memory(size)
            # Create numpy view of the pinned memory
            arr = np.frombuffer(mem, dtype=np.uint8, count=size).reshape(H, W, 3)
            self.ping_pong_frames.append(mem)
            self.ping_pong_arrays.append(arr)
            try:
                ptr = mem.ptr
            except Exception:
                ptr = int(arr.__array_interface__['data'][0])
            self.ping_pong_mem_ptrs.append(ptr)
            self.streams.append(cp.cuda.Stream()) # Separate stream for each buffer
        self.buffer_cycle = cycle(range(PING_PONG_BUFFERS))
        self.frame_gpu = cp.empty((H, W, 3), dtype=cp.uint8) # Main GPU frame buffer
        self.main_stream = cp.cuda.Stream()

    # ---------- color helpers ----------
    def _ensure_min_saturation(self, bgr, min_sat=MIN_GLOW_SAT):
        """
        Ensures a color has a minimum saturation.
        Used to prevent glow effects from looking washed out (gray/white).
        """
        try:
            arr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
            hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0,0].astype(np.int32)
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            changed = False
            if s < min_sat:
                s = min_sat
                changed = True
            new_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0,0]
            return (int(new_bgr[0]), int(new_bgr[1]), int(new_bgr[2])), changed
        except Exception:
            return tuple(int(x) for x in bgr), False

    def _ensure_min_brightness(self, bgr, min_val=MIN_GLOW_VAL):
        """
        Ensures a color has a minimum brightness (Value in HSV).
        Used to prevent glow effects from being invisible (black).
        """
        try:
            arr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
            hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0,0].astype(np.int32)
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            changed = False
            if v < min_val:
                v = min_val
                changed = True
            new_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0,0]
            return (int(new_bgr[0]), int(new_bgr[1]), int(new_bgr[2])), changed
        except Exception:
            return tuple(int(x) for x in bgr), False

    def _compute_dominant_color(self, img_bgr, k=3, sample=SAMPLE_GRID):
        """
        Computes the dominant color of an image (album cover).
        1. Resizes image to small grid for speed.
        2. Filters out pixels that are too dark, too bright, or too desaturated.
        3. Uses K-Means clustering or random sampling to find the most representative color.
        """
        try:
            small = cv2.resize(img_bgr, (sample, sample), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            v = hsv[..., 2].reshape(-1).astype(np.float32)
            s = hsv[..., 1].reshape(-1).astype(np.float32)

            # Filter out extreme brightness values
            p_low = np.percentile(v, P_LOW_PCT)
            p_high = np.percentile(v, P_HIGH_PCT)
            mask_bv = (v >= p_low) & (v <= p_high)
            mask_sat = (s >= MIN_SAT)
            mask = mask_bv & mask_sat

            bgr_flat = small.reshape(-1, 3)
            indices = np.nonzero(mask)[0]

            candidates = []
            if indices.size > 0:
                # Random sampling (Monte Carlo) to pick candidates
                tries = min(RANDOM_SAMPLE_TRIES, indices.size * 4 + RANDOM_SAMPLE_TRIES // 4)
                rng = np.random.default_rng()
                for _ in range(tries):
                    idx = rng.choice(indices)
                    candidates.append(bgr_flat[idx])
                    if len(candidates) >= RANDOM_CANDIDATES_MIN:
                        break

            if len(candidates) >= RANDOM_CANDIDATES_MIN:
                arr = np.array(candidates, dtype=np.float32)
                med = np.median(arr, axis=0)
                dom = tuple(int(x) for x in med)
                return dom

            # Fallback to K-Means if random sampling didn't yield enough candidates
            filtered = bgr_flat[mask]
            if filtered.shape[0] >= max(30, RANDOM_CANDIDATES_MIN // 2):
                data = filtered.astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                flags = cv2.KMEANS_PP_CENTERS
                try:
                    ret = cv2.kmeans(data, k, None, criteria, 8, flags)
                    labels = ret[1]
                    centers = ret[2]
                    counts = np.bincount(labels.flatten())
                    dominant = centers[np.argmax(counts)]
                    dom = tuple(int(c) for c in dominant)
                    return dom
                except Exception:
                    mean_col = data.mean(axis=0)
                    dom = tuple(int(c) for c in mean_col)
                    return dom

            # Ultimate fallback: mean color of the whole image
            mean_col = small.reshape(-1, 3).mean(axis=0)
            dom = tuple(int(c) for c in mean_col)
            return dom
        except Exception:
            mean_col = img_bgr.mean(axis=(0,1))
            return tuple(int(c) for c in mean_col)

    def _adjust_color_brightness(self, bgr, boost_v=1.12, boost_s=1.04):
        """
        Slightly boosts the brightness and saturation of a color.
        Used to make the glow color pop more against the background.
        """
        try:
            arr = np.uint8([[list(bgr)]])
            hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).astype(np.float32)[0,0]
            hsv[1] = min(255.0, hsv[1] * boost_s)
            hsv[2] = min(255.0, hsv[2] * boost_v)
            new_bgr = cv2.cvtColor(np.uint8([[hsv.astype(np.uint8)]]), cv2.COLOR_HSV2BGR)[0,0]
            return tuple(int(c) for c in new_bgr)
        except Exception:
            return bgr

    def render_video(self, audio_file, lrc_file, output_file):
        """
        Main entry point for rendering a single video.
        - Extracts metadata and cover art.
        - Parses lyrics (if available).
        - Prepares visual elements (cover, text, background).
        - Launches the frame rendering loop.
        """
        print("[INFO]", "Processing:", os.path.basename(audio_file))
        
        cover_data, title, artist, album, date = extract_metadata(audio_file)
        if not cover_data:
            print("Cover not found, skipping.")
            return False
        has_lyrics = bool(lrc_file and os.path.exists(lrc_file))
        lyrics = parse_lyrics(lrc_file) if has_lyrics else []
        duration = get_audio_duration(audio_file)
        base_frames = int(round(duration * FPS))
        total_frames = base_frames
        # Title/Artist/Year: removed from logs
        print("[INFO]", f"Duration: {duration:.3f}s ({total_frames} frames @ {FPS} FPS)")
        print("[INFO]", "Lyrics:", "available" if has_lyrics else "not available", len(lyrics) if has_lyrics else 0)

        if not self._prepare_visual_elements(cover_data, title, artist, album, date, has_lyrics):
            return False

        if has_lyrics:
            self._prepare_text_overlays(lyrics, total_frames, duration)

        return self._render_frames(audio_file, lyrics, output_file, total_frames, duration, has_lyrics)

    def _prepare_visual_elements(self, cover_data, title, artist, album, date, has_lyrics):
        """
        Loads the cover art from bytes, computes dominant colors for glow effects,
        and prepares the fixed visual elements (text, background, assets).
        """
        try:
            # Decode image from memory
            nparr = np.frombuffer(cover_data, np.uint8)
            cover = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cover is None:
                raise RuntimeError("Cover could not be decoded")
            h, w = cover.shape[:2]
            
            # Compute dominant color for dynamic glow
            dom = self._compute_dominant_color(cover, k=3, sample=SAMPLE_GRID)
            dom_adj = self._adjust_color_brightness(dom, boost_v=1.12, boost_s=1.04)

            # Ensure minimum saturation/value and use the resulting color for glow
            sat_checked, sat_changed = self._ensure_min_saturation(dom_adj, min_sat=MIN_GLOW_SAT)
            bright_checked, bright_changed = self._ensure_min_brightness(sat_checked, min_val=MIN_GLOW_VAL)
            self.glow_color = bright_checked
            if sat_changed or bright_changed:
                debug_print("Adjusted glow color for S/V minimum -> (B,G,R):", self.glow_color)
            else:
                debug_print("Selected glow color (B,G,R):", self.glow_color)

            # Compute cover size and position depending on whether lyrics are present
            if has_lyrics:
                new_height = int(800 * 0.8)
                new_width = int(w * (new_height / h))
                # Double the horizontal gap from the left wall to cover+text:
                # previous left margin was LEFT_MARGIN; we place cover at 2 * LEFT_MARGIN
                self.cover_x = LEFT_MARGIN * 2
                # Apply configured vertical offset for cover when lyrics are present.
                # The user requested "+40" relative adjustment; this constant centralizes it.
                self.cover_y = (H - new_height) // 2 - 50 -40
            else:
                scale_factor = 1.2
                new_height = int(800 * 0.8 * scale_factor)
                new_width = int(w * (new_height / h))

                # Reduce cover size by 10% in 'no lyrics' mode while keeping the same center.
                # The original center is (W/2, H/2 - 70). Compute scaled size and place so center unchanged.
                center_x = float(W) / 2.0
                center_y = float(H) / 2.0 - 70.0
                new_width = int(round(new_width * 0.90))
                new_height = int(round(new_height * 0.90))
                self.cover_x = int(round(center_x - (new_width / 2.0)))
                # Move cover up by 10 pixels in no-lyrics mode (decrease y)
                self.cover_y = int(round(center_y - (new_height / 2.0))) - 20

            small = cv2.resize(cover, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            self.cover_width, self.cover_height = new_width, new_height
            
            # Create the fixed text overlay (Title, Artist, Album, Year)
            self._create_fixed_text(title, artist, album, date, has_lyrics)
            # Create the blurred background
            self._create_background(small)
            
            # Upload to GPU
            self.cover_gpu = cp.asarray(small)
            self.fixed_text_gpu = cp.asarray(self.text_image)
            
            # Upload fixed text glow if created
            if hasattr(self, 'fixed_text_glow_image'):
                self.fixed_text_glow_gpu = cp.asarray(self.fixed_text_glow_image)
            else:
                self.fixed_text_glow_gpu = None

            # Upload assets glow if created
            if hasattr(self, 'assets_glow_image'):
                self.assets_glow_gpu = cp.asarray(self.assets_glow_image)
            else:
                self.assets_glow_gpu = None
                
            return True
        except Exception as e:
            print("[INFO]", "Error preparing visuals:", e)
            return False

    def _create_fixed_text(self, title, artist, album, date, has_lyrics):
        """
        Draws the static text elements (Title, Artist, Album, Year) onto a transparent overlay.
        Also handles the placement of small icons/assets (title.png, artist.png, etc.).
        Generates a glow map for these elements.
        """
        # Use a sans-serif Hershey font to approximate "sans sheriff"
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_thickness = 3 if has_lyrics else 6
        frame_x1 = self.cover_x - frame_thickness
        frame_y1 = self.cover_y - frame_thickness
        frame_x2 = self.cover_x + self.cover_width + frame_thickness
        frame_y2 = self.cover_y + self.cover_height + frame_thickness
        # Use 4-channel image for text to support alpha blending
        self.text_image = np.zeros((H, W, 4), dtype=np.uint8)
        self.assets_to_blit = []
        # Draw white frame around cover
        cv2.rectangle(self.text_image, (frame_x1, frame_y1), (frame_x2, frame_y2),
                      (255, 255, 255, 255), frame_thickness)
        if has_lyrics:
            title_scale, title_thickness = 1.7, 5
            label_scale, label_thickness = 1.1, 3
            text_y_offset = 80
        else:
            title_scale, title_thickness = 1.7, 5
            label_scale = 1.1
            label_thickness = 4
            text_y_offset = 95

        title_s = sanitize_text(title) if title is not None else ""
        artist_s = sanitize_text(artist) if artist is not None else ""
        
        # --- Line 1: Title ---
        (title_w, title_h), _ = cv2.getTextSize(title_s, font, title_scale, title_thickness)
        title_x = ((self.cover_x + self.cover_width // 2 - title_w // 2) if has_lyrics else (W // 2 - title_w // 2)) + 20
        title_y = self.cover_y + self.cover_height + text_y_offset

        # Place specific PNG assets to the left of their corresponding text if present:
        # title.png -> to the left of Title text
        # artist.png -> to the left of Artist text
        # album.png -> to the left of Album text
        try:
            assets_dir = "assets"
            gap = 12
            # Title asset
            title_asset = os.path.join(assets_dir, "title.png")
            if os.path.isfile(title_asset):
                a = cv2.imread(title_asset, cv2.IMREAD_UNCHANGED)
                if a is not None:
                    (t_w, t_h) = (title_w, title_h)
                    target_h = max(6, int(round(t_h)))
                    h0, w0 = a.shape[:2]
                    target_w = max(6, int(round((w0 / h0) * target_h)))
                    # Convert to float and premultiply alpha before resizing to avoid dark halos
                    if a.shape[2] == 4:
                        a_f = a.astype(np.float32) / 255.0
                        # Premultiply
                        a_f[..., :3] *= a_f[..., 3:4]
                        # Resize
                        resized_f = cv2.resize(a_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        # Convert back to uint8 (still premultiplied)
                        resized = (resized_f * 255.0).clip(0, 255).astype(np.uint8)
                        res_rgba = resized
                    else:
                        resized = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        res_rgba = resized

                    dst_x = int(title_x - target_w - gap) - 8
                    dst_y = int(title_y - title_h) + 4
                    if dst_x >= 0 and dst_y >= 0 and dst_y + target_h <= H:
                        self.assets_to_blit.append((res_rgba, int(dst_x), int(dst_y)))
        except Exception:
            pass
        
        # Debug: print computed positions once to help verify layout changes
        try:
                if not hasattr(self, '_coords_logged'):
                    debug_print(f"DEBUG_POSITIONS: cover_x={self.cover_x}, cover_y={self.cover_y}, cover_h={self.cover_height}, title_y={title_y}")
                    self._coords_logged = True
        except Exception:
            pass

        # Draw Title Shadow
        cv2.putText(self.text_image, title_s, (title_x + 4, title_y + 4), font,
                    title_scale, (30, 30, 30), title_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(self.text_image, title_s, (title_x + 4, title_y + 4), font,
                    title_scale, (30, 30, 30, 255), title_thickness + 2, lineType=cv2.LINE_AA)
        # Draw Title Text
        cv2.putText(self.text_image, title_s, (title_x, title_y), font,
                    title_scale, (255, 255, 255, 255), title_thickness, lineType=cv2.LINE_AA)

        # --- Line 2: Artist ---
        (artist_w, artist_h), _ = cv2.getTextSize(artist_s, font, label_scale, label_thickness)
        artist_x = ((self.cover_x + self.cover_width // 2 - artist_w // 2) if has_lyrics else (W // 2 - artist_w // 2)) + 20
        artist_y = title_y + title_h + 25
        
        # Artist asset placement
        try:
            assets_dir = "assets"
            artist_asset = os.path.join(assets_dir, "artist.png")
            gap_asset = 12
            if os.path.isfile(artist_asset):
                a = cv2.imread(artist_asset, cv2.IMREAD_UNCHANGED)
                if a is not None:
                    (a_w, a_h) = (artist_w, artist_h)
                    target_h = max(6, int(round(a_h)))
                    h0, w0 = a.shape[:2]
                    target_w = max(6, int(round((w0 / h0) * target_h)))
                    # Convert to float and premultiply alpha before resizing to avoid dark halos
                    if a.shape[2] == 4:
                        a_f = a.astype(np.float32) / 255.0
                        # Premultiply
                        a_f[..., :3] *= a_f[..., 3:4]
                        # Resize
                        resized_f = cv2.resize(a_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        # Convert back to uint8 (still premultiplied)
                        resized = (resized_f * 255.0).clip(0, 255).astype(np.uint8)
                        res_rgba = resized
                    else:
                        resized = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        res_rgba = resized

                    dst_x = int(artist_x - target_w - gap_asset) - 4
                    dst_y = int(artist_y - artist_h) + 4
                    if dst_x >= 0 and dst_y >= 0 and dst_y + target_h <= H:
                        self.assets_to_blit.append((res_rgba, int(dst_x), int(dst_y)))
        except Exception:
            pass

        # Draw Artist Shadow
        cv2.putText(self.text_image, artist_s, (artist_x + 4, artist_y + 4), font,
                    label_scale, (30, 30, 30), label_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(self.text_image, artist_s, (artist_x + 4, artist_y + 4), font,
                    label_scale, (30, 30, 30, 255), label_thickness + 2, lineType=cv2.LINE_AA)
        # Draw Artist Text
        cv2.putText(self.text_image, artist_s, (artist_x, artist_y), font,
                    label_scale, (200, 200, 200, 255), label_thickness, lineType=cv2.LINE_AA)

        # --- Line 3: Album - Year ---
        album_s = sanitize_text(album) if album is not None else ""
        year_s = date[:4] if date is not None else ""
        
        (alb_w, alb_h), _ = cv2.getTextSize(album_s, font, label_scale, label_thickness)
        (year_w, year_h), _ = cv2.getTextSize(year_s, font, label_scale, label_thickness)
        
        # Calculate total width: album + gap + circle + gap + year
        gap = 15
        circle_radius = 4
        total_w = alb_w + gap + (circle_radius * 2) + gap + year_w
        
        start_x = ((self.cover_x + self.cover_width // 2 - total_w // 2) if has_lyrics else (W // 2 - total_w // 2)) + 20
        album_y = artist_y + artist_h + 25
        
        # Album asset placement
        try:
            assets_dir = "assets"
            album_asset = os.path.join(assets_dir, "album.png")
            gap_asset = 12
            if os.path.isfile(album_asset):
                a = cv2.imread(album_asset, cv2.IMREAD_UNCHANGED)
                if a is not None:
                    (al_w, al_h) = (alb_w, alb_h)
                    target_h = max(6, int(round(al_h)))
                    h0, w0 = a.shape[:2]
                    target_w = max(6, int(round((w0 / h0) * target_h)))
                    
                    # Convert to float and premultiply alpha before resizing to avoid dark halos
                    if a.shape[2] == 4:
                        a_f = a.astype(np.float32) / 255.0
                        # Premultiply
                        a_f[..., :3] *= a_f[..., 3:4]
                        # Resize
                        resized_f = cv2.resize(a_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        # Convert back to uint8 (still premultiplied)
                        resized = (resized_f * 255.0).clip(0, 255).astype(np.uint8)
                        res_rgba = resized
                    else:
                        resized = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        res_rgba = resized

                    dst_x = int(start_x - target_w - gap_asset) - 4
                    dst_y = int(album_y - alb_h) + 4
                    if dst_x >= 0 and dst_y >= 0 and dst_y + target_h <= H:
                        self.assets_to_blit.append((res_rgba, int(dst_x), int(dst_y)))
        except Exception:
            pass

        # Draw Album Shadow
        cv2.putText(self.text_image, album_s, (start_x + 4, album_y + 4), font,
                    label_scale, (30, 30, 30), label_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(self.text_image, album_s, (start_x + 4, album_y + 4), font,
                    label_scale, (30, 30, 30, 255), label_thickness + 2, lineType=cv2.LINE_AA)
        # Draw Album Text
        cv2.putText(self.text_image, album_s, (start_x, album_y), font,
                    label_scale, (200, 200, 200, 255), label_thickness, lineType=cv2.LINE_AA)
        
        # Draw Separator Circle
        circle_cx = start_x + alb_w + gap + circle_radius
        circle_cy = album_y - alb_h // 2 + 5 # Adjust vertical alignment
        cv2.circle(self.text_image, (circle_cx + 2, circle_cy + 2), circle_radius, (30, 30, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(self.text_image, (circle_cx + 2, circle_cy + 2), circle_radius, (30, 30, 30, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(self.text_image, (circle_cx, circle_cy), circle_radius, (200, 200, 200, 255), -1, lineType=cv2.LINE_AA)
        
        # Draw Year Shadow
        year_x = start_x + alb_w + gap + (circle_radius * 2) + gap
        cv2.putText(self.text_image, year_s, (year_x + 4, album_y + 4), font,
                    label_scale, (30, 30, 30), label_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(self.text_image, year_s, (year_x + 4, album_y + 4), font,
                    label_scale, (30, 30, 30, 255), label_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(self.text_image, year_s, (year_x, album_y), font,
                    label_scale, (200, 200, 200, 255), label_thickness, lineType=cv2.LINE_AA)

        # (removed previous generic asset placements that overlapped the cover;
        #  assets are now placed specifically to the left of their corresponding texts)

        # Create glow map for fixed text
        glow_mask = np.zeros((H, W, 3), dtype=np.uint8)
        # Title glow
        cv2.putText(glow_mask, title_s, (title_x, title_y), font,
                    title_scale, (255, 255, 255), title_thickness, lineType=cv2.LINE_AA)
        # Artist glow
        cv2.putText(glow_mask, artist_s, (artist_x, artist_y), font,
                    label_scale, (255, 255, 255), label_thickness, lineType=cv2.LINE_AA)
        # Album glow
        cv2.putText(glow_mask, album_s, (start_x, album_y), font,
                    label_scale, (255, 255, 255), label_thickness, lineType=cv2.LINE_AA)
        # Circle glow
        cv2.circle(glow_mask, (circle_cx, circle_cy), circle_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        # Year glow
        cv2.putText(glow_mask, year_s, (year_x, album_y), font,
                    label_scale, (255, 255, 255), label_thickness, lineType=cv2.LINE_AA)
        
        # Create separate glow mask for assets
        assets_glow_mask = np.zeros((H, W, 3), dtype=np.uint8)

        # Render assets onto text_image and assets_glow_mask
        for item in self.assets_to_blit:
            # item is (img, x, y)
            img, x, y = item
            h_img, w_img = img.shape[:2]
            
            # Ensure bounds
            if x < 0 or y < 0 or x + w_img > W or y + h_img > H:
                continue
                
            # Overlay onto assets_glow_mask (solid white for glow)
            # We want the shape of the asset to be white in the glow mask
            if img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                # Create a white version of the image shape
                white_asset = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255
                for c in range(3):
                    # Blend white asset onto glow mask using the alpha channel
                    assets_glow_mask[y:y+h_img, x:x+w_img, c] = (1.0 - alpha) * assets_glow_mask[y:y+h_img, x:x+w_img, c] + alpha * white_asset[:, :, c]
            else:
                assets_glow_mask[y:y+h_img, x:x+w_img] = (255, 255, 255)

        # Also include the white frame surrounding the cover in the fixed glow mask
        # (use same thickness as rectangle drawn in `self.text_image`)
        cv2.rectangle(glow_mask, (frame_x1, frame_y1), (frame_x2, frame_y2),
                      (255, 255, 255), frame_thickness)
        # Blur and colorize fixed text glow
        blur_k = GLOW_BLUR_KSIZE if GLOW_BLUR_KSIZE % 2 == 1 else GLOW_BLUR_KSIZE + 1
        blurred = cv2.GaussianBlur(glow_mask, (blur_k, blur_k), 0)
        blurred_f = blurred.astype(np.float32) / 255.0
        color_f = np.array(self.glow_color, dtype=np.float32).reshape((1,1,3))
        glow_f = blurred_f * color_f * float(GLOW_BOOST)
        glow_f = np.clip(glow_f, 0.0, 255.0)
        self.fixed_text_glow_image = glow_f.astype(np.uint8)

        # Blur and colorize assets glow
        blurred_assets = cv2.GaussianBlur(assets_glow_mask, (blur_k, blur_k), 0)
        blurred_assets_f = blurred_assets.astype(np.float32) / 255.0
        glow_assets_f = blurred_assets_f * color_f * float(GLOW_BOOST)
        glow_assets_f = np.clip(glow_assets_f, 0.0, 255.0)
        self.assets_glow_image = glow_assets_f.astype(np.uint8)

    def _create_background(self, small):
        """
        Creates the large, blurred background image from the album cover.
        - Scales up the cover.
        - Applies heavy Gaussian blur.
        - Uploads to GPU.
        """
        # Scale to 3840x3840 (1920*2 x 1920*2)
        target_size = 3840
        # Resize cover to target size
        bg_large = cv2.resize(small, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        # Blur
        # Increased blur kernel to make background softer and text stand out more
        blur_k = BLUR_STRENGTH
        bg_blurred = cv2.GaussianBlur(bg_large, (blur_k, blur_k), 0)
        
        # Darken slightly to make text pop? Optional, but good for backgrounds
        # bg_blurred = (bg_blurred * 0.7).astype(np.uint8)

        self.background_large_height = target_size
        self.background_large_width = target_size
        self.background_large_gpu = cp.asarray(bg_blurred)

    def _precompute_rotation_angles(self, total_frames):
        """
        Precomputes rotation angles for the background animation.
        - Creates a slow, steady rotation effect over time.
        """
        # Rotation speed: e.g. 1 full rotation every 60 seconds?
        # 20 FPS * 60s = 1200 frames.
        # 2pi / 1200 approx 0.005 rad/frame
        # Let's make it slow and steady.
        speed = 0.005
        angles = []
        for i in range(total_frames):
            angle = i * speed
            angles.append(float(angle))
        return angles

    def _lyric_multiplier(self, offset: int) -> float:
        """
        Compute vertical multiplier for lyric offsets.
        Keeps consistent spacing between adjacent lines and slightly
        increases the separation for lines further from center.
        Returns 0.0 for the active line (offset == 0).
        """
        if offset == 0:
            return 0.0
        sign = -1.0 if offset < 0 else 1.0
        return float(offset) + sign * 0.25

    def _prepare_text_overlays(self, lyrics, total_frames, audio_duration):
        """
        Pre-renders all lyric lines onto transparent overlays.
        - Converts timestamps to frame indices.
        - Generates text images for active and surrounding lines.
        - Creates glow maps for each text overlay.
        - Uploads everything to GPU memory to avoid per-frame CPU rendering.
        - Builds a mapping of frame index -> active overlays.
        """
        # Use a sans-serif Hershey font to approximate "sans sheriff"
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Convert second intervals -> frames (round) and clamp to valid range
        lyrics_frames = []
        for (s, e, text) in lyrics:
            s_frame = max(0, int(round(s * FPS)))
            if total_frames > 0:
                s_frame = min(s_frame, total_frames - 1)
            if isinf(e):
                e_frame = max(0, total_frames - 1) if total_frames > 0 else s_frame
            else:
                e_frame = max(0, int(round(e * FPS)))
                e_frame = min(e_frame, total_frames - 1)
            if e_frame < s_frame:
                e_frame = s_frame
            text = sanitize_text(text)
            lyrics_frames.append((s_frame, e_frame, text))

        overlay_images = {}
        overlay_glow_images = {}
        self.key_frame_map.clear()

        # Calculate horizontal center for lyrics
        # They should be centered in the remaining space to the right of the cover
        left_start = self.cover_x + self.cover_width + 45
        right_margin = max(left_start + 100, W - 45)
        axis_x = (left_start + right_margin) // 2

        # Iterate through all lyric lines
        for idx in range(len(lyrics_frames)):
            # Generate overlays for the active line and its neighbors (-2 to +2)
            for offset in range(-2, 3):
                lyr_idx = idx + offset
                if lyr_idx < 0 or lyr_idx >= len(lyrics_frames):
                    continue
                text = lyrics_frames[lyr_idx][2]
                
                # Unique key for caching: (lyric_index, type)
                key = (lyr_idx, 'active') if offset == 0 else (lyr_idx, f'near_{offset}')
                if key in overlay_images:
                    continue
                    
                overlay = np.zeros((H, W, 4), dtype=np.uint8)
                
                # Styling based on whether it's the active line or a neighbor
                if offset == 0:
                    scale, thickness, color = 1.66, 8, (255, 255, 255, 255)
                    shadow_th = 8
                else:
                    scale, thickness, color = 1.29, 6, (180, 180, 180, 255)
                    shadow_th = 6

                mult = self._lyric_multiplier(offset)
                # Apply configured lyric vertical offset (negative moves lyrics up).
                y_pos = H // 2 + int(round(mult * 70)) + 9

                # Debug: print lyric y_pos for the first active lyric generated
                try:
                    if offset == 0 and not hasattr(self, '_lyric_coords_logged'):
                        debug_print(f"DEBUG_LYRIC_POS: lyric_index={lyr_idx}, y_pos={y_pos}, mult={mult}")
                        self._lyric_coords_logged = True
                except Exception:
                    pass

                (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                text_x = int(axis_x - (text_w // 2))
                
                # Draw shadow
                cv2.putText(overlay, text, (text_x + 3, y_pos + 3),
                            font, scale, (20, 20, 20), shadow_th, lineType=cv2.LINE_AA)
                cv2.putText(overlay, text, (text_x + 3, y_pos + 3),
                            font, scale, (20, 20, 20, 255), shadow_th, lineType=cv2.LINE_AA)
                # Draw text
                cv2.putText(overlay, text, (text_x, y_pos),
                            font, scale, color, thickness, lineType=cv2.LINE_AA)
                overlay_images[key] = overlay

                # Generate glow map for the active line
                if offset == 0:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.putText(mask, text, (text_x, y_pos),
                                font, scale, 255, thickness, lineType=cv2.LINE_AA)
                    blur_k = GLOW_BLUR_KSIZE if GLOW_BLUR_KSIZE % 2 == 1 else GLOW_BLUR_KSIZE + 1
                    blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
                    blurred_f = blurred.astype(np.float32) / 255.0
                    color_f = np.array(self.glow_color, dtype=np.float32).reshape((1,1,3))
                    glow_f = blurred_f[..., None] * color_f * float(GLOW_BOOST)
                    glow_f = np.clip(glow_f, 0.0, 255.0)
                    glow = glow_f.astype(np.uint8)
                    overlay_glow_images[key] = glow

                    # Store timing info for glow animation
                    s_f, e_f, _ = lyrics_frames[lyr_idx]
                    if e_f < s_f:
                        e_f = s_f
                    self.key_frame_map[key] = (s_f, e_f, int(text_x), int(text_w))

        # Upload overlays and glow images to GPU caches
        for k, img in overlay_images.items():
            self.overlay_gpu_cache[k] = cp.asarray(img)
        for k, img in overlay_glow_images.items():
            self.overlay_glow_gpu_cache[k] = cp.asarray(img)

        # Build mapping: frame index -> list of overlay keys
        # This allows O(1) lookup during the render loop
        for frame_index in range(total_frames):
            keys = []
            active_idx = -1
            for idx, (s_frame, e_frame, _) in enumerate(lyrics_frames):
                if s_frame <= frame_index <= e_frame:
                    active_idx = idx
                    break
            if active_idx >= 0:
                for offset in range(-2, 3):
                    lyr_idx = active_idx + offset
                    if 0 <= lyr_idx < len(lyrics_frames):
                        key = (lyr_idx, 'active') if offset == 0 else (lyr_idx, f'near_{offset}')
                        keys.append(key)
            self.frame_overlay_indices[frame_index] = keys

    def _render_frames(self, audio_file, lyrics, output_file, total_frames, audio_duration, has_lyrics):
        """
        The core rendering loop.
        - Starts an FFmpeg subprocess to encode the video.
        - Starts a background thread to write rendered frames to FFmpeg's stdin.
        - Iterates through every frame:
            1. Selects a pinned memory buffer (Ping-Pong).
            2. Launches GPU kernels to:
                - Sample the background (with rotation).
                - Render the cover art.
                - Blend the fixed text overlay.
                - Blend dynamic lyric overlays (with glow and reveal effects).
                - Composite small assets.
            3. Asynchronously copies the rendered frame from GPU to CPU (pinned memory).
            4. Queues the frame for the writer thread.
        """
        try:
            # Setup FFmpeg command
            # Input: Raw video (BGR24) from pipe, Audio from file
            # Output: H.264 (NVENC accelerated) + AAC audio
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                '-i', audio_file,
                '-c:v', 'h264_nvenc', '-preset', 'p7', '-b:v', BITRATE_V,
                '-c:a', 'aac', '-b:a', BITRATE_A,
                '-pix_fmt', 'yuv420p', '-shortest', output_file
            ]
            
            # Launch ffmpeg. Suppress stdout/stderr unless debugging.
            if DEBUG_TRIGGER:
                ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=10**7)
            else:
                ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=10**7,
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Initialize Ping-Pong buffer queues
            available_buffers = Queue(maxsize=PING_PONG_BUFFERS)
            for i in range(PING_PONG_BUFFERS):
                available_buffers.put(i)
            frame_queue = Queue(maxsize=QUEUE_MAX)

            # Background thread to write frames to FFmpeg
            # This allows the main thread to continue issuing GPU commands without waiting for I/O
            def writer_thread_fn():
                try:
                    while True:
                        item = frame_queue.get()
                        if item is None:
                            break
                        buf_idx, event = item
                        # Wait for the GPU copy to complete before writing
                        if event is not None:
                            try:
                                event.synchronize()
                            except Exception:
                                pass
                        arr = self.ping_pong_arrays[buf_idx]
                        try:
                            ffmpeg_process.stdin.write(memoryview(arr))
                        except BrokenPipeError:
                            print("[ERROR] FFmpeg stdin pipe broken. FFmpeg likely crashed.")
                            # Try to consume any remaining items to unblock main thread if it's waiting on queue?
                            # But main thread checks for success return.
                            # We should probably just exit the loop.
                            frame_queue.task_done()
                            break
                        
                        frame_queue.task_done()
                        # Return buffer to pool
                        available_buffers.put(buf_idx)
                except Exception as e:
                    print(f"[ERROR] Writer thread exception: {e}")

            writer_thread = Thread(target=writer_thread_fn, daemon=True)
            writer_thread.start()

            rotation_angles = self._precompute_rotation_angles(total_frames)
            start_time = time.time()
            processed_count = 0

            # Grid dimensions for CUDA kernels
            blocks_x = (W + BLOCK_SIZE - 1) // BLOCK_SIZE
            blocks_y = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_grid = (blocks_x, blocks_y)
            thread_block = (BLOCK_SIZE, BLOCK_SIZE)

            for frame_index in range(total_frames):
                if not writer_thread.is_alive():
                    print("[ERROR] Writer thread is no longer alive. Stopping render.")
                    break

                # Get a free buffer (blocks if none available)
                try:
                    buf_idx = available_buffers.get(timeout=2.0)
                except Empty:
                    if not writer_thread.is_alive():
                        print("[ERROR] Writer thread died. Stopping render.")
                        break
                    print("[WARN] Buffer queue empty for 2s, but writer thread is alive. Retrying...")
                    continue
                stream = self.streams[buf_idx]
                
                # Issue commands to the specific CUDA stream for this buffer
                with stream:
                    # Debug: print the cover coordinates used at render time once
                    try:
                            if not hasattr(self, '_render_coords_logged'):
                                debug_print(f"RENDER_COVER_COORDS: cover_x={self.cover_x}, cover_y={self.cover_y}, cover_w={self.cover_width}, cover_h={self.cover_height}, frame_index={frame_index}")
                                self._render_coords_logged = True
                    except Exception:
                        pass
                        
                    angle = rotation_angles[frame_index]
                    
                    # 1. Draw Background (Rotated)
                    try:
                        self.sample_background(block_grid, thread_block,
                                               (self.frame_gpu, self.background_large_gpu,
                                                np.int32(self.background_large_height), np.int32(self.background_large_width),
                                                np.float32(angle), np.int32(W), np.int32(H)))
                    except Exception:
                        self.frame_gpu[:] = 0

                    # 2. Draw Cover Art
                    self.render_cover(block_grid, thread_block,
                                      (self.frame_gpu, self.cover_gpu,
                                       np.int32(self.cover_x), np.int32(self.cover_y),
                                       np.int32(self.cover_width), np.int32(self.cover_height),
                                       np.int32(W), np.int32(H)))
                                       
                    # 3. Draw Fixed Text Overlay (Title, Artist, etc.)
                    self.blend_overlay_alpha(block_grid, thread_block, (self.frame_gpu, self.fixed_text_gpu, np.int32(W), np.int32(H)))

                    # 4. Draw Fixed Text Glow
                    if hasattr(self, 'fixed_text_glow_gpu') and self.fixed_text_glow_gpu is not None:
                        alpha = FIXED_TEXT_GLOW_INTENSITY
                        if alpha < ALPHA_MIN:
                            alpha = ALPHA_MIN
                        alpha32 = np.float32(alpha)
                        reveal32 = np.float32(1.0)
                        edge32 = np.float32(GLOW_EDGE_PX)
                        try:
                            self.blend_overlay_add(block_grid, thread_block,
                                                   (self.frame_gpu, self.fixed_text_glow_gpu, alpha32, reveal32, edge32,
                                                    np.int32(0), np.int32(W), np.int32(W), np.int32(H)))
                        except Exception:
                            pass

                    # 5. Draw Assets Glow
                    if hasattr(self, 'assets_glow_gpu') and self.assets_glow_gpu is not None:
                        alpha = ASSETS_GLOW_INTENSITY
                        if alpha < ALPHA_MIN:
                            alpha = ALPHA_MIN
                        alpha32 = np.float32(alpha)
                        reveal32 = np.float32(1.0)
                        edge32 = np.float32(GLOW_EDGE_PX)
                        try:
                            self.blend_overlay_add(block_grid, thread_block,
                                                   (self.frame_gpu, self.assets_glow_gpu, alpha32, reveal32, edge32,
                                                    np.int32(0), np.int32(W), np.int32(W), np.int32(H)))
                        except Exception:
                            pass

                    # 6. Draw Assets (Icons)
                    # Composite prepared small assets onto the final frame GPU buffer
                    try:
                        if hasattr(self, 'assets_to_blit') and len(self.assets_to_blit) > 0:
                            if cp is not None and isinstance(self.frame_gpu, cp.ndarray):
                                for (img_np, dst_x, dst_y) in self.assets_to_blit:
                                    try:
                                        img_cp = cp.asarray(img_np)
                                        h_a, w_a = int(img_cp.shape[0]), int(img_cp.shape[1])
                                        if img_cp.shape[2] == 4:
                                            # img_cp is premultiplied alpha
                                            alpha = img_cp[..., 3].astype(cp.float32) / 255.0
                                            rgb_premul = img_cp[..., :3].astype(cp.float32)
                                            dst = self.frame_gpu[dst_y:dst_y+h_a, dst_x:dst_x+w_a].astype(cp.float32)
                                            # Porter-Duff 'Source Over'
                                            comp = (rgb_premul + dst * (1.0 - alpha[..., None]))
                                            self.frame_gpu[dst_y:dst_y+h_a, dst_x:dst_x+w_a] = comp.astype(cp.uint8)
                                        else:
                                            self.frame_gpu[dst_y:dst_y+h_a, dst_x:dst_x+w_a] = img_cp[..., :3]
                                    except Exception:
                                        pass
                            else:
                                # CPU fallback (should not happen in GPU renderer)
                                pass
                    except Exception:
                        pass

                    # 7. Draw Lyrics (if available)
                    if has_lyrics:
                        keys = self.frame_overlay_indices.get(frame_index, [])
                        # Calculate global pulse for glow
                        t = frame_index / float(FPS)
                        sin_val = sin(2.0 * pi * GLOW_PULSE_FREQ * t)
                        alpha = GLOW_INTENSITY_BASE + (GLOW_PULSE_AMP * (sin_val - 1.0) / 2.0)
                        
                        if alpha < ALPHA_MIN:
                            alpha = ALPHA_MIN
                        alpha32 = np.float32(alpha)

                        for k in keys:
                            # Draw Text Overlay
                            overlay_gpu = self.overlay_gpu_cache.get(k)
                            if overlay_gpu is not None:
                                self.blend_overlay_alpha(block_grid, thread_block, (self.frame_gpu, overlay_gpu, np.int32(W), np.int32(H)))

                            # Draw Glow Overlay (with Karaoke reveal effect)
                            glow_gpu = self.overlay_glow_gpu_cache.get(k)
                            if glow_gpu is not None:
                                frame_pair = self.key_frame_map.get(k)
                                if frame_pair is None:
                                    # Non-active lines (neighbors) get full reveal but lower intensity
                                    reveal32 = np.float32(1.0)
                                    edge32 = np.float32(GLOW_EDGE_PX)
                                    try:
                                        self.blend_overlay_add(block_grid, thread_block,
                                                               (self.frame_gpu, glow_gpu, alpha32, reveal32, edge32,
                                                                np.int32(0), np.int32(W), np.int32(W), np.int32(H)))
                                    except Exception:
                                        pass
                                    continue

                                # Active line: calculate reveal progress
                                s_frame, e_frame, left_x, text_w = frame_pair
                                if not (s_frame <= frame_index <= e_frame):
                                    continue

                                # Force full reveal for pulsating effect (Karaoke fill is optional/disabled here for style)
                                # To enable karaoke fill, uncomment/modify:
                                # progress = (frame_index - s_frame) / max(1, float(e_frame - s_frame))
                                # reveal_frac = progress
                                reveal_frac = 1.0
                                reveal32 = np.float32(reveal_frac)
                                edge32 = np.float32(GLOW_EDGE_PX)

                                try:
                                    self.blend_overlay_add(block_grid, thread_block,
                                                           (self.frame_gpu, glow_gpu, alpha32, reveal32, edge32,
                                                            np.int32(left_x), np.int32(text_w), np.int32(W), np.int32(H)))
                                except Exception:
                                    pass

                    # 8. Async Copy to CPU (Pinned Memory)
                    size_bytes = H * W * 3
                    src_ptr = int(self.frame_gpu.data.ptr)
                    dst_ptr = int(self.ping_pong_mem_ptrs[buf_idx])
                    try:
                        cp.cuda.runtime.memcpyAsync(dst_ptr, src_ptr, size_bytes,
                                                    cp.cuda.runtime.cudaMemcpyDeviceToHost,
                                                    stream.ptr)
                        # Record event to signal completion
                        ev = cp.cuda.Event()
                        stream.record_event(ev)
                        self.events[buf_idx] = ev
                        frame_queue.put((buf_idx, ev))
                    except Exception:
                        # Fallback synchronous copy
                        cp.asnumpy(self.frame_gpu, out=self.ping_pong_arrays[buf_idx])
                        frame_queue.put((buf_idx, None))

                processed_count += 1

                # Periodic progress logging and memory cleanup
                if frame_index % 2500 == 0 and frame_index > 0:
                    elapsed = time.time() - start_time
                    fps_rate = processed_count / elapsed if processed_count>0 else 0
                    remaining = total_frames - processed_count
                    eta = (remaining / fps_rate) if fps_rate > 0 else 0
                    print("[INFO]", f"Progress: {processed_count}/{total_frames} frames ({processed_count/total_frames*100:.1f}%) - {fps_rate:.1f} FPS - ETA: {eta:.0f}s")
                    try:
                        if self.memory_pool is not None:
                            self.memory_pool.free_all_blocks()
                    except Exception:
                        pass

            # Signal writer thread to stop if it's still running
            if writer_thread.is_alive():
                try:
                    frame_queue.put(None, timeout=1.0)
                    writer_thread.join(timeout=60.0)
                except Exception:
                    pass
            
            # Close FFmpeg
            try:
                ffmpeg_process.stdin.close()
            except Exception:
                pass
            ffmpeg_process.wait()

            elapsed = time.time() - start_time
            print("[INFO]", "Completed in {:.2f}s -> {:.2f} FPS".format(elapsed, processed_count / elapsed if processed_count>0 else 0))
            
            if processed_count < total_frames:
                print(f"[WARN] Render incomplete: {processed_count}/{total_frames} frames. Marking as failed.")
                return False
                
            return True
        except Exception as e:
            print("Rendering error:", e)
            return False

    def cleanup(self):
        """
        Releases GPU resources and clears caches.
        """
        try:
            if hasattr(self, 'zoom_cache'):
                self.zoom_cache.clear()
            if hasattr(self, 'overlay_gpu_cache'):
                self.overlay_gpu_cache.clear()
            if hasattr(self, 'overlay_glow_gpu_cache'):
                self.overlay_glow_gpu_cache.clear()
            try:
                del self.main_stream
            except Exception:
                pass
            if hasattr(self, 'memory_pool') and self.memory_pool is not None:
                self.memory_pool.free_all_blocks()
            if hasattr(self, 'pinned_memory_pool') and self.pinned_memory_pool is not None:
                self.pinned_memory_pool.free_all_blocks()
        except Exception:
            pass

class CPUVideoRenderer:
    """
    Fallback renderer for systems without NVIDIA GPUs.
    Currently a stub - prompts the user to install CUDA.
    """
    def render_video(self, audio_file, lrc_file, output_file):
        print("[ERROR] CPU rendering is not implemented yet.")
        print("Please use an NVIDIA GPU with CUDA installed.")
        return False
    
    def cleanup(self):
        pass

def setup_directories():
    """
    Creates necessary input/output directories if they don't exist.
    """
    if not os.path.exists(MUSIC_FOLDER):
        os.makedirs(MUSIC_FOLDER)
        print(f"Created input folder: {MUSIC_FOLDER}")
        print("Put your music files (.flac, .mp3, .wav) here.")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Create assets folder if it doesn't exist
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

def validate_environment():
    """
    Checks if all required external tools and libraries are available.
    - FFmpeg and FFprobe (system PATH).
    - Mutagen (Python library).
    - CuPy (Python library + CUDA).
    """
    try:
        subprocess.check_call(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(['ffprobe', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("FFmpeg/FFprobe not found. Please install FFmpeg and add it to your PATH.")
        return False
    try:
        import importlib
        importlib.import_module('mutagen.flac')
    except Exception:
        print("Mutagen not installed. Run: pip install mutagen")
        return False
        
    if CUPY_AVAILABLE and cp is not None:
        try:
            debug_print("CuPy version:", cp.__version__)
            debug_print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
        except Exception:
            pass
    else:
        print("CuPy not available. GPU rendering disabled.")
        
    return True

def get_renderer():
    """
    Factory function to return the appropriate renderer.
    """
    if CUPY_AVAILABLE:
        return GPUVideoRenderer()
    else:
        return CPUVideoRenderer()

def main():
    """
    Main execution flow.
    1. Sets up directories.
    2. Scans for music files.
    3. Initializes the renderer.
    4. Iterates through each music file and renders a video.
    5. Prints a summary of the process.
    """
    print("Starting batch music video processing")
    setup_directories()
    
    music_files = find_music_files()
    if not music_files:
        debug_print(f"No music files found in '{MUSIC_FOLDER}'.")
        return
        
    print(f"Found {len(music_files)} music file(s):")
    for i, p in enumerate(music_files, 1):
        print(f"  {i:2d}. {os.path.basename(p)}")
        
    renderer = get_renderer()
    total_start = time.time()
    processed = failed = skipped = 0
    
    # Create log file
    log_name = f"000 output {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_filename = os.path.join(OUTPUT_FOLDER, log_name)
    
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"Execution Log - {datetime.now()}\n")
        log_file.write("========================================\n")

        try:
            for i, audio_file in enumerate(music_files, 1):
                print("=" * 60)
                print(f"Processing file {i}/{len(music_files)}")
                
                lrc_file = find_lyrics_file(audio_file)
                base = os.path.splitext(os.path.basename(audio_file))[0]
                out_path = os.path.join(OUTPUT_FOLDER, f"{base}.mp4")
                
                if os.path.exists(out_path):
                    print("Output already exists:", out_path)
                    skipped += 1
                    log_file.write(f"[ALREADY PROCESSED] {os.path.basename(audio_file)}\n")
                    log_file.flush()
                    continue
                    
                success = renderer.render_video(audio_file, lrc_file, out_path)
                
                if success:
                    processed += 1
                    print("[INFO]", "Video generated:", out_path)
                    log_file.write(f"[PROPERLY PROCESSED] {os.path.basename(audio_file)}\n")
                else:
                    failed += 1
                    print("Failed to render:", os.path.basename(audio_file))
                    log_file.write(f"[FAILED] {os.path.basename(audio_file)}\n")
                
                log_file.flush()
                
                # Aggressive cleanup between videos to prevent VRAM fragmentation
                if hasattr(renderer, 'zoom_cache'):
                    renderer.zoom_cache.clear()
                if hasattr(renderer, 'memory_pool'):
                    try:
                        renderer.memory_pool.free_all_blocks()
                    except Exception:
                        pass
                        
                elapsed_total = time.time() - total_start
                avg_time = elapsed_total / i if i > 0 else 0
                remaining = len(music_files) - i
                eta_total = remaining * avg_time
                
                print("[INFO]", f"Overall: {i}/{len(music_files)} processed - success {processed}, skipped {skipped}, failed {failed}")
                if remaining > 0:
                    print("[INFO]", f"ETA remaining: {eta_total/60:.1f} minutes")
                
        except KeyboardInterrupt:
            print("Processing interrupted by user.")
        except Exception as e:
            print("Critical error during batch processing:", e)
        finally:
            try:
                if 'renderer' in locals():
                    renderer.cleanup()
            except Exception:
                pass
            
            total_elapsed = time.time() - total_start
            print("=" * 60)
            print("FINAL SUMMARY")
            print("=" * 60)
            print(f"Total time: {total_elapsed/60:.1f} minutes")
            print(f"Files found: {len(music_files)}")
            print(f"Videos generated: {processed}")
            print(f"Already existed: {skipped}")
            print(f"Failed: {failed}")
            if processed > 0:
                print(f"Average time per video: {total_elapsed/processed:.1f}s")
                print("Videos written to:", os.path.abspath(OUTPUT_FOLDER))
            print("Processing finished.")

if __name__ == "__main__":
    import sys
    if not validate_environment():
        print("Environment not configured correctly. Install dependencies and try again.")
        sys.exit(1)
        
    setup_directories()
    
    try:
        main()
    except KeyboardInterrupt:
        print("Process cancelled by user.")
    except Exception as e:
        print("Unhandled critical error:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)