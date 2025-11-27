import glob
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import zoom
from mutagen.flac import FLAC

# -------------------------
# MAIN CONFIGURATION
# -------------------------

# Folder structure (relative to script)
MUSIC_FOLDER = "songs"      # Audio files folder (FLAC, MP3, WAV, etc.)
LYRICS_FOLDER = "lyrics"    # .lrc files folder (synchronized lyrics)
OUTPUT_FOLDER = "output"    # Output folder for generated videos
TEMP_DIR = "temp"           # Temporary directory for intermediate files

# Video configuration
W, H = 1920, 1080           # Output resolution (Full HD)
FPS = 5                     # Frames per second (low for lyrics sync)
BITRATE_V = "1500k"         # Video bitrate
BITRATE_A = "320k"          # Audio bitrate
MAX_ZOOM = 4                # Maximum zoom multiplier for background animation
DS_FACTOR = 0.05            # Downscale factor for background optimization

# GPU/CPU performance tuning
BLOCK_SIZE = 16
BATCH_SIZE = 3
NUM_WORKERS = min(6, os.cpu_count() or 1)
PREFETCH_FRAMES = 6
PING_PONG_BUFFERS = 8
QUEUE_MAX = 100

# -------------------------
# FILE / PATH UTILITIES
# -------------------------

def find_music_files():
    """
    Search and sort audio files that start with a 3-digit prefix (001, 002, ...).
    This enforces deterministic ordering when batching files.
    Returns:
        list[str]: ordered list of audio file paths
    """
    pattern = os.path.join(MUSIC_FOLDER, "[0-9][0-9][0-9]*")
    entries = glob.glob(pattern)
    audio_exts = {'.flac', '.mp3', '.wav', '.m4a', '.ogg'}
    audio_files = [p for p in entries if os.path.splitext(p.lower())[1] in audio_exts]

    def _extract_number(path):
        base = os.path.basename(path)
        m = re.match(r'^(\d{3})', base)
        return int(m.group(1)) if m else 999

    audio_files.sort(key=_extract_number)
    return audio_files


def find_lyrics_file(audio_file):
    """
    Try to find a corresponding .lrc file for a given audio file.
    Strategy:
    1. Exact basename match (songname.lrc)
    2. Fallback: search by leading 3-digit number pattern
    Returns None if no lyrics file found.
    """
    base = os.path.splitext(os.path.basename(audio_file))[0]
    exact = os.path.join(LYRICS_FOLDER, base + ".lrc")
    if os.path.exists(exact):
        return exact

    m = re.match(r'^(\d{3})', base)
    if m:
        num = m.group(1)
        pattern = os.path.join(LYRICS_FOLDER, f"{num}*.lrc")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

# -------------------------
# SETUP / CLEANUP UTILITIES
# -------------------------

def setup_gpu():
    """
    Initialize GPU device and memory pools for CuPy.
    Returns the default memory pool and pinned memory pool objects.
    """
    # Select device 0 explicitly for consistent behavior
    try:
        cp.cuda.Device(0).use()
    except Exception:
        # If using context manager is required, it's still fine to proceed;
        # keep this guarded so scripts don't fail on machines without a GPU.
        pass

    # Configure allocators to use memory pools for better throughput
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    return cp.get_default_memory_pool(), cp.get_default_pinned_memory_pool()


def cleanup_temp():
    """Remove temporary folder contents and recreate an empty temp directory."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------
# METADATA AND LYRICS PARSING
# -------------------------

def extract_metadata(audio_file):
    """
    Extract metadata and cover art from a FLAC audio file using Mutagen.
    Returns:
        tuple: (cover_path, title, artist, date) or (None, None, None, None) on failure
    Notes:
        - If the FLAC has embedded pictures, the first picture is saved to TEMP_DIR.
        - Artist strings with many contributors are truncated to the first two to avoid overflow.
    """
    try:
        audio = FLAC(audio_file)
        if not getattr(audio, 'pictures', None):
            print(f"Cover not found in {audio_file}")
            return None, None, None, None

        cover_out = os.path.join(TEMP_DIR, "cover_temp.jpg")
        with open(cover_out, "wb") as f:
            f.write(audio.pictures[0].data)

        tags = audio.tags or {}
        title = (tags.get("title", [os.path.splitext(os.path.basename(audio_file))[0]]))[0]
        artist_raw = (tags.get("artist", ["Unknown"]))[0]
        artist_parts = [p.strip() for p in artist_raw.split(',')]
        artist = ', '.join(artist_parts[:2]) if len(artist_parts) > 2 else artist_raw.strip()
        date = (tags.get("date", tags.get("year", ["Unknown"])))[0]

        return cover_out, title, artist, date
    except Exception as e:
        print(f"Error extracting metadata from {audio_file}: {e}")
        return None, None, None, None


def parse_lyrics(lrc_file):
    """
    Parse a .lrc file to extract timestamped lyrics.
    Returns:
        list[tuple]: list of (start_sec, end_sec, text) tuples
    Behavior:
        - Handles multiple timestamps per line
        - Removes duplicates or extremely close repeated timestamps
        - The last line's end_time is set to float('inf') to indicate 'until end'
    """
    timestamp_regex = re.compile(r'\[(\d+):(\d+\.?\d*)\]')
    entries = []

    with open(lrc_file, encoding="utf-8") as f:
        for line in f:
            matches = timestamp_regex.findall(line)
            if not matches:
                continue
            text = timestamp_regex.sub("", line).strip()
            if not text:
                continue
            for mm, ss in matches:
                entries.append((int(mm) * 60 + float(ss), text))

    # Sort by time and filter duplicates (allow small deltas)
    entries.sort(key=lambda x: x[0])
    filtered = []
    for i, (t, txt) in enumerate(entries):
        if i == 0 or txt != entries[i - 1][1] or t - entries[i - 1][0] > 0.1:
            filtered.append((t, txt))

    lyrics = []
    for i in range(len(filtered)):
        start = filtered[i][0]
        end = filtered[i + 1][0] if i + 1 < len(filtered) else float('inf')
        lyrics.append((start, end, filtered[i][1]))

    return lyrics


def get_audio_duration(audio_file):
    """
    Use ffprobe to retrieve audio duration in seconds.
    Returns 0 on failure.
    """
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
        ])
        return float(out)
    except Exception as e:
        print(f"Error getting duration for {audio_file}: {e}")
        return 0.0

# -------------------------
# MAIN RENDERER CLASS
# -------------------------

class VideoRenderer:
    """
    Renderer class that constructs video frames from cover art and optional lyrics,
    and pipes raw frames into FFmpeg for encoding.

    Key features:
      - Uses CuPy (CUDA) to accelerate image composition
      - Uses pinned memory and ping-pong buffers for efficient CPU<->GPU transfers
      - Caches zoom results to avoid repeated expensive resampling
      - Supports two layouts: with lyrics (side layout) and without lyrics (centered cover)
    """

    def __init__(self):
        """Initialize GPU resources, compile kernels and allocate buffers."""
        self.memory_pool, self.pinned_memory_pool = setup_gpu()
        self._setup_cuda_kernels()
        self._setup_buffers()
        self.zoom_cache = {}

    def _setup_cuda_kernels(self):
        """Compile minimal CUDA kernels for cover rendering and overlay blending."""
        kernel_code = r'''
        extern "C" __global__
        void render_cover(unsigned char* frame, const unsigned char* cover,
                          int cover_x, int cover_y, int cover_width, int cover_height,
                          int frame_width, int frame_height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= frame_width || y >= frame_height) return;
            int dx = x - cover_x, dy = y - cover_y;
            int dst_idx = (y * frame_width + x) * 3;
            if (dx >= 0 && dx < cover_width && dy >= 0 && dy < cover_height) {
                int src_idx = (dy * cover_width + dx) * 3;
                frame[dst_idx] = cover[src_idx];
                frame[dst_idx + 1] = cover[src_idx + 1];
                frame[dst_idx + 2] = cover[src_idx + 2];
            }
        }

        extern "C" __global__
        void blend_overlay(unsigned char* frame, const unsigned char* overlay,
                           int frame_width, int frame_height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= frame_width || y >= frame_height) return;
            int idx = (y * frame_width + x) * 3;
            if (overlay[idx] != 0 || overlay[idx + 1] != 0 || overlay[idx + 2] != 0) {
                frame[idx] = overlay[idx];
                frame[idx + 1] = overlay[idx + 1];
                frame[idx + 2] = overlay[idx + 2];
            }
        }
        '''
        # Compile kernels separately by kernel name
        self.render_cover = cp.RawKernel(kernel_code, 'render_cover', options=('--std=c++14',))
        self.blend_overlay_kernel = cp.RawKernel(kernel_code, 'blend_overlay', options=('--std=c++14',))

    def _setup_buffers(self):
        """
        Allocate ping-pong pinned CPU buffers and a GPU frame buffer.
        Pinned memory accelerates host->device and device->host transfers.
        """
        self.ping_pong_frames = []
        self.ping_pong_arrays = []
        self.frame_ready = [threading.Event() for _ in range(PING_PONG_BUFFERS)]

        for _ in range(PING_PONG_BUFFERS):
            size = H * W * 3
            mem = cp.cuda.alloc_pinned_memory(size)
            self.ping_pong_frames.append(mem)
            arr = np.frombuffer(mem, dtype=np.uint8, count=size).reshape(H, W, 3)
            self.ping_pong_arrays.append(arr)

        self.buffer_cycle = cycle(range(PING_PONG_BUFFERS))
        self.frame_gpu = cp.empty((H, W, 3), dtype=cp.uint8)
        self.main_stream = cp.cuda.Stream()

    def render_video(self, audio_file, lrc_file, output_file):
        """
        High-level function: prepare assets and render the video file.
        Returns True if successful.
        """
        print(f"\nProcessing: {os.path.basename(audio_file)}")

        cleanup_temp()

        cover_file, title, artist, date = extract_metadata(audio_file)
        if not cover_file:
            return False

        has_lyrics = bool(lrc_file and os.path.exists(lrc_file))
        lyrics = parse_lyrics(lrc_file) if has_lyrics else []

        duration = get_audio_duration(audio_file)
        base_frames = int(duration * FPS)
        total_frames = base_frames

        # Ensure total_frames is a multiple of FPS by adding the needed frames
        remainder = base_frames % FPS if FPS > 0 else 0
        if remainder != 0 and FPS > 0:
            add = FPS - remainder
            total_frames += add
            print(f"   Added {add} extra frame(s) to reach a multiple of FPS ({FPS}): {total_frames}")

        if total_frames % 2 != 0:
            # keep optional historical behavior (still okay) but not required; leave note
            pass

        print(f"   Title: {title}")
        print(f"   Artist: {artist}")
        print(f"   Year: {date}")
        print(f"   Duration: {duration:.1f}s ({total_frames} frames)")
        print(f"   Lyrics: {'available' if has_lyrics else 'not available'} — "
              f"{len(lyrics) if has_lyrics else 0} lines")

        success = self._prepare_visual_elements(cover_file, title, artist, date, has_lyrics)
        if not success:
            return False

        return self._render_frames(audio_file, lyrics, output_file, total_frames, duration, has_lyrics)

    def _prepare_visual_elements(self, cover_file, title, artist, date, has_lyrics):
        """
        Load and resize the cover, compose fixed text overlay, and create a blurred animated background.
        Upload prepared elements to GPU arrays for fast rendering.
        """
        try:
            cover = cv2.imread(cover_file)
            if cover is None:
                raise RuntimeError("Cover could not be loaded with OpenCV")

            h, w = cover.shape[:2]

            if has_lyrics:
                # Side layout: modest cover size on the left
                new_height = int(800 * 0.8)
                new_width = int(w * (new_height / h))
                self.cover_x, self.cover_y = 50, (H - new_height) // 2 - 50
            else:
                # Centered large cover for songs without lyrics
                scale_factor = 1.2
                new_height = int(800 * 0.8 * scale_factor)
                new_width = int(w * (new_height / h))
                self.cover_x = (W - new_width) // 2
                self.cover_y = ((H - new_height) // 2) - 70

            small = cv2.resize(cover, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            self.cover_width, self.cover_height = new_width, new_height

            self._create_fixed_text(title, artist, date, has_lyrics)
            self._create_background(small)

            # Upload to GPU for kernels to consume
            self.cover_gpu = cp.asarray(small)
            self.fixed_text_gpu = cp.asarray(self.text_image)

            return True
        except Exception as e:
            print(f"Error preparing visual elements: {e}")
            return False

    def _create_fixed_text(self, title, artist, date, has_lyrics):
        """
        Create a fixed overlay image containing title, artist and decorative frame.
        This overlay is blended onto each frame with the blending kernel.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        frame_thickness = 3 if has_lyrics else 6
        frame_x1 = self.cover_x - frame_thickness
        frame_y1 = self.cover_y - frame_thickness
        frame_x2 = self.cover_x + self.cover_width + frame_thickness
        frame_y2 = self.cover_y + self.cover_height + frame_thickness

        self.text_image = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.rectangle(self.text_image, (frame_x1, frame_y1), (frame_x2, frame_y2),
                      (255, 255, 255), frame_thickness)

        if has_lyrics:
            title_scale, title_thickness = 1.4, 3
            label_scale, label_thickness = 1.2, 2
            text_y_offset = 70
        else:
            title_scale, title_thickness = 1.68, 4
            label_scale, label_thickness = 1.44, 3
            text_y_offset = 84

        (text_w, text_h), _ = cv2.getTextSize(title, font, title_scale, title_thickness)
        text_x = (self.cover_x + self.cover_width // 2 - text_w // 2) if has_lyrics else (W // 2 - text_w // 2)
        text_y = self.cover_y + self.cover_height + text_y_offset

        # Shadow then main text
        cv2.putText(self.text_image, title, (text_x + 4, text_y + 4), font,
                    title_scale, (30, 30, 30), title_thickness + 2)
        cv2.putText(self.text_image, title, (text_x, text_y), font,
                    title_scale, (255, 255, 255), title_thickness)

        label = f"{date[:4]} - {artist}"
        (label_w, label_h), _ = cv2.getTextSize(label, font, label_scale, label_thickness)
        label_x = (self.cover_x + self.cover_width // 2 - label_w // 2) if has_lyrics else (W // 2 - label_w // 2)
        label_y = text_y + text_h + 13

        cv2.putText(self.text_image, label, (label_x + 4, label_y + 4), font,
                    label_scale, (30, 30, 30), label_thickness + 2)
        cv2.putText(self.text_image, label, (label_x, label_y), font,
                    label_scale, (200, 200, 200), label_thickness)

    def _create_background(self, small):
        """
        Generate an animated blurred background from the cover art:
          1. tile cover to 3x3 mosaic to avoid border artifacts on rotation
          2. rotate 45 degrees and heavily blur
          3. crop and scale to cover the full frame
          4. create a downscaled version used for efficient zooming on GPU
        """
        tile = np.tile(small, (3, 3, 1))
        center = (tile.shape[1] / 2, tile.shape[0] / 2)
        rot_mat = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(tile, rot_mat, (tile.shape[1], tile.shape[0]))
        blur = cv2.GaussianBlur(rotated, (101, 101), 0)

        # compute crop region approximately matching a single tile area
        corners = np.hstack([
            np.array([[self.cover_width, self.cover_height],
                      [self.cover_width * 2, self.cover_height],
                      [self.cover_width * 2, self.cover_height * 2],
                      [self.cover_width, self.cover_height * 2]], float),
            np.ones((4, 1), float)
        ])
        rotated_corners = (rot_mat @ corners.T).T
        min_x, max_x = int(rotated_corners[:, 0].min()), int(rotated_corners[:, 0].max())
        min_y, max_y = int(rotated_corners[:, 1].min()), int(rotated_corners[:, 1].max())
        background_crop = blur[min_y:max_y, min_x:max_x]

        scale_factor = max(W / background_crop.shape[1], H / background_crop.shape[0])
        new_w = int(background_crop.shape[1] * scale_factor)
        new_h = int(background_crop.shape[0] * scale_factor)
        background_full = cv2.resize(background_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # small downsampled version to accelerate zoom computations on GPU
        self.small_height = int(background_full.shape[0] * DS_FACTOR)
        self.small_width = int(background_full.shape[1] * DS_FACTOR)
        background_small = cv2.resize(background_full, (self.small_width, self.small_height), interpolation=cv2.INTER_LINEAR)

        self.background_gpu = cp.asarray(background_small)

    def _render_frames(self, audio_file, lyrics, output_file, total_frames, audio_duration, has_lyrics):
        """
        Core loop that composes frames and encodes via FFmpeg.
        Uses a dedicated writer thread to stream raw frames to FFmpeg stdin.
        """
        try:
            base_frames = int(audio_duration * FPS)
            buffer_size = min(QUEUE_MAX * H * W * 3, 32 * 1024 * 1024)
            ffmpeg_process = subprocess.Popen([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                '-i', audio_file,
                '-c:v', 'h264_nvenc', '-preset', 'p7', '-b:v', BITRATE_V,
                '-c:a', 'aac', '-b:a', BITRATE_A,
                '-pix_fmt', 'yuv420p', '-shortest', output_file
            ], stdin=subprocess.PIPE, bufsize=buffer_size)

            frame_queue = Queue(maxsize=QUEUE_MAX)

            def frame_writer():
                """Thread that writes frames to ffmpeg stdin."""
                while True:
                    frame_data = frame_queue.get()
                    if frame_data is None:
                        break
                    ffmpeg_process.stdin.write(frame_data)
                    frame_queue.task_done()

            writer_thread = Thread(target=frame_writer, daemon=True)
            writer_thread.start()

            zoom_factors = self._precompute_zoom_factors(total_frames)
            text_cache = self._precompute_text_properties(lyrics, total_frames, audio_duration) if has_lyrics else {}

            start_time = time.time()
            processed_count = 0
            last_frame_data = None

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = []

                for frame_index in range(total_frames):
                    current_buffer = next(self.buffer_cycle)
                    self.frame_ready[current_buffer].clear()

                    # detect if this frame is part of the added "fill" (duplicate) frames:
                    is_duplicate_frame = (frame_index >= base_frames and last_frame_data is not None)

                    if is_duplicate_frame and last_frame_data is not None:
                        # fill with previously saved last real frame
                        self.ping_pong_arrays[current_buffer][:] = last_frame_data
                        text_params = text_cache.get(base_frames - 1, []) if has_lyrics and base_frames > 0 else []
                    else:
                        frame_progress, zoom_factor = zoom_factors[frame_index]

                        with self.main_stream:
                            zoomed_background = self._compute_zoom_safely(zoom_factor)
                            # copy zoomed background into GPU frame buffer
                            self.frame_gpu[:] = zoomed_background

                            # configure kernel grid and blocks
                            blocks_x = (W + BLOCK_SIZE - 1) // BLOCK_SIZE
                            blocks_y = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
                            block_grid = (blocks_x, blocks_y)
                            thread_block = (BLOCK_SIZE, BLOCK_SIZE)

                            # invoke kernels (grid, block, args)
                            self.render_cover(block_grid, thread_block,
                                              (self.frame_gpu, self.cover_gpu, self.cover_x, self.cover_y,
                                               self.cover_width, self.cover_height, W, H))
                            self.blend_overlay_kernel(block_grid, thread_block,
                                                      (self.frame_gpu, self.fixed_text_gpu, W, H))

                            # synchronize and copy to pinned host memory
                            self.main_stream.synchronize()
                            cp.asnumpy(self.frame_gpu, out=self.ping_pong_arrays[current_buffer])

                        text_params = text_cache.get(frame_index, []) if has_lyrics else []

                    if has_lyrics:
                        # submit text rendering job (it will draw into the pinned CPU buffer)
                        future = executor.submit(self._render_text_worker,
                                                 (frame_index, current_buffer, text_params))
                    else:
                        # no lyrics: mark buffer ready immediately
                        self.frame_ready[current_buffer].set()
                        future = executor.submit(lambda: (frame_index, current_buffer))

                    futures.append((frame_index, future))

                    # Flush completed futures in small batches to keep memory usage bounded
                    while len(futures) >= PREFETCH_FRAMES:
                        batch = futures[:BATCH_SIZE]
                        futures = futures[BATCH_SIZE:]
                        for f_idx, fut in batch:
                            try:
                                frame_idx, buffer_idx = fut.result(timeout=5.0)
                                self.frame_ready[buffer_idx].wait(timeout=5.0)
                                frame_queue.put(self.ping_pong_arrays[buffer_idx].tobytes())
                                processed_count += 1
                                # only update last_frame_data for real frames (not duplicates)
                                if frame_idx < base_frames:
                                    last_frame_data = self.ping_pong_arrays[buffer_idx].copy()
                            except Exception as e:
                                print(f"Error processing frame {f_idx}: {e}")

                    # periodic progress report
                    if frame_index % 500 == 0 and frame_index > 0:
                        elapsed = time.time() - start_time
                        fps_rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total_frames - processed_count) / fps_rate if fps_rate > 0 else 0
                        print(f"   Progress: {processed_count}/{total_frames} frames "
                              f"({processed_count / total_frames * 100:.1f}%) - "
                              f"{fps_rate:.1f} FPS - ETA: {eta:.0f}s")
                        self.memory_pool.free_all_blocks()
                        if len(self.zoom_cache) > 15:
                            # prune oldest entries
                            keys = list(self.zoom_cache.keys())[:7]
                            for k in keys:
                                del self.zoom_cache[k]

                # finish pending futures
                for frame_idx, fut in futures:
                    try:
                        f_idx, buffer_idx = fut.result(timeout=10.0)
                        self.frame_ready[buffer_idx].wait(timeout=10.0)
                        frame_queue.put(self.ping_pong_arrays[buffer_idx].tobytes())
                        processed_count += 1
                    except Exception as e:
                        print(f"Error finalizing frame {frame_idx}: {e}")

            # signal writer thread and finish ffmpeg
            frame_queue.put(None)
            writer_thread.join(timeout=30.0)
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

            elapsed = time.time() - start_time
            print(f"   Completed: {elapsed:.2f}s → {processed_count / elapsed:.2f} FPS" if elapsed > 0 else "   Completed")

            return True

        except Exception as e:
            print(f"Rendering error: {e}")
            return False

    def _precompute_zoom_factors(self, total_frames):
        """
        Precompute zoom factors for smooth non-linear animated zoom.
        Returns list of tuples: (progress, zoom_factor).
        """
        zooms = []
        for i in range(total_frames):
            progress = min(i / (total_frames * 3 / 2), 1.0)
            zoom_factor = 1.0 + progress * MAX_ZOOM
            zooms.append((progress, zoom_factor))
        return zooms

    def _precompute_text_properties(self, lyrics, total_frames, audio_duration):
        """
        Precompute all text rendering properties to avoid repeated heavy calls
        to getTextSize or other OpenCV text metrics in the hot loop.
        Stores per-frame text parameter lists.
        """
        text_cache = {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_frames = int(audio_duration * FPS)

        for frame_index in range(total_frames):
            # If this frame index is in the "fill" area (>= base_frames), map it to the last real-frame time
            if frame_index >= base_frames:
                current_time = (base_frames - 1) / FPS if base_frames > 0 else 0.0
            else:
                current_time = frame_index / FPS

            active_line = -1
            for idx, (start, end, _) in enumerate(lyrics):
                if start <= current_time < end:
                    active_line = idx
                    break

            frame_params = []
            if active_line >= 0:
                for offset in range(-2, 3):
                    lyr_idx = active_line + offset
                    if 0 <= lyr_idx < len(lyrics):
                        text = lyrics[lyr_idx][2]
                        y_pos = H // 2 + offset * 70
                        if offset == 0:
                            scale, thickness, color = 1.458, 6, (255, 255, 255)
                            shadow_th = 8
                        else:
                            scale, thickness, color = 1.134, 4, (180, 180, 180)
                            shadow_th = 6

                        (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thickness)
                        text_x = self.cover_x + self.cover_width + 45  # right-side offset
                        frame_params.append((text, text_x, y_pos, scale, thickness, shadow_th, color))
            text_cache[frame_index] = frame_params

        return text_cache

    def _render_text_worker(self, params):
        """
        Worker that paints lyric lines directly into the CPU pinned buffer for a frame.
        After drawing, it sets the corresponding event to mark the buffer ready.
        """
        if params is None:
            return None
        frame_index, buffer_index, text_params = params
        frame = self.ping_pong_arrays[buffer_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        shadow_offset_x, shadow_offset_y = 3, 3

        for text, x, y, scale, thickness, shadow_thickness, color in text_params:
            # draw shadow
            sx, sy = x + shadow_offset_x, y + shadow_offset_y
            cv2.putText(frame, text, (sx, sy), font, scale, (20, 20, 20), shadow_thickness, lineType=cv2.LINE_AA)
            # draw main text
            cv2.putText(frame, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

        self.frame_ready[buffer_index].set()
        return (frame_index, buffer_index)

    def _compute_zoom_safely(self, zoom_factor):
        """
        Compute a GPU zoomed background using the downsampled background as source.
        Uses a cache keyed by rounded zoom factor to avoid recomputation.
        Returns a cp.ndarray of shape (H, W, 3) dtype=uint8.
        """
        cache_key = int(zoom_factor * 1000)
        if cache_key in self.zoom_cache:
            return self.zoom_cache[cache_key]

        scale_v = zoom_factor * (H / self.small_height)
        scale_h = zoom_factor * (W / self.small_width)
        zoomed = zoom(self.background_gpu, (scale_v, scale_h, 1.0), order=1).astype(cp.uint8)

        zh, zw, _ = zoomed.shape
        yoff, xoff = (zh - H) // 2, (zw - W) // 2
        cropped = zoomed[yoff:yoff + H, xoff:xoff + W].copy()

        if len(self.zoom_cache) >= 30:
            keys = list(self.zoom_cache.keys())[:15]
            for k in keys:
                del self.zoom_cache[k]

        self.zoom_cache[cache_key] = cropped
        return cropped

    def cleanup(self):
        """Free CUDA resources, clear caches and memory pools to avoid leaks."""
        if hasattr(self, 'zoom_cache'):
            self.zoom_cache.clear()
        if hasattr(self, 'main_stream'):
            del self.main_stream
        if hasattr(self, 'memory_pool'):
            self.memory_pool.free_all_blocks()
        if hasattr(self, 'pinned_memory_pool'):
            self.pinned_memory_pool.free_all_blocks()

# -------------------------
# BATCH PROCESSING / CLI
# -------------------------

def setup_directories():
    """Ensure required directories exist (creates them if missing)."""
    for d in (MUSIC_FOLDER, LYRICS_FOLDER, OUTPUT_FOLDER, TEMP_DIR):
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")


def validate_environment():
    """
    Check presence of required system tools and Python packages:
      - ffmpeg and ffprobe
      - CUDA / CuPy (best-effort)
      - OpenCV
      - Mutagen
    Returns True when the environment looks usable.
    """
    print("Validating runtime environment...")
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("ffmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not found. Install from https://ffmpeg.org/")
        return False

    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        print("ffprobe found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffprobe not found. It ships with ffmpeg.")
        return False

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            print(f"CUDA available ({device_count} GPU device(s))")
        else:
            print("CUDA runtime present but no GPU devices detected")
    except Exception:
        print("CuPy/CUDA not fully available — GPU acceleration may fail")

    try:
        print(f"OpenCV found (version {cv2.__version__})")
    except Exception:
        print("OpenCV not installed. pip install opencv-python")
        return False

    try:
        _ = FLAC
        print("Mutagen found")
    except Exception:
        print("Mutagen not installed. pip install mutagen")
        return False

    return True


def print_usage():
    """Print detailed usage instructions for the script."""
    print("""
Batch Music Video Renderer - Usage
=================================
- Place audio files in the 'songs/' folder (recommended FLAC).
- Place optional synchronized .lrc files in 'lyrics/'.
- Output videos will be written to 'output/'.
- Files should ideally start with a 3-digit number for ordering:
    001 - Song.flac
    002 - AnotherSong.mp3
Commands:
    python script.py            # run batch
    python script.py --help     # show help
    python script.py --config   # interactive config
""")


def interactive_config():
    """Allow the user to change the folder configuration interactively."""
    global MUSIC_FOLDER, LYRICS_FOLDER, OUTPUT_FOLDER
    print("\nInteractive configuration (press Enter to keep current value)\n")
    resp = input(f"Music folder (current: '{MUSIC_FOLDER}'): ").strip()
    if resp:
        MUSIC_FOLDER = resp
    resp = input(f"Lyrics folder (current: '{LYRICS_FOLDER}'): ").strip()
    if resp:
        LYRICS_FOLDER = resp
    resp = input(f"Output folder (current: '{OUTPUT_FOLDER}'): ").strip()
    if resp:
        OUTPUT_FOLDER = resp
    print(f"Configuration updated: Music='{MUSIC_FOLDER}', Lyrics='{LYRICS_FOLDER}', Output='{OUTPUT_FOLDER}'")


def main():
    """Main coordinator: find music files, initialize renderer and process each file sequentially."""
    print("Starting batch music video processing")
    print("=" * 60)
    setup_directories()

    music_files = find_music_files()
    if not music_files:
        print(f"No music files found in '{MUSIC_FOLDER}'. Files should start with numbers (e.g. 001...).")
        return

    print(f"Found {len(music_files)} music file(s):")
    for i, p in enumerate(music_files, 1):
        print(f"   {i:2d}. {os.path.basename(p)}")

    renderer = VideoRenderer()
    total_start = time.time()
    processed = failed = skipped = 0

    try:
        for i, audio_file in enumerate(music_files, 1):
            print("\n" + "=" * 60)
            print(f"Processing file {i}/{len(music_files)}")
            print("=" * 60)

            lrc_file = find_lyrics_file(audio_file)
            base = os.path.splitext(os.path.basename(audio_file))[0]
            out_path = os.path.join(OUTPUT_FOLDER, f"{base}.mp4")

            if os.path.exists(out_path):
                print(f"Output already exists: {out_path}")
                skipped += 1
                continue

            success = renderer.render_video(audio_file, lrc_file, out_path)
            if success:
                processed += 1
                print(f"Video generated: {out_path}")
            else:
                failed += 1
                print(f"Failed to render video for: {os.path.basename(audio_file)}")

            # free caches to avoid memory growth between files
            if hasattr(renderer, 'zoom_cache'):
                renderer.zoom_cache.clear()
            if hasattr(renderer, 'memory_pool'):
                renderer.memory_pool.free_all_blocks()
            if hasattr(renderer, 'pinned_memory_pool'):
                renderer.pinned_memory_pool.free_all_blocks()

            elapsed_total = time.time() - total_start
            avg_time = elapsed_total / i if i > 0 else 0
            remaining = len(music_files) - i
            eta_total = remaining * avg_time

            print(f"Overall progress: {i}/{len(music_files)} processed")
            print(f"Success: {processed}")
            print(f"Skipped: {skipped}")
            print(f"Failed: {failed}")
            print(f"Avg time/file: {avg_time:.1f}s")
            if remaining > 0:
                print(f"ETA remaining: {eta_total/60:.1f} minutes")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"Critical error during batch processing: {e}")
    finally:
        if 'renderer' in locals():
            renderer.cleanup()
        cleanup_temp()

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"Files found: {len(music_files)}")
        print(f"Videos generated: {processed}")
        print(f"Already existed: {skipped}")
        print(f"Failed: {failed}")
        if processed > 0:
            print(f"Average time per video: {total_elapsed/processed:.1f}s")
            print(f"Videos written to: {os.path.abspath(OUTPUT_FOLDER)}")
        print("Processing finished!")

# -------------------------
# PROGRAM ENTRYPOINT
# -------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print_usage()
            sys.exit(0)
        elif sys.argv[1] in ['--config', '-c']:
            interactive_config()

    if not validate_environment():
        print("Environment is not configured correctly. Please install dependencies and try again.")
        sys.exit(1)

    setup_directories()

    try:
        main()
    except KeyboardInterrupt:
        print("Process cancelled by user.")
    except Exception as e:
        print(f"Unhandled critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
