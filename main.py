"""CLI entrypoint, Web Studio launcher, and backward-compatible facade for Beautiful Lyric Video Creator.

Supports both batch CLI execution (`python main.py`) and the interactive Web Studio daemon
(`python main.py --ui` or `run_ui.bat`), matching the Namida-Wrapped background launcher architecture.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import webbrowser

from lyric_video_creator import (
    CUPY_AVAILABLE,
    DEFAULT_CONFIG,
    CPUVideoRenderer,
    GPUVideoRenderer,
    RenderConfig,
    VideoRenderer,
    configure_logging,
    debug_print,
    extract_metadata,
    find_lyrics_file,
    find_music_files,
    get_audio_duration,
    get_logger,
    get_renderer,
    parse_lyrics,
    run_batch_pipeline,
    sanitize_text,
    setup_directories,
    strip_title_extra,
    validate_environment,
)

ROOT_DIR: Path = Path(__file__).resolve().parent
LOGS_DIR: Path = ROOT_DIR / "logs"
SERVER_LOG_PATH: Path = LOGS_DIR / "studio_server.log"

# ------------------------- Legacy Module-Level Constants -------------------------
MUSIC_FOLDER: str = str(DEFAULT_CONFIG.music_folder)
LYRICS_FOLDER: str = str(DEFAULT_CONFIG.lyrics_folder)
OUTPUT_FOLDER: str = str(DEFAULT_CONFIG.output_folder)

W: int = DEFAULT_CONFIG.width
H: int = DEFAULT_CONFIG.height
FPS: int = DEFAULT_CONFIG.fps
BITRATE_V: str = DEFAULT_CONFIG.video_bitrate
BITRATE_A: str = DEFAULT_CONFIG.audio_bitrate
LEFT_MARGIN: int = DEFAULT_CONFIG.left_margin

BLOCK_SIZE: int = DEFAULT_CONFIG.block_size
PING_PONG_BUFFERS: int = DEFAULT_CONFIG.ping_pong_buffers
QUEUE_MAX: int = DEFAULT_CONFIG.queue_max
MAX_LYRICS_LENGTH: int = DEFAULT_CONFIG.max_lyrics_length
MAX_LYRICS_LENGHT: int = MAX_LYRICS_LENGTH

GLOW_BLUR_KSIZE: int = DEFAULT_CONFIG.glow_blur_ksize
GLOW_INTENSITY_BASE: float = DEFAULT_CONFIG.glow_intensity_base
BLUR_STRENGTH: int = DEFAULT_CONFIG.blur_strength
FIXED_TEXT_GLOW_INTENSITY: float = DEFAULT_CONFIG.fixed_text_glow_intensity
ASSETS_GLOW_INTENSITY: float = DEFAULT_CONFIG.assets_glow_intensity
GLOW_PULSE_AMP: float = DEFAULT_CONFIG.glow_pulse_amp
GLOW_PULSE_FREQ: float = DEFAULT_CONFIG.glow_pulse_freq
GLOW_EDGE_PX: float = DEFAULT_CONFIG.glow_edge_px
GLOW_BOOST: float = DEFAULT_CONFIG.glow_boost
ALPHA_MIN: float = DEFAULT_CONFIG.alpha_min
DEFAULT_LAST_LINE_DURATION: float = DEFAULT_CONFIG.default_last_line_duration

P_LOW_PCT: float = DEFAULT_CONFIG.p_low_pct
P_HIGH_PCT: float = DEFAULT_CONFIG.p_high_pct
MIN_SAT: int = DEFAULT_CONFIG.min_sat
RANDOM_SAMPLE_TRIES: int = DEFAULT_CONFIG.random_sample_tries
RANDOM_CANDIDATES_MIN: int = DEFAULT_CONFIG.random_candidates_min
SAMPLE_GRID: int = DEFAULT_CONFIG.sample_grid

MIN_GLOW_SAT: int = DEFAULT_CONFIG.min_glow_sat
MIN_GLOW_VAL: int = DEFAULT_CONFIG.min_glow_val
DEBUG_TRIGGER: bool = DEFAULT_CONFIG.debug


def free_port(port: int) -> None:
    """Find and terminate any stale process currently listening on `port`.

    Why necessary:
    Prevents `[Errno 10048]` socket binding failures when relaunching Web Studio.
    """
    try:
        if os.name == "nt":
            no_win = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            cmd = f"netstat -ano | findstr LISTENING | findstr :{port}"
            output = subprocess.check_output(
                cmd, shell=True, text=True, stderr=subprocess.DEVNULL, creationflags=no_win
            )
            pids: set[int] = set()
            current_pid = os.getpid()
            for line in output.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1]:
                    try:
                        pid = int(parts[-1])
                        if pid != current_pid and pid > 0:
                            pids.add(pid)
                    except ValueError:
                        pass
            for pid in pids:
                print(f"[Web Studio] Port {port} is occupied by PID {pid}. Terminating process {pid}...")
                subprocess.run(
                    f"taskkill /F /PID {pid}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=no_win,
                )
        else:
            cmd = f"lsof -t -i:{port}"
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            current_pid = os.getpid()
            for line in output.strip().splitlines():
                try:
                    pid = int(line)
                    if pid != current_pid:
                        os.kill(pid, signal.SIGKILL)
                except ValueError:
                    pass
    except Exception:
        pass


def _wait_for_port(host: str, port: int, timeout: float = 6.0) -> bool:
    """Poll TCP socket until the background Uvicorn server accepts connections."""
    deadline = time.monotonic() + timeout
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((check_host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.12)
    return False


def _close_parent_shell_if_interactive() -> None:
    """Close the parent PowerShell/CMD console window if launched interactively on Windows."""
    if os.name != "nt":
        return
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return
        ppid = os.getppid()
        if ppid <= 0 or ppid == os.getpid():
            return
        no_win = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            f'tasklist /FI "PID eq {ppid}" /FO CSV /NH',
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=no_win,
        ).lower()
        if any(shell_name in out for shell_name in ("powershell.exe", "pwsh.exe", "cmd.exe")):
            subprocess.Popen(
                f"taskkill /PID {ppid} /F",
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=no_win | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
    except Exception:
        pass


def run_studio(
    host: str = "127.0.0.1",
    port: int = 8686,
    foreground: bool = False,
    keep_shell: bool = False,
) -> int:
    """Launch the interactive Web Studio server in background daemon or foreground mode.

    Why windowless background daemon by default:
    Matches the Namida-Wrapped user experience—spawns Uvicorn without a visible console window,
    waits for port readiness, opens the user's default browser, and closes the launcher shell.
    """
    setup_directories(DEFAULT_CONFIG)
    url = f"http://{host}:{port}"

    if not foreground:
        free_port(port)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--ui",
            "--host",
            host,
            "--port",
            str(port),
            "--foreground",
        ]

        popen_kwargs: dict[str, object] = {
            "cwd": str(ROOT_DIR),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            base_flags = (
                subprocess.CREATE_NO_WINDOW
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            breakaway_flag = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
            try:
                popen_kwargs["creationflags"] = base_flags | breakaway_flag
                proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
            except OSError:
                popen_kwargs["creationflags"] = base_flags
                proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
        else:
            popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]

        print(f"\n[Web Studio] Starting background server (PID {proc.pid}) on {url} ...")
        ready = _wait_for_port(host, port, timeout=6.0)
        if ready:
            print(f"[Web Studio] Server ready at {url} (logs: {SERVER_LOG_PATH})")
            if not os.environ.get("LYRIC_STUDIO_NO_BROWSER"):
                webbrowser.open(url)
            if not keep_shell:
                _close_parent_shell_if_interactive()
            return 0

        print(
            f"[Web Studio] Warning: Server did not respond within 6s. Check {SERVER_LOG_PATH}",
            file=sys.stderr,
        )
        return 1

    import uvicorn
    import threading

    free_port(port)

    # If running windowless under pythonw.exe or without attached console, direct output to server log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if sys.executable.lower().endswith("pythonw.exe") or not (sys.stdout and sys.stdout.isatty()):
        log_file = open(SERVER_LOG_PATH, "a", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file

    if not os.environ.get("LYRIC_STUDIO_NO_BROWSER"):
        def _open_when_ready() -> None:
            if _wait_for_port(host, port, timeout=6.0):
                webbrowser.open(url)

        threading.Thread(target=_open_when_ready, daemon=True, name="BrowserAutoOpener").start()

    print(f"\n[Web Studio] Starting Web Studio server on {url} ...")
    uvicorn.run(
        "lyric_video_creator.server.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for both batch CLI execution and Web Studio UI mode."""
    parser = argparse.ArgumentParser(
        prog="Beautiful-Lyric-Video-Creator",
        description=(
            "Generate synchronized lyric videos with dynamic glow effects using CUDA, "
            "PyTorch (ROCm/MPS/XPU), OpenCL, or CPU fallback."
        ),
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Web Studio UI in the browser.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the Web Studio UI server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8686,
        help="Port number for the Web Studio UI server (default: 8686).",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run the Web Studio server attached to the current terminal.",
    )
    parser.add_argument(
        "--keep-shell",
        action="store_true",
        help="Do not auto-close the terminal window after launching the background Web Studio.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "torch", "opencl", "cpu"),
        default=DEFAULT_CONFIG.backend,
        help="Compute backend selection (default: 'auto' probes CuPy -> Torch -> OpenCL -> CPU).",
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "nvenc", "amf", "qsv", "videotoolbox", "vaapi", "libx264"),
        default=DEFAULT_CONFIG.encoder,
        help="FFmpeg H.264 encoder selection (default: 'auto' probes GPU encoders -> libx264).",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=DEFAULT_CONFIG.music_folder,
        help="Directory containing input audio files (default: songs).",
    )
    parser.add_argument(
        "--lyrics-dir",
        type=Path,
        default=DEFAULT_CONFIG.lyrics_folder,
        help="Directory containing .lrc files (default: lyrics).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG.output_folder,
        help="Directory for generated .mp4 files (default: output).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_CONFIG.fps,
        help="Output video frame rate (default: 20).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CONFIG.width,
        help="Output video width in pixels (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CONFIG.height,
        help="Output video height in pixels (default: 1080).",
    )
    parser.add_argument(
        "--video-bitrate",
        type=str,
        default=DEFAULT_CONFIG.video_bitrate,
        help="FFmpeg video bitrate, e.g., '8000k' (default: 8000k).",
    )
    parser.add_argument(
        "--audio-bitrate",
        type=str,
        default=DEFAULT_CONFIG.audio_bitrate,
        help="FFmpeg audio bitrate, e.g., '320k' (default: 320k).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=DEFAULT_CONFIG.debug,
        help="Enable verbose debug logging and FFmpeg console output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, validate the environment, and launch either Web Studio or batch CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.ui:
        return run_studio(
            host=args.host,
            port=args.port,
            foreground=args.foreground,
            keep_shell=args.keep_shell,
        )

    config = RenderConfig(
        music_folder=args.music_dir,
        lyrics_folder=args.lyrics_dir,
        output_folder=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        video_bitrate=args.video_bitrate,
        audio_bitrate=args.audio_bitrate,
        backend=args.backend,
        encoder=args.encoder,
        debug=args.debug,
    )

    configure_logging(debug=config.debug)
    logger = get_logger()

    if not validate_environment():
        logger.error("Environment validation failed. Install required dependencies and try again.")
        return 1

    setup_directories(config)
    return run_batch_pipeline(config)


__all__ = [
    "CUPY_AVAILABLE",
    "CPUVideoRenderer",
    "GPUVideoRenderer",
    "RenderConfig",
    "VideoRenderer",
    "build_arg_parser",
    "debug_print",
    "extract_metadata",
    "find_lyrics_file",
    "find_music_files",
    "free_port",
    "get_audio_duration",
    "get_renderer",
    "main",
    "parse_lyrics",
    "run_studio",
    "sanitize_text",
    "setup_directories",
    "strip_title_extra",
    "validate_environment",
]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Process cancelled by user.")
        sys.exit(130)
    except Exception as exc:
        print("Unhandled critical error:", exc)
        traceback.print_exc()
        sys.exit(1)
