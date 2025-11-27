# Beautiful animated music video with lyrics

## Project Overview

This tool automatically generates music videos with album cover animations and synchronized lyrics (if available, LRC format). It processes a batch of audio files (FLAC format, revise code to change) in numerical order, applies GPU-accelerated effects, and produces MP4 videos with a smooth zooming background. To use it just just add the music files to the "songs" folder and the synchronized lyrics to the "lyrics" folders and run [the notebook](BLVF.ipynb).

## Key Features

- **GPU-Accelerated Processing**: Utilizes CUDA/CuPy kernels for fast image blending and zoom animations.
- **Dual Modes**:
  - **With Lyrics**: Side-aligned cover with synchronized subtitle-style lyrics.
  - **Without Lyrics**: Centered cover, enlarged by 20%, with decorative frame and text overlay.
- **Batch Workflow**: Scans a folder of numbered audio files, skips existing outputs, and reports progress and statistics.
- **High Performance**: Ping-pong buffers, prefetching, thread pools, and hardware NVENC encoding.
- **Assembling**: To merge all generated outputs into a single file, it is recommended to use `LosslessCut` program.

## Variants

1. **Cover-Only Mode**  
   - For tracks lacking `.lrc` files.  
   - Displays a centered, enhanced cover with title/artist text below.
2. **Lyrics Mode**  
   - For tracks with synchronized `.lrc` lyrics.
   - Renders up to five lines of lyrics per frame, highlighting the active line.

## Screenshots

- **Cover-Only Mode** 

  ![Cover Only Example](rm/cover_only.png)

- **Lyrics Mode**

  ![Lyrics Mode Example](rm/lyrics_mode.png)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/albertnica/BeautifulLyricVideoForge
   cd BeautifulLyricVideoForge

2. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv BLVF
   BLVF\Scripts\activate

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt

   # Beautiful animated music video with lyrics

## Lyrics Retriever

Automatically download lyrics from Genius using their API (you must create an API client on the Genius website and add your credentials to the script). The lyrics are then synchronized using OpenAI’s Whisper with GPU acceleration. This is intended as a fallback when LRC files are not available via the “LRCGET” program, which remains the preferred option.

## Installation

1. **Install Python dependencies**  
   ```bash
   pip install -r requirements_l.txt

2. Install PyTorch with CUDA support (adjust CUDA version as needed):
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129