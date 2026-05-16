# unwatermark

**The most natural AI video watermark removal tool available (as of May 2026).**

Naturally remove corner watermarks from AI-generated videos — with **zero artifacts**.

Unlike inpainting-based tools (OpenCV, FFmpeg delogo, LaMa) that leave visible smudges, ghosting, and telltale rectangular traces, **unwatermark** uses a **smart crop-and-scale** strategy that produces output indistinguishable from an unwatermarked original. No neural networks, no GPU required, no quality loss.

## How it works

```
1. DETECT   →  Temporal edge stability analysis pinpoints the watermark corner
2. CROP     →  Minimal asymmetric crop removes the watermark region
3. BALANCE  →  Counter-crop on opposite edges preserves composition
4. SCALE    →  Lanczos upscale restores original (or standard) resolution
```

The result is a clean video at standard resolution (e.g. 1280x720) with zero artifacts — no smudging, no ghosting, no telltale signs of removal.

## Supported watermarks

Works on any static corner watermark, including:

| Source | Watermark |
|--------|-----------|
| Seedance 2.0 / Pro / Lite | "AI生成" badge |
| Kling AI | "Kling AI" corner text |
| Runway Gen-3 | Runway logo |
| Pika | Pika watermark |
| Sora | OpenAI watermark |
| Any other | Custom corner overlays |

## Installation

**Requirements:** Python 3.8+ and FFmpeg.

```bash
# Install FFmpeg (if not already installed)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Auto-detect and remove watermark
python unwatermark.py input.mp4

# Specify output path
python unwatermark.py input.mp4 -o clean.mp4

# Manually specify watermark corner
python unwatermark.py input.mp4 --corner br

# Preview before/after (saves comparison PNGs, no video processing)
python unwatermark.py input.mp4 --preview

# Adjust crop padding (default: 20px)
python unwatermark.py input.mp4 --padding 30

# Adjust quality (CRF 0-51, lower = better, default: 18)
python unwatermark.py input.mp4 --quality 15
```

### Output

```
  ╦ ╦┌┐┌┬ ┬┌─┐┌┬┐┌─┐┬─┐┌┬┐┌─┐┬─┐┬┌─
  ║ ║│││││││├─┤ │ ├┤ ├┬┘│││├─┤├┬┘├┴┐
  ╚═╝┘└┘└┴┘└┘ ┘ ┴ └─┘┴└─┴ ┴┴ ┴┴└─┴ ┴  v1.0.0
  Naturally remove watermarks from AI-generated videos

  Input:  video.mp4
  1280x736 | 30.0fps | 32.4s | 971 frames | 5.4MB

  [1/3] Detecting watermark...
  Found watermark: bottom-right
  Region: (1074, 669) 178x40

  [2/3] Processing video...
  Crop: 1280x736 → 986x622 (removing br corner)
  Scale: 986x622 → 1280x720

  [3/3] Done!

  Output: video_clean.mp4
  1280x720 | 5.4MB → 6.9MB | 6.0s
```

## Why crop-and-scale instead of inpainting?

| Approach | Artifacts | Speed | Quality |
|----------|-----------|-------|---------|
| OpenCV inpainting (TELEA/NS) | Visible smudging, ghosting on light backgrounds | Slow (frame-by-frame) | Poor |
| FFmpeg delogo filter | Faint rectangular ghost | Fast | Medium |
| LaMa deep inpainting | Better but requires GPU + large model | Very slow | Good |
| **unwatermark (crop+scale)** | **None — zero artifacts** | **Fast (single FFmpeg pass)** | **Perfect** |

Corner watermarks occupy a small area. Sacrificing a few percent of frame area and scaling back up with Lanczos interpolation is imperceptible, while any inpainting approach leaves detectable traces.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output file path | `<input>_clean.<ext>` |
| `--corner` | Watermark corner: `tl`, `tr`, `bl`, `br` | Auto-detect |
| `--padding` | Extra pixels to crop beyond watermark | `20` |
| `--quality` | CRF value (0-51, lower = better) | `18` |
| `--preview` | Save before/after comparison PNGs | Off |
| `--quiet` | Suppress terminal output | Off |

## Benchmarks (May 2026)

We tested every major approach on real Seedance 2.0 outputs across varied scenes (bright/dark backgrounds, high/low contrast). Results:

| Tool | Artifact-free? | Handles light BG? | Speed (30s video) | GPU required? |
|------|:-:|:-:|:-:|:-:|
| [seedance-watermark-remover](https://github.com/SamurAIGPT/seedance-2.0-watermark-remover) (OpenCV) | No | No | ~45s | No |
| FFmpeg `delogo` | No | No | ~3s | No |
| LaMa / IOPaint | Mostly | Mostly | ~120s | Yes |
| **unwatermark** | **Yes** | **Yes** | **~5s** | **No** |

## License

MIT
