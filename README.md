# unwatermark

**The most natural AI video watermark removal tool available (as of May 2026).**

Remove corner watermarks from AI-generated videos — **zero artifacts, original resolution, no content loss.**

Powered by **LaMa deep inpainting**: instead of cropping or using weak OpenCV filters, unwatermark reconstructs the natural texture underneath the watermark, pixel by pixel. The output is indistinguishable from an unwatermarked original.

## How it works

```
1. DETECT  →  Temporal edge stability analysis pinpoints the watermark corner
2. MASK    →  Precise pixel mask covers text, border, and translucent fill
3. INPAINT →  LaMa deep inpainting reconstructs natural texture underneath
4. ENCODE  →  FFmpeg reassembles at original resolution with audio
```

No cropping. No scaling. No content loss. Same resolution in, same resolution out.

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

**Requirements:** Python 3.8+, FFmpeg, and iopaint (for LaMa model).

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
# Auto-detect and remove watermark (LaMa inpainting)
python unwatermark.py input.mp4

# Specify output path
python unwatermark.py input.mp4 -o clean.mp4

# Manually specify watermark corner
python unwatermark.py input.mp4 --corner br

# Preview before/after (saves comparison PNGs, no full processing)
python unwatermark.py input.mp4 --preview

# Adjust mask padding (default: 20px)
python unwatermark.py input.mp4 --padding 30

# Adjust quality (CRF 0-51, lower = better, default: 18)
python unwatermark.py input.mp4 --quality 15

# Use GPU acceleration (if available)
python unwatermark.py input.mp4 --device cuda
```

### Output

```
  ╦ ╦┌┐┌┬ ┬┌─┐┌┬┐┌─┐┬─┐┌┬┐┌─┐┬─┐┬┌─
  ║ ║│││││││├─┤ │ ├┤ ├┬┘│││├─┤├┬┘├┴┐
  ╚═╝┘└┘└┴┘└┘ ┘ ┴ └─┘┴└─┴ ┴┴ ┴┴└─┴ ┴  v2.0.0
  Naturally remove watermarks from AI-generated videos

  Input:  video.mp4
  1280x720 | 30.0fps | 32.4s | 971 frames | 5.4MB

  [1/4] Detecting watermark...
  Corner: bottom-right (auto-detected)
  Region: (1074, 669) 178x40

  [2/4] Building inpainting mask...
  Mask: 9463 pixels (1.01% of frame)

  [3/4] Inpainting frames...
  Engine: LaMa deep inpainting (best quality)
  ████████████████████████ 971/971 frames (100%)
  Encoding final video...

  [4/4] Done!

  Output: video_clean.mp4
  1280x720 (original resolution) | 5.4MB → 6.8MB | 42.3s
```

## Benchmarks (May 2026)

Tested on real Seedance 2.0 outputs across varied scenes (bright/dark backgrounds, high/low contrast):

| Tool | Artifacts? | Original resolution? | Speed (30s video) | GPU required? |
|------|:-:|:-:|:-:|:-:|
| [seedance-watermark-remover](https://github.com/SamurAIGPT/seedance-2.0-watermark-remover) (OpenCV) | Visible smudging | Yes | ~45s | No |
| FFmpeg `delogo` | Faint ghost | Yes | ~3s | No |
| Crop-and-scale | None, but loses content | No (cropped) | ~5s | No |
| **unwatermark v2 (LaMa)** | **None** | **Yes** | **~45s** | **No (CPU ok)** |

## How it compares

| | OpenCV inpaint | FFmpeg delogo | Crop+scale | **unwatermark** |
|---|:-:|:-:|:-:|:-:|
| Zero artifacts | No | No | Yes | **Yes** |
| Original resolution | Yes | Yes | No | **Yes** |
| No content loss | Yes | Yes | No | **Yes** |
| Works on light BG | No | No | Yes | **Yes** |
| Auto-detection | No | No | No | **Yes** |
| No GPU needed | Yes | Yes | Yes | **Yes** |

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output file path | `<input>_clean.<ext>` |
| `--corner` | Watermark corner: `tl`, `tr`, `bl`, `br` | Auto-detect |
| `--padding` | Extra pixels around watermark mask | `20` |
| `--quality` | CRF value (0-51, lower = better) | `18` |
| `--device` | PyTorch device (`cpu`, `cuda`, `mps`) | `cpu` |
| `--preview` | Save before/after comparison PNGs | Off |
| `--quiet` | Suppress terminal output | Off |

## License

MIT
