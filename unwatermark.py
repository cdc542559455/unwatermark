#!/usr/bin/env python3
"""
unwatermark — naturally remove corner watermarks from AI-generated videos.

Instead of inpainting (which leaves visible artifacts), this tool uses a
smart crop-and-scale strategy:

  1. Auto-detect watermark position via temporal edge stability analysis
  2. Crop minimally to eliminate the watermark
  3. Distribute the crop to preserve composition balance
  4. Scale back to the original (or a standard) resolution

The result is indistinguishable from an unwatermarked original.

Supports: Seedance, Kling, Runway, Pika, Sora, and any corner watermark.

Usage:
    python unwatermark.py input.mp4
    python unwatermark.py input.mp4 -o clean.mp4
    python unwatermark.py input.mp4 --corner br --preview
"""

import argparse
import os
import sys
import time
import subprocess
import shutil
import tempfile

import cv2
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

VERSION = "1.0.0"

CORNERS = {
    "tl": "top-left",
    "tr": "top-right",
    "bl": "bottom-left",
    "br": "bottom-right",
}

# Standard resolutions to snap to (width, height)
STANDARD_RESOLUTIONS = [
    (3840, 2160),  # 4K
    (2560, 1440),  # 2K
    (1920, 1080),  # 1080p
    (1280, 720),   # 720p
    (854, 480),    # 480p
]

# ANSI colors
class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    RESET   = "\033[0m"


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg, color=C.RESET):
    print(f"{color}{msg}{C.RESET}")


def log_step(step, total, msg):
    bar_w = 20
    filled = int(bar_w * step / total)
    bar = f"{'█' * filled}{'░' * (bar_w - filled)}"
    print(f"\r  {C.CYAN}{bar}{C.RESET} {C.DIM}{step}/{total}{C.RESET} {msg}", end="", flush=True)
    if step == total:
        print()


def get_video_info(path):
    """Return (width, height, fps, frame_count, has_audio) for a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Check for audio stream
    has_audio = False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        has_audio = "audio" in r.stdout
    except Exception:
        pass

    return w, h, fps, count, has_audio


def find_nearest_standard(w, h):
    """Find the nearest standard resolution that fits within (w, h)."""
    best = None
    best_diff = float("inf")
    for sw, sh in STANDARD_RESOLUTIONS:
        if sw <= w and sh <= h:
            diff = (w - sw) + (h - sh)
            if diff < best_diff:
                best_diff = diff
                best = (sw, sh)
    return best


# ── Core: Watermark Detection ───────────────────────────────────────────────

def detect_watermark(video_path, num_samples=50):
    """
    Detect which corner contains a static watermark overlay.

    Strategy: sample many frames, run edge detection, and accumulate.
    Static overlays (watermarks) produce edges in the same position across
    all frames, while scene content changes. The corner with the highest
    concentration of temporally-stable edges contains the watermark.

    Returns: (corner, bbox) where corner is 'tl'|'tr'|'bl'|'br'
             and bbox is (x, y, w, h) of the watermark region.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames evenly across the video
    indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int)
    accum = np.zeros((vh, vw), dtype=np.float64)
    count = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 80)
        accum += (edges > 0).astype(np.float64)
        count += 1

    cap.release()

    if count == 0:
        return None, None

    # Normalize: pixels that have edges in >30% of frames are "static"
    stability = accum / count
    static_mask = (stability > 0.30).astype(np.uint8) * 255

    # Analyze each corner quadrant
    mid_y, mid_x = vh // 2, vw // 2
    # Scan a generous region (25% of frame) per corner
    scan_h, scan_w = vh // 4, vw // 4

    corners = {
        "tl": static_mask[0:scan_h, 0:scan_w],
        "tr": static_mask[0:scan_h, vw - scan_w:vw],
        "bl": static_mask[vh - scan_h:vh, 0:scan_w],
        "br": static_mask[vh - scan_h:vh, vw - scan_w:vw],
    }

    # Score each corner by density of static edges
    scores = {}
    for corner, region in corners.items():
        # Use connected components to find clusters (watermarks are clustered)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        # Find the largest non-background component
        max_area = 0
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
        scores[corner] = max_area

    best_corner = max(scores, key=scores.get)

    if scores[best_corner] < 100:
        # No significant watermark detected
        return None, None

    # Get bounding box of the watermark in the detected corner
    region = corners[best_corner]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    max_area = 0
    best_idx = -1
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            best_idx = i

    if best_idx < 0:
        return best_corner, None

    rx = stats[best_idx, cv2.CC_STAT_LEFT]
    ry = stats[best_idx, cv2.CC_STAT_TOP]
    rw = stats[best_idx, cv2.CC_STAT_WIDTH]
    rh = stats[best_idx, cv2.CC_STAT_HEIGHT]

    # Convert from corner-local coords to full-frame coords
    offset_x, offset_y = 0, 0
    if best_corner in ("tr", "br"):
        offset_x = vw - scan_w
    if best_corner in ("bl", "br"):
        offset_y = vh - scan_h

    bbox = (offset_x + rx, offset_y + ry, rw, rh)

    return best_corner, bbox


# ── Core: Smart Crop ────────────────────────────────────────────────────────

def compute_crop(vw, vh, corner, bbox, padding=20):
    """
    Compute the optimal crop region that removes the watermark naturally.

    Strategy:
      1. Determine how many pixels to remove from the watermark edge(s).
      2. Distribute a proportional crop to the opposite side for composition balance.
      3. Maintain aspect ratio as close to original as possible.
      4. Return the crop box and target scale resolution.
    """
    bx, by, bw, bh = bbox

    # How much to cut from each edge (0 = no cut)
    cut_top = 0
    cut_bottom = 0
    cut_left = 0
    cut_right = 0

    pad = padding  # extra margin beyond the watermark

    if corner == "br":
        cut_bottom = vh - by + pad
        cut_right = vw - bx + pad
    elif corner == "bl":
        cut_bottom = vh - by + pad
        cut_left = bx + bw + pad
    elif corner == "tr":
        cut_top = by + bh + pad
        cut_right = vw - bx + pad
    elif corner == "tl":
        cut_top = by + bh + pad
        cut_left = bx + bw + pad

    # Distribute a smaller counter-crop to the opposite side for natural composition
    # Rule: opposite side gets 30% of the primary cut (subtle rebalancing)
    balance_ratio = 0.3

    if cut_top > 0:
        cut_bottom = max(cut_bottom, int(cut_top * balance_ratio))
    if cut_bottom > 0:
        cut_top = max(cut_top, int(cut_bottom * balance_ratio))
    if cut_left > 0:
        cut_right = max(cut_right, int(cut_left * balance_ratio))
    if cut_right > 0:
        cut_left = max(cut_left, int(cut_right * balance_ratio))

    # Ensure we don't crop more than 20% from any single edge
    max_v = int(vh * 0.20)
    max_h = int(vw * 0.20)
    cut_top = min(cut_top, max_v)
    cut_bottom = min(cut_bottom, max_v)
    cut_left = min(cut_left, max_h)
    cut_right = min(cut_right, max_h)

    # Crop box (in ffmpeg crop filter terms: w:h:x:y)
    crop_x = cut_left
    crop_y = cut_top
    crop_w = vw - cut_left - cut_right
    crop_h = vh - cut_top - cut_bottom

    # Make dimensions even (required for most codecs)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    return crop_x, crop_y, crop_w, crop_h


def compute_scale_target(crop_w, crop_h, orig_w, orig_h):
    """
    Decide the final output resolution.

    Priority:
      1. If a standard resolution fits cleanly, use it.
      2. Otherwise scale back to original dimensions.

    Returns (target_w, target_h).
    """
    # Try to find a standard resolution close to the original
    std = find_nearest_standard(orig_w, orig_h)
    if std:
        sw, sh = std
        # Only use standard if it's close to original (within 15%)
        w_ratio = sw / orig_w
        h_ratio = sh / orig_h
        if w_ratio > 0.85 and h_ratio > 0.85:
            # Make sure it's even
            return sw - (sw % 2), sh - (sh % 2)

    # Fall back to original resolution
    return orig_w - (orig_w % 2), orig_h - (orig_h % 2)


# ── Core: Video Processing ──────────────────────────────────────────────────

def process_video(input_path, output_path, corner, bbox, padding=20, quality=18):
    """
    Process the video: crop watermark, scale back to target resolution.
    Uses ffmpeg for fast, high-quality processing (no frame-by-frame Python loop).
    """
    vw, vh, fps, frame_count, has_audio = get_video_info(input_path)

    # Compute crop
    cx, cy, cw, ch = compute_crop(vw, vh, corner, bbox, padding)
    log(f"  Crop: {vw}x{vh} → {cw}x{ch} (removing {corner} corner)", C.DIM)

    # Compute scale target
    tw, th = compute_scale_target(cw, ch, vw, vh)
    log(f"  Scale: {cw}x{ch} → {tw}x{th}", C.DIM)

    # Build ffmpeg filter chain
    vf = f"crop={cw}:{ch}:{cx}:{cy},scale={tw}:{th}:flags=lanczos"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(quality),
        "-preset", "slow",       # better compression quality
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # web-optimized
    ]

    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    return tw, th


# ── Preview Mode ─────────────────────────────────────────────────────────────

def save_preview(input_path, corner, bbox, padding, output_dir=None):
    """Save before/after preview frames as PNGs for visual comparison."""
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."

    vw, vh, fps, total, _ = get_video_info(input_path)
    cx, cy, cw, ch = compute_crop(vw, vh, corner, bbox, padding)
    tw, th = compute_scale_target(cw, ch, vw, vh)

    cap = cv2.VideoCapture(input_path)

    # Sample 3 frames: start, middle, end
    sample_frames = [
        int(total * 0.1),
        int(total * 0.5),
        int(total * 0.9),
    ]

    previews = []
    for i, fidx in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw watermark bbox on original
        orig_annotated = frame.copy()
        bx, by, bw, bh = bbox
        cv2.rectangle(orig_annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(orig_annotated, "WATERMARK", (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Simulate crop + scale
        cropped = frame[cy:cy + ch, cx:cx + cw]
        scaled = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        # Resize both to same height for side-by-side
        display_h = 400
        scale_factor = display_h / vh
        orig_display = cv2.resize(orig_annotated, (int(vw * scale_factor), display_h))
        clean_display = cv2.resize(scaled, (int(tw * scale_factor * vw / tw), display_h))

        # Side by side with label
        canvas_w = orig_display.shape[1] + clean_display.shape[1] + 20
        canvas = np.zeros((display_h + 40, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Place images
        canvas[0:display_h, 0:orig_display.shape[1]] = orig_display
        x_off = orig_display.shape[1] + 20
        canvas[0:display_h, x_off:x_off + clean_display.shape[1]] = clean_display

        # Labels
        cv2.putText(canvas, "BEFORE", (10, display_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(canvas, "AFTER", (x_off + 10, display_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        preview_path = os.path.join(output_dir, f"preview_frame_{i + 1}.png")
        cv2.imwrite(preview_path, canvas)
        previews.append(preview_path)

    cap.release()
    return previews


# ── CLI ──────────────────────────────────────────────────────────────────────

BANNER = f"""{C.MAGENTA}
  ╦ ╦┌┐┌┬ ┬┌─┐┌┬┐┌─┐┬─┐┌┬┐┌─┐┬─┐┬┌─
  ║ ║│││││││├─┤ │ ├┤ ├┬┘│││├─┤├┬┘├┴┐
  ╚═╝┘└┘└┴┘└┘ ┘ ┴ └─┘┴└─┴ ┴┴ ┴┴└─┴ ┴  {C.DIM}v{VERSION}{C.RESET}
{C.DIM}  Naturally remove watermarks from AI-generated videos{C.RESET}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Remove corner watermarks from AI-generated videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python unwatermark.py video.mp4\n"
               "  python unwatermark.py video.mp4 -o clean.mp4 --corner br\n"
               "  python unwatermark.py video.mp4 --preview\n",
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output file path (default: <input>_clean.<ext>)")
    parser.add_argument("--corner", choices=["tl", "tr", "bl", "br"],
                        help="Watermark corner (auto-detected if omitted)")
    parser.add_argument("--padding", type=int, default=20,
                        help="Extra pixels to crop beyond watermark bounds (default: 20)")
    parser.add_argument("--quality", type=int, default=18,
                        help="CRF quality (lower=better, 0-51, default: 18)")
    parser.add_argument("--preview", action="store_true",
                        help="Save before/after preview images instead of processing")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--version", action="version", version=f"unwatermark {VERSION}")

    args = parser.parse_args()

    if not args.quiet:
        print(BANNER)

    # Validate input
    if not os.path.isfile(args.input):
        log(f"  Error: File not found: {args.input}", C.RED)
        sys.exit(1)

    # Default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"

    # Step 1: Read video info
    t0 = time.time()
    log(f"  {C.BOLD}Input:{C.RESET}  {args.input}")
    vw, vh, fps, frame_count, has_audio = get_video_info(args.input)
    duration = frame_count / fps if fps > 0 else 0
    in_size = os.path.getsize(args.input) / (1024 * 1024)
    log(f"  {C.DIM}{vw}x{vh} | {fps:.1f}fps | {duration:.1f}s | {frame_count} frames | {in_size:.1f}MB{C.RESET}")
    print()

    # Step 2: Detect watermark
    log(f"  {C.CYAN}[1/3]{C.RESET} {C.BOLD}Detecting watermark...{C.RESET}")

    if args.corner:
        # Manual corner — still detect bbox
        corner = args.corner
        _, bbox = detect_watermark(args.input)
        if bbox is None:
            # Fallback: assume a typical watermark size in the specified corner
            margin = 15
            wm_w, wm_h = 180, 55
            if corner == "br":
                bbox = (vw - wm_w - margin, vh - wm_h - margin, wm_w, wm_h)
            elif corner == "bl":
                bbox = (margin, vh - wm_h - margin, wm_w, wm_h)
            elif corner == "tr":
                bbox = (vw - wm_w - margin, margin, wm_w, wm_h)
            elif corner == "tl":
                bbox = (margin, margin, wm_w, wm_h)
        log(f"  Using specified corner: {C.YELLOW}{CORNERS[corner]}{C.RESET}")
    else:
        corner, bbox = detect_watermark(args.input)
        if corner is None:
            log(f"  {C.YELLOW}No watermark detected.{C.RESET} Use --corner to specify manually.")
            sys.exit(0)
        log(f"  Found watermark: {C.YELLOW}{CORNERS[corner]}{C.RESET}")

    bx, by, bw, bh = bbox
    log(f"  {C.DIM}Region: ({bx}, {by}) {bw}x{bh}{C.RESET}")
    print()

    # Step 3: Preview mode
    if args.preview:
        log(f"  {C.CYAN}[2/3]{C.RESET} {C.BOLD}Generating previews...{C.RESET}")
        previews = save_preview(args.input, corner, bbox, args.padding)
        for p in previews:
            log(f"  {C.GREEN}Saved:{C.RESET} {p}")
        print()
        log(f"  {C.GREEN}Done!{C.RESET} Review previews, then run without --preview to process.")
        return

    # Step 4: Process video
    log(f"  {C.CYAN}[2/3]{C.RESET} {C.BOLD}Processing video...{C.RESET}")
    tw, th = process_video(args.input, args.output, corner, bbox, args.padding, args.quality)
    print()

    # Step 5: Report
    log(f"  {C.CYAN}[3/3]{C.RESET} {C.BOLD}Done!{C.RESET}")
    out_size = os.path.getsize(args.output) / (1024 * 1024)
    elapsed = time.time() - t0

    print()
    log(f"  {C.BOLD}Output:{C.RESET} {args.output}")
    log(f"  {C.DIM}{tw}x{th} | {in_size:.1f}MB → {out_size:.1f}MB | {elapsed:.1f}s{C.RESET}")
    print()


if __name__ == "__main__":
    main()
