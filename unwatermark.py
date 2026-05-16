#!/usr/bin/env python3
"""
unwatermark — naturally remove corner watermarks from AI-generated videos.

Uses LaMa deep inpainting for pixel-perfect watermark removal at original
resolution. No cropping, no artifacts, no content loss.

  1. Auto-detect watermark position via temporal edge stability analysis
  2. Build a precise pixel mask (text + border + translucent fill)
  3. LaMa inpaint each frame — reconstructs natural texture underneath
  4. Reassemble with original audio via FFmpeg

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

VERSION = "2.0.0"

CORNERS = {
    "tl": "top-left",
    "tr": "top-right",
    "bl": "bottom-left",
    "br": "bottom-right",
}

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

    stability = accum / count
    static_mask = (stability > 0.30).astype(np.uint8) * 255

    # Analyze each corner quadrant (25% of frame)
    scan_h, scan_w = vh // 4, vw // 4

    corners_regions = {
        "tl": static_mask[0:scan_h, 0:scan_w],
        "tr": static_mask[0:scan_h, vw - scan_w:vw],
        "bl": static_mask[vh - scan_h:vh, 0:scan_w],
        "br": static_mask[vh - scan_h:vh, vw - scan_w:vw],
    }

    # Score each corner by largest connected component
    scores = {}
    for corner, region in corners_regions.items():
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        max_area = 0
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
        scores[corner] = max_area

    best_corner = max(scores, key=scores.get)

    if scores[best_corner] < 100:
        return None, None

    # Get bounding box
    region = corners_regions[best_corner]
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

    # Convert from corner-local to full-frame coords
    offset_x, offset_y = 0, 0
    if best_corner in ("tr", "br"):
        offset_x = vw - scan_w
    if best_corner in ("bl", "br"):
        offset_y = vh - scan_h

    bbox = (offset_x + rx, offset_y + ry, rw, rh)
    return best_corner, bbox


# ── Core: Mask Building ─────────────────────────────────────────────────────

def build_mask(video_path, corner, bbox, padding=20, num_samples=50):
    """
    Build a precise inpainting mask that covers:
      - The watermark text (detected via temporal edge stability)
      - The rounded-rect border
      - The semi-transparent fill inside the border

    Returns a binary mask (uint8, 0 or 255) at full frame resolution.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int)
    edge_accum = np.zeros((vh, vw), dtype=np.float64)
    count = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 80)
        edge_accum += (edges > 0).astype(np.float64)
        count += 1
    cap.release()

    # Static edges = watermark outline
    threshold = count * 0.30
    edge_mask = (edge_accum >= threshold).astype(np.uint8) * 255

    # Zero out everything except the watermark corner region (with padding)
    bx, by, bw, bh = bbox
    pad = padding
    roi_x1 = max(0, bx - pad)
    roi_y1 = max(0, by - pad)
    roi_x2 = min(vw, bx + bw + pad)
    roi_y2 = min(vh, by + bh + pad)

    isolated = np.zeros_like(edge_mask)
    isolated[roi_y1:roi_y2, roi_x1:roi_x2] = edge_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    # Close gaps in edges and fill the interior of the watermark badge
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.dilate(isolated, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Find contours and fill the largest one (the badge outline)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((vh, vw), dtype=np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 300:
            cv2.drawContours(filled, [c], -1, 255, -1)

    # Combine filled interior with original edge pixels
    combined = cv2.bitwise_or(filled, isolated)

    # Dilate slightly to catch anti-aliased edges and semi-transparent fringe
    combined = cv2.dilate(combined, kernel, iterations=2)

    return combined


# ── Core: LaMa Inpainting ───────────────────────────────────────────────────

def _try_import_iopaint():
    """Check if iopaint (LaMa) is available."""
    try:
        from iopaint.model import models as iopaint_models
        return "lama" in iopaint_models
    except ImportError:
        return False


def inpaint_frame_lama(model, frame_rgb, mask):
    """Inpaint a single frame using LaMa via iopaint."""
    from iopaint.schema import HDStrategy, InpaintRequest

    config = InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1280,
    )
    result = model(frame_rgb, mask, config)
    return result


def inpaint_frame_opencv(frame, mask, radius=6):
    """Fallback: OpenCV NS inpainting."""
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_NS)


def load_lama_model(device="cpu"):
    """Load the LaMa model directly from iopaint's model registry."""
    from iopaint.model import models as iopaint_models
    model = iopaint_models["lama"](device)
    return model


# ── Core: Video Processing ──────────────────────────────────────────────────

def process_video(input_path, output_path, mask, quality=18, device="cpu"):
    """
    Process every frame: inpaint the watermark region, reassemble video.
    Uses LaMa if available, falls back to OpenCV.
    """
    vw, vh, fps, frame_count, has_audio = get_video_info(input_path)

    use_lama = _try_import_iopaint()
    if use_lama:
        log(f"  Engine: {C.GREEN}LaMa deep inpainting{C.RESET} (best quality)")
        model = load_lama_model(device)
    else:
        log(f"  Engine: {C.YELLOW}OpenCV NS inpainting{C.RESET} (install iopaint for better results)")
        model = None

    # Work in a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="unwatermark_")
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir)

    cap = cv2.VideoCapture(input_path)
    idx = 0

    # Find the bounding rect of the mask to only process that region
    mask_points = cv2.findNonZero(mask)
    mx, my, mw, mh = cv2.boundingRect(mask_points)
    # Add margin for context
    margin = 40
    rx1 = max(0, mx - margin)
    ry1 = max(0, my - margin)
    rx2 = min(vw, mx + mw + margin)
    ry2 = min(vh, my + mh + margin)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_lama:
            # LaMa expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process only the ROI for speed
            roi_frame = frame_rgb[ry1:ry2, rx1:rx2].copy()
            roi_mask = mask[ry1:ry2, rx1:rx2].copy()

            if np.any(roi_mask > 0):
                roi_result = inpaint_frame_lama(model, roi_frame, roi_mask)
                frame_rgb[ry1:ry2, rx1:rx2] = roi_result
            result = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            result = inpaint_frame_opencv(frame, mask)

        cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}.png"), result)
        idx += 1
        if idx % 30 == 0 or idx == frame_count:
            pct = min(100, int(idx / frame_count * 100))
            bar_w = 24
            filled = int(bar_w * idx / frame_count)
            bar = f"{'█' * filled}{'░' * (bar_w - filled)}"
            print(f"\r  {C.CYAN}{bar}{C.RESET} {C.DIM}{idx}/{frame_count} frames ({pct}%){C.RESET}", end="", flush=True)

    cap.release()
    print()

    # Reassemble with FFmpeg
    log(f"  {C.DIM}Encoding final video...{C.RESET}")
    temp_video = os.path.join(tmp_dir, "noaudio.mp4")

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%06d.png"),
        "-c:v", "libx264",
        "-crf", str(quality),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temp_video,
    ], capture_output=True, text=True, timeout=600)

    # Mux with original audio
    if has_audio:
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", temp_video,
            "-i", input_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0?",
            output_path,
        ], capture_output=True, text=True, timeout=600)
    else:
        shutil.move(temp_video, output_path)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return vw, vh


# ── Preview Mode ─────────────────────────────────────────────────────────────

def save_preview(input_path, mask, device="cpu", output_dir=None):
    """Save before/after preview frames for visual comparison."""
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."

    vw, vh, fps, total, _ = get_video_info(input_path)

    use_lama = _try_import_iopaint()
    model = load_lama_model(device) if use_lama else None

    mask_points = cv2.findNonZero(mask)
    mx, my, mw, mh = cv2.boundingRect(mask_points)
    margin = 40
    rx1, ry1 = max(0, mx - margin), max(0, my - margin)
    rx2, ry2 = min(vw, mx + mw + margin), min(vh, my + mh + margin)

    cap = cv2.VideoCapture(input_path)
    sample_frames = [int(total * 0.1), int(total * 0.5), int(total * 0.9)]
    previews = []

    for i, fidx in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Inpaint this frame
        if use_lama:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_frame = frame_rgb[ry1:ry2, rx1:rx2].copy()
            roi_mask = mask[ry1:ry2, rx1:rx2].copy()
            if np.any(roi_mask > 0):
                roi_result = inpaint_frame_lama(model, roi_frame, roi_mask)
                frame_rgb[ry1:ry2, rx1:rx2] = roi_result
            clean = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            clean = inpaint_frame_opencv(frame, mask)

        # Draw watermark bbox on original
        orig_annotated = frame.copy()
        cv2.rectangle(orig_annotated, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        # Side by side
        display_h = 400
        sf = display_h / vh
        orig_r = cv2.resize(orig_annotated, (int(vw * sf), display_h))
        clean_r = cv2.resize(clean, (int(vw * sf), display_h))

        canvas_w = orig_r.shape[1] + clean_r.shape[1] + 20
        canvas = np.full((display_h + 40, canvas_w, 3), 30, dtype=np.uint8)
        canvas[0:display_h, 0:orig_r.shape[1]] = orig_r
        x_off = orig_r.shape[1] + 20
        canvas[0:display_h, x_off:x_off + clean_r.shape[1]] = clean_r

        cv2.putText(canvas, "BEFORE", (10, display_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(canvas, "AFTER", (x_off + 10, display_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        path = os.path.join(output_dir, f"preview_frame_{i + 1}.png")
        cv2.imwrite(path, canvas)
        previews.append(path)

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
                        help="Extra pixels around watermark mask (default: 20)")
    parser.add_argument("--quality", type=int, default=18,
                        help="CRF quality (lower=better, 0-51, default: 18)")
    parser.add_argument("--device", default="cpu", help="Device for LaMa model (default: cpu)")
    parser.add_argument("--preview", action="store_true",
                        help="Save before/after preview images without full processing")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--version", action="version", version=f"unwatermark {VERSION}")

    args = parser.parse_args()

    if not args.quiet:
        print(BANNER)

    if not os.path.isfile(args.input):
        log(f"  Error: File not found: {args.input}", C.RED)
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"

    # ── Step 1: Video info ──
    t0 = time.time()
    log(f"  {C.BOLD}Input:{C.RESET}  {args.input}")
    vw, vh, fps, frame_count, has_audio = get_video_info(args.input)
    duration = frame_count / fps if fps > 0 else 0
    in_size = os.path.getsize(args.input) / (1024 * 1024)
    log(f"  {C.DIM}{vw}x{vh} | {fps:.1f}fps | {duration:.1f}s | {frame_count} frames | {in_size:.1f}MB{C.RESET}")
    print()

    # ── Step 2: Detect watermark ──
    log(f"  {C.CYAN}[1/4]{C.RESET} {C.BOLD}Detecting watermark...{C.RESET}")

    if args.corner:
        corner = args.corner
        _, bbox = detect_watermark(args.input)
        if bbox is None:
            margin = 15
            wm_w, wm_h = 180, 55
            positions = {
                "br": (vw - wm_w - margin, vh - wm_h - margin, wm_w, wm_h),
                "bl": (margin, vh - wm_h - margin, wm_w, wm_h),
                "tr": (vw - wm_w - margin, margin, wm_w, wm_h),
                "tl": (margin, margin, wm_w, wm_h),
            }
            bbox = positions[corner]
        log(f"  Corner: {C.YELLOW}{CORNERS[corner]}{C.RESET} (manual)")
    else:
        corner, bbox = detect_watermark(args.input)
        if corner is None:
            log(f"  {C.YELLOW}No watermark detected.{C.RESET} Use --corner to specify manually.")
            sys.exit(0)
        log(f"  Corner: {C.YELLOW}{CORNERS[corner]}{C.RESET} (auto-detected)")

    bx, by, bw, bh = bbox
    log(f"  {C.DIM}Region: ({bx}, {by}) {bw}x{bh}{C.RESET}")
    print()

    # ── Step 3: Build mask ──
    log(f"  {C.CYAN}[2/4]{C.RESET} {C.BOLD}Building inpainting mask...{C.RESET}")
    mask = build_mask(args.input, corner, bbox, args.padding)
    mask_pixels = np.count_nonzero(mask)
    mask_pct = mask_pixels / (vw * vh) * 100
    log(f"  {C.DIM}Mask: {mask_pixels} pixels ({mask_pct:.2f}% of frame){C.RESET}")
    print()

    # ── Preview mode ──
    if args.preview:
        log(f"  {C.CYAN}[3/4]{C.RESET} {C.BOLD}Generating previews...{C.RESET}")
        previews = save_preview(args.input, mask, args.device)
        for p in previews:
            log(f"  {C.GREEN}Saved:{C.RESET} {p}")
        print()
        log(f"  {C.GREEN}Done!{C.RESET} Review previews, then run without --preview to process.")
        return

    # ── Step 4: Inpaint video ──
    log(f"  {C.CYAN}[3/4]{C.RESET} {C.BOLD}Inpainting frames...{C.RESET}")
    tw, th = process_video(args.input, args.output, mask, args.quality, args.device)
    print()

    # ── Report ──
    log(f"  {C.CYAN}[4/4]{C.RESET} {C.BOLD}Done!{C.RESET}")
    out_size = os.path.getsize(args.output) / (1024 * 1024)
    elapsed = time.time() - t0

    print()
    log(f"  {C.BOLD}Output:{C.RESET} {args.output}")
    log(f"  {C.DIM}{tw}x{th} (original resolution) | {in_size:.1f}MB → {out_size:.1f}MB | {elapsed:.1f}s{C.RESET}")
    print()


if __name__ == "__main__":
    main()
