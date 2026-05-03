"""
Split a video into its constituent clips by detecting scene cuts.

Uses frame-to-frame histogram difference to find large visual changes
that indicate a cut between clips.

Usage:
    python split_clips.py <input_video> [--threshold 0.7] [--output-dir clips/]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def compute_hist_diff(frame_a, frame_b):
    """Compare two frames using histogram correlation (0 = identical, 1 = totally different)."""
    hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)

    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return 1.0 - score  # invert so higher = more different


def detect_cuts(video_path: str, threshold: float = 0.7) -> list[float]:
    """Return list of timestamps (seconds) where scene cuts are detected."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.1f} fps, {total_frames} frames, {total_frames / fps:.1f}s")

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")

    cuts = []
    frame_idx = 1
    diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        diff = compute_hist_diff(prev_frame, frame)
        diffs.append((frame_idx, diff))

        if diff >= threshold:
            timestamp = frame_idx / fps
            cuts.append(timestamp)
            print(f"  Cut at {timestamp:.2f}s (frame {frame_idx}, diff={diff:.3f})")

        prev_frame = frame
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    # Print diff stats to help tune threshold
    if diffs:
        all_diffs = [d for _, d in diffs]
        print(f"\nDiff stats: min={min(all_diffs):.3f}, max={max(all_diffs):.3f}, "
              f"mean={np.mean(all_diffs):.3f}, median={np.median(all_diffs):.3f}")
        # Show top 20 biggest diffs to help identify the right threshold
        top = sorted(diffs, key=lambda x: x[1], reverse=True)[:20]
        print("Top 20 diffs:")
        for fidx, d in top:
            print(f"  frame {fidx} ({fidx / fps:.2f}s): {d:.3f}")

    return cuts


def split_video(video_path: str, cuts: list[float], output_dir: str):
    """Use ffmpeg to split the video at detected cut points."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(video_path).stem

    # Build segment boundaries: [0, cut1, cut2, ..., end]
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    boundaries = [0.0] + cuts + [duration]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        clip_duration = end - start

        if clip_duration < 0.1:
            continue

        out_path = os.path.join(output_dir, f"{stem}_clip{i + 1:03d}.mp4")
        cmd = (
            f'ffmpeg -y -ss {start:.3f} -i "{video_path}" '
            f'-t {clip_duration:.3f} -c copy -avoid_negative_ts 1 "{out_path}" '
            f'-loglevel warning'
        )
        print(f"Clip {i + 1}: {start:.2f}s - {end:.2f}s ({clip_duration:.1f}s) -> {out_path}")
        os.system(cmd)

    print(f"\nDone! {len(boundaries) - 1} clips saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Split a video into clips at scene cuts")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Histogram diff threshold for cut detection (0-1, default 0.7)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <video_name>_clips/)")
    parser.add_argument("--detect-only", action="store_true",
                        help="Only detect cuts, don't split the video")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(args.video).stem + "_clips"

    print(f"Detecting cuts (threshold={args.threshold})...")
    cuts = detect_cuts(args.video, threshold=args.threshold)
    print(f"\nFound {len(cuts)} cuts")

    if cuts and not args.detect_only:
        split_video(args.video, cuts, args.output_dir)
    elif args.detect_only:
        print("Timestamps:", [f"{t:.2f}s" for t in cuts])


if __name__ == "__main__":
    main()
