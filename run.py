"""
CLIP Video Scene Search — CLI

Usage:
    python run.py <video_path> --query "a person walking" [--top_k 5] [--output ./output]

Examples:
    python run.py demo.mp4 --query "a dog running"
    python run.py demo.mp4 --query "car driving" --top_k 3 --output results
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from clip_search import SearchEngine


def save_thumbnail(path: str, frame: np.ndarray) -> None:
    """Save a single RGB frame as a JPEG thumbnail."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(description="Search video scenes with CLIP text queries.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--query", "-q", required=True, help="Text query to search for")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of top scenes to show (default: 5)")
    parser.add_argument("--output", "-o", default="output", help="Directory to save result thumbnails (default: output/)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    # ── Step 1: process video ──────────────────────────────
    print(f"Processing video: {video_path}")
    print("Loading CLIP encoder...")
    engine = SearchEngine()

    print("Detecting scenes...")
    scenes = engine.process_video(video_path)
    print(f"  → {len(scenes)} scenes found")

    if not scenes:
        print("No scenes detected.")
        sys.exit(0)

    # ── Step 2: search ─────────────────────────────────────
    print(f'\nSearching for: "{args.query}"')
    results = engine.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching scenes found.")
        sys.exit(0)

    # ── Step 3: display results ────────────────────────────
    print(f"\nTop {len(results)} results:")
    print("-" * 72)
    print(f"  {'Rank':<6} {'Scene':<6} {'Time range':<22} {'Score':<10}  Thumbnail")
    print("-" * 72)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, r in enumerate(results, 1):
        time_range = f"{r['start_sec']:.1f}s → {r['end_sec']:.1f}s"
        thumb_name = f"rank{rank:02d}_scene{r['scene_idx']:03d}.jpg"
        thumb_path = str(out_dir / thumb_name)

        save_thumbnail(thumb_path, r["thumbnail"])

        print(f"  {rank:<6} {r['scene_idx']:<6} {time_range:<22} {r['score']:.4f}    {thumb_path}")

    print("-" * 72)

    # ── Step 4: summary per query ──────────────────────────
    best = results[0]
    print(f'\nBest match: Scene {best["scene_idx"]} '
          f'({best["start_sec"]:.1f}s → {best["end_sec"]:.1f}s), '
          f'score={best["score"]:.4f}')
    print(f"Thumbnails saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
