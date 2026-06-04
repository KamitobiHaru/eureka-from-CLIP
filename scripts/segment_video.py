"""
Segment a video into scenes and save all keyframes to a directory.

Output structure:
    <output-dir>/<video_stem>/
        metadata.json           # [{scene_idx, start_sec, end_sec, num_frames}, ...]
        scene_000/
            thumbnail.jpg       # first frame
            frame_000.jpg ...   # all uniformly sampled frames
        scene_001/
        ...

Usage:
    python scripts/segment_video.py demo.mp4 --output-dir ./segments
    python scripts/segment_video.py demo.mp4 -o ./segments --num-frames 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.segmenter import detect_scenes


def main():
    parser = argparse.ArgumentParser(description="Segment video into scenes and save keyframes.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--output-dir", "-o", default="./segments",
                        help="Root directory to save segmented scenes (default: ./segments)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    # Detect scenes
    print(f"Detecting scenes in: {video_path}")
    scenes = detect_scenes(str(video_path))
    print(f"  → {len(scenes)} scenes found")

    if not scenes:
        print("No scenes detected.")
        sys.exit(0)

    # Output directory
    out_root = Path(args.output_dir) / video_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Save each scene
    metadata = []
    for scene in scenes:
        scene_dir = out_root / f"scene_{scene.scene_idx:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Save all frames as JPEGs
        for i, frame in enumerate(scene.frames):
            frame_path = scene_dir / f"frame_{i:03d}.jpg"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Save thumbnail (first frame)
        thumb_path = scene_dir / "thumbnail.jpg"
        bgr = cv2.cvtColor(scene.frames[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(thumb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        info = {
            "scene_idx": scene.scene_idx,
            "start_sec": scene.start_sec,
            "end_sec": scene.end_sec,
            "num_frames": len(scene.frames),
        }
        metadata.append(info)
        print(f"  Scene {scene.scene_idx:03d}: {scene.start_sec:.1f}s → {scene.end_sec:.1f}s  "
              f"({len(scene.frames)} frames)")

    # Save metadata
    meta_path = out_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll scenes saved to: {out_root.resolve()}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
