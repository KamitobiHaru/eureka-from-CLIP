#!/usr/bin/env python3
"""Precompute video embeddings and save to a cache directory.

Usage:
    python cache_videos.py /path/to/videos --output ./data

The cache directory will contain:

    embeddings.npy    — (N, 512) float32 video-level embeddings
    metadata.json     — list of {video_id, start_sec, end_sec}
    video_map.json    — {video_id: "/absolute/path/to/video"}
    config.json       — processing parameters
"""
import argparse
import json
from pathlib import Path

import numpy as np

from clip_search import SearchEngine


def main():
    parser = argparse.ArgumentParser(description="Precompute video embeddings for search")
    parser.add_argument("video_dir", help="Path to folder containing video files")
    parser.add_argument("--output", "-o", default="./data",
                        help="Output cache directory (default: ./data)")
    parser.add_argument("--device", default=None,
                        help="Device (cuda:0, cpu, etc.)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = SearchEngine(device=args.device, uniform_sample=True)
    engine.process_folder(args.video_dir)

    if not engine.scenes:
        print("No videos processed. Nothing to cache.")
        return

    np.save(str(output_dir / "embeddings.npy"), engine.scene_embs)
    metadata = [
        {
            "video_id": s.video_id,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
        }
        for s in engine.scenes
    ]
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "video_map.json", "w") as f:
        json.dump(dict(engine.video_map), f, indent=2)

    print(f"\n✓ Cached {len(engine.scenes)} videos to {output_dir.resolve()}")
    print(f"  embeddings.npy  → {engine.scene_embs.shape}")
    print(f"  metadata.json   → {len(metadata)} entries")
    print(f"  video_map.json  → {len(engine.video_map)} entries")


if __name__ == "__main__":
    main()
