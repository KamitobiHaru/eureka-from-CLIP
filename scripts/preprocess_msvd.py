"""
Preprocess MSVD annotations: parse AllVideoDescriptions.txt, group captions by
video, and write a single JSON annotation file for zero-shot evaluation.

Output: {output_dir}/msvd_all.json
  Array of {video_id, captions: [str]} — all 1970 videos with grouped captions.
All videos are used as a single test set (zero-shot evaluation only).
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Preprocess MSVD annotations.")
    parser.add_argument("--input", default="./datasets/MSVD/AllVideoDescriptions.txt")
    parser.add_argument("--output_dir", default="./data/msvd")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse lines: "video_id description"
    captions_by_video = defaultdict(list)
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split on first space to separate video_id from caption
            idx = line.index(" ")
            video_id = line[:idx].strip()
            caption = line[idx:].strip()
            captions_by_video[video_id].append(caption)

    # Build annotation array
    annotations = [
        {"video_id": vid, "captions": caps}
        for vid, caps in captions_by_video.items()
    ]
    annotations.sort(key=lambda x: x["video_id"])

    out_path = os.path.join(args.output_dir, "msvd_all.json")
    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)

    n_captions = sum(len(a["captions"]) for a in annotations)
    print(f"MSVD: {len(annotations)} videos, {n_captions} captions")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
