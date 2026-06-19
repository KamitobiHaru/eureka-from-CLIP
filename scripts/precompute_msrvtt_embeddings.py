"""
Precompute CLIP frame embeddings for MSR-VTT test videos only.

Reads the test annotation JSON to find which videos to process, samples
uniform frames from each, encodes via CLIP ViT-B/32, and saves per-video
(N_frames, 512) .npy files.

Usage:
    python scripts/precompute_msrvtt_embeddings.py \
        --video_root ./datasets/MSR-VTT \
        --cache_dir ./data/msrvtt/clip_embeddings \
        --num_frames 12 \
        --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder


def sample_frames(video_path: str, num_frames: int):
    """Uniformly sample *num_frames* frames from a video file.

    Returns list of RGB arrays (H, W, 3, uint8) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            for _ in range(int(idx)):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP frame embeddings for MSR-VTT test videos."
    )
    parser.add_argument("--video_root", default="./datasets/MSR-VTT",
                        help="Root dir containing video/ subdirectory")
    parser.add_argument("--cache_dir", default="./data/msrvtt/clip_embeddings",
                        help="Where to save .npy files")
    parser.add_argument("--annotation",
                        default="msrvtt_test_1k.json",
                        help="Test annotation JSON (relative to video_root)")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to sample per video")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Frames per CLIP batch")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    video_dir = Path(args.video_root) / "video"
    if not video_dir.is_dir():
        print(f"Video directory not found: {video_dir}")
        print("Run: unzip MSRVTT_Videos.zip -d <video_root>")
        sys.exit(1)

    # ── Read annotation to get test video IDs ────────────────────
    ann_path = Path(args.video_root) / args.annotation
    with open(ann_path) as f:
        annotations = json.load(f)
    test_ids = {entry["video_id"] for entry in annotations}
    print(f"Test videos in annotation: {len(test_ids)}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Only process test videos that aren't cached yet
    to_process = []
    already = 0
    for vid in sorted(test_ids):
        if (cache_dir / f"{vid}.npy").exists():
            already += 1
        else:
            mp4 = video_dir / f"{vid}.mp4"
            if mp4.exists():
                to_process.append(mp4)

    print(f"Cached: {already}, Remaining: {len(to_process)}")
    if not to_process:
        print("All test videos already cached.")
        return

    encoder = CLIPEncoder(model_type="openai", device=args.device)
    device = encoder.device
    print(f"CLIP device: {device}, batch_size={args.batch_size}, num_frames={args.num_frames}")

    processed = 0
    pbar = tqdm(to_process, desc="MSR-VTT test", unit="video")
    for vpath in pbar:
        frames = sample_frames(str(vpath), args.num_frames)
        if frames is None or len(frames) == 0:
            continue

        all_embs = []
        for i in range(0, len(frames), args.batch_size):
            batch = frames[i:i + args.batch_size]
            embs = encoder.encode_frames(batch)  # (B, 512), L2-normed
            all_embs.append(embs)
        video_emb = np.concatenate(all_embs, axis=0)  # (N, 512)

        np.save(cache_dir / f"{vpath.stem}.npy", video_emb)
        processed += 1

    pbar.close()
    total = len(list(cache_dir.glob("*.npy")))
    print(f"Done. Processed={processed}, Total test cached={total}")


if __name__ == "__main__":
    main()
