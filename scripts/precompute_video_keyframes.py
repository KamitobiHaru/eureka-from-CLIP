"""
Extract key frames from MSR-VTT / MSVD videos via uniform frame sampling,
encode them with frozen CLIP, and save per-video embeddings.

Usage (MSR-VTT):
    python scripts/precompute_video_keyframes.py \
        --video_dir ./datasets/MSR-VTT/video \
        --output_dir ./data/msrvtt/clip_keyframes \
        --pattern "*.mp4"

Usage (MSVD):
    python scripts/precompute_video_keyframes.py \
        --video_dir "./datasets/MSVD/YouTubeClips" \
        --output_dir ./data/msvd/clip_keyframes \
        --pattern "*.avi"
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"


def select_key_frames(total_frames: int, max_frames: int = 12) -> list:
    """Uniformly sample ``max_frames`` frame indices from the video."""
    if total_frames <= max_frames:
        return list(range(total_frames))
    return [int((i + 0.5) * total_frames / max_frames) for i in range(max_frames)]


def process_video(video_path: str, encoder, max_frames: int = 12, batch_size: int = 64):
    """Uniformly sample key frames from a video and encode them with CLIP.
    Returns (N, 512) float32 array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return np.empty((0, 512), dtype=np.float32)

    key_indices = select_key_frames(total_frames, max_frames)

    frames = []
    for idx in key_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        return np.empty((0, 512), dtype=np.float32)

    embs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        batch_embs = encoder.encode_frames(batch)
        embs.append(batch_embs)

    return np.concatenate(embs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP key-frame embeddings for videos.")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", default="openai")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from clip_search.encoder import CLIPEncoder
    print(f"Loading CLIP model: {args.model}")
    encoder = CLIPEncoder(model_type=args.model)

    video_paths = sorted(video_dir.glob(args.pattern))
    if not video_paths:
        print(f"No videos found matching '{args.pattern}' in {video_dir}")
        return

    to_process = []
    for vp in video_paths:
        out_path = output_dir / f"{vp.stem}.npy"
        if not out_path.exists():
            to_process.append(vp)

    print(f"Total: {len(video_paths)}, cached: {len(video_paths) - len(to_process)}, to process: {len(to_process)}")
    if not to_process:
        print("All done.")
        return

    for vp in tqdm(to_process, desc="Extracting key frames"):
        try:
            emb = process_video(str(vp), encoder, args.max_frames, args.batch_size)
            np.save(str(output_dir / f"{vp.stem}.npy"), emb)
        except Exception as e:
            print(f"  Error {vp.name}: {e}")

    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    main()
