"""
Extract key frames from MSR-VTT / MSVD videos via histogram-based frame
analysis, encode them with frozen CLIP, and save per-video embeddings.

Usage (MSR-VTT):
    python scripts/precompute_video_keyframes.py \
        --video_dir /data2/zsy/datasets/MSR-VTT/video \
        --output_dir ./data/msrvtt/clip_keyframes \
        --pattern "*.mp4"

Usage (MSVD):
    python scripts/precompute_video_keyframes.py \
        --video_dir /data2/zsy/datasets/MSVD/OpenDataLab___MSVD/raw/MSVD/YouTubeClips \
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


def _hist_diff(gray1, gray2):
    """Chi-squared histogram distance between two grayscale frames."""
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)


def select_key_frames(total_frames: int, max_frames: int, diffs: list) -> list:
    """Select key frames via segment-based max-diff selection.

    Divides the video into ``max_frames`` segments and picks the frame with
    the largest histogram difference inside each segment, ensuring temporal
    coverage while favouring regions of high visual change.

    ``diffs[i]`` = histogram diff between frame *i* and *i+1*.
    """
    if total_frames <= max_frames:
        return list(range(total_frames))

    selected = []
    seg_size = total_frames / max_frames

    for seg_idx in range(max_frames):
        start = int(seg_idx * seg_size)
        end = int((seg_idx + 1) * seg_size)

        best_idx = start
        best_diff = -1.0
        for i in range(start, min(end, total_frames - 1)):
            if diffs[i] > best_diff:
                best_diff = diffs[i]
                best_idx = i
        selected.append(best_idx)

    return selected


def process_video(video_path: str, encoder, max_frames: int = 12, batch_size: int = 32):
    """Extract key frames from a video and encode them with CLIP.

    Returns (key_frame_indices, embeddings) where embeddings is [N, 512].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return [], np.empty((0, 512), dtype=np.float32)

    # ── 1st pass: read all frames and compute frame diffs ──
    frames = []
    diffs = []
    prev_gray = None

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diffs.append(_hist_diff(prev_gray, gray))
        prev_gray = gray
        frames.append(frame)

    cap.release()
    actual_frames = len(frames)
    if actual_frames == 0:
        return [], np.empty((0, 512), dtype=np.float32)

    # ── 2nd pass: select key frames ──
    key_indices = select_key_frames(actual_frames, max_frames, diffs)

    # ── 3rd pass: encode key frames with CLIP ──
    embs = []
    for i in range(0, len(key_indices), batch_size):
        batch_indices = key_indices[i : i + batch_size]
        batch_frames = [frames[idx] for idx in batch_indices]
        batch_embs = encoder.encode_frames(batch_frames)
        embs.append(batch_embs)

    all_embs = np.concatenate(embs, axis=0)  # [N_keyframes, 512]

    return key_indices, all_embs


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP key-frame embeddings for videos.")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for video files (default: *.mp4)")
    parser.add_argument("--max_frames", type=int, default=12, help="Max key frames per video (default: 12)")
    parser.add_argument("--batch_size", type=int, default=32, help="CLIP encoding batch size (default: 32)")
    parser.add_argument("--model", default="openai", help="CLIP model variant (default: openai)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CLIP Encoder ──
    from clip_search.encoder import CLIPEncoder
    print(f"Loading CLIP model: {args.model}")
    encoder = CLIPEncoder(model_type=args.model)

    # ── Gather video files ──
    video_paths = sorted(video_dir.glob(args.pattern))
    if not video_paths:
        print(f"No videos found matching '{args.pattern}' in {video_dir}")
        return

    # Filter already processed
    to_process = []
    for vp in video_paths:
        stem = vp.stem  # filename without extension
        out_path = output_dir / f"{stem}.npy"
        if not out_path.exists():
            to_process.append((vp, out_path))

    print(f"Total videos: {len(video_paths)}, already cached: {len(video_paths) - len(to_process)}")
    if not to_process:
        print("All videos already processed.")
        return

    # ── Process ──
    for vp, out_path in tqdm(to_process, desc="Extracting key frames"):
        try:
            _, emb = process_video(str(vp), encoder, max_frames=args.max_frames,
                                   batch_size=args.batch_size)
            np.save(str(out_path), emb)
        except Exception as e:
            print(f"  Error processing {vp.name}: {e}")

    print(f"Done. Embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
