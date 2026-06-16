"""
Precompute CLIP frame embeddings for MSVD / MSR-VTT videos using
scene-detection-based key frame selection.

Instead of uniform sampling, this script detects scene boundaries with
scenedetect ContentDetector and selects one representative frame per scene.
The intuition: scene-aware sampling reduces redundancy from long static shots
and better captures diverse visual content for video retrieval.

Output format: per-video (N, 512) .npy files, same as uniform-sampling
precomputation, so existing evaluation code can load them directly.

Usage (MSVD):
    python scripts/precompute_video_keyframes_scenedetect.py \
        --video_dir "/path/to/MSVD/YouTubeClips" \
        --output_dir ./data/msvd/clip_keyframes_scenedetect \
        --pattern "*.avi"

Usage (MSR-VTT):
    python scripts/precompute_video_keyframes_scenedetect.py \
        --video_dir /path/to/MSR-VTT/video \
        --output_dir ./data/msrvtt/clip_keyframes_scenedetect \
        --pattern "*.mp4"
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder
from scenedetect import open_video, ContentDetector, SceneManager

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"


def _detect_scenes(video_path: str, threshold: float = 27.0):
    """Run scene detection, return list of (start, end) FrameTimecode pairs."""
    try:
        video = open_video(str(video_path))
        scene_mgr = SceneManager()
        scene_mgr.add_detector(ContentDetector(threshold=threshold))
        scene_mgr.detect_scenes(video)
        return scene_mgr.get_scene_list()
    except Exception:
        return []


def select_key_frames(video_path: str, max_frames: int = 12) -> list:
    """Select frames using scene detection + per-scene uniform sampling.

    Detects scene boundaries with ContentDetector, then distributes the
    *max_frames* budget across scenes proportionally to each scene's duration.
    Within each scene, frames are sampled uniformly.  This produces more
    representative coverage than global uniform sampling when a video has
    distinct scenes of varying length.

    If no scene boundaries are detected (single continuous shot), falls back
    to global uniform sampling.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))

    # Uniform helper (same formula as the original precompute script)
    def _uniform(n=None, start=0, length=None):
        n = n or max_frames
        length = length or total_frames
        return [int(start + (i + 0.5) * length / n) for i in range(n)]

    scene_list = _detect_scenes(video_path)
    if not scene_list:
        return _uniform()

    # Proportional allocation: each scene gets frames ∝ its duration
    durations = np.array([end.frame_num - start.frame_num
                          for start, end in scene_list])
    total_dur = durations.sum()

    # Fractional allocation, at least 1 frame per scene
    n_per_scene = np.maximum(1, (max_frames * durations / total_dur).round().astype(int))

    # Adjust if sum overshoots or undershoots max_frames
    while n_per_scene.sum() > max_frames:
        # Reduce frame count from scene with most frames beyond 1
        candidates = [(i, n) for i, n in enumerate(n_per_scene) if n > 1]
        if not candidates:
            break
        i = max(candidates, key=lambda x: x[1])[0]
        n_per_scene[i] -= 1

    while n_per_scene.sum() < max_frames:
        # Give extra frame to longest scene
        i = durations.argmax()
        n_per_scene[i] += 1

    # Uniformly sample within each scene
    selected = []
    for (start, end), n in zip(scene_list, n_per_scene):
        scene_len = end.frame_num - start.frame_num
        n = min(n, scene_len)  # clamp to available frames
        if scene_len <= n:
            indices = list(range(start.frame_num, end.frame_num))
        else:
            indices = [int(start.frame_num + (i + 0.5) * scene_len / n)
                       for i in range(n)]
        selected.extend(indices)

    return sorted(selected[:max_frames])


def process_video(video_path: str, encoder, max_frames: int = 12, batch_size: int = 64):
    """Detect scenes, select key frames, encode with CLIP -> (N, 512) array."""
    key_indices = select_key_frames(video_path, max_frames)
    if not key_indices:
        return np.empty((0, 512), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0, 512), dtype=np.float32)

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
        batch_embs = encoder.encode_frames(frames[i:i + batch_size])
        embs.append(batch_embs)

    return np.concatenate(embs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP key-frame embeddings with scene detection."
    )
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.avi")
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", default="openai")
    parser.add_argument("--threshold", type=float, default=27.0,
                        help="ContentDetector sensitivity (lower = more scene changes)")
    parser.add_argument("--test_list", default=None,
                        help="Optional file with video IDs (one per line) to filter which videos to process")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test list filter if provided
    allowed_ids = None
    if args.test_list:
        with open(args.test_list) as f:
            allowed_ids = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(allowed_ids)} video IDs from {args.test_list}")

    print(f"Loading CLIP model: {args.model}")
    encoder = CLIPEncoder(model_type=args.model)

    video_paths = sorted(video_dir.glob(args.pattern))
    if not video_paths:
        print(f"No videos found matching '{args.pattern}' in {video_dir}")
        return

    if allowed_ids is not None:
        video_paths = [vp for vp in video_paths if vp.stem in allowed_ids]
        print(f"Matched {len(video_paths)} / {len(allowed_ids)} IDs in video_dir")

    to_process = [vp for vp in video_paths
                  if not (output_dir / f"{vp.stem}.npy").exists()]

    print(f"Total videos: {len(video_paths)}")
    print(f"Already cached: {len(video_paths) - len(to_process)}")
    print(f"To process: {len(to_process)}")

    if not to_process:
        print("All done.")
        return

    successful = 0
    for vp in tqdm(to_process, desc="Scene-detect key frames"):
        try:
            emb = process_video(str(vp), encoder, args.max_frames, args.batch_size)
            np.save(str(output_dir / f"{vp.stem}.npy"), emb)
            successful += 1
        except Exception as e:
            print(f"\n  Error on {vp.name}: {e}")

    print(f"\nDone. {successful}/{len(to_process)} processed. Saved to {output_dir}")


if __name__ == "__main__":
    main()
