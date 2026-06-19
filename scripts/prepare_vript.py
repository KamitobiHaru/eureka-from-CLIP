"""
Prepare VRIPT dataset for evaluation: unzip clips, precompute CLIP frame
embeddings, and create a simplified annotation file.

Usage:
    python scripts/prepare_vript.py \
        --vript_root ./datasets/vript \
        --output_dir ./data/vript

Step 1 extracts all video clips from the 42 zip archives.
Step 2 encodes each clip's frames through CLIP ViT-B/32 (12 uniform frames).
Step 3 writes a compact annotation JSON for the evaluator.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

MAX_FRAMES = 12


def select_key_frames(total_frames: int, max_frames: int = MAX_FRAMES) -> list:
    """Uniformly sample ``max_frames`` frame indices from the video."""
    if total_frames <= max_frames:
        return list(range(total_frames))
    return [int((i + 0.5) * total_frames / max_frames) for i in range(max_frames)]


def extract_clip_frames(clip_path: str, encoder, max_frames: int = MAX_FRAMES, batch_size: int = 64):
    """Uniformly sample frames from a video clip and encode with CLIP.
    Uses decord for fast GPU-accelerated seeking.
    Returns (N, 512) float32 array or None on error.
    """
    import decord
    decord.bridge.set_bridge('torch')
    try:
        vr = decord.VideoReader(str(clip_path))
    except Exception:
        return None

    total_frames = len(vr)
    if total_frames == 0:
        return None

    key_indices = select_key_frames(total_frames, max_frames)
    # decord.get_batch returns torch uint8 tensor (N, H, W, 3), RGB
    frames_tensor = vr.get_batch(key_indices)  # (N, H, W, 3)
    # Convert to list of numpy arrays for CLIPEncoder
    frames = [f.numpy() for f in frames_tensor]

    embs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i: i + batch_size]
        batch_embs = encoder.encode_frames(batch)
        embs.append(batch_embs)
    return np.concatenate(embs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Prepare VRIPT dataset")
    parser.add_argument("--vript_root", default="./datasets/vript",
                        help="VRIPT dataset root (with zip files and jsonl)")
    parser.add_argument("--output_dir", default="./data/vript",
                        help="Output directory for extracted data (use a drive with space)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip CLIP frame encoding (only unzip + annotations)")
    parser.add_argument("--skip_unzip", action="store_true",
                        help="Skip unzipping (only encode + annotations)")
    args = parser.parse_args()

    vript_root = Path(args.vript_root)
    output_dir = Path(args.output_dir)

    clips_dir = output_dir / "clips"
    keyframes_dir = output_dir / "clip_keyframes"
    ann_path = output_dir / "annotations.json"

    clips_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Unzip all clips ──────────────────────────────────
    if not args.skip_unzip:
        zip_dir = vript_root / "vript_short_videos_clips"
        zip_paths = sorted(zip_dir.glob("*.zip"))
        print(f"Found {len(zip_paths)} zip files ({sum(p.stat().st_size for p in zip_paths) / 1024**3:.1f} GB)")

        for zip_path in tqdm(zip_paths, desc="Unzipping clips"):
            with zipfile.ZipFile(zip_path) as z:
                for member in z.infolist():
                    # Only extract .mp4 files
                    if not member.filename.endswith(".mp4"):
                        continue
                    # member.filename is like "shorts_clips_10_of_42/6904271117783928065/6904271117783928065-Scene-001.mp4"
                    # We want just the clip_id filename
                    clip_name = Path(member.filename).name
                    out_path = clips_dir / clip_name
                    if out_path.exists():
                        continue
                    with z.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())

        extracted = list(clips_dir.glob("*.mp4"))
        print(f"Extracted {len(extracted)} clips ({sum(p.stat().st_size for p in extracted) / 1024**3:.2f} GB)")

    # ── Step 2: Precompute CLIP frame embeddings ─────────────────
    if not args.skip_extraction:
        from clip_search.encoder import CLIPEncoder
        print("Loading CLIP encoder...")
        encoder = CLIPEncoder(model_type="openai")

        # Warm-up: encode dummy frames to stabilise CUDA context before decord
        print("  Warming up CUDA context...")
        _warmup = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
        _ = encoder.encode_frames(_warmup)
        del _warmup
        print("  CUDA ready")

        video_paths = sorted(clips_dir.glob("*.mp4"))
        to_process = [vp for vp in video_paths if not (keyframes_dir / f"{vp.stem}.npy").exists()]
        print(f"Total clips: {len(video_paths)}, cached: {len(video_paths) - len(to_process)}, "
              f"to process: {len(to_process)}")

        for vp in tqdm(to_process, desc="Extracting CLIP keyframes"):
            try:
                emb = extract_clip_frames(str(vp), encoder)
                if emb is not None:
                    np.save(str(keyframes_dir / f"{vp.stem}.npy"), emb)
            except Exception as e:
                print(f"  Error {vp.name}: {e}")

        encoded = len(list(keyframes_dir.glob("*.npy")))
        print(f"Encoded {encoded} clips")

    # ── Step 3: Create simplified annotation JSON ────────────────
    jsonl_path = vript_root / "vript_short_videos_captions.jsonl"
    print(f"Processing annotations from {jsonl_path}...")
    annotations = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            annotations.append({
                "clip_id": entry["clip_id"],
                "video_id": entry["meta"]["video_id"],
                "caption": entry["caption"]["content"],
                "shot_type": entry["caption"]["shot_type"],
                "camera_movement": entry["caption"]["camera_movement"],
                "scene_title": entry["caption"]["scene_title"],
            })

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved {len(annotations)} annotations to {ann_path}")

    # Stats
    clip_ids_in_ann = {a["clip_id"] for a in annotations}
    npy_stems = {p.stem for p in keyframes_dir.glob("*.npy")}
    overlap = clip_ids_in_ann & npy_stems
    print(f"Annotations: {len(annotations)}, Encoded frames: {len(npy_stems)}, "
          f"Match: {len(overlap)}")


if __name__ == "__main__":
    main()
