"""
CLIP Video Scene Search — CLI

Accepts a pre-segmented scene directory (from ``scripts/segment_video.py``) or
a folder of raw videos (with ``--folder-mode``) and searches with CLIP, COCO
BERT, or Stack LoRA BERT.

Usage:
    # Search with CLIP (folder mode — auto-segment)
    python run.py ./videos --folder-mode --query "a person walking"

    # Search with COCO BERT (folder mode)
    python run.py ./videos --folder-mode --model bert-coco \\
        --bert-checkpoint checkpoints/bert_epoch26_t2i60.5.pt -q "dog"

    # Search with Stack LoRA BERT (folder mode)
    python run.py ./videos --folder-mode --model bert-stack-lora \\
        --bert-checkpoint checkpoints/domain_adapted.pt -q "car"

    # Legacy: load pre-segmented scene directory (no --folder-mode)
    python run.py ./segments/demo -q "cat"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from clip_search import SearchEngine, load_text_encoder
from clip_search.segmenter import Scene


def load_scene_dir(scene_dir: str):
    """Load pre-segmented scenes from a directory created by ``segment_video.py``.

    Returns a list of Scene objects.
    """
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / "metadata.json"

    if not meta_path.exists():
        print(f"Error: metadata not found in {scene_dir}")
        print("Run scripts/segment_video.py first to create the scene directory.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    video_id = scene_dir.stem
    scenes = []

    for info in metadata:
        sid = info["scene_idx"]
        scene_path = scene_dir / f"scene_{sid:03d}"

        frames = []
        for i in range(info["num_frames"]):
            frame_path = scene_path / f"frame_{i:03d}.jpg"
            if frame_path.exists():
                img = cv2.imread(str(frame_path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        scenes.append(Scene(
            video_id=video_id,
            scene_idx=sid,
            start_sec=info["start_sec"],
            end_sec=info["end_sec"],
            frames=frames,
        ))

    return scenes


def save_thumbnail(path: str, frame: np.ndarray) -> None:
    """Save a single RGB frame as a JPEG thumbnail."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(
        description="Search video scenes with CLIP or BERT text encoder."
    )
    parser.add_argument("scene_dir", help="Path to pre-segmented scene directory "
                                          "(e.g. ./segments/demo) or, with --folder-mode, "
                                          "a folder of raw videos.")
    parser.add_argument("--query", "-q", required=True, help="Text query to search for")
    parser.add_argument("--top_k", "-k", type=int, default=5,
                        help="Number of top scenes to show (default: 5)")
    parser.add_argument("--output", "-o", default="output",
                        help="Directory to save result thumbnails (default: output/)")
    parser.add_argument("--folder-mode", action="store_true",
                        help="Treat scene_dir as a folder of raw videos to auto-segment "
                             "(instead of a pre-segmented scene directory).")
    parser.add_argument("--model", default="clip",
                        choices=["clip", "bert-coco", "bert-stack-lora"],
                        help="Text encoder model type (default: clip).")
    parser.add_argument("--bert_checkpoint", "-b", default=None,
                        help="Path to BERT checkpoint .pt file. "
                             "Required when --model is bert-coco or bert-stack-lora.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Config file for BERT model path and stack_lora parameters "
                             "(default: config/default.yaml).")
    parser.add_argument("--device", default=None,
                        help="Device to run on: 'cpu' or 'cuda'. Default: auto-detect.")
    args = parser.parse_args()

    # ── Validation ───────────────────────────────────────────────
    if args.model in ("bert-coco", "bert-stack-lora") and not args.bert_checkpoint:
        parser.error(f"--bert-checkpoint is required when --model={args.model}")

    engine = SearchEngine(device=args.device)

    # ── Step 1: load scenes ──────────────────────────────────────
    if args.folder_mode:
        print(f"Folder mode — processing videos from: {args.scene_dir}")
        engine.process_folder(args.scene_dir)
        if not engine.scenes:
            print("No scenes found in any video.")
            sys.exit(0)
    else:
        print(f"Loading scenes from: {args.scene_dir}")
        scenes = load_scene_dir(args.scene_dir)
        print(f"  → {len(scenes)} scenes loaded")
        if not scenes:
            print("No scenes found.")
            sys.exit(0)

        # Encode scenes manually (legacy path)
        print("Loading CLIP encoder (vision)...")
        engine.scenes = scenes
        scene_embs = []
        for scene in scenes:
            emb = (engine.scene_encoder(scene.frames)
                   if engine.scene_encoder
                   else engine.encoder.encode_scene(scene.frames))
            scene_embs.append(emb)
        engine.scene_embs = np.stack(scene_embs)

        # Set up video_map for clip extraction (legacy: one video per run)
        engine.video_map[scenes[0].video_id] = str(Path(args.scene_dir).resolve())

    # ── Step 2: set up text encoder ──────────────────────────────
    if args.model != "clip":
        text_enc = load_text_encoder(
            model=args.model,
            bert_checkpoint=args.bert_checkpoint,
            config_path=args.config,
            device=args.device,
        )
        engine.text_encoder = text_enc

    # ── Step 3: search ───────────────────────────────────────────
    encoder_name = args.model.upper()
    print(f'\nSearching for: "{args.query}"  (encoder: {encoder_name})')
    results = engine.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching scenes found.")
        sys.exit(0)

    # ── Step 4: display results ──────────────────────────────────
    print(f"\nTop {len(results)} results:")
    print("-" * 88)
    print(f"  {'Rank':<6} {'Video':<14} {'Scene':<6} {'Time range':<22} {'Score':<10}  Thumbnail")
    print("-" * 88)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, r in enumerate(results, 1):
        time_range = f"{r['start_sec']:.1f}s → {r['end_sec']:.1f}s"
        thumb_name = f"rank{rank:02d}_{r['video_id']}_scene{r['scene_idx']:03d}.jpg"
        thumb_path = str(out_dir / thumb_name)

        save_thumbnail(thumb_path, r["thumbnail"])

        print(f"  {rank:<6} {r['video_id']:<14} {r['scene_idx']:<6} "
              f"{time_range:<22} {r['score']:.4f}    {thumb_path}")

    print("-" * 88)

    best = results[0]
    print(f'\nBest match: [{best["video_id"]}] Scene {best["scene_idx"]} '
          f'({best["start_sec"]:.1f}s → {best["end_sec"]:.1f}s), '
          f'score={best["score"]:.4f}')
    print(f"Thumbnails saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
