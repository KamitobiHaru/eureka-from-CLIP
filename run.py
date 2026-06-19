"""
CLIP Video Scene Search — CLI

Accepts a folder of raw videos, uniformly samples frames per video, mean-pools
CLIP embeddings, and searches with CLIP, COCO BERT, or Stack LoRA BERT.

Usage:
    # Search with CLIP (default)
    python run.py ./videos --query "a person walking"

    # Search with COCO BERT
    python run.py ./videos --model bert-coco \
        --bert-checkpoint checkpoints/bert_epoch26_t2i60.5.pt -q "dog"

    # Search with Stack LoRA BERT
    python run.py ./videos --model bert-stack-lora \
        --bert-checkpoint checkpoints/domain_adapted.pt -q "car"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from clip_search import SearchEngine, load_text_encoder


def save_thumbnail(path: str, frame: np.ndarray) -> None:
    """Save a single RGB frame as a JPEG thumbnail."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(
        description="Search videos with CLIP or BERT text encoder."
    )
    parser.add_argument("video_folder", help="Path to a folder of video files")
    parser.add_argument("--query", "-q", required=True, help="Text query to search for")
    parser.add_argument("--top_k", "-k", type=int, default=5,
                        help="Number of top results to show (default: 5)")
    parser.add_argument("--output", "-o", default="output",
                        help="Directory to save result thumbnails (default: output/)")
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

    # ── Step 1: process all videos in folder ─────────────────────
    print(f"Processing videos from: {args.video_folder}")
    engine.process_folder(args.video_folder)
    if not engine.scenes:
        print("No scenes found in any video.")
        sys.exit(0)

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
