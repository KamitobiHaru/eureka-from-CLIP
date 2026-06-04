"""
CLIP Video Scene Search — CLI

Accepts a pre-segmented scene directory (from scripts/segment_video.py) and
searches with either CLIP's default text encoder or a trained BERT encoder.

Usage:
    # Segment once
    python scripts/segment_video.py demo.mp4 -o ./segments

    # Search with CLIP
    python run.py ./segments/demo --query "a person walking"

    # Search with BERT
    python run.py ./segments/demo --query "car" --bert_checkpoint checkpoints/bert_epoch02_val1.5993.pt

    # Change query without re-segmenting
    python run.py ./segments/demo -q "dog" --top_k 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from clip_search import SearchEngine
from clip_search.segmenter import Scene


def load_scene_dir(scene_dir: str):
    """Load pre-segmented scenes from a directory created by segment_video.py.

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


def load_bert_text_encoder(bert_checkpoint: str, config_path: str = "config/default.yaml") -> callable:
    """Load a trained BertEncoder checkpoint and return a text encoding function."""
    import yaml
    from transformers import BertTokenizer
    from src.models import BertEncoder

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_path = cfg["model"]["bert_model_path"]
    print(f"Loading BERT from: {bert_path}")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
    ).to(device)

    print(f"Loading checkpoint: {bert_checkpoint}")
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    @torch.no_grad()
    def encode_text(text: str) -> np.ndarray:
        tokens = tokenizer([text], padding=True, truncation=True, max_length=77, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        emb = model(input_ids, attention_mask)
        return emb.cpu().numpy().flatten().astype(np.float32)

    return encode_text


def main():
    parser = argparse.ArgumentParser(
        description="Search video scenes with CLIP or BERT text encoder."
    )
    parser.add_argument("scene_dir", help="Path to pre-segmented scene directory "
                                          "(e.g. ./segments/demo)")
    parser.add_argument("--query", "-q", required=True, help="Text query to search for")
    parser.add_argument("--top_k", "-k", type=int, default=5,
                        help="Number of top scenes to show (default: 5)")
    parser.add_argument("--output", "-o", default="output",
                        help="Directory to save result thumbnails (default: output/)")
    parser.add_argument("--bert_checkpoint", "-b", default=None,
                        help="Path to trained BERT checkpoint .pt file. "
                             "If not set, uses CLIP's default text encoder.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Config file for BERT model path (default: config/default.yaml)")
    args = parser.parse_args()

    # ── Step 1: load pre-segmented scenes ──────────────────
    print(f"Loading scenes from: {args.scene_dir}")
    scenes = load_scene_dir(args.scene_dir)
    print(f"  → {len(scenes)} scenes loaded")

    if not scenes:
        print("No scenes found.")
        sys.exit(0)

    # ── Step 2: encode scenes ──────────────────────────────
    print("Loading CLIP encoder (vision)...")
    engine = SearchEngine()

    if args.bert_checkpoint:
        text_encoder = load_bert_text_encoder(args.bert_checkpoint, args.config)
        engine.text_encoder = text_encoder

    # Encode all scenes
    engine.scenes = scenes
    scene_embs = []
    for scene in scenes:
        emb = engine.encoder.encode_scene(scene.frames)
        scene_embs.append(emb)
    engine.scene_embs = np.stack(scene_embs)

    # ── Step 3: search ─────────────────────────────────────
    encoder_name = "BERT" if args.bert_checkpoint else "CLIP"
    print(f'\nSearching for: "{args.query}"  (encoder: {encoder_name})')
    results = engine.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching scenes found.")
        sys.exit(0)

    # ── Step 4: display results ────────────────────────────
    print(f"\nTop {len(results)} results:")
    print("-" * 72)
    print(f"  {'Rank':<6} {'Scene':<6} {'Time range':<22} {'Score':<10}  Thumbnail")
    print("-" * 72)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, r in enumerate(results, 1):
        time_range = f"{r['start_sec']:.1f}s → {r['end_sec']:.1f}s"
        thumb_name = f"rank{rank:02d}_scene{r['scene_idx']:03d}.jpg"
        thumb_path = str(out_dir / thumb_name)

        save_thumbnail(thumb_path, r["thumbnail"])

        print(f"  {rank:<6} {r['scene_idx']:<6} {time_range:<22} {r['score']:.4f}    {thumb_path}")

    print("-" * 72)

    best = results[0]
    print(f'\nBest match: Scene {best["scene_idx"]} '
          f'({best["start_sec"]:.1f}s → {best["end_sec"]:.1f}s), '
          f'score={best["score"]:.4f}')
    print(f"Thumbnails saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
