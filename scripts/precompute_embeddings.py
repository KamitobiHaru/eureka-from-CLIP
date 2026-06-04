"""
Precompute CLIP vision embeddings for all COCO images.

Usage:
    python scripts/precompute_embeddings.py [--config config/default.yaml] [--split val2017]

This processes images in batches through the CLIP vision encoder and saves
512-dim L2-normalized embeddings as .npy files (one per image_id).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP vision embeddings for COCO.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val2017", help="COCO split: val2017 or train2017")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    coco_root = Path(cfg["data"]["coco_root"])
    split_dir = coco_root / args.split
    cache_dir = Path(cfg["data"]["embedding_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.is_dir():
        print(f"Split directory not found: {split_dir}")
        print("Update coco_root in config/default.yaml once the data is available.")
        sys.exit(1)

    # Load image list from caption annotations
    ann_dir = cfg["data"].get("annotations_dir", "annotations")
    ann_file = coco_root / ann_dir / f"captions_{args.split}.json"
    if not ann_file.exists():
        print(f"Annotation file not found: {ann_file}")
        sys.exit(1)

    with open(ann_file) as f:
        data = json.load(f)

    images = data["images"]
    print(f"Precomputing CLIP embeddings for {args.split}: {len(images)} images")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Batch size: {args.batch_size}")

    encoder = CLIPEncoder()
    already_done = len(list(cache_dir.glob("*.npy")))

    processed = 0
    skipped = 0
    pbar = tqdm(range(0, len(images), args.batch_size), desc="Encoding")

    for i in pbar:
        batch = images[i : i + args.batch_size]
        batch_pils = []
        batch_ids = []

        for img_info in batch:
            img_path = split_dir / img_info["file_name"]
            cache_path = cache_dir / f"{img_info['id']}.npy"

            if cache_path.exists():
                skipped += 1
                continue

            if not img_path.exists():
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")
                batch_pils.append(pil_img)
                batch_ids.append(img_info["id"])
            except Exception as e:
                tqdm.write(f"  Error loading {img_path}: {e}")

        if batch_pils:
            embs = encoder.encode_images(batch_pils)
            for img_id, emb in zip(batch_ids, embs):
                np_save_path = cache_dir / f"{img_id}.npy"
                np.save(np_save_path, emb)

            processed += len(batch_pils)

        pbar.set_postfix(processed=processed, skipped=skipped, cached=already_done)

    total_in_cache = len(list(cache_dir.glob("*.npy")))
    print(f"\nDone. Processed={processed}, Skipped={skipped}, Total cached={total_in_cache}")


if __name__ == "__main__":
    main()
