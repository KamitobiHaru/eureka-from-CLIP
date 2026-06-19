"""
Precompute CLIP vision embeddings for all COCO images.

Usage:
    python scripts/precompute_embeddings.py [--config config/default.yaml] [--split val2017]

Processes images through the CLIP vision encoder and saves 512-dim L2-normalized
embeddings as .npy files (one per image_id). Uses DataLoader workers for parallel
image loading and preprocessing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder


class CocoPrecomputationDataset(Dataset):
    """Yields (preprocessed_tensor, image_id) for uncached images only."""
    def __init__(self, images, split_dir, cache_dir, transform):
        self.items = [
            (info["id"], split_dir / info["file_name"])
            for info in images
            if not (cache_dir / f"{info['id']}.npy").exists()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, img_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_id


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP vision embeddings for COCO.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--split", default="val2017", help="COCO split: val2017 or train2017")
    parser.add_argument("--batch_size", type=int, default=64, help="Images per batch")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    clip_model = cfg.get("clip", {}).get("model", "openai")

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
    already_cached = len(list(cache_dir.glob("*.npy")))
    print(f"Precomputing CLIP embeddings for {args.split}: {len(images)} images")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Already cached: {already_cached}")

    encoder = CLIPEncoder(model_type=clip_model, device=args.device)
    device = encoder.device

    dataset = CocoPrecomputationDataset(images, split_dir, cache_dir, encoder.preprocess)
    if len(dataset) == 0:
        print("  All images already cached. Nothing to do.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    processed = 0
    pbar = tqdm(total=len(dataset), desc="Encoding", unit="img")
    for imgs, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        embs = encoder.model.encode_image(imgs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.detach().cpu().numpy().astype(np.float32)
        for img_id, emb in zip(ids, embs):
            np.save(cache_dir / f"{img_id}.npy", emb)
        processed += len(ids)
        pbar.update(len(ids))
    pbar.close()

    total_in_cache = len(list(cache_dir.glob("*.npy")))
    print(f"\nDone. Processed={processed}, Total cached={total_in_cache}")


if __name__ == "__main__":
    main()
