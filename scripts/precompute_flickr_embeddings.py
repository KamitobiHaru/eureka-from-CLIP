"""
Precompute CLIP vision embeddings for all Flickr30k images.

Usage:
    python scripts/precompute_flickr_embeddings.py [--config config/default.yaml]

Reads images from the Flickr30k zip archive, processes them through the CLIP
vision encoder, and saves 512-dim L2-normalized embeddings as .npy files (one
per filename-stem). Uses DataLoader workers for parallel image loading.
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder


class FlickrPrecomputationDataset(Dataset):
    """Yields (preprocessed_tensor, stem) for uncached images from a zip archive.

    Each DataLoader worker opens its own ZipFile handle (lazily) for safe
    multi-process reading.
    """
    def __init__(self, image_keys, cache_dir, transform, zip_path, zip_stem_map):
        self.items = [
            stem for stem, _ in image_keys
            if stem in zip_stem_map
            and not (cache_dir / f"{stem}.npy").exists()
        ]
        self.cache_dir = cache_dir
        self.transform = transform
        self.zip_path = zip_path
        self.zip_stem_map = zip_stem_map
        self._zip = None  # per-process lazy handle

    def __len__(self):
        return len(self.items)

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __getitem__(self, idx):
        stem = self.items[idx]
        z = self._get_zip()
        with z.open(self.zip_stem_map[stem]) as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img), stem


def _find_zip(flickr_root):
    """Locate the Flickr30k zip archive (may be hash-named without .zip)."""
    candidates = list(flickr_root.glob("*"))
    for p in candidates:
        if p.is_file() and "zip" in p.suffix.lower():
            return p
    for p in candidates:
        if p.is_file() and len(p.name) == 64 and not p.name.endswith(".csv"):
            import magic
            if "zip" in magic.from_file(str(p), mime=True):
                return p
    for p in candidates:
        if p.is_file() and p.stat().st_size > 1_000_000_000:
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP vision embeddings for Flickr30k.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--batch_size", type=int, default=64, help="Images per batch")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    clip_model = cfg.get("clip", {}).get("model", "laion")

    flickr_cfg = cfg.get("flickr", {})
    if not flickr_cfg:
        print("Error: 'flickr' section not found in config.")
        sys.exit(1)

    flickr_root = Path(flickr_cfg["root"])
    cache_dir = Path(flickr_cfg["embedding_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Locate zip
    zip_path = _find_zip(flickr_root)
    if zip_path is None:
        print(f"Error: no zip archive found in {flickr_root}")
        sys.exit(1)
    print(f"Found image archive: {zip_path} ({zip_path.stat().st_size / 1024**3:.1f} GB)")

    # Read CSV to get all unique filenames
    ann_file = flickr_root / flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv")
    if not ann_file.exists():
        print(f"Annotation file not found: {ann_file}")
        sys.exit(1)

    with open(ann_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    image_keys = []
    seen = set()
    for r in rows:
        fname = r["filename"].strip()
        if fname not in seen:
            seen.add(fname)
            image_keys.append((Path(fname).stem, fname))

    already_cached = len(list(cache_dir.glob("*.npy")))
    print(f"Flickr30k: {len(image_keys)} unique images to process")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Already cached: {already_cached}")

    # Build zip entry lookup (stem → entry name)
    with zipfile.ZipFile(zip_path) as z:
        zip_stem_map = {
            Path(n).stem: n
            for n in z.namelist()
            if n.endswith(".jpg") and not n.startswith("__MACOSX")
        }

    encoder = CLIPEncoder(model_type=clip_model)
    device = encoder.device

    dataset = FlickrPrecomputationDataset(
        image_keys, cache_dir, encoder.preprocess, zip_path, zip_stem_map,
    )
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
    for imgs, stems in loader:
        imgs = imgs.to(device, non_blocking=True)
        embs = encoder.model.encode_image(imgs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.detach().cpu().numpy().astype(np.float32)
        for stem, emb in zip(stems, embs):
            np.save(cache_dir / f"{stem}.npy", emb)
        processed += len(stems)
        pbar.update(len(stems))
    pbar.close()

    total_in_cache = len(list(cache_dir.glob("*.npy")))
    print(f"\nDone. Processed={processed}, Total cached={total_in_cache}")


if __name__ == "__main__":
    main()
