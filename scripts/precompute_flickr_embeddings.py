"""
Precompute CLIP vision embeddings for all Flickr30k images.

Usage:
    python scripts/precompute_flickr_embeddings.py [--config config/default.yaml]

Processes images read directly from the Flickr30k zip archive through the CLIP
vision encoder and saves 512-dim L2-normalized embeddings as .npy files (one
per filename-stem).
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP vision embeddings for Flickr30k.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    flickr_cfg = cfg.get("flickr", {})
    if not flickr_cfg:
        print("Error: 'flickr' section not found in config.")
        sys.exit(1)

    flickr_root = Path(flickr_cfg["root"])
    cache_dir = Path(flickr_cfg["embedding_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Locate the zip file containing images
    zip_candidates = list(flickr_root.glob("*"))
    zip_path = None
    for p in zip_candidates:
        if p.is_file() and "zip" in p.suffix.lower():
            zip_path = p
            break
    if zip_path is None:
        # fallback: check the SHA-hash-named file (no .zip extension)
        for p in zip_candidates:
            if p.is_file() and len(p.name) == 64 and not p.name.endswith(".csv"):
                # likely the hash-named zip
                import magic
                if "zip" in magic.from_file(str(p), mime=True):
                    zip_path = p
                    break
    if zip_path is None:
        # last resort: try the hash-named file directly
        for p in zip_candidates:
            if p.is_file() and p.stat().st_size > 1_000_000_000:
                zip_path = p
                break
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

    # Build set of (filename_stem, filename) pairs
    image_keys = []  # list of (stem, filename)
    seen = set()
    for r in rows:
        fname = r["filename"].strip()
        if fname not in seen:
            seen.add(fname)
            stem = Path(fname).stem
            image_keys.append((stem, fname))

    print(f"Flickr30k: {len(image_keys)} unique images to process")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Batch size: {args.batch_size}")

    encoder = CLIPEncoder()

    processed = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as z:
        # Build a quick lookup: zip entry name -> stem
        zip_entries = {}
        for name in z.namelist():
            if name.endswith(".jpg") and not name.startswith("__MACOSX"):
                zip_entries[Path(name).stem] = name

        pbar = tqdm(range(0, len(image_keys), args.batch_size), desc="Encoding")
        for i in pbar:
            batch = image_keys[i : i + args.batch_size]
            batch_pils = []
            batch_stems = []

            for stem, fname in batch:
                cache_path = cache_dir / f"{stem}.npy"
                if cache_path.exists():
                    skipped += 1
                    continue

                entry_name = zip_entries.get(stem)
                if entry_name is None:
                    continue

                try:
                    with z.open(entry_name) as imgf:
                        pil_img = Image.open(imgf).convert("RGB")
                        batch_pils.append(pil_img)
                        batch_stems.append(stem)
                except Exception as e:
                    tqdm.write(f"  Error loading {fname}: {e}")

            if batch_pils:
                embs = encoder.encode_images(batch_pils)
                for stem, emb in zip(batch_stems, embs):
                    np.save(cache_dir / f"{stem}.npy", emb)
                processed += len(batch_pils)

            pbar.set_postfix(processed=processed, skipped=skipped)

    total_in_cache = len(list(cache_dir.glob("*.npy")))
    print(f"\nDone. Processed={processed}, Skipped={skipped}, Total cached={total_in_cache}")


if __name__ == "__main__":
    main()
