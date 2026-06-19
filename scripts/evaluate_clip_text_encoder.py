"""
Evaluate CLIP's OWN text encoder on Flickr30k test (1K).

Loads the same open_clip ViT-B/32 model used as the frozen image encoder
during BERT training, and runs it in zero-shot mode (images via vision
encoder, captions via text encoder) to get the baseline recall numbers.

This tells you: what performance does CLIP's own text encoder achieve
with the exact same image encoder you're aligning BERT to?

Usage:
    python scripts/evaluate_clip_text_encoder.py [--config config/default.yaml]
"""

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder, _find_checkpoint
from src.training.evaluation import compute_recall_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP text encoder on Flickr30k")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    clip_model = cfg.get("clip", {}).get("model", "openai")

    flickr_cfg = cfg.get("flickr", {})
    flickr_root = Path(flickr_cfg["root"])
    ann_file = flickr_root / flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load CLIP model ──────────────────────────────────────
    ckpt = _find_checkpoint()
    if ckpt:
        print(f"Loading OpenAI CLIP ViT-B/32 from: {ckpt}")
    else:
        print("No local checkpoint found. Using open_clip default.")
    encoder = CLIPEncoder(device=device)

    # ── Parse test annotations ───────────────────────────────
    print(f"\nParsing: {ann_file}")
    with open(ann_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # (filename_stem, caption) pairs for test split
    pairs: list[tuple[str, str]] = []
    for r in rows:
        if r["split"].strip() != "test":
            continue
        stem = Path(r["filename"].strip()).stem
        captions = json.loads(r["raw"])
        for cap in captions:
            pairs.append((stem, cap))

    print(f"  Test pairs: {len(pairs)} ({len(pairs)//5} unique images × 5 captions)")

    # ── Locate image zip ─────────────────────────────────────
    zip_candidates = list(flickr_root.glob("*"))
    zip_path = None
    for p in zip_candidates:
        if p.is_file() and (".zip" in p.suffix.lower() or (len(p.name) == 64 and p.stat().st_size > 1_000_000_000)):
            zip_path = p
            break
    if zip_path is None:
        print("Error: no zip archive found in flickr_root")
        sys.exit(1)
    print(f"Image archive: {zip_path}")

    # ── Build stem → zip entry lookup ─────────────────────────
    with zipfile.ZipFile(zip_path) as zf:
        stem_to_entry = {}
        for name in zf.namelist():
            if name.endswith(".jpg") and not name.startswith("__MACOSX"):
                stem_to_entry[Path(name).stem] = name

        # ── Encode all unique test images ──────────────────────
        unique_stems = list(dict.fromkeys(stem for stem, _ in pairs))
        print(f"\nEncoding {len(unique_stems)} unique test images...")
        stem_to_emb: dict[str, np.ndarray] = {}
        batch_size = 32
        for i in tqdm(range(0, len(unique_stems), batch_size), desc="Images"):
            batch_stems = unique_stems[i:i + batch_size]
            batch_pils = []
            valid_stems = []
            for stem in batch_stems:
                entry = stem_to_entry.get(stem)
                if entry is None:
                    continue
                with zf.open(entry) as imgf:
                    batch_pils.append(Image.open(imgf).convert("RGB"))
                    valid_stems.append(stem)
            if batch_pils:
                embs = encoder.encode_images(batch_pils)  # (N, 512), L2-normalized
                for stem, emb in zip(valid_stems, embs):
                    stem_to_emb[stem] = emb

        # ── Build paired arrays ────────────────────────────────
        all_image_embs = []
        all_text_embs = []
        all_image_ids = []
        captions_batch = []
        print(f"\nEncoding {len(pairs)} captions with CLIP text encoder...")
        for stem, cap in tqdm(pairs, desc="Captions"):
            img_emb = stem_to_emb.get(stem)
            if img_emb is None:
                continue
            captions_batch.append(cap)
            all_image_embs.append(img_emb)
            all_image_ids.append(f"flickr_{stem}")

            # Encode texts in mini-batches for efficiency
            if len(captions_batch) >= 64:
                with torch.no_grad():
                    text_embs = encoder.model.encode_text(
                        encoder.tokenizer(captions_batch).to(device)
                    ).detach().cpu().numpy()
                text_embs = text_embs / np.linalg.norm(text_embs, axis=-1, keepdims=True)
                all_text_embs.extend(text_embs)
                captions_batch = []

        if captions_batch:
            with torch.no_grad():
                text_embs = encoder.model.encode_text(
                    encoder.tokenizer(captions_batch).to(device)
                ).detach().cpu().numpy()
            text_embs = text_embs / np.linalg.norm(text_embs, axis=-1, keepdims=True)
            all_text_embs.extend(text_embs)

        image_embs_t = torch.from_numpy(np.stack(all_image_embs))
        text_embs_t = torch.from_numpy(np.stack(all_text_embs))

        print(f"\n  Image embeddings: {image_embs_t.shape}")
        print(f"  Text embeddings:  {text_embs_t.shape}")

        # ── Compute recall ─────────────────────────────────────
        recall = compute_recall_metrics(image_embs_t, text_embs_t, all_image_ids, ks=(1, 5, 10))

        print("\n" + "=" * 50)
        print("CLIP zero-shot on Flickr30k test (1K)")
        print("=" * 50)
        for key in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
            if key in recall:
                print(f"  {key}: {recall[key]:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
