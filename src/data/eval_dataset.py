"""Combined evaluation dataset for CLIP-style retrieval evaluation.

Builds a validation set from COCO val2017 + Flickr30k test (when available),
giving a ~6000-image / ~30000-caption pool — closer to CLIP's evaluation
protocol than COCO-only validation.
"""

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import BertTokenizer

from .coco_dataset import CocoDataset, make_collate_fn
from .flickr_dataset import FlickrDataset


def build_combined_val_loader(
    cfg: dict,
    tokenizer: BertTokenizer,
    batch_size: int = 64,
    num_workers: int = 4,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader that evaluates on COCO val2017 + Flickr30k test combined.

    *NOTE:* This pools images from both datasets into one gallery, which is useful
    for measuring generalisation under a larger candidate pool, but is NOT how CLIP's
    paper reports numbers (CLIP evaluates each dataset separately).

    For per-dataset CLIP-style evaluation, use ``build_split_loaders()`` instead.

    Returns a standard DataLoader yielding batches of
    (image_emb, input_ids, attention_mask, image_ids).
    """
    datasets = []

    # ── COCO val2017 (always present) ────────────────────────
    coco_root = cfg["data"]["coco_root"]
    coco_cache = cfg["data"]["embedding_cache"]
    ann_dir = cfg["data"].get("annotations_dir", "annotations")
    coco_val = CocoDataset(coco_root, "val2017", coco_cache, ann_dir)
    datasets.append(coco_val)
    print(f"  COCO val2017: {len(coco_val):,} captions")

    # ── Flickr30k test (optional) ────────────────────────────
    flickr_cfg = cfg.get("flickr", {})
    if flickr_cfg.get("root"):
        flickr_cache = flickr_cfg["embedding_cache"]
        flickr_cache_path = Path(flickr_cache)
        if flickr_cache_path.is_dir() and len(list(flickr_cache_path.glob("*.npy"))) > 0:
            flickr_test = FlickrDataset(
                flickr_cfg["root"], "test",
                flickr_cache,
                annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
            )
            datasets.append(flickr_test)
            print(f"  Flickr30k test: {len(flickr_test):,} captions")
        else:
            print("  Flickr30k test: skipped (no embeddings found)")

    combined = ConcatDataset(datasets)
    print(f"  Combined val set: {len(combined):,} captions total")

    _collate_fn = collate_fn or make_collate_fn(tokenizer)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )


def build_split_loaders(
    cfg: dict,
    tokenizer: BertTokenizer,
    batch_size: int = 64,
    num_workers: int = 4,
    collate_fn=None,
) -> dict:
    """Build separate DataLoaders for each validation split.

    Returns a dict::

        {
            "coco": DataLoader,
            "flickr": DataLoader | None,
        }

    Useful for per-dataset breakdown in standalone eval scripts.
    """
    _collate_fn = collate_fn or make_collate_fn(tokenizer)
    loaders = {}

    # COCO val2017
    coco_root = cfg["data"]["coco_root"]
    coco_cache = cfg["data"]["embedding_cache"]
    ann_dir = cfg["data"].get("annotations_dir", "annotations")
    coco_val = CocoDataset(coco_root, "val2017", coco_cache, ann_dir)
    loaders["coco"] = DataLoader(
        coco_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    # Flickr30k test (optional)
    flickr_cfg = cfg.get("flickr", {})
    if flickr_cfg.get("root"):
        flickr_cache = flickr_cfg["embedding_cache"]
        flickr_cache_path = Path(flickr_cache)
        if flickr_cache_path.is_dir() and len(list(flickr_cache_path.glob("*.npy"))) > 0:
            flickr_test = FlickrDataset(
                flickr_cfg["root"], "test",
                flickr_cache,
                annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
            )
            loaders["flickr"] = DataLoader(
                flickr_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_collate_fn,
                pin_memory=True,
            )

    return loaders
