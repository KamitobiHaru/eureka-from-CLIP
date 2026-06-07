import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from .coco_dataset import make_collate_fn


class FlickrDataset(Dataset):
    """Loads precomputed CLIP embeddings + caption text for Flickr30k.

    Each caption is a separate (image_emb, caption_text) pair,
    matching the COCO dataset interface for seamless concatenation.

    Expects:
      - Annotations in a CSV file (``flickr_annotations_30k.csv``) with
        columns: ``raw`` (JSON list of 5 captions), ``split``, ``filename``.
      - Precomputed ``.npy`` CLIP embeddings in ``embedding_cache``, named
        ``{filename_stem}.npy`` (created by ``scripts/precompute_flickr_embeddings.py``).
    """

    def __init__(
        self,
        flickr_root: str,
        split: str,
        embedding_cache: str,
        annotation_file: str = "flickr_annotations_30k.csv",
    ):
        self.embedding_cache = Path(embedding_cache)

        ann_path = Path(flickr_root) / annotation_file
        with open(ann_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Filter by split and flatten (image, caption) pairs
        self.pairs: List[Tuple[str, str]] = []  # (filename_stem, caption)
        for r in rows:
            if r["split"].strip() != split:
                continue
            stem = Path(r["filename"].strip()).stem
            captions = json.loads(r["raw"])
            for cap in captions:
                self.pairs.append((stem, cap))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        stem, caption = self.pairs[idx]
        emb_path = self.embedding_cache / f"{stem}.npy"
        emb = np.load(emb_path)
        return torch.from_numpy(emb).float(), caption, f"flickr_{stem}"


def get_flickr_dataloader(
    flickr_root: str,
    split: str,
    embedding_cache: str,
    tokenizer: BertTokenizer,
    annotation_file: str = "flickr_annotations_30k.csv",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = FlickrDataset(flickr_root, split, embedding_cache, annotation_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer),
        pin_memory=True,
    )
