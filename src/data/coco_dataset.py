import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class CocoDataset(Dataset):
    """Loads precomputed CLIP embeddings + caption text.

    Each caption is a separate (image_emb, caption_text) pair,
    so an image with 5 captions produces 5 entries.
    """

    def __init__(self, coco_root: str, split: str, embedding_cache: str, annotations_dir: str = "annotations"):
        self.embedding_cache = Path(embedding_cache)

        ann_file = Path(coco_root) / annotations_dir / f"captions_{split}.json"
        with open(ann_file, "r") as f:
            data = json.load(f)

        # (image_id, caption) pairs — one per annotation
        self.pairs: List[Tuple[int, str]] = [
            (ann["image_id"], ann["caption"]) for ann in data["annotations"]
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        img_id, caption = self.pairs[idx]
        emb_path = self.embedding_cache / f"{img_id}.npy"
        emb = np.load(emb_path)
        return torch.from_numpy(emb).float(), caption, f"coco_{img_id}"


def make_collate_fn(tokenizer: BertTokenizer):
    """Returns a collate function that tokenizes captions on the fly (BERT)."""

    def collate(batch):
        embs, texts, image_ids = zip(*batch)
        image_emb = torch.stack(embs)  # (B, 512)
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return image_emb, tokens["input_ids"], tokens["attention_mask"], list(image_ids)

    return collate


def make_clip_collate_fn(tokenizer):
    """Returns a collate function for CLIP text encoder tokenizer.

    CLIP's transformer uses causal masking internally, so attention_mask
    is returned as a dummy all-ones tensor (accepted but ignored).
    """

    def collate(batch):
        embs, texts, image_ids = zip(*batch)
        image_emb = torch.stack(embs)
        tokens = tokenizer(list(texts))
        input_ids = tokens.clone() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.long)
        B = len(batch)
        # Dummy attention_mask (all 1s) — CLIP ignores it, but Trainer
        # calls .to(device) on it so it can't be None.
        attention_mask = torch.ones(B, input_ids.size(1), dtype=torch.long)
        return image_emb, input_ids, attention_mask, list(image_ids)

    return collate


def get_dataloader(
    coco_root: str,
    split: str,
    embedding_cache: str,
    tokenizer: BertTokenizer,
    annotations_dir: str = "annotations",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CocoDataset(coco_root, split, embedding_cache, annotations_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer),
        pin_memory=True,
    )
