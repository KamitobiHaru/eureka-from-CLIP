import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class CocoDataset(Dataset):
    """Loads precomputed CLIP embeddings + caption text (or precomputed text
    embeddings when ``text_cache_dir`` is provided).

    Each caption is a separate (image_emb, caption_text) pair,
    so an image with 5 captions produces 5 entries.
    """

    def __init__(self, coco_root: str, split: str, embedding_cache: str,
                 annotations_dir: str = "annotations", text_cache_dir: Optional[str] = None):
        self.embedding_cache = Path(embedding_cache)

        ann_file = Path(coco_root) / annotations_dir / f"captions_{split}.json"
        with open(ann_file, "r") as f:
            data = json.load(f)

        # (image_id, caption) pairs — one per annotation
        self.pairs: List[Tuple[int, str]] = [
            (ann["image_id"], ann["caption"]) for ann in data["annotations"]
        ]

        # Precomputed text embeddings (optional)
        self.text_embs: Optional[torch.Tensor] = None
        if text_cache_dir is not None:
            text_path = Path(text_cache_dir) / f"coco_{split}.pt"
            self.text_embs = torch.load(text_path, weights_only=True)
            if len(self.text_embs) != len(self.pairs):
                raise ValueError(
                    f"Text embedding count ({len(self.text_embs)}) "
                    f"doesn't match dataset size ({len(self.pairs)}) for {split}"
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, str], str]:
        img_id, caption = self.pairs[idx]
        emb_path = self.embedding_cache / f"{img_id}.npy"
        emb = np.load(emb_path)
        if self.text_embs is not None:
            return torch.from_numpy(emb).float(), self.text_embs[idx], f"coco_{img_id}"
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


def make_text_emb_collate_fn():
    """Collate function for precomputed text embeddings — no tokenization needed.

    Expects each batch item to be (image_emb, text_emb, image_id) where
    text_emb is already a ``(512,)`` tensor.
    Returns ``(image_emb, text_emb, image_ids)``.
    """

    def collate(batch):
        image_embs, text_embs, image_ids = zip(*batch)
        return torch.stack(image_embs), torch.stack(text_embs), list(image_ids)

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
