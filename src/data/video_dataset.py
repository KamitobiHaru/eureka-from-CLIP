"""
Video datasets for retrieval with precomputed CLIP key-frame embeddings.

Supports MSR-VTT and MSVD annotation formats.  Frame embeddings are stored
as per-video ``.npy`` files (N_keyframes, 512).
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class VideoDataset(Dataset):
    """Loads precomputed video key-frame embeddings + captions.

    Expects ``annotation_file`` to be a JSON array of objects with keys
    ``video_id`` and ``caption`` (single string) **or** ``captions`` (list of
    strings).  Each caption produces a separate entry, so a video with 20
    captions creates 20 rows.

    Frame embeddings are loaded from ``frame_cache / {video_id}.npy`` (shape
    ``[N_keyframes, 512]``).
    """

    def __init__(self, annotation_file: str, frame_cache: str):
        self.frame_cache = Path(frame_cache)

        with open(annotation_file) as f:
            raw = json.load(f)

        pairs: List[Tuple[str, str]] = []  # (video_id, caption)
        for entry in raw:
            vid = entry["video_id"]
            caps = entry.get("captions")
            if caps is None:
                caps = entry.get("caption")
            if isinstance(caps, str):
                caps = [caps]

            for c in caps:
                if c and c.strip():
                    pairs.append((vid, c.strip()))

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        video_id, caption = self.pairs[idx]
        emb_path = self.frame_cache / f"{video_id}.npy"
        if not emb_path.exists():
            # Fallback: empty single frame (should not happen with proper precomputation)
            emb = np.zeros((1, 512), dtype=np.float32)
        else:
            emb = np.load(str(emb_path))
        return torch.from_numpy(emb).float(), caption, video_id


def video_collate_fn(tokenizer: BertTokenizer, max_text_len: int = 77):
    """Collate function for VideoDataset.

    Returns::

        frame_embs    [B, T, 512]   padded to max T in batch
        padding_mask  [B, T]        True = padded positions
        input_ids     [B, L]
        attention_mask [B, L]
        video_ids     List[str]
    """
    def collate(batch):
        embs, texts, video_ids = zip(*batch)

        # ── Pad frame embeddings to max length in batch ──
        max_T = max(e.size(0) for e in embs)
        B = len(embs)
        frame_embs = torch.zeros(B, max_T, embs[0].size(1), dtype=embs[0].dtype)
        padding_mask = torch.ones(B, max_T, dtype=torch.bool)  # True = padded

        for i, e in enumerate(embs):
            T = e.size(0)
            frame_embs[i, :T] = e
            padding_mask[i, :T] = False  # valid positions

        # ── Tokenize texts ──
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )

        return frame_embs, padding_mask, tokens["input_ids"], tokens["attention_mask"], list(video_ids)

    return collate


def video_emb_collate_fn():
    """Collate for datasets that return precomputed text embeddings.

    Expects each batch item to be (frame_embs, text_emb, video_id).

    Returns::

        frame_embs    [B, T, 512]
        padding_mask  [B, T]
        text_embs     [B, 512]
        video_ids     List[str]
    """
    def collate(batch):
        embs, text_embs, video_ids = zip(*batch)

        max_T = max(e.size(0) for e in embs)
        B = len(embs)
        frame_embs = torch.zeros(B, max_T, embs[0].size(1), dtype=embs[0].dtype)
        padding_mask = torch.ones(B, max_T, dtype=torch.bool)

        for i, e in enumerate(embs):
            T = e.size(0)
            frame_embs[i, :T] = e
            padding_mask[i, :T] = False

        return frame_embs, padding_mask, torch.stack(text_embs), list(video_ids)

    return collate
