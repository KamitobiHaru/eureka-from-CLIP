import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# ── Connector vocabulary ──────────────────────────────────────────────
INDEX_WORDS = [
    "First", "Second", "Third", "Fourth", "Fifth",
    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
]
LAST_WORDS = ["Last", "Finally", "At last"]

SEQUENTIAL_STYLES: List[List[str]] = [
    ["First", "Then", "Finally"],
    ["To begin with", "After that", "At last"],
    ["First", "After that", "Finally"],
    ["First", "Next", "Last"],
]

MIDDLE_WORDS = ["Then", "After that", "What's more", "Next"]


def _build_connectors(K: int, style: str) -> List[str]:
    """Build a list of K connector words for the given style."""
    if style == "index":
        words = list(INDEX_WORDS[: K - 1])
        words.append(random.choice(LAST_WORDS))
        return words[:K]
    else:
        base = random.choice(SEQUENTIAL_STYLES)
        if K <= 3:
            return base[:K]
        mid = [random.choice(MIDDLE_WORDS) for _ in range(K - 2)]
        return [base[0]] + mid + [base[-1]]


def _format_caption(connectors: List[str], captions: List[str]) -> str:
    """Join connector-word pairs into a caption string."""
    parts = [f"{w}, {c}" for w, c in zip(connectors, captions)]
    return ". ".join(parts) + "."


# ── Dataset ───────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Synthesise pseudo-video sequences from COCO still images.

    Each sample groups K=random(3,10) images, retrieves their precomputed
    CLIP embeddings, and builds two temporally-connectorized captions:

      * correct  — captions ordered to match the image sequence
      * shuffled — same captions in a permuted order (wrong order, used as
                   a negative for the temporal-order loss)

    The image embeddings are preloaded into memory at construction time
    (total ~240 MiB for train2017, ~10 MiB for val2017).
    """

    def __init__(
        self,
        coco_root: str,
        split: str,
        embedding_cache: str,
        annotations_dir: str = "annotations",
        min_len: int = 3,
        max_len: int = 10,
    ):
        self.embedding_cache = Path(embedding_cache)
        self.min_len = min_len
        self.max_len = max_len

        # Load COCO captions
        ann_file = Path(coco_root) / annotations_dir / f"captions_{split}.json"
        with open(ann_file) as f:
            data = json.load(f)

        # Build image_id -> [caption, ...] mapping
        img_to_captions: dict[int, List[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            img_to_captions.setdefault(img_id, []).append(ann["caption"])

        # Filter to images that have a precomputed embedding
        valid_ids: List[int] = []
        for img_id in img_to_captions:
            if (self.embedding_cache / f"{img_id}.npy").exists():
                valid_ids.append(img_id)

        self.img_ids: List[int] = valid_ids
        self.img_to_captions: dict[int, List[str]] = img_to_captions

        # Preload all embeddings into memory (~240 MiB)
        self._embeddings: dict[int, np.ndarray] = {}
        for img_id in self.img_ids:
            path = self.embedding_cache / f"{img_id}.npy"
            self._embeddings[img_id] = np.load(path).astype(np.float32)

    def __len__(self) -> int:
        return len(self.img_ids)  # number of unique images

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return (frame_embs, positions, correct_caption).

        Positions are zeros (no motion data) — the temporal training script
        masks the position loss for non-motion samples via pos_mask.
        """
        K = random.randint(self.min_len, self.max_len)
        sampled = random.sample(self.img_ids, K)

        # Frame embeddings in sampled order → [K, 512]
        frame_embs = torch.from_numpy(np.stack([self._embeddings[i] for i in sampled]))

        # Placeholder positions (all zeros) — no motion data for this dataset
        positions = torch.zeros(K, 3)

        # One caption per image
        captions = [random.choice(self.img_to_captions[i]) for i in sampled]

        # Connector style
        style = random.choice(["index", "sequential"])
        connectors = _build_connectors(K, style)

        # -- Correct caption: captions in image order --
        correct_caption = _format_caption(connectors, captions)

        return frame_embs, positions, correct_caption


# ── Collate ───────────────────────────────────────────────────────────

def sequence_collate_fn(tokenizer, max_text_len: int = 77):
    """Return a collate function for temporal sequence batches.

    Expects each dataset element as ``(frame_embs, positions, correct_text)``
    where ``frame_embs`` is [K, 512], ``positions`` is [K, 3] (cx, cy, scale
    normalised to [0,1], all-zero for non-motion samples), and ``correct_text``
    is the temporally ordered caption.
    """

    def collate(batch):
        frame_embs, positions, correct_texts = zip(*batch)

        # Pad frame embeddings
        lengths = [e.size(0) for e in frame_embs]
        max_T = max(lengths)
        B = len(batch)
        padded = torch.zeros(B, max_T, frame_embs[0].size(-1), dtype=torch.float32)
        mask = torch.ones(B, max_T, dtype=torch.bool)  # True = padding
        pos_padded = torch.zeros(B, max_T, 3, dtype=torch.float32)

        for i, (emb, pos, L) in enumerate(zip(frame_embs, positions, lengths)):
            padded[i, :L] = emb
            mask[i, :L] = False
            pos_padded[i, :L] = pos

        # Position mask: True for frames where real position data exists
        # (connector / original sequences have all-zero placeholder positions)
        pos_mask = (pos_padded.abs().sum(dim=-1) > 1e-8) & ~mask  # [B, max_T]

        correct_tokens = tokenizer(
            list(correct_texts),
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        return (
            padded,                       # [B, max_T, 512]
            mask,                         # [B, max_T]  True=padding
            correct_tokens["input_ids"],
            correct_tokens["attention_mask"],
            pos_padded,                   # [B, max_T, 3]
            pos_mask,                     # [B, max_T]  True=valid position
        )

    return collate


def get_sequence_dataloader(
    coco_root: str,
    split: str,
    embedding_cache: str,
    tokenizer: BertTokenizer,
    annotations_dir: str = "annotations",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    min_len: int = 3,
    max_len: int = 10,
) -> DataLoader:
    dataset = SequenceDataset(
        coco_root=coco_root,
        split=split,
        embedding_cache=embedding_cache,
        annotations_dir=annotations_dir,
        min_len=min_len,
        max_len=max_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn(tokenizer),
        pin_memory=True,
    )
