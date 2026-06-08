"""Chinese Flickr30k test dataset for cross-lingual evaluation.

The image embeddings are identical to the English Flickr30k (same .npy files);
only the caption text is replaced with Chinese translations.
"""

import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FlickrZhDataset(Dataset):
    """Chinese Flickr30k test dataset.

    Expects:
      - ``annotation_file``: English CSV with columns ``raw``, ``split``, ``filename``.
      - ``zh_file``: JSON mapping filename_stem → list of 5 Chinese captions.
      - ``embedding_cache``: Precomputed ``.npy`` files named ``{stem}.npy``.
    """

    def __init__(
        self,
        flickr_root: str,
        split: str,
        embedding_cache: str,
        zh_file: str = "flickr_test_zh.json",
        annotation_file: str = "flickr_annotations_30k.csv",
    ):
        self.embedding_cache = Path(embedding_cache)

        # Load Chinese translations
        zh_path = Path(zh_file)
        if not zh_path.is_absolute():
            zh_path = Path(flickr_root) / zh_path
        with open(zh_path, encoding="utf-8") as f:
            self.zh_map: dict = json.load(f)

        # Read annotations to get (stem, caption) pairs, but only
        # keep stems that have Chinese translations available.
        ann_path = Path(flickr_root) / annotation_file
        with open(ann_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.pairs: List[Tuple[str, str]] = []
        for r in rows:
            if r["split"].strip() != split:
                continue
            stem = Path(r["filename"].strip()).stem
            zh_captions = self.zh_map.get(stem)
            if zh_captions is None:
                continue
            for cap in zh_captions:
                self.pairs.append((stem, cap))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        stem, caption = self.pairs[idx]
        emb_path = self.embedding_cache / f"{stem}.npy"
        emb = np.load(emb_path)
        return torch.from_numpy(emb).float(), caption, f"flickr_{stem}"
