from typing import List, Optional, Tuple

import torch


class ContrastiveQueue:
    """GPU-resident FIFO queue of (image_emb, text_emb, image_id) entries.

    Maintains a **contiguous** buffer — ``get()`` returns slices directly
    (zero-copy), avoiding the ``torch.cat`` overhead that the old circular
    buffer required on every forward pass.

    Enqueue happens AFTER loss computation to prevent current-batch
    items from acting as their own negatives.
    """

    def __init__(self, max_size: int = 65536, embed_dim: int = 512, device: str = "cuda"):
        self.max_size = max_size
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.image_embs = torch.empty(0, self.embed_dim, device=self.device)
        self.text_embs = torch.empty(0, self.embed_dim, device=self.device)
        self.image_ids: List[str] = []

    def enqueue(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_ids: List[str],
    ) -> None:
        """Append B entries. Trim oldest when over ``max_size``."""
        B = image_emb.size(0)
        self.image_embs = torch.cat([self.image_embs, image_emb.detach()])
        self.text_embs = torch.cat([self.text_embs, text_emb.detach()])
        self.image_ids.extend(image_ids)

        excess = len(self.image_embs) - self.max_size
        if excess > 0:
            # Clone so the old storage is freed (a slice would be a view)
            self.image_embs = self.image_embs[excess:].clone()
            self.text_embs = self.text_embs[excess:].clone()
            self.image_ids = self.image_ids[excess:]

    def get(
        self, device: Optional[torch.device] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
        """Return (image_embs, text_embs, image_ids) for the full queue.

        The returned tensors are views (zero-copy). The ``device`` argument
        is accepted for backward compatibility (data is already on GPU).
        """
        N = len(self.image_embs)
        if N == 0:
            return None, None, None
        return self.image_embs[:N], self.text_embs[:N], self.image_ids[:N]

    def __len__(self) -> int:
        return len(self.image_embs)

    def state_dict(self) -> Optional[dict]:
        """Serializable state for checkpointing. Returns None when empty."""
        if len(self.image_embs) == 0:
            return None
        return {
            "image_embs": self.image_embs.cpu(),
            "text_embs": self.text_embs.cpu(),
            "image_ids": self.image_ids,
        }

    def load_state_dict(self, state: Optional[dict]) -> None:
        """Restore queue state from a checkpoint."""
        if state is None:
            self.reset()
            return

        imgs = state["image_embs"].to(self.device)
        txts = state["text_embs"].to(self.device)
        ids: list = list(state["image_ids"])

        if len(ids) > self.max_size:
            excess = len(ids) - self.max_size
            imgs = imgs[excess:]
            txts = txts[excess:]
            ids = ids[excess:]

        self.image_embs = imgs
        self.text_embs = txts
        self.image_ids = ids
