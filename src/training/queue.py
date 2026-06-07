from typing import List, Optional, Tuple

import torch


class ContrastiveQueue:
    """GPU-resident FIFO queue of (image_emb, text_emb, image_id) entries.

    Uses a pre-allocated circular buffer on GPU, avoiding PCIe transfers
    and Python list overhead on every enqueue/get.  Enqueue is O(B) zero-copy
    writes into a pre-allocated tensor.

    Enqueue happens AFTER loss computation to prevent current-batch
    items from acting as their own negatives.
    """

    def __init__(self, max_size: int = 65536, embed_dim: int = 512, device: str = "cuda"):
        self.max_size = max_size
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.ptr = 0  # next write position
        self.size = 0  # number of valid entries
        self.image_embs = torch.empty(
            self.max_size, self.embed_dim, device=self.device
        )
        self.text_embs = torch.empty(
            self.max_size, self.embed_dim, device=self.device
        )
        self.image_ids: List[str] = [""] * self.max_size

    def enqueue(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_ids: List[str],
    ) -> None:
        """Add B entries to the GPU-resident circular buffer. O(B)."""
        B = image_emb.size(0)
        end = self.ptr + B

        if end <= self.max_size:
            # No wrap — single contiguous write
            self.image_embs[self.ptr : end] = image_emb.detach()
            self.text_embs[self.ptr : end] = text_emb.detach()
            for k in range(B):
                self.image_ids[self.ptr + k] = image_ids[k]
        else:
            # Wrap around the end of the buffer
            first = self.max_size - self.ptr
            self.image_embs[self.ptr :] = image_emb.detach()[:first]
            self.text_embs[self.ptr :] = text_emb.detach()[:first]
            self.image_embs[: end - self.max_size] = image_emb.detach()[first:]
            self.text_embs[: end - self.max_size] = text_emb.detach()[first:]
            for k in range(first):
                self.image_ids[self.ptr + k] = image_ids[k]
            for k in range(end - self.max_size):
                self.image_ids[k] = image_ids[first + k]

        self.ptr = end % self.max_size
        self.size = min(self.size + B, self.max_size)

    def get(
        self, device: Optional[torch.device] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
        """Return (image_embs, text_embs, image_ids) for the full queue.

        The ``device`` argument is accepted for backward compatibility
        (the data is already on the correct GPU).
        """
        if self.size == 0:
            return None, None, None

        if self.size < self.max_size:
            # Not yet full — [0 .. size) are valid
            return (
                self.image_embs[: self.size],
                self.text_embs[: self.size],
                self.image_ids[: self.size],
            )

        # Full circular buffer — linearise so [0] = oldest
        imgs = torch.cat(
            [self.image_embs[self.ptr :], self.image_embs[: self.ptr]]
        )
        txts = torch.cat(
            [self.text_embs[self.ptr :], self.text_embs[: self.ptr]]
        )
        ids = self.image_ids[self.ptr :] + self.image_ids[: self.ptr]
        return imgs, txts, ids

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> Optional[dict]:
        """Serializable state for checkpointing. Returns None when empty.

        Tensors are moved to CPU for serialisation (torch.save handles this
        transparently).
        """
        if self.size == 0:
            return None
        imgs, txts, ids = self.get(None)
        return {
            "image_embs": imgs.cpu(),
            "text_embs": txts.cpu(),
            "image_ids": ids,
        }

    def load_state_dict(self, state: Optional[dict]) -> None:
        """Restore queue state from a checkpoint."""
        if state is None:
            self.reset()
            return

        imgs = state["image_embs"].to(self.device)
        txts = state["text_embs"].to(self.device)
        ids: list = list(state["image_ids"])
        N = len(ids)

        self.size = min(N, self.max_size)
        self.ptr = self.size % self.max_size

        if N > self.max_size:
            # Truncate oldest excess entries
            excess = N - self.max_size
            imgs = imgs[excess:]
            txts = txts[excess:]
            ids = ids[excess:]

        self.image_embs[: len(imgs)] = imgs
        self.text_embs[: len(txts)] = txts
        for i in range(len(ids)):
            self.image_ids[i] = ids[i]
