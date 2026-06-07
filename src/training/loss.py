from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .queue import ContrastiveQueue


class SymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE (NT-Xent) loss for contrastive learning.

    Loss = (1 - t2i_weight) * CE(sim * scale, labels)
         + t2i_weight * CE(sim.T * scale, labels)
    where labels = [0, 1, ..., B-1] (diagonal = positive pairs)
    and   scale = logit_scale.exp()   (learnable, CLIP-style)

    When ``logit_scale`` is not provided (e.g. evaluation without access
    to the model parameter), falls back to the fixed ``temperature`` set
    at construction time.

    The ``t2i_weight`` controls the asymmetry: 0.5 = symmetric.
    """

    def __init__(self, temperature: float = 0.07, t2i_weight: float = 0.5):
        super().__init__()
        self.temp = temperature
        self.t2i_weight = t2i_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor,
                logit_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emb_a.size(0)
        labels = torch.arange(B, device=emb_a.device)
        if logit_scale is not None:
            sim = emb_a @ emb_b.T * logit_scale.exp()  # (B, B)
        else:
            sim = emb_a @ emb_b.T / self.temp  # (B, B)
        w = self.t2i_weight
        return (1 - w) * self.ce(sim, labels) + w * self.ce(sim.T, labels)


class OrderConsistencyLoss(nn.Module):
    """Triplet margin loss for temporal order.

    Penalizes when sim(anchor, positive) is not closer than sim(anchor, negative)
    by at least a margin. Operates on L2-normed embeddings (cosine via dot product).

    Loss = max(0, margin - sim(anchor, pos) + sim(anchor, neg)).mean()
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos = (anchor * positive).sum(dim=-1)   # [B], cosine similarity
        sim_neg = (anchor * negative).sum(dim=-1)   # [B]
        return F.relu(self.margin - sim_pos + sim_neg).mean()


class QueueInfoNCE(nn.Module):
    """Symmetric InfoNCE with extra negatives from a ContrastiveQueue
    and false-negative masking based on image_id.

    When the queue is empty or None, falls back to standard symmetric
    InfoNCE (identical to SymmetricInfoNCE).

    False-negative masking: queue entries with the same image_id as a
    batch item are masked out (set to -inf) so they don't contribute to
    the softmax denominator. In-batch false negatives (same image,
    different caption) are also masked.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        queue: Optional[ContrastiveQueue] = None,
        t2i_weight: float = 0.5,
        uniform_weight: float = 0.0,
    ):
        super().__init__()
        self.temp = temperature
        self.queue = queue
        self.t2i_weight = t2i_weight
        self.uniform_weight = uniform_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        image_ids: Optional[List[str]] = None,
        logit_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = image_emb.size(0)
        labels = torch.arange(B, device=image_emb.device)

        # ── Scaling factor ───────────────────────────────────────
        if logit_scale is not None:
            scale = logit_scale.exp()
        else:
            scale = 1.0 / self.temp  # fixed fallback

        # ── Fallback: standard in-batch symmetric InfoNCE ──────────
        if self.queue is None or len(self.queue) == 0:
            sim = image_emb @ text_emb.T * scale
            w = self.t2i_weight
            contrastive = (1 - w) * self.ce(sim, labels) + w * self.ce(sim.T, labels)
            return contrastive

        # ── Queue-assisted loss ────────────────────────────────────
        q_image, q_text, q_ids = self.queue.get(image_emb.device)
        Q = q_image.size(0)

        # All candidates: current batch + queue
        all_text = torch.cat([text_emb, q_text], dim=0)   # (B+Q, D)
        all_image = torch.cat([image_emb, q_image], dim=0) # (B+Q, D)

        sim_i2t = image_emb @ all_text.T * scale    # (B, B+Q)
        sim_t2i = text_emb @ all_image.T * scale    # (B, B+Q)

        # ── False-negative mask (O(B+Q+M) dict-based) ──────────────
        if image_ids is not None:
            mask = torch.zeros(B, B + Q, dtype=torch.bool, device=image_emb.device)

            # image_id → positions in current batch
            id_to_bpos: dict[str, list] = {}
            for i, img_id in enumerate(image_ids):
                id_to_bpos.setdefault(img_id, []).append(i)

            # image_id → positions in queue
            id_to_qpos: dict[str, list] = {}
            for q_idx, q_id in enumerate(q_ids):
                id_to_qpos.setdefault(q_id, []).append(q_idx)

            # Only iterate over IDs that appear in BOTH batch and queue
            for img_id, b_positions in id_to_bpos.items():
                q_positions = id_to_qpos.get(img_id)
                if q_positions:
                    for b_idx in b_positions:
                        for q_idx in q_positions:
                            mask[b_idx, B + q_idx] = True

                # In-batch false negatives (same image, different caption)
                if len(b_positions) > 1:
                    for b_idx in b_positions:
                        for other in b_positions:
                            if b_idx != other:
                                mask[b_idx, other] = True

            sim_i2t = sim_i2t.masked_fill(mask, float("-inf"))
            sim_t2i = sim_t2i.masked_fill(mask, float("-inf"))

        w = self.t2i_weight
        contrastive = (1 - w) * self.ce(sim_i2t, labels) + w * self.ce(sim_t2i, labels)

        # ── Text uniformity regulariser ──────────────────────────
        if self.uniform_weight > 0 and B > 1 and image_ids is not None:
            # Pairwise text-text cosine similarity
            text_sim = text_emb @ text_emb.T  # (B, B)

            # Build mask: exclude diagonal and same-image pairs
            id_to_idx = {img_id: i for i, img_id in enumerate(set(image_ids))}
            id_indices = torch.tensor(
                [id_to_idx[img_id] for img_id in image_ids], device=text_emb.device
            )
            same_image = id_indices[:, None] == id_indices[None, :]  # (B, B)

            uni_mask = (~torch.eye(B, dtype=torch.bool, device=text_emb.device)
                        & ~same_image)

            n_pairs = uni_mask.sum()
            if n_pairs > 0:
                uniformity = torch.exp(text_sim[uni_mask]).mean().log()
                contrastive = contrastive + self.uniform_weight * uniformity
                self._last_uniformity = uniformity.item()
            else:
                self._last_uniformity = 0.0
        else:
            self._last_uniformity = 0.0

        return contrastive
