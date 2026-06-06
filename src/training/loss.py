import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE (NT-Xent) loss for contrastive learning.

    Loss = 0.5 * CE(sim / temp, labels) + 0.5 * CE(sim.T / temp, labels)
    where labels = [0, 1, ..., B-1] (diagonal = positive pairs).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        B = emb_a.size(0)
        labels = torch.arange(B, device=emb_a.device)
        sim = emb_a @ emb_b.T / self.temp  # (B, B)
        return 0.5 * self.ce(sim, labels) + 0.5 * self.ce(sim.T, labels)


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
