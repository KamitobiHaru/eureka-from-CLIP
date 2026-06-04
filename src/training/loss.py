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
