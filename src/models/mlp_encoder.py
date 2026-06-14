"""
Frozen BERT + trainable MLP projection head.

Ablation control for Eureka-from-CLIP: replaces LoRA + ProjectionHead
with a deeper MLP of comparable parameter count (~1.40M) on frozen BERT
features.  Same forward interface as BertEncoder so it drops into the
existing training and evaluation pipeline.

MLP: 768 --Linear--ReLU--> 704 --Linear--ReLU--> 704 --Linear--> 512 --LayerNorm--> L2
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_BERT_DIR = os.path.join(_PROJECT_ROOT, "models", "bert")


class MLPProjection(nn.Module):
    """3-layer MLP: 768 → 704 → 704 → 512 → LayerNorm → L2Norm."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 704, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ln(self.net(x)), dim=-1)


class MLPBertEncoder(nn.Module):
    """Frozen BERT-base → [CLS] → 3-layer MLP → 512-dim L2-normalized embedding.

    BERT parameters are fully frozen (``requires_grad=False`` + ``eval()``).
    Only the MLP projection head and logit_scale are trainable.
    """

    def __init__(
        self,
        model_path: str | None = None,
        embed_dim: int = 512,
        mlp_hidden: int = 704,
        initial_temperature: float = 0.07,
    ):
        super().__init__()
        model_path = model_path or _DEFAULT_BERT_DIR
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self._freeze_bert()

        self.projection = MLPProjection(
            input_dim=self.bert.config.hidden_size,
            hidden_dim=mlp_hidden,
            output_dim=embed_dim,
        )

        # Learnable temperature (CLIP-style log-parameterized)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / initial_temperature))
        )

        self._log_params()

    def _freeze_bert(self):
        """Freeze all BERT parameters and set to eval mode."""
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.bert.eval()

    @torch.no_grad()
    def _encode_bert(self, input_ids, attention_mask):
        """Run frozen BERT, return [CLS] token."""
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self._encode_bert(input_ids, attention_mask)  # (B, 768), no grad
        return self.projection(cls)                          # (B, 512)

    def get_temperature(self) -> float:
        with torch.no_grad():
            return float(1.0 / self.logit_scale.exp().item())

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  MLPBertEncoder: {total:,} total, {trainable:,} trainable, {frozen:,} frozen")
        if trainable > 0:
            print(f"  MLP projection: {sum(p.numel() for p in self.projection.parameters()):,} params")
