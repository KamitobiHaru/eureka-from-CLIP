"""Frozen CLIP text encoder (ViT-B-32) as a trainable-compatible nn.Module.

Wraps ``open_clip`` text encoder.  **Always frozen** — no gradients flow
through this module.  Output is 512-dim L2-normalised, compatible with
the ``BertEncoder`` embedding space (after contrastive alignment).

Usage::

    encoder = CLIPTextEncoder()
    tokens = open_clip.get_tokenizer("ViT-B-32")(["a cat"])   # [1, 77]
    emb = encoder(tokens)                                       # [1, 512]
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CKPT_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "models", "clip", "open_clip_model.safetensors"),
    os.path.join(_PROJECT_ROOT, "models", "clip", "open_clip_pytorch_model.bin"),
]


def _find_clip_checkpoint() -> str | None:
    for p in _CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder (ViT-B-32 from open_clip).

    Input:  ``[B, T]`` token IDs (from ``open_clip.get_tokenizer("ViT-B-32")``)
    Output: ``[B, 512]`` L2-normalised text embeddings

    The module is always in eval mode and requires no gradients.
    ``attention_mask`` is accepted but **ignored** — CLIP's transformer
    uses causal masking internally and pools at the EOS token.
    """

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        import open_clip

        ckpt = _find_clip_checkpoint()
        if ckpt is None:
            print("[CLIPTextEncoder] No local checkpoint found, using random init.")

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=ckpt if ckpt else "",
        )
        if device is not None:
            model = model.to(device)
        model.eval()

        # Store sub-modules so we can run a differentiable forward pass
        # (open_clip's ``encode_text`` may or may not have no_grad).
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        self._device = device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard CLIP text forward — differentiable but all params frozen."""
        x = self.token_embedding(input_ids)                     # [B, T, D]
        x = x + self.positional_embedding                       # broadcast
        x = x.permute(1, 0, 2)                                  # [T, B, D]
        x = self.transformer(x)                                 # [T, B, D]
        x = x.permute(1, 0, 2)                                  # [B, T, D]
        x = self.ln_final(x)

        # Pool at EOS token (the token with the highest id)
        eos_pos = input_ids.argmax(dim=-1)                      # [B]
        x = x[torch.arange(x.shape[0], device=x.device), eos_pos]  # [B, D]
        x = x @ self.text_projection                              # [B, 512]
        return F.normalize(x, dim=-1)
