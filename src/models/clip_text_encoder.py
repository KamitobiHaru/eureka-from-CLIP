"""Frozen CLIP text encoder (ViT-B-32, OpenAI WIT-400M) as a trainable-compatible nn.Module.

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
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_OPENAI_CKPT_CANDIDATES = [
    # HF clone (converted) — preferred source
    os.path.join(_PROJECT_ROOT, "models", "clip-vit-base-patch32", "open_clip_model.safetensors"),
    # Legacy cached open-clip checkpoints (same weights)
    os.path.join(_PROJECT_ROOT, "models", "clip", "openai_open_clip_model.safetensors"),
    os.path.join(_PROJECT_ROOT, "models", "clip", "openai_pytorch_model.bin"),
]


def _find_clip_checkpoint() -> str | None:
    """Return the first existing checkpoint path for OpenAI CLIP ViT-B/32."""
    for p in _OPENAI_CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder (ViT-B-32 from open_clip, OpenAI WIT-400M), optionally trainable via LoRA.

    Input:  ``[B, T]`` token IDs (from ``open_clip.get_tokenizer("ViT-B-32")``)
    Output: ``[B, 512]`` L2-normalised text embeddings

    ``attention_mask`` is accepted but **ignored** — CLIP's transformer
    uses causal masking internally and pools at the EOS token.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ):
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
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.register_buffer("attn_mask", model.attn_mask, persistent=False)

        # Freeze all base params first
        for p in self.parameters():
            p.requires_grad = False

        # Fixed logit_scale (CLIP default), registered as buffer so
        # it follows model.to(device), matching BertEncoder interface.
        self.register_buffer(
            "logit_scale",
            torch.tensor(np.log(1.0 / 0.07)),
        )

        # ── LoRA ────────────────────────────────────────────────────
        if lora_cfg and lora_cfg.get("enabled"):
            self._apply_lora(lora_cfg)

        self._device = device

    def _apply_lora(self, lora_cfg: Dict[str, Any]) -> None:
        """Manually replace target Linear layers with PEFT LoRA layers.

        ``get_peft_model`` won't work here because PEFT's wrapper expects
        HuggingFace-style ``input_ids`` kwargs, but CLIP's ``Transformer``
        takes a plain tensor positional arg.
        """
        from peft.tuners.lora import Linear as LoRALinear

        r = lora_cfg.get("r", 8)
        alpha = lora_cfg.get("alpha", 16)
        dropout = lora_cfg.get("dropout", 0.1)
        target_patterns = lora_cfg.get(
            "target_modules", ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
        )

        def _find_and_replace(module, path=""):
            for name, child in list(module.named_children()):
                full = f"{path}.{name}" if path else name
                if isinstance(child, nn.Linear) and any(
                    pat in full for pat in target_patterns
                ):
                    lora_linear = LoRALinear(
                        child,       # base_layer
                        "default",   # adapter_name
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                    )
                    setattr(module, name, lora_linear)
                else:
                    _find_and_replace(child, full)

        _find_and_replace(self.transformer)

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  CLIP LoRA enabled: {n_train:,} trainable / {n_total:,} total")

    def get_temperature(self) -> torch.Tensor:
        """Fixed temperature (CLIP's default 1/0.07), matching BertEncoder interface."""
        return torch.tensor(1.0 / 0.07)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard CLIP text forward — differentiable but all params frozen.

        Matches ``open_clip.CLIP.encode_text`` (``batch_first=True``).
        """
        x = self.token_embedding(input_ids)                     # [B, T, D]
        x = x + self.positional_embedding                       # [B, T, D]
        x = self.transformer(x, attn_mask=self.attn_mask)       # [B, T, D]
        x = self.ln_final(x)

        # Pool at EOS token (the token with the highest id)
        eos_pos = input_ids.argmax(dim=-1)                      # [B]
        x = x[torch.arange(x.shape[0], device=x.device), eos_pos]  # [B, D]
        x = x @ self.text_projection                              # [B, 512]
        return F.normalize(x, dim=-1)
