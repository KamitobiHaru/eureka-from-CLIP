import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_BERT_DIR = os.path.join(_PROJECT_ROOT, "models", "bert")


class ProjectionHead(nn.Module):
    """Linear(768, 512) → LayerNorm → L2Norm"""

    def __init__(self, input_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ln(self.proj(x)), dim=-1)


class BertEncoder(nn.Module):
    """BERT-base → [CLS] → ProjectionHead → 512-dim L2-normalized embedding.

    Supports optional LoRA fine-tuning via the ``lora_cfg`` argument.
    When LoRA is enabled the underlying BERT weights are frozen and only
    low-rank adapters + the projection head are trained.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: int = 512,
        lora_cfg: Optional[Dict[str, Any]] = None,
        initial_temperature: float = 0.07,
    ):
        super().__init__()
        model_path = model_path or _DEFAULT_BERT_DIR
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.projection = ProjectionHead(self.bert.config.hidden_size, embed_dim)

        # Learnable temperature (CLIP-style log-parameterized)
        #   effective temperature = 1 / logit_scale.exp()
        #   initialised so that temperature ≈ initial_temperature
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / initial_temperature))
        )

        # Optional LoRA adaptation ------------------------------------------------
        self.lora_config: Optional[Dict[str, Any]] = None       # serializable dict from config
        self._peft_config = None                                 # peft LoraConfig object (not serialized)
        if lora_cfg and lora_cfg.get("enabled", False):
            self.lora_config = lora_cfg
            self._apply_lora(lora_cfg)

        self._stack_lora_config: Optional[Dict[str, Any]] = None

        self._log_params()

    # ── LoRA ─────────────────────────────────────────────────────────────────────

    def _apply_lora(self, cfg: Dict[str, Any]) -> None:
        """Wrap ``self.bert`` in a PEFT LoRA model."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA is enabled but `peft` is not installed.\n"
                "  pip install peft>=0.12.0"
            )

        lora_config = LoraConfig(
            r=cfg.get("r", 8),
            lora_alpha=cfg.get("alpha", 16),
            lora_dropout=cfg.get("dropout", 0.1),
            target_modules=cfg.get("target_modules", ["query", "value"]),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.bert = get_peft_model(self.bert, lora_config)
        self.bert.gradient_checkpointing_enable()
        self._peft_config = lora_config  # keep a reference for checkpoint metadata

    def load_and_stack_lora(self, checkpoint_path: str, stack_lora_cfg: Dict[str, Any]) -> None:
        """Load pretrained LoRA checkpoint and stack a new smaller LoRA on top.

        The model must already have a LoRA adapter ``"default"`` applied (via
        :meth:`_apply_lora` in ``__init__``).  This method:

        1. Loads checkpoint weights into the ``"default"`` adapter, projection
           head, and logit scale.
        2. Merges the ``"default"`` LoRA adapter into the base BERT weights
           via ``merge_and_unload``, so the base LoRA's knowledge is baked
           into the model and doesn't need to stay as a separate adapter.
        3. Adds a **single** new LoRA adapter (the "stack") with a smaller
           rank (e.g. r=2) on top of the merged weights.
        4. Only the stack LoRA + projection head remain trainable.

        This avoids depending on PEFT's multi-adapter forward API, which
        changed across PEFT versions (``active_adapter`` no longer accepts
        a list in PEFT >= 0.13).
        """
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("LoRA stacking requires `peft`.\n  pip install peft>=0.12.0")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.load_state_dict(ckpt["model_state_dict"], strict=False)

        # ── Step 1: Merge "default" LoRA into BERT weights ──
        self.bert = self.bert.merge_and_unload()
        # Now self.bert is a plain BertModel (no PEFT wrapper) with
        # the base LoRA's contribution baked into the weights.

        # ── Step 2: Re-wrap with only the stack LoRA ──
        stack_config = LoraConfig(
            r=stack_lora_cfg["r"],
            lora_alpha=stack_lora_cfg["alpha"],
            lora_dropout=stack_lora_cfg.get("dropout", 0.1),
            target_modules=stack_lora_cfg.get("target_modules", ["query", "value"]),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.bert = get_peft_model(self.bert, stack_config)
        self.bert.gradient_checkpointing_enable()
        self._peft_config = stack_config

        self._stack_lora_config = stack_lora_cfg

        self._log_params()

    # ── Forward ──────────────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # (B, 768) [CLS] token
        return self.projection(pooled)  # (B, 512)

    def get_temperature(self) -> float:
        """Return the current effective temperature (for logging / monitoring).

        The forward pass uses ``logit_scale.exp()`` directly in the loss
        rather than dividing by temperature; this method is provided solely
        for human-readable logging of the temperature value.
        """
        with torch.no_grad():
            return float(1.0 / self.logit_scale.exp().item())

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _log_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  BERT params: {total:,} total, {trainable:,} trainable, {frozen:,} frozen")
        if self.lora_config:
            print(f"  Base LoRA config: r={self.lora_config.get('r', 8)}, "
                  f"alpha={self.lora_config.get('alpha', 16)}, "
                  f"target_modules={self.lora_config.get('target_modules', ['query', 'value'])}")
        if self._stack_lora_config:
            sc = self._stack_lora_config
            print(f"  Stack LoRA config: r={sc['r']}, alpha={sc['alpha']}, "
                  f"target_modules={sc.get('target_modules', ['query', 'value'])}")
