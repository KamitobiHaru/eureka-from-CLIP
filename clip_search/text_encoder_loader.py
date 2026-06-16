"""Shared text encoder loader for all three model types (CLIP, COCO BERT, Stack LoRA BERT).

Usage::

    from clip_search.text_encoder_loader import load_text_encoder

    encode_text = load_text_encoder(
        model="bert-stack-lora",
        bert_checkpoint="checkpoints/domain_adapted.pt",
        config_path="config/bert_domain.yaml",
        device="cuda:0",
    )
"""

import os
from typing import Callable, Optional

import numpy as np
import torch
import yaml

from src.models import BertEncoder

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_BERT_DIR = os.path.join(_PROJECT_ROOT, "models", "bert")

# Default stack LoRA parameters (r=2, alpha=4 as per bert_domain.yaml).
DEFAULT_STACK_LORA_CFG = {
    "enabled": True,
    "r": 2,
    "alpha": 4,
    "dropout": 0.1,
    "target_modules": ["key", "query", "value", "output.dense"],
}


def _get_bert_model_config(config_path: str = "config/default.yaml") -> tuple:
    """Read BERT model path and embed dim from config. Returns defaults on failure."""
    default_path = _DEFAULT_BERT_DIR
    default_dim = 512
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return (
            cfg.get("model", {}).get("bert_model_path", default_path),
            cfg.get("model", {}).get("embed_dim", default_dim),
        )
    except FileNotFoundError:
        return default_path, default_dim


def _read_stack_lora_config(config_path: Optional[str]) -> dict:
    """Read stack_lora section from a YAML config, falling back to defaults."""
    if config_path:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            stack_cfg = cfg.get("stack_lora")
            if stack_cfg:
                stack_cfg["enabled"] = True
                return stack_cfg
        except FileNotFoundError:
            pass
    return dict(DEFAULT_STACK_LORA_CFG)


def _load_bert_coco(
    bert_checkpoint: str,
    bert_model_path: str,
    embed_dim: int,
    device: torch.device,
) -> Callable:
    """Load a COCO-trained BERT checkpoint and return an encode_text function."""
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    lora_cfg = ckpt.get("lora_config", None)

    model = BertEncoder(
        model_path=bert_model_path,
        embed_dim=embed_dim,
        lora_cfg=lora_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)

    @torch.no_grad()
    def encode_text(text: str) -> np.ndarray:
        tokens = tokenizer(
            [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        emb = model(
            tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
        )
        return emb.cpu().numpy().flatten().astype(np.float32)

    return encode_text


def _load_bert_stack_lora(
    bert_checkpoint: str,
    bert_model_path: str,
    embed_dim: int,
    stack_lora_cfg: dict,
    device: torch.device,
) -> Callable:
    """Load a domain-adapted Stack LoRA BERT checkpoint and return an encode_text function.

    The checkpoint state dict has merged BERT weights (base BERT + COCO LoRA baked
    in) and a stack LoRA adapter with r=2.  We wrap BertEncoder with the corresponding
    stack LoRA config so that the saved adapter weights load correctly.
    """
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)

    # Wrap with stack LoRA so the state dict's lora_A/lora_B keys match.
    model = BertEncoder(
        model_path=bert_model_path,
        embed_dim=embed_dim,
        lora_cfg=stack_lora_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)

    @torch.no_grad()
    def encode_text(text: str) -> np.ndarray:
        tokens = tokenizer(
            [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        emb = model(
            tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
        )
        return emb.cpu().numpy().flatten().astype(np.float32)

    return encode_text


def load_text_encoder(
    model: str = "clip",
    bert_checkpoint: Optional[str] = None,
    config_path: str = "config/default.yaml",
    stack_lora_cfg: Optional[dict] = None,
    device: Optional[str] = None,
) -> Optional[Callable[[str], np.ndarray]]:
    """Load a text encoder based on the model type.

    Parameters
    ----------
    model:
        ``"clip"`` (default CLIP text encoder), ``"bert-coco"`` (COCO-trained BERT),
        or ``"bert-stack-lora"`` (domain-adapted BERT with stacked LoRA).
    bert_checkpoint:
        Path to the BERT checkpoint ``.pt`` file.  Required for ``bert-coco`` and
        ``bert-stack-lora``.
    config_path:
        YAML config used to look up ``model.bert_model_path`` and ``model.embed_dim``.
        For ``bert-stack-lora`` the ``stack_lora`` section is also read (fallback to
        defaults r=2, alpha=4).
    stack_lora_cfg:
        Optional override dict for stack LoRA parameters.  If given, takes precedence
        over values from ``config_path`` and built-in defaults.
    device:
        Torch device string (e.g. ``"cuda:0"``, ``"cpu"``).  Auto-detected if ``None``.

    Returns
    -------
    callable or None
        A function ``encode_text(text: str) -> np.ndarray``, or ``None`` for CLIP-only
        mode (the engine falls back to ``CLIPEncoder.encode_text()``).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if model == "clip":
        return None

    if bert_checkpoint is None:
        raise ValueError(
            f"--bert-checkpoint is required when --model={model}"
        )

    bert_model_path, embed_dim = _get_bert_model_config(config_path)

    if model == "bert-coco":
        return _load_bert_coco(bert_checkpoint, bert_model_path, embed_dim, device)

    if model == "bert-stack-lora":
        resolved_cfg = stack_lora_cfg or _read_stack_lora_config(config_path)
        return _load_bert_stack_lora(
            bert_checkpoint, bert_model_path, embed_dim, resolved_cfg, device
        )

    raise ValueError(f"Unknown model type: {model}. Expected clip, bert-coco, or bert-stack-lora.")
