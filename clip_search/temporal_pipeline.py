"""Factory for building the temporal inference pipeline (CLIP + TemporalTransformer + BERT).

Requires two separate checkpoints:
  - BERT checkpoint from ``train_bert.py`` (COCO) or ``train_bert_domain.py`` (Stack LoRA)
  - TemporalTransformer checkpoint from ``train_temporal.py``
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import yaml

from .encoder import CLIPEncoder
from .text_encoder_loader import load_text_encoder
from src.models import TemporalTransformer, BertEncoder


@dataclass
class TemporalPipeline:
    """Holds callables for scene encoding and text encoding using the temporal pipeline."""
    scene_encoder: Callable[[List[np.ndarray]], np.ndarray]  # frames -> [512]
    text_encoder: Callable[[str], np.ndarray]                 # text -> [512]


def build_temporal_pipeline(
    temporal_checkpoint: str,
    bert_checkpoint: str,
    model: str = "bert-coco",
    stack_lora_cfg: Optional[dict] = None,
    config_path: str = "config/default.yaml",
    device: str = None,
) -> TemporalPipeline:
    """Load CLIP + TemporalTransformer + BERT from separate checkpoints.

    Args:
        temporal_checkpoint: Path to temporal-transformer .pt file.
        bert_checkpoint: Path to BERT .pt file.
        model: ``"bert-coco"`` (COCO-trained BERT) or ``"bert-stack-lora"``
            (domain-adapted BERT with stacked LoRA).
        stack_lora_cfg: Stack LoRA config override (only used when
            ``model="bert-stack-lora"``).
        config_path: YAML config for model paths / architecture.
        device: Device for inference.

    Returns a TemporalPipeline with scene_encoder and text_encoder callables.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ── CLIP encoder (frozen, for per-frame features) ────────────────
    clip_encoder = CLIPEncoder(device=device)

    # ── TemporalTransformer ──────────────────────────────────────────
    t_cfg = cfg.get("temporal", {})
    temporal = TemporalTransformer(
        d_model=t_cfg.get("d_model", 512),
        nhead=t_cfg.get("nhead", 8),
        num_layers=t_cfg.get("num_layers", 2),
        dim_feedforward=t_cfg.get("dim_feedforward", 1024),
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg.get("max_frames", 16),
    ).to(device)

    temporal_ckpt = torch.load(temporal_checkpoint, map_location=device, weights_only=True)
    temporal.load_state_dict(temporal_ckpt["temporal_state_dict"])
    temporal.eval()

    # ── Text encoder (BERT) via shared loader ────────────────────────
    encode_text = load_text_encoder(
        model=model,
        bert_checkpoint=bert_checkpoint,
        config_path=config_path,
        stack_lora_cfg=stack_lora_cfg,
        device=device,
    )
    if encode_text is None:
        raise ValueError(
            "build_temporal_pipeline requires a BERT-based model "
            f"(got {model!r})"
        )

    # ── Scene encoder: CLIP frames → TemporalTransformer ─────────────
    @torch.no_grad()
    def encode_scene(frames: List[np.ndarray]) -> np.ndarray:
        """Encode a list of RGB frames into a single [512] scene embedding."""
        frame_embs = clip_encoder.encode_frames(frames)  # [T, 512]
        if len(frame_embs) == 0:
            return np.zeros(512, dtype=np.float32)

        frame_tensor = torch.from_numpy(frame_embs).unsqueeze(0).to(device)
        mask = torch.zeros(1, frame_tensor.size(1), dtype=torch.bool, device=device)
        video_emb = temporal(frame_tensor, mask)
        return video_emb.cpu().numpy().flatten().astype(np.float32)

    return TemporalPipeline(scene_encoder=encode_scene, text_encoder=encode_text)
