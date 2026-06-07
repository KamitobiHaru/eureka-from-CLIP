"""Factory for building the temporal inference pipeline (CLIP + TemporalTransformer + BERT).

Requires two separate checkpoints:
  - BERT checkpoint from ``train_bert.py``
  - TemporalTransformer checkpoint from ``train_temporal.py``
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import yaml
from transformers import BertTokenizer

from .encoder import CLIPEncoder
from src.models import TemporalTransformer, BertEncoder


@dataclass
class TemporalPipeline:
    """Holds callables for scene encoding and text encoding using the temporal pipeline."""
    scene_encoder: Callable[[List[np.ndarray]], np.ndarray]  # frames -> [512]
    text_encoder: Callable[[str], np.ndarray]                 # text -> [512]


def build_temporal_pipeline(
    temporal_checkpoint: str,
    bert_checkpoint: str,
    config_path: str = "config/default.yaml",
    device: str = None,
) -> TemporalPipeline:
    """Load CLIP + TemporalTransformer + BERT from separate checkpoints.

    Args:
        temporal_checkpoint: Path to temporal-transformer .pt file.
        bert_checkpoint: Path to BERT .pt file from ``train_bert.py``.
        config_path: YAML config for model paths / architecture.
        device: Device for inference.

    Returns a TemporalPipeline with scene_encoder and text_encoder callables.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP encoder (frozen, for per-frame features)
    clip_encoder = CLIPEncoder(device=device)

    # TemporalTransformer
    t_cfg = cfg.get("temporal", {})
    temporal = TemporalTransformer(
        d_model=t_cfg.get("d_model", 512),
        nhead=t_cfg.get("nhead", 8),
        num_layers=t_cfg.get("num_layers", 2),
        dim_feedforward=t_cfg.get("dim_feedforward", 1024),
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg.get("max_frames", 16),
    ).to(device)

    # BERT text encoder (load LoRA config from checkpoint metadata)
    bert_ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    lora_cfg = bert_ckpt.get("lora_config", None)
    bert = BertEncoder(
        model_path=cfg["model"]["bert_model_path"],
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)
    bert.load_state_dict(bert_ckpt["model_state_dict"])
    bert.eval()

    # Temporal weights
    temporal_ckpt = torch.load(temporal_checkpoint, map_location=device, weights_only=True)
    temporal.load_state_dict(temporal_ckpt["temporal_state_dict"])
    temporal.eval()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        cfg["model"]["bert_model_path"], local_files_only=True
    )

    @torch.no_grad()
    def encode_scene(frames: List[np.ndarray]) -> np.ndarray:
        """Encode a list of RGB frames into a single [512] scene embedding."""
        # Step 1: CLIP per-frame embeddings
        frame_embs = clip_encoder.encode_frames(frames)  # [T, 512]
        if len(frame_embs) == 0:
            return np.zeros(512, dtype=np.float32)

        # Step 2: TemporalTransformer
        frame_tensor = torch.from_numpy(frame_embs).unsqueeze(0).to(device)  # [1, T, 512]
        mask = torch.zeros(1, frame_tensor.size(1), dtype=torch.bool, device=device)
        video_emb = temporal(frame_tensor, mask)  # [1, 512]
        return video_emb.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_text(text: str) -> np.ndarray:
        """Encode a text query into a [512] embedding via BERT."""
        tokens = tokenizer(
            [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        emb = bert(input_ids, attention_mask)
        return emb.cpu().numpy().flatten().astype(np.float32)

    return TemporalPipeline(scene_encoder=encode_scene, text_encoder=encode_text)
