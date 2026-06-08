"""
Train TemporalTransformer with frozen pretrained BERT on pseudo-video sequences.

COCO still images are assembled into pseudo-video sequences with connectorized
captions (e.g. "First, a dog. Then, a cat. Finally, a car.").  The TemporalTransformer
aggregates per-frame CLIP embeddings into a single video embedding, and the frozen
BERT provides the corresponding text embedding.

Loss functions:
  * SymmetricInfoNCE — contrastive alignment between video_emb and text_emb.
  * PositionPredictionReward — frame-level MSE on (cx, cy, scale) for motion
    samples (the "RL reward" that teaches temporal understanding).
  * AnchorMSE — light regulariser that keeps the video embedding anchored in
    CLIP embedding space.

Prerequisites:
  1. Run scripts/precompute_embeddings.py first (generates .npy files)
  2. Run scripts/precompute_motion_sequences.py (generates motion data)
  3. A trained BERT checkpoint from train_bert.py

Usage:
    python scripts/train_temporal.py \\
        --bert_checkpoint checkpoints/bert_best.pt \\
        --config config/default3_temporal.yaml \\
        --use_motion
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder, TemporalTransformer
from src.data import get_sequence_dataloader, get_mixed_dataloader
from src.training import SymmetricInfoNCE, PositionPredictionReward
from src.training.trainer import _get_cosine_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: build frozen BERT text encoder
# ═══════════════════════════════════════════════════════════════════════════

def _build_bert_encoder(cfg, device, bert_checkpoint):
    """Build BertEncoder, load pretrained checkpoint, freeze all params.
    Returns ``(text_encoder, tokenizer)``.
    """
    bert_path = cfg["model"]["bert_model_path"]
    if not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        sys.exit(1)

    lora_cfg = cfg.get("lora", {})
    print("Building BERT text encoder (frozen)...")
    encoder = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)

    if not bert_checkpoint:
        print("ERROR: --bert_checkpoint is required")
        sys.exit(1)
    if not Path(bert_checkpoint).exists():
        print(f"BERT checkpoint not found: {bert_checkpoint}")
        sys.exit(1)
    print(f"Loading pretrained BERT from: {bert_checkpoint}")
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("  BERT frozen.")

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    return encoder, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: build optimizer
# ═══════════════════════════════════════════════════════════════════════════

def _build_optimizer(temporal, weight_decay, t_cfg):
    """Build AdamW optimising only temporal transformer params."""
    temporal_lr = t_cfg.get("temporal_lr", 1e-4)
    return torch.optim.AdamW(
        temporal.parameters(), lr=temporal_lr, weight_decay=weight_decay
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: run one training step
# ═══════════════════════════════════════════════════════════════════════════

def _train_step(batch, temporal, text_encoder, losses,
                device, scaler, optimizer, max_grad_norm):
    """Execute one forward/backward pass.  Returns dict of per-component losses."""
    frame_embs, frame_mask, corr_ids, corr_mask, positions, pos_mask = batch
    frame_embs = frame_embs.to(device)
    frame_mask = frame_mask.to(device)
    corr_mask = corr_mask.to(device)
    positions = positions.to(device)
    pos_mask = pos_mask.to(device)

    # ── Forward ──────────────────────────────────────────────────
    def _forward():
        video_emb, per_frame = temporal(
            frame_embs, frame_mask,
            positions=positions, return_per_frame=True,
        )

        correct_emb = text_encoder(corr_ids.to(device), corr_mask)   # [B, 512]

        # Symmetric InfoNCE: align video_emb ↔ correct_text_emb
        sNCE = losses["sNCE"](video_emb, correct_emb)

        # Position prediction reward: only motion frames contribute
        pos_reward = losses["pos_reward"](per_frame, positions, pos_mask)

        # Anchoring loss: keep video_emb near mean-pooled frame embeddings
        # This prevents the video embedding from drifting too far from CLIP space.
        valid = (~frame_mask).unsqueeze(-1).float()      # [B, T, 1], 1=valid
        frame_mean = (frame_embs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        frame_mean = F.normalize(frame_mean, dim=-1)     # [B, 512]
        anchor_mse = F.mse_loss(video_emb, frame_mean.detach())

        total = (losses["sNCE_w"] * sNCE
                 + losses["pos_w"] * pos_reward
                 + losses["anchor_w"] * anchor_mse)
        return total, {"sNCE": sNCE, "pos_reward": pos_reward, "anchor_mse": anchor_mse}

    if scaler:
        with torch.amp.autocast("cuda"):
            total, result = _forward()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(temporal.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total, result = _forward()
        total.backward()
        torch.nn.utils.clip_grad_norm_(temporal.parameters(), max_grad_norm)
        optimizer.step()

    optimizer.zero_grad()
    return {k: v.item() if hasattr(v, "item") else v for k, v in result.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(temporal, text_encoder, loader, sNCE_fn, pos_reward_fn, device, use_amp):
    """Epoch-end validation. Returns dict of average sNCE and pos MSE losses."""
    temporal.eval()
    text_encoder.eval()
    sNCE_total = 0.0
    pos_total = 0.0
    count = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        frame_embs, frame_mask, corr_ids, corr_mask, positions, pos_mask = batch
        frame_embs = frame_embs.to(device)
        frame_mask = frame_mask.to(device)
        corr_mask = corr_mask.to(device)
        positions = positions.to(device)
        pos_mask = pos_mask.to(device)

        ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with ctx:
            video_emb, per_frame = temporal(
                frame_embs, frame_mask, positions=positions, return_per_frame=True,
            )
            correct_emb = text_encoder(corr_ids.to(device), corr_mask)
            sNCE_loss = sNCE_fn(video_emb, correct_emb)
            pos_loss = pos_reward_fn(per_frame, positions, pos_mask)

        sNCE_total += sNCE_loss.item()
        pos_total += pos_loss.item()
        count += 1

    return {"sNCE": sNCE_total / count, "pos": pos_total / count}


class nullcontext:
    """No-op context manager for the non-AMP code path."""
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def _save_checkpoint(ckpt_dir, epoch, val_loss, temporal,
                     optimizer, scheduler, scaler, global_step):
    """Save temporal-only checkpoint (BERT is loaded separately at inference)."""
    path = os.path.join(ckpt_dir, f"temporal_epoch{epoch:02d}_val{val_loss:.4f}.pt")
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "temporal_state_dict": temporal.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler:
        state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train TemporalTransformer with frozen BERT on synthetic sequences."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--bert_checkpoint", required=True,
                        help="Path to trained BERT checkpoint from train_bert.py")
    parser.add_argument("--resume", default=None,
                        help="Resume from a temporal checkpoint .pt file")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: from config)")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    parser.add_argument("--use_motion", action="store_true",
                        help="Use precomputed motion+connector mixed dataset")
    parser.add_argument("--motion_cache", default=None,
                        help="Override motion cache directory (default: from config)")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    t_cfg = cfg.get("temporal", {})

    # ── Data checks ───────────────────────────────────────────────
    cache_dir = cfg["data"]["embedding_cache"]
    coco_root = cfg["data"]["coco_root"]
    if not Path(cache_dir).is_dir() or len(list(Path(cache_dir).glob("*.npy"))) == 0:
        print(f"No precomputed embeddings found in {cache_dir}.")
        print("Run scripts/precompute_embeddings.py first.")
        sys.exit(1)
    ann_dir = cfg["data"].get("annotations_dir", "annotations")

    # ── Build models ──────────────────────────────────────────────
    print("Building TemporalTransformer...")
    temporal = TemporalTransformer(
        d_model=t_cfg.get("d_model", 512),
        nhead=t_cfg.get("nhead", 8),
        num_layers=t_cfg.get("num_layers", 2),
        dim_feedforward=t_cfg.get("dim_feedforward", 1024),
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg.get("max_frames", 16),
    ).to(device)
    print(f"  Temporal params: {sum(p.numel() for p in temporal.parameters()):,}")

    text_encoder, tokenizer = _build_bert_encoder(cfg, device, args.bert_checkpoint)

    # ── Data loaders ──────────────────────────────────────────────
    print("Loading data...")
    if args.use_motion:
        motion_cache = args.motion_cache or cfg.get("motion", {}).get("cache_dir",
                                                                       "data/coco/mixed_sequences")
        print(f"  Using mixed-sequence cache: {motion_cache}")
        if not Path(motion_cache, "train2017", "samples.json").exists():
            print(f"ERROR: No precomputed sequences found at {motion_cache}.")
            print("Run scripts/precompute_motion_sequences.py first.")
            sys.exit(1)
        batch_size = cfg["training"]["batch_size"]
        num_workers = cfg["training"]["num_workers"]
        train_loader = get_mixed_dataloader(
            motion_cache, "train2017", tokenizer,
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = get_mixed_dataloader(
            motion_cache, "val2017", tokenizer,
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
    else:
        train_loader = get_sequence_dataloader(
            coco_root, "train2017", cache_dir, tokenizer,
            annotations_dir=ann_dir,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True, num_workers=cfg["training"]["num_workers"],
            min_len=t_cfg.get("sequence_min_len", 3),
            max_len=t_cfg.get("sequence_max_len", 10),
        )
        val_loader = get_sequence_dataloader(
            coco_root, "val2017", cache_dir, tokenizer,
            annotations_dir=ann_dir,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False, num_workers=cfg["training"]["num_workers"],
            min_len=t_cfg.get("sequence_min_len", 3),
            max_len=t_cfg.get("sequence_max_len", 10),
        )
    print(f"  Train: {len(train_loader.dataset)} sequences")
    print(f"  Val:   {len(val_loader.dataset)} sequences")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = _build_optimizer(temporal, cfg["training"]["weight_decay"], t_cfg)

    # ── Losses ────────────────────────────────────────────────────
    sNCE_fn = SymmetricInfoNCE(temperature=cfg["training"]["temperature"])
    pos_reward_fn = PositionPredictionReward(d_model=t_cfg.get("d_model", 512)).to(device)
    losses = {
        "sNCE": sNCE_fn,
        "pos_reward": pos_reward_fn,
        "sNCE_w": t_cfg.get("order_consistency_weight", 1.0),
        "pos_w": t_cfg.get("position_prediction_weight", 0.1),
        "anchor_w": t_cfg.get("anchor_mse_weight", 0.05),
    }

    # ── Scheduler ─────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup = t_cfg.get("temporal_warmup", cfg["training"].get("warmup_steps", 0))
    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    # ── AMP ───────────────────────────────────────────────────────
    use_amp = cfg["training"]["amp"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpoint dir ────────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir or cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────
    global_step = 0
    start_epoch = 1
    best_val_loss = math.inf

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        temporal.load_state_dict(ckpt["temporal_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", math.inf)
        print(f"  Resumed at epoch {ckpt.get('epoch', 0)} (val_loss={best_val_loss:.4f})")

    # ── Training setup ────────────────────────────────────────────
    max_grad_norm = cfg["training"]["max_grad_norm"]

    print(f"\nTraining: {epochs} epochs, {warmup} warmup steps, cosine decay")
    print(f"  Temporal LR: {t_cfg.get('temporal_lr', 1e-4)} (frozen BERT)")
    print(f"  Losses: {losses['sNCE_w']}×SymmetricInfoNCE + "
          f"{losses['pos_w']}×PositionPrediction + "
          f"{losses['anchor_w']}×AnchorMSE")
    print(f"  Save best + every 5 epochs")
    print("-" * 60)

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        temporal.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for batch in pbar:
            result = _train_step(
                batch, temporal, text_encoder, losses,
                device, scaler, optimizer, max_grad_norm,
            )
            scheduler.step()
            global_step += 1
            total_loss = (losses["sNCE_w"] * result["sNCE"]
                          + losses["pos_w"] * result["pos_reward"]
                          + losses["anchor_w"] * result["anchor_mse"])
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                sNCE=f"{result['sNCE']:.4f}",
                pos=f"{result['pos_reward']:.4f}",
                anchor=f"{result['anchor_mse']:.4f}",
            )

        # ── End-of-epoch validation ──────────────────────────
        val_metrics = evaluate(
            temporal, text_encoder, val_loader, sNCE_fn, pos_reward_fn, device, use_amp,
        )
        lr = optimizer.param_groups[0]["lr"]
        val_snce = val_metrics["sNCE"]
        val_pos = val_metrics["pos"]
        is_best = val_snce < best_val_loss
        if is_best:
            best_val_loss = val_snce

        log = (
            f"Epoch {epoch:02d}/{epochs} done  |  "
            f"Val sNCE: {val_snce:.4f}  |  Val pos: {val_pos:.4f}  |  "
            f"Best sNCE: {best_val_loss:.4f}  |  LR: {lr:.2e}"
        )

        # Save (best always, otherwise every 5 epochs)
        if is_best or epoch % 5 == 0:
            ckpt_path = _save_checkpoint(
                ckpt_dir, epoch, val_snce,
                temporal, optimizer, scheduler, scaler, global_step,
            )
            suffix = f"  → {Path(ckpt_path).name}"
            if is_best:
                suffix += "  ★ New best!"
            log += suffix

        print("  " + log)
        print("-" * 60)

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
