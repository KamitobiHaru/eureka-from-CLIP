"""
Train TemporalTransformer for pseudo-video temporal understanding.

Three training modes:

  ``joint`` (default)
      Train TemporalTransformer + BERT jointly with full loss
      (NCE + order + CLIP anchor + text-order).  BERT and LoRA
      adapters are trained from scratch.

  ``bert_frozen``  (Phase 2)
      Load a pretrained BERT checkpoint (from ``train_bert.py``),
      freeze all BERT + LoRA parameters, and train only the
      TemporalTransformer.  Loss is simplified — no CLIP anchor
      needed because BERT is already in CLIP space.

  ``clip_frozen``  (Phase 2 baseline)
      Use a frozen open_clip ViT-B/32 **text encoder** instead
      of BERT.  Train only the TemporalTransformer.  Provides a
      direct comparison: how well does CLIP's own text encoder
      perform on temporal understanding vs a fine-tuned BERT?

Prerequisites:
    1. Run ``scripts/precompute_embeddings.py`` first to generate .npy files
    2. For ``bert_frozen``: a trained BERT checkpoint from ``train_bert.py``
    3. For ``clip_frozen``: a local open_clip checkpoint (download via
       ``scripts/download_clip_model.py``) — falls back to random init

Usage:
    # Joint training (original)
    python scripts/train_temporal.py

    # Phase 2 with frozen pretrained BERT
    python scripts/train_temporal.py --mode bert_frozen \\
        --bert_checkpoint /path/to/bert_best.pt

    # Phase 2 with frozen CLIP text encoder
    python scripts/train_temporal.py --mode clip_frozen
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder, TemporalTransformer, CLIPTextEncoder
from src.data import get_sequence_dataloader
from src.training import SymmetricInfoNCE, OrderConsistencyLoss
from src.training.trainer import _get_cosine_schedule_with_warmup


# ── Supported training modes ────────────────────────────────────────────
MODES = ("joint", "bert_frozen", "clip_frozen")


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: build text encoder + tokenizer
# ═══════════════════════════════════════════════════════════════════════════

def _build_text_encoder(mode, cfg, device, bert_checkpoint, freeze=True):
    """Return ``(text_encoder, tokenizer, use_clip_tokenizer)``."""
    bert_path = cfg["model"]["bert_model_path"]
    lora_cfg = cfg.get("lora", {})

    if mode == "clip_frozen":
        import open_clip
        print("Building CLIP text encoder (frozen)...")
        encoder = CLIPTextEncoder(device=device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return encoder, tokenizer, True

    # ── BERT text encoder ──────────────────────────────────────────
    if not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        sys.exit(1)

    print("Building BERT text encoder...")
    encoder = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # Load pretrained checkpoint for frozen mode
    if mode == "bert_frozen":
        if not bert_checkpoint:
            print("ERROR: --bert_checkpoint is required in bert_frozen mode")
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

    if freeze and mode != "joint":
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        print("  Text encoder frozen.")

    return encoder, tokenizer, False


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: build optimizer
# ═══════════════════════════════════════════════════════════════════════════

def _build_optimizer(temporal, text_encoder, mode, weight_decay, t_cfg):
    """Build AdamW with appropriate param groups.

    In frozen modes only temporal params are optimised.
    """
    temporal_lr = t_cfg.get("temporal_lr", 1e-4)
    bert_lr = t_cfg.get("bert_lr", 3e-5)

    if mode == "joint":
        return torch.optim.AdamW([
            {"params": temporal.parameters(), "lr": temporal_lr},
            {"params": text_encoder.parameters(), "lr": bert_lr},
        ], weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(
            temporal.parameters(), lr=temporal_lr, weight_decay=weight_decay
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: run one training step
# ═══════════════════════════════════════════════════════════════════════════

def _train_step(batch, temporal, text_encoder, mode, losses,
                device, scaler, optimizer, max_grad_norm):
    """Execute one forward/backward pass.  Returns dict of per-component losses."""
    frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask = batch
    frame_embs = frame_embs.to(device)
    frame_mask = frame_mask.to(device)

    # CLIP anchor (only needed in joint mode) — always compute for logging
    valid = (~frame_mask).unsqueeze(-1).float()
    clip_anchor = F.normalize(
        (frame_embs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1),
        dim=-1,
    )

    # ── Forward ──────────────────────────────────────────────────
    def _forward():
        video_emb = temporal(frame_embs, frame_mask)                # [B, 512]

        # Text encoder handles both BERT (needs mask) and CLIP (ignores mask)
        if corr_mask is not None:
            corr_mask_d = corr_mask.to(device)
            wrong_mask_d = wrong_mask.to(device)
        else:
            corr_mask_d = wrong_mask_d = None

        correct_emb = text_encoder(corr_ids.to(device), corr_mask_d)   # [B, 512]
        wrong_emb = text_encoder(wrong_ids.to(device), wrong_mask_d)   # [B, 512]

        # ── Loss components ──────────────────────────────────────────
        nce = losses["nce"](video_emb, correct_emb)
        order = losses["order"](video_emb, correct_emb, wrong_emb)

        result = {
            "nce": nce,
            "order": order,
            "clip": torch.tensor(0.0, device=video_emb.device),
            "text_order": torch.tensor(0.0, device=video_emb.device),
        }

        if mode == "joint":
            result["clip"] = losses["nce"](correct_emb, clip_anchor)
            result["text_order"] = losses["order"](
                correct_emb.detach(), correct_emb.detach(), wrong_emb
            )

        total = result["nce"] \
                + losses["order_w"] * result["order"] \
                + losses["clip_w"] * result["clip"] \
                + losses["text_order_w"] * result["text_order"]

        return total, result

    if scaler:
        with torch.amp.autocast("cuda"):
            total, result = _forward()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(temporal.parameters(), max_grad_norm)
        if mode == "joint":
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total, result = _forward()
        total.backward()
        torch.nn.utils.clip_grad_norm_(temporal.parameters(), max_grad_norm)
        if mode == "joint":
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_grad_norm)
        optimizer.step()

    optimizer.zero_grad()
    return {k: v.item() if hasattr(v, "item") else v for k, v in result.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(temporal, text_encoder, loader, losses, mode, device, use_amp):
    """Epoch-end validation. Returns dict of average loss components."""
    temporal.eval()
    text_encoder.eval()
    totals = {"nce": 0.0, "order": 0.0, "clip": 0.0}

    for batch in tqdm(loader, desc="Val", leave=False):
        frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask = batch
        frame_embs = frame_embs.to(device)
        frame_mask = frame_mask.to(device)
        corr_mask_d = corr_mask.to(device) if corr_mask is not None else None
        wrong_mask_d = wrong_mask.to(device) if wrong_mask is not None else None

        valid = (~frame_mask).unsqueeze(-1).float()
        clip_anchor = F.normalize(
            (frame_embs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1), dim=-1,
        )

        with torch.amp.autocast("cuda") if use_amp else nullcontext():
            video_emb = temporal(frame_embs, frame_mask)
            correct_emb = text_encoder(corr_ids.to(device), corr_mask_d)
            wrong_emb = text_encoder(wrong_ids.to(device), wrong_mask_d)

        totals["nce"] += losses["nce"](video_emb, correct_emb).item()
        totals["order"] += losses["order"](video_emb, correct_emb, wrong_emb).item()
        if mode == "joint":
            totals["clip"] += losses["nce"](correct_emb, clip_anchor).item()

    N = len(loader)
    return {k: v / N for k, v in totals.items()}


class nullcontext:
    """No-op context manager for the non-AMP code path."""
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def _save_checkpoint(ckpt_dir, epoch, val_loss, temporal, text_encoder,
                     mode, optimizer, scheduler, scaler, global_step):
    """Save checkpoint.

    In ``joint`` mode the checkpoint includes both the temporal and
    BERT state dicts (compatible with ``clip_search/temporal_pipeline.py``).

    In frozen modes a **temporal-only** checkpoint is written, so it can
    be loaded independently of which text encoder was used.
    """
    path = os.path.join(ckpt_dir, f"temporal_epoch{epoch:02d}_val{val_loss:.4f}.pt")
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "mode": mode,
        "temporal_state_dict": temporal.state_dict(),
    }

    if mode == "joint":
        # Full checkpoint compatible with temporal_pipeline.py
        state["bert_state_dict"] = text_encoder.state_dict()
        state["lora_config"] = getattr(text_encoder, "lora_config", None)

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
        description="Train TemporalTransformer on synthetic sequences."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", default=None,
                        choices=MODES,
                        help="Training mode (default: from config or 'joint')")
    parser.add_argument("--bert_checkpoint", default=None,
                        help="Path to a trained BERT checkpoint (required for "
                             "bert_frozen mode)")
    parser.add_argument("--resume", default=None,
                        help="Resume from a temporal checkpoint .pt file")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: from config)")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    t_cfg = cfg.get("temporal", {})
    mode = args.mode or t_cfg.get("mode", "joint")
    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from {MODES}")
        sys.exit(1)
    print(f"Mode:   {mode}")

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

    freeze_text = t_cfg.get("freeze_text_encoder", True)
    text_encoder, tokenizer, use_clip_tokenizer = _build_text_encoder(
        mode, cfg, device, args.bert_checkpoint,
        freeze=freeze_text,
    )

    # ── Data loaders ──────────────────────────────────────────────
    print("Loading data...")
    train_loader = get_sequence_dataloader(
        coco_root, "train2017", cache_dir, tokenizer,
        annotations_dir=ann_dir,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        min_len=t_cfg.get("sequence_min_len", 3),
        max_len=t_cfg.get("sequence_max_len", 10),
    )
    val_loader = get_sequence_dataloader(
        coco_root, "val2017", cache_dir, tokenizer,
        annotations_dir=ann_dir,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        min_len=t_cfg.get("sequence_min_len", 3),
        max_len=t_cfg.get("sequence_max_len", 10),
    )
    print(f"  Train: {len(train_loader.dataset)} unique images")
    print(f"  Val:   {len(val_loader.dataset)} unique images")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = _build_optimizer(
        temporal, text_encoder, mode,
        cfg["training"]["weight_decay"], t_cfg,
    )

    # ── Losses ────────────────────────────────────────────────────
    nce_loss_fn = SymmetricInfoNCE(temperature=cfg["training"]["temperature"])
    order_loss_fn = OrderConsistencyLoss(
        margin=t_cfg.get("order_consistency_margin", 0.2)
    )
    losses = {
        "nce": nce_loss_fn,
        "order": order_loss_fn,
        "order_w": t_cfg.get("order_consistency_weight", 0.5),
        "clip_w": t_cfg.get("clip_anchor_weight", 0.3),
        "text_order_w": t_cfg.get("text_order_weight", 0.1),
    }

    # ── Scheduler ─────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup = cfg["training"]["warmup_steps"]
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
        if "bert_state_dict" in ckpt and mode == "joint":
            text_encoder.load_state_dict(ckpt["bert_state_dict"])
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
    print(f"  Temporal LR: {t_cfg.get('temporal_lr', 1e-4)}", end="")
    if mode == "joint":
        print(f", BERT LR: {t_cfg.get('bert_lr', 3e-5)}")
    else:
        print(" (frozen text encoder)")
    print(f"  Losses: NCE + {losses['order_w']}×Order", end="")
    if mode == "joint":
        print(f" + {losses['clip_w']}×ClipAnchor + {losses['text_order_w']}×TextOrder")
    else:
        print()
    print(f"  Save best + every 5 epochs")
    print("-" * 60)

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        temporal.train()
        text_encoder.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for batch in pbar:
            result = _train_step(
                batch, temporal, text_encoder, mode, losses,
                device, scaler, optimizer, max_grad_norm,
            )
            scheduler.step()
            global_step += 1
            pbar.set_postfix(
                loss=f"{result['nce'] + losses['order_w'] * result['order']:.4f}",
                nce=f"{result['nce']:.4f}",
                order=f"{result['order']:.4f}",
                clip=f"{result['clip']:.4f}",
                txt_ord=f"{result['text_order']:.4f}",
            )

        # ── End-of-epoch validation ──────────────────────────
        val_metrics = evaluate(
            temporal, text_encoder, val_loader, losses, mode, device, use_amp,
        )
        lr = optimizer.param_groups[0]["lr"]
        val_nce = val_metrics["nce"]
        is_best = val_nce < best_val_loss
        if is_best:
            best_val_loss = val_nce

        log = (
            f"Epoch {epoch:02d}/{epochs} done  |  "
            f"Val NCE: {val_nce:.4f}  |  "
            f"Val Order: {val_metrics['order']:.4f}"
        )
        if mode == "joint":
            log += f"  |  Val Clip: {val_metrics['clip']:.4f}"
        log += f"  |  Best NCE: {best_val_loss:.4f}  |  LR: {lr:.2e}"

        # Save (best always, otherwise every 5 epochs)
        if is_best or epoch % 5 == 0:
            ckpt_path = _save_checkpoint(
                ckpt_dir, epoch, val_nce,
                temporal, text_encoder, mode,
                optimizer, scheduler, scaler, global_step,
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
