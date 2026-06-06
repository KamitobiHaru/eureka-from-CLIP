"""
Jointly train TemporalTransformer + BertEncoder on synthetic pseudo-video sequences.

Usage:
    python scripts/train_temporal.py [--config config/default.yaml]

Prerequisites:
    1. Run scripts/precompute_embeddings.py first to generate .npy files
    2. Set coco_root in config/default.yaml
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
from src.models import BertEncoder, TemporalTransformer
from src.data import get_sequence_dataloader
from src.training import SymmetricInfoNCE, OrderConsistencyLoss
from src.training.trainer import _get_cosine_schedule_with_warmup


class JointTemporalModel(nn.Module):
    """Combined TemporalTransformer + BertEncoder for joint training.

    Forward returns (video_emb, correct_text_emb, wrong_text_emb), each [B, 512].
    """

    def __init__(self, temporal: TemporalTransformer, bert: BertEncoder):
        super().__init__()
        self.temporal = temporal
        self.bert = bert

    def forward(self, frame_embs, frame_mask, correct_ids, correct_mask,
                wrong_ids, wrong_mask):
        video_emb = self.temporal(frame_embs, frame_mask)               # [B, 512]
        correct_text_emb = self.bert(correct_ids, correct_mask)         # [B, 512]
        wrong_text_emb = self.bert(wrong_ids, wrong_mask)               # [B, 512]
        return video_emb, correct_text_emb, wrong_text_emb


def main():
    parser = argparse.ArgumentParser(
        description="Train TemporalTransformer + BERT on synthetic sequences."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint .pt file")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # ── Model ────────────────────────────────────────────────────
    bert_path = cfg["model"]["bert_model_path"]
    if not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        sys.exit(1)

    t_cfg = cfg.get("temporal", {})
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

    lora_cfg = cfg.get("lora", {})
    print("Loading BERT model...")
    bert = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)

    model = JointTemporalModel(temporal, bert)

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # ── Data ──────────────────────────────────────────────────────
    cache_dir = cfg["data"]["embedding_cache"]
    coco_root = cfg["data"]["coco_root"]

    if not Path(cache_dir).is_dir() or len(list(Path(cache_dir).glob("*.npy"))) == 0:
        print(f"No precomputed embeddings found in {cache_dir}.")
        print("Run scripts/precompute_embeddings.py first.")
        sys.exit(1)

    ann_dir = cfg["data"].get("annotations_dir", "annotations")

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
    temporal_lr = t_cfg.get("temporal_lr", 1e-4)
    bert_lr = t_cfg.get("bert_lr", 3e-5)
    weight_decay = cfg["training"]["weight_decay"]

    optimizer = torch.optim.AdamW([
        {"params": model.temporal.parameters(), "lr": temporal_lr},
        {"params": model.bert.parameters(), "lr": bert_lr},
    ], weight_decay=weight_decay)

    # ── Losses ────────────────────────────────────────────────────
    nce_loss_fn = SymmetricInfoNCE(temperature=cfg["training"]["temperature"])
    order_loss_fn = OrderConsistencyLoss(
        margin=t_cfg.get("order_consistency_margin", 0.2)
    )
    order_weight = t_cfg.get("order_consistency_weight", 0.5)
    clip_anchor_weight = t_cfg.get("clip_anchor_weight", 0.3)
    text_order_weight = t_cfg.get("text_order_weight", 0.2)

    # ── Scheduler ─────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup = cfg["training"]["warmup_steps"]
    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    # ── AMP ────────────────────────────────────────────────────────
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
        model.temporal.load_state_dict(ckpt["temporal_state_dict"])
        model.bert.load_state_dict(ckpt["bert_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", math.inf)
        print(f"  Resumed at epoch {ckpt.get('epoch', 0)} (val_loss={best_val_loss:.4f})")

    # ── Training Loop ─────────────────────────────────────────────
    max_grad_norm = cfg["training"]["max_grad_norm"]
    save_every_epoch = t_cfg.get("save_every_epoch", True)

    print(f"\nTraining: {epochs} epochs, {warmup} warmup steps, cosine decay")
    print(f"  Temporal LR: {temporal_lr}, BERT LR: {bert_lr}")
    print(f"  Order loss weight: {order_weight}")
    print(f"  Save every epoch: {save_every_epoch}")
    print("-" * 60)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for batch in pbar:
            frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask = batch
            frame_embs = frame_embs.to(device)
            frame_mask = frame_mask.to(device)
            corr_ids = corr_ids.to(device)
            corr_mask = corr_mask.to(device)
            wrong_ids = wrong_ids.to(device)
            wrong_mask = wrong_mask.to(device)

            # CLIP anchor: masked mean pool over valid frames, re-normalize
            valid = (~frame_mask).unsqueeze(-1).float()  # [B, T, 1], 1=valid
            clip_anchor = F.normalize(
                (frame_embs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1),
                dim=-1,
            )

            if scaler:
                with torch.amp.autocast("cuda"):
                    video_emb, correct_emb, wrong_emb = model(
                        frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask
                    )
                    nce_loss = nce_loss_fn(video_emb, correct_emb)
                    order_loss = order_loss_fn(video_emb, correct_emb, wrong_emb)
                    clip_nce_loss = nce_loss_fn(correct_emb, clip_anchor)
                    text_order = order_loss_fn(correct_emb.detach(), correct_emb.detach(), wrong_emb)
                    loss = nce_loss + order_weight * order_loss + clip_anchor_weight * clip_nce_loss + text_order_weight * text_order

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                video_emb, correct_emb, wrong_emb = model(
                    frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask
                )
                nce_loss = nce_loss_fn(video_emb, correct_emb)
                order_loss = order_loss_fn(video_emb, correct_emb, wrong_emb)
                clip_nce_loss = nce_loss_fn(correct_emb, clip_anchor)
                text_order = order_loss_fn(correct_emb.detach(), correct_emb.detach(), wrong_emb)
                loss = nce_loss + order_weight * order_loss + clip_anchor_weight * clip_nce_loss + text_order_weight * text_order

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                nce=f"{nce_loss.item():.4f}",
                order=f"{order_loss.item():.4f}",
                clip=f"{clip_nce_loss.item():.4f}",
                txt_ord=f"{text_order.item():.4f}",
            )

        # ── End-of-epoch validation ──────────────────────────
        val_metrics = evaluate(model, val_loader, nce_loss_fn, order_loss_fn, device, use_amp)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:02d}/{epochs} done  |  "
              f"Val NCE: {val_metrics['nce']:.4f}  |  "
              f"Val Order: {val_metrics['order']:.4f}  |  "
              f"Val Clip: {val_metrics['clip']:.4f}  |  "
              f"Best NCE: {best_val_loss:.4f}  |  LR: {lr:.2e}")

        val_nce = val_metrics["nce"]
        is_best = val_nce < best_val_loss
        if is_best:
            best_val_loss = val_nce

        if is_best or save_every_epoch:
            _save_checkpoint(ckpt_dir, epoch, val_nce, model,
                             optimizer, scheduler, scaler, global_step)
            suffix = " ★ New best!" if is_best else ""
            print(f"  → Checkpoint saved.{suffix}")

        print("-" * 60)

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate(model, loader, nce_loss_fn, order_loss_fn, device, use_amp):
    model.eval()
    total_nce = 0.0
    total_order = 0.0
    total_clip = 0.0
    for batch in tqdm(loader, desc="Val", leave=False):
        frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask = batch
        frame_embs = frame_embs.to(device)
        frame_mask = frame_mask.to(device)
        corr_ids = corr_ids.to(device)
        corr_mask = corr_mask.to(device)
        wrong_ids = wrong_ids.to(device)
        wrong_mask = wrong_mask.to(device)

        # CLIP anchor
        valid = (~frame_mask).unsqueeze(-1).float()
        clip_anchor = F.normalize(
            (frame_embs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1), dim=-1,
        )

        if use_amp:
            with torch.amp.autocast("cuda"):
                video_emb, correct_emb, wrong_emb = model(
                    frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask
                )
        else:
            video_emb, correct_emb, wrong_emb = model(
                frame_embs, frame_mask, corr_ids, corr_mask, wrong_ids, wrong_mask
            )

        total_nce += nce_loss_fn(video_emb, correct_emb).item()
        total_order += order_loss_fn(video_emb, correct_emb, wrong_emb).item()
        total_clip += nce_loss_fn(correct_emb, clip_anchor).item()

    N = len(loader)
    return {"nce": total_nce / N, "order": total_order / N, "clip": total_clip / N}


def _save_checkpoint(ckpt_dir, epoch, val_loss, model, optimizer, scheduler, scaler, global_step):
    path = os.path.join(ckpt_dir, f"temporal_epoch{epoch:02d}_val{val_loss:.4f}.pt")
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "temporal_state_dict": model.temporal.state_dict(),
        "bert_state_dict": model.bert.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "lora_config": getattr(model.bert, "lora_config", None),
    }
    if scaler:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)
    return path


if __name__ == "__main__":
    main()
