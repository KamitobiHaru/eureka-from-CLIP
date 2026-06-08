"""
Train TemporalTransformer on COCO retrieval with a frozen BERT text encoder.

Inserts the TemporalTransformer between CLIP image embeddings and the
contrastive loss (QueueInfoNCE).  BERT (base + LoRA) is fully frozen.
This serves as Phase-1 "safe projection" training — the temporal module
learns to re-project CLIP embeddings to better match the fixed BERT text
space without disrupting retrieval performance.

Usage:
    python scripts/train_bert_temporal.py \\
        --config config/default.yaml \\
        --bert_checkpoint path/to/bert_best.pt \\
        [--checkpoint_dir ...] \\
        [--device cuda:0]

Prerequisites (same as train_bert.py):
    1. Run scripts/precompute_embeddings.py for COCO
    2. (optional) Run scripts/precompute_flickr_embeddings.py for Flickr30k
    3. A trained BERT checkpoint from train_bert.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
import yaml
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder, TemporalTransformer
from src.data import (
    CocoDataset, FlickrDataset,
    make_text_emb_collate_fn,
)
from src.training import (
    QueueInfoNCE, ContrastiveQueue, SymmetricInfoNCE,
)
from src.training.temporal_trainer import TemporalTrainer
from src.training.evaluation import compute_recall_metrics


def _build_frozen_bert(cfg, device, bert_checkpoint):
    """Load BertEncoder, load pretrained weights, freeze ALL params."""
    te_cfg = cfg.get("text_encoder", {})
    bert_path = (
        te_cfg.get("bert_model_path")
        or cfg["model"].get("bert_model_path")
    )
    if not bert_path or not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        sys.exit(1)

    lora_cfg = cfg.get("lora", {})
    print("Loading BERT model (will be frozen)...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
        initial_temperature=cfg["training"]["temperature"],
    ).to(device)

    if not bert_checkpoint:
        print("ERROR: --bert_checkpoint is required")
        sys.exit(1)
    if not Path(bert_checkpoint).exists():
        print(f"BERT checkpoint not found: {bert_checkpoint}")
        sys.exit(1)
    print(f"Loading pretrained BERT from: {bert_checkpoint}")
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  BERT frozen (base + LoRA).")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train TemporalTransformer on COCO retrieval (frozen BERT)."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--bert_checkpoint", required=True,
                        help="Path to trained BERT checkpoint from train_bert.py")
    parser.add_argument("--resume", default=None,
                        help="Resume from a temporal_checkpoint .pt file")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: from config)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    t_cfg = cfg.get("temporal", {})

    # ── Frozen BERT ────────────────────────────────────────────
    bert_model = _build_frozen_bert(cfg, device, args.bert_checkpoint)
    n_bert = sum(p.numel() for p in bert_model.parameters())
    print(f"  BERT total params: {n_bert:,} (all frozen)")

    # ── TemporalTransformer ────────────────────────────────────
    print("Building TemporalTransformer...")
    temporal = TemporalTransformer(
        d_model=t_cfg.get("d_model", 512),
        nhead=t_cfg.get("nhead", 8),
        num_layers=t_cfg.get("num_layers", 2),
        dim_feedforward=t_cfg.get("dim_feedforward", 1024),
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg.get("max_frames", 16),
    ).to(device)
    n_temporal = sum(p.numel() for p in temporal.parameters())
    print(f"  TemporalTransformer params: {n_temporal:,}")

    # ── Data ───────────────────────────────────────────────────
    collate_fn = make_text_emb_collate_fn()
    coco_cache = cfg["data"]["embedding_cache"]
    coco_text_cache = coco_cache + "_text"
    coco_root = cfg["data"]["coco_root"]

    if not Path(coco_cache).is_dir() or len(list(Path(coco_cache).glob("*.npy"))) == 0:
        print(f"\nNo precomputed CLIP embeddings found in {coco_cache}.")
        print("Run scripts/precompute_embeddings.py first.")
        sys.exit(1)

    ann_dir = cfg["data"].get("annotations_dir", "annotations")
    coco_train = CocoDataset(coco_root, "train2017", coco_cache, ann_dir,
                             text_cache_dir=coco_text_cache)
    datasets_train = [coco_train]

    flickr_cfg = cfg.get("flickr", {})
    if flickr_cfg.get("root") and flickr_cfg.get("train_on", True):
        flickr_cache = flickr_cfg["embedding_cache"]
        flickr_text_cache = flickr_cache + "_text"
        if Path(flickr_cache).is_dir() and len(list(Path(flickr_cache).glob("*.npy"))) > 0:
            flickr_train = FlickrDataset(
                flickr_cfg["root"], "train",
                flickr_cache,
                annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
                text_cache_dir=flickr_text_cache,
            )
            datasets_train.append(flickr_train)
            print(f"  Flickr train: {len(flickr_train):,} pairs")

    train_dataset = ConcatDataset(datasets_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Validation ─────────────────────────────────────────────
    if flickr_cfg.get("root"):
        flickr_cache = flickr_cfg["embedding_cache"]
        flickr_text_cache = flickr_cache + "_text"
        if Path(flickr_cache).is_dir() and len(list(Path(flickr_cache).glob("*.npy"))) > 0:
            val_dataset = FlickrDataset(
                flickr_cfg["root"], "test",
                flickr_cache,
                annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
                text_cache_dir=flickr_text_cache,
            )
            val_label = "Flickr30k test (1K)"
        else:
            val_dataset = CocoDataset(coco_root, "val2017", coco_cache, ann_dir,
                                     text_cache_dir=coco_text_cache)
            val_label = "COCO val2017 (fallback)"
    else:
        val_dataset = CocoDataset(coco_root, "val2017", coco_cache, ann_dir,
                                 text_cache_dir=coco_text_cache)
        val_label = "COCO val2017"
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  Train: {len(train_dataset):,} pairs  ({'+'.join(str(len(d)) for d in datasets_train)})")
    print(f"  Val:   {len(val_dataset):,} captions  ({val_label})")

    # ── Queue & Loss ───────────────────────────────────────────
    queue_max_size = cfg.get("queue", {}).get("max_size", 0)
    t2i_weight = cfg["training"].get("t2i_weight", 0.5)
    uniform_weight = cfg["training"].get("uniformity_weight", 0.0)
    queue = None
    if queue_max_size > 0:
        print(f"  ContrastiveQueue: max_size={queue_max_size:,}")
        print(f"  mask_stale_texts: False (both sides frozen → no staleness)")
        queue = ContrastiveQueue(max_size=queue_max_size, device=device)
        loss_fn = QueueInfoNCE(
            temperature=cfg["training"]["temperature"],
            queue=queue,
            t2i_weight=t2i_weight,
            uniform_weight=uniform_weight,
            mask_stale_texts=False,
        )
    else:
        loss_fn = SymmetricInfoNCE(
            temperature=cfg["training"]["temperature"],
            t2i_weight=t2i_weight,
        )

    # ── Optimizer (temporal ONLY) ──────────────────────────────
    temporal_lr = t_cfg.get("temporal_lr", cfg["training"]["lr"])
    optimizer = torch.optim.AdamW(
        temporal.parameters(),
        lr=temporal_lr,
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── Trainer ────────────────────────────────────────────────
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
    trainer = TemporalTrainer(
        bert_model, train_loader, val_loader, optimizer, loss_fn, cfg,
        queue=queue, temporal=temporal,
    )

    # ── Resume ─────────────────────────────────────────────────
    start_epoch = 1
    best_t2i_r1 = -1.0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resumed_epoch, resumed_val_loss, resumed_t2i, resumed_best = trainer.load_checkpoint(args.resume)
        start_epoch = resumed_epoch + 1
        if resumed_t2i is not None:
            best_t2i_r1 = resumed_best if resumed_best is not None else resumed_t2i
            print(f"  Resumed at epoch {resumed_epoch} (best t2i_R@1={best_t2i_r1:.2f})")
        else:
            print(f"  Resumed at epoch {resumed_epoch} (val_loss={resumed_val_loss:.4f})")

    # ── Training Loop ──────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    eval_interval = cfg["training"].get("eval_interval", 6000)
    step_level_val = cfg.get("training", {}).get("step_level_val", True)
    recall_keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]

    print(f"\nTraining: {epochs} epochs, temporal LR: {temporal_lr:.2e}")
    print(f"  TemporalTransformer params: {n_temporal:,}")
    if step_level_val:
        print(f"  Step-level validation every {eval_interval} steps")
    print(f"  Save best + every 5 epochs")
    print("-" * 60)

    global_step = trainer.global_step
    for epoch in range(start_epoch, epochs + 1):
        epoch_losses = []
        epoch_queue_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)
        for batch in pbar:
            result = trainer.train_batch(*batch)
            global_step += 1
            loss_val = result["loss"]
            inbatch_val = result.get("loss_inbatch")
            epoch_queue_losses.append(loss_val)
            monitor_loss = inbatch_val if inbatch_val is not None else loss_val
            epoch_losses.append(monitor_loss)
            postfix = {"loss": f"{monitor_loss:.4f}"}
            if inbatch_val is not None:
                postfix["qLoss"] = f"{loss_val:.4f}"
            if trainer.queue is not None:
                postfix["Q"] = len(trainer.queue)
            uniform_val = result.get("uniformity")
            if uniform_val is not None:
                postfix["U"] = f"{uniform_val:.4f}"
            i2t_q = result.get("i2t_q")
            t2i_q = result.get("t2i_q")
            if i2t_q is not None:
                postfix["i2t_q"] = f"{i2t_q:.4f}"
                postfix["t2i_q"] = f"{t2i_q:.4f}"
            pbar.set_postfix(**postfix)

            # Step-level validation
            if step_level_val and global_step % eval_interval == 0:
                eval_results = trainer.evaluate()
                val_loss = eval_results["val_loss"]
                lr = trainer.optimizer.param_groups[0]["lr"]

                is_best = eval_results.get("t2i_R@1", -1.0) > best_t2i_r1
                if is_best:
                    best_t2i_r1 = eval_results["t2i_R@1"]

                log_parts = [
                    f"Step {global_step}",
                    f"Val loss: {val_loss:.4f}",
                    f"Best t2i_R@1: {best_t2i_r1:.2f}" if best_t2i_r1 > 0 else "Best: —",
                ]
                for key in recall_keys:
                    if key in eval_results:
                        log_parts.append(f"{key}: {eval_results[key]:.2f}")
                log_parts.append(f"LR: {lr:.2e}")

                if is_best:
                    ckpt_path = trainer.save_checkpoint(
                        epoch, val_loss,
                        best_t2i_r1=best_t2i_r1,
                        **{k: v for k, v in eval_results.items() if k != "val_loss"},
                    )
                    log_parts.append(f"★ New best!  → {Path(ckpt_path).name}")

                print("  " + " | ".join(log_parts))

        # ── Epoch-end validation & save ─────────────────────
        eval_results = trainer.evaluate()
        val_loss = eval_results["val_loss"]
        avg_queue_loss = sum(epoch_queue_losses) / len(epoch_queue_losses)
        lr = trainer.optimizer.param_groups[0]["lr"]

        is_best = eval_results.get("t2i_R@1", -1.0) > best_t2i_r1
        if is_best:
            best_t2i_r1 = eval_results["t2i_R@1"]

        log_parts = [
            f"Epoch {epoch:02d}/{epochs} done",
            f"Val:   {val_loss:.4f}",
            f"Best t2i_R@1: {best_t2i_r1:.2f}" if best_t2i_r1 > 0 else "Best: —",
        ]
        if trainer.queue is not None:
            avg_monitor_loss = sum(epoch_losses) / len(epoch_losses)
            log_parts.insert(1, f"Train: {avg_monitor_loss:.4f}")
            log_parts.insert(2, f"qLoss: {avg_queue_loss:.4f}")
        for key in recall_keys:
            if key in eval_results:
                log_parts.append(f"{key}: {eval_results[key]:.2f}")
        log_parts.append(f"LR: {lr:.2e}")

        if is_best or epoch % 5 == 0:
            ckpt_path = trainer.save_checkpoint(
                epoch, val_loss,
                best_t2i_r1=best_t2i_r1,
                **{k: v for k, v in eval_results.items() if k != "val_loss"},
            )
            suffix = f"  → {Path(ckpt_path).name}"
            if is_best:
                suffix += "  ★ New best!"
            log_parts[-1] = log_parts[-1] + suffix

        print("  " + " | ".join(log_parts))
        print("-" * 60)

    print(f"Training complete. Best t2i_R@1: {best_t2i_r1:.2f}")


if __name__ == "__main__":
    main()
