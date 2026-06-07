"""
Train BERT-base + projection head to align with CLIP vision embeddings.

Supports COCO only, or combined COCO + Flickr30k training.

Usage:
    python scripts/train_bert.py [--config config/default.yaml]

Prerequisites:
    1. Run scripts/precompute_embeddings.py for COCO (and optionally
       scripts/precompute_flickr_embeddings.py for Flickr30k)
    2. Set paths in config/default.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder
from src.data import CocoDataset, FlickrDataset, make_collate_fn
from src.training import (
    SymmetricInfoNCE, QueueInfoNCE, Trainer, ContrastiveQueue,
)


def main():
    parser = argparse.ArgumentParser(description="Train BERT alignment with CLIP.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint .pt file (e.g. checkpoints/bert_best.pt)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # ── Model ────────────────────────────────────────────────
    bert_path = cfg["model"]["bert_model_path"]
    if not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        print("Run the download script first: bash scripts/download_bert.sh")
        sys.exit(1)

    lora_cfg = cfg.get("lora", {})
    print("Loading BERT model...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
        initial_temperature=cfg["training"]["temperature"],
    ).to(device)

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # ── Data ─────────────────────────────────────────────────
    coco_cache = cfg["data"]["embedding_cache"]
    coco_root = cfg["data"]["coco_root"]

    if not Path(coco_cache).is_dir() or len(list(Path(coco_cache).glob("*.npy"))) == 0:
        print(f"\nNo precomputed embeddings found in {coco_cache}.")
        print("Run scripts/precompute_embeddings.py first.")
        sys.exit(1)

    ann_dir = cfg["data"].get("annotations_dir", "annotations")

    # Build training dataset(s) ────────────────────────────────
    coco_train = CocoDataset(coco_root, "train2017", coco_cache, ann_dir)
    datasets_train = [coco_train]

    flickr_cfg = cfg.get("flickr", {})
    if flickr_cfg.get("root") and flickr_cfg.get("train_on", True):
        flickr_cache = flickr_cfg["embedding_cache"]
        if not Path(flickr_cache).is_dir() or len(list(Path(flickr_cache).glob("*.npy"))) == 0:
            print(f"\nPrecomputed Flickr embeddings not found in {flickr_cache}.")
            print("Run scripts/precompute_flickr_embeddings.py first, or remove 'flickr' from config.")
            sys.exit(1)
        flickr_train = FlickrDataset(
            flickr_cfg["root"], "train",
            flickr_cache,
            annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
        )
        datasets_train.append(flickr_train)
        print(f"  Flickr train: {len(flickr_train):,} pairs")

    train_dataset = ConcatDataset(datasets_train)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Validation: Flickr30k test (CLIP-style) ─────────────
    # CLIP evaluates on each dataset independently.  Flickr30k test
    # (1K images / 5K captions) is the primary val set when available.
    if flickr_cfg.get("root"):
        flickr_cache = flickr_cfg["embedding_cache"]
        if Path(flickr_cache).is_dir() and len(list(Path(flickr_cache).glob("*.npy"))) > 0:
            val_dataset = FlickrDataset(
                flickr_cfg["root"], "test",
                flickr_cache,
                annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
            )
            val_label = "Flickr30k test (1K)"
        else:
            val_dataset = CocoDataset(coco_root, "val2017", coco_cache, ann_dir)
            val_label = "COCO val2017 (fallback)"
    else:
        val_dataset = CocoDataset(coco_root, "val2017", coco_cache, ann_dir)
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

    # ── Queue & Loss ─────────────────────────────────────
    queue_max_size = cfg.get("queue", {}).get("max_size", 0)
    mask_stale_texts = cfg.get("queue", {}).get("mask_stale_texts", False)
    t2i_weight = cfg["training"].get("t2i_weight", 0.5)
    uniform_weight = cfg["training"].get("uniformity_weight", 0.0)
    queue = None
    if queue_max_size > 0:
        print(f"  ContrastiveQueue: max_size={queue_max_size:,}")
        if mask_stale_texts:
            print(f"  mask_stale_texts: True (i2t uses in-batch negatives only)")
        queue = ContrastiveQueue(max_size=queue_max_size)
        loss_fn = QueueInfoNCE(
            temperature=cfg["training"]["temperature"],
            queue=queue,
            t2i_weight=t2i_weight,
            uniform_weight=uniform_weight,
            mask_stale_texts=mask_stale_texts,
        )
    else:
        loss_fn = SymmetricInfoNCE(
            temperature=cfg["training"]["temperature"],
            t2i_weight=t2i_weight,
        )

    # ── Optimizer ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── Trainer ──────────────────────────────────────────────
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, cfg, queue=queue)

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 1
    best_t2i_r1 = -1.0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resumed_epoch, resumed_val_loss, resumed_t2i = trainer.load_checkpoint(args.resume)
        start_epoch = resumed_epoch + 1
        if resumed_t2i is not None:
            best_t2i_r1 = resumed_t2i
            print(f"  Resumed at epoch {resumed_epoch} (best t2i_R@1={resumed_t2i:.2f})")
        else:
            print(f"  Resumed at epoch {resumed_epoch} (val_loss={resumed_val_loss:.4f})")

    # ── Training Loop (step-level validation) ──────────────
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"]["warmup_steps"]
    eval_interval = cfg["training"].get("eval_interval", 5000)
    step_level_val = cfg.get("training", {}).get("step_level_val", True)
    recall_keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]

    print(f"\nTraining: {epochs} epochs, {warmup} warmup steps, cosine decay")
    if step_level_val:
        print(f"  Step-level validation every {eval_interval} steps")
    else:
        print(f"  Epoch-level validation only")
    print(f"  Save best + every 5 epochs")
    print("-" * 60)

    global_step = trainer.global_step
    for epoch in range(start_epoch, epochs + 1):
        epoch_losses = []
        epoch_queue_losses = []
        epoch_uniform_vals = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)
        for batch in pbar:
            result = trainer.train_batch(*batch)
            global_step += 1
            loss_val = result["loss"]
            inbatch_val = result.get("loss_inbatch")
            epoch_queue_losses.append(loss_val)
            # Use in-batch loss for monitoring (comparable to val loss)
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
                epoch_uniform_vals.append(uniform_val)
            # Raw i2t/t2i components from QueueInfoNCE (before weighting)
            # These are the two values being combined in the weighted loss
            i2t_q = result.get("i2t_q")
            t2i_q = result.get("t2i_q")
            if i2t_q is not None:
                postfix["i2t_q"] = f"{i2t_q:.4f}"
                postfix["t2i_q"] = f"{t2i_q:.4f}"
            pbar.set_postfix(**postfix)

            # ── Step-level validation ─────────────────────
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
                    trainer.save_checkpoint(
                        epoch, val_loss,
                        **{k: v for k, v in eval_results.items() if k != "val_loss"},
                    )
                    log_parts.append("★ New best!")

                print("  " + " | ".join(log_parts))

        # ── Epoch-end validation & save ─────────────────────
        eval_results = trainer.evaluate()
        val_loss = eval_results["val_loss"]
        avg_monitor_loss = sum(epoch_losses) / len(epoch_losses)
        avg_queue_loss = sum(epoch_queue_losses) / len(epoch_queue_losses)
        lr = trainer.optimizer.param_groups[0]["lr"]

        is_best = eval_results.get("t2i_R@1", -1.0) > best_t2i_r1
        if is_best:
            best_t2i_r1 = eval_results["t2i_R@1"]

        log_parts = [
            f"Epoch {epoch:02d}/{epochs} done",
            f"Train: {avg_monitor_loss:.4f} (iLoss avg)",
            f"Val:   {val_loss:.4f}",
            f"Best t2i_R@1: {best_t2i_r1:.2f}" if best_t2i_r1 > 0 else "Best: —",
        ]
        if trainer.queue is not None:
            log_parts.insert(2, f"qLoss: {avg_queue_loss:.4f}")
        if epoch_uniform_vals:
            avg_uniform = sum(epoch_uniform_vals) / len(epoch_uniform_vals)
            log_parts.insert(3, f"U: {avg_uniform:.4f}")
        for key in recall_keys:
            if key in eval_results:
                log_parts.append(f"{key}: {eval_results[key]:.2f}")
        log_parts.append(f"LR: {lr:.2e}")

        if is_best or epoch % 5 == 0:
            ckpt_path = trainer.save_checkpoint(
                epoch, val_loss,
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
