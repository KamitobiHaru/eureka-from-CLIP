"""
Train BERT-base + projection head to align with CLIP vision embeddings.

Usage:
    python scripts/train_bert.py [--config config/default.yaml]

Prerequisites:
    1. Run scripts/precompute_embeddings.py first to generate .npy files
    2. Set coco_root in config/default.yaml (or pass --config)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder
from src.data import get_dataloader
from src.training import SymmetricInfoNCE, Trainer


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
    ).to(device)

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # ── Data ─────────────────────────────────────────────────
    cache_dir = cfg["data"]["embedding_cache"]
    coco_root = cfg["data"]["coco_root"]

    if not Path(cache_dir).is_dir() or len(list(Path(cache_dir).glob("*.npy"))) == 0:
        print(f"\nNo precomputed embeddings found in {cache_dir}.")
        print("Run scripts/precompute_embeddings.py first.")
        sys.exit(1)

    ann_dir = cfg["data"].get("annotations_dir", "annotations")

    print("Loading data...")
    train_loader = get_dataloader(
        coco_root, "train2017", cache_dir, tokenizer,
        annotations_dir=ann_dir,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = get_dataloader(
        coco_root, "val2017", cache_dir, tokenizer,
        annotations_dir=ann_dir,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    print(f"  Train: {len(train_loader.dataset)} pairs")
    print(f"  Val:   {len(val_loader.dataset)} pairs")

    # ── Optimizer & Loss ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    loss_fn = SymmetricInfoNCE(temperature=cfg["training"]["temperature"])

    # ── Trainer ──────────────────────────────────────────────
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, cfg)

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = math.inf
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resumed_epoch, resumed_val_loss = trainer.load_checkpoint(args.resume)
        start_epoch = resumed_epoch + 1
        best_val_loss = resumed_val_loss
        print(f"  Resumed at epoch {resumed_epoch} (val_loss={resumed_val_loss:.4f})")

    # ── Training Loop (batch-level) ──────────────────────────
    epochs = cfg["training"]["epochs"]
    eval_interval = cfg["training"]["eval_interval"]
    total_steps = len(train_loader) * epochs
    warmup = cfg["training"]["warmup_steps"]

    print(f"\nTraining: {epochs} epochs, {eval_interval} steps/val, "
          f"{warmup} warmup steps, cosine decay")
    print("-" * 60)

    global_step = trainer.global_step
    for epoch in range(start_epoch, epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)
        for batch in pbar:
            loss = trainer.train_batch(*batch)
            global_step += 1
            pbar.set_postfix(loss=f"{loss:.4f}")

            # ── Validation ───────────────────────────────────
            if global_step % eval_interval == 0:
                val_loss = trainer.evaluate()
                lr = trainer.optimizer.param_groups[0]["lr"]
                print(f"  Step {global_step:>6}/{total_steps}  |  "
                      f"Train loss: {loss:.4f}  |  "
                      f"Val loss:   {val_loss:.4f}  |  "
                      f"LR: {lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = trainer.save_checkpoint(epoch, val_loss)
                    print(f"  ★ New best! Checkpoint saved: {ckpt_path}")

        # ── End-of-epoch validation ──────────────────────────
        val_loss = trainer.evaluate()
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:02d}/{epochs} done  |  "
              f"Val loss: {val_loss:.4f}  |  "
              f"Best: {best_val_loss:.4f}  |  LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = trainer.save_checkpoint(epoch, val_loss)
            print(f"  ★ New best! Checkpoint saved: {ckpt_path}")

        print("-" * 60)

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
