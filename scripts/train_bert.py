"""
Train BERT-base + projection head to align with CLIP vision embeddings.

Usage:
    python scripts/train_bert.py [--config config/default.yaml]

Prerequisites:
    1. Run scripts/precompute_embeddings.py first to generate .npy files
    2. Set coco_root in config/default.yaml (or pass --config)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder
from src.data import get_dataloader
from src.training import SymmetricInfoNCE, Trainer


def main():
    parser = argparse.ArgumentParser(description="Train BERT alignment with CLIP.")
    parser.add_argument("--config", default="config/default.yaml")
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

    print("Loading BERT model...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

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
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, cfg)

    # ── Training Loop ────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch()
        val_loss = trainer.evaluate()

        print(f"Epoch {epoch:02d}/{epochs}  |  "
              f"Train loss: {train_loss:.4f}  |  "
              f"Val loss:   {val_loss:.4f}")

        ckpt_path = trainer.save_checkpoint(epoch, val_loss)
        print(f"  Checkpoint saved: {ckpt_path}")
        print("-" * 60)

    print("Training complete.")


if __name__ == "__main__":
    main()
