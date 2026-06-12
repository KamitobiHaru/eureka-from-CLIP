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
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder
from src.data import (
    CocoDataset, FlickrDataset,
    make_collate_fn, make_clip_collate_fn,
)
from src.training import (
    SymmetricInfoNCE, QueueInfoNCE, Trainer, ContrastiveQueue,
)
from src.training.evaluation import compute_recall_metrics
from scripts.evaluate_multilingual import _compute_median_rank


def main():
    parser = argparse.ArgumentParser(description="Train BERT alignment with CLIP.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint .pt file (e.g. checkpoints/bert_best.pt)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: from config)")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # ── Seed ─────────────────────────────────────────────────
    seed = cfg.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seed: {seed}")
        train_generator = torch.Generator().manual_seed(seed)
    else:
        train_generator = None

    # ── Model + Tokenizer ────────────────────────────────────
    te_cfg = cfg.get("text_encoder", {})
    encoder_type = te_cfg.get("type", "bert")

    if encoder_type == "clip":
        # ── CLIP text encoder + optional LoRA ────────────
        from src.models.clip_text_encoder import CLIPTextEncoder
        import open_clip

        clip_lora = te_cfg.get("clip_lora", {})
        clip_model_type = cfg.get("clip", {}).get("model", "openai")
        print(f"Loading CLIP text encoder ({clip_model_type})...")
        model = CLIPTextEncoder(
            device=device, lora_cfg=clip_lora,
            model_type=clip_model_type,
        ).to(device)

        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        collate_fn = make_clip_collate_fn(tokenizer)
    else:
        # ── BERT text encoder ────────────────────────────
        bert_path = (
            te_cfg.get("bert_model_path")
            or cfg["model"].get("bert_model_path")
        )
        if not bert_path or not os.path.isdir(bert_path):
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

        tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
        collate_fn = make_collate_fn(tokenizer)

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        generator=train_generator,
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

    # ── Chinese Flickr30k validation (cross-lingual) ──────────
    zh_loader = None
    zh_flickr_cfg = cfg.get("flickr_zh", {})
    if zh_flickr_cfg.get("zh_file"):
        zh_rel = zh_flickr_cfg["zh_file"]
        # Resolve relative to config location (train_bert.py runs from project root)
        zh_abs = Path(zh_rel).resolve()
        if zh_abs.exists():
            from src.data.flickr_zh_dataset import FlickrZhDataset
            zh_dataset = FlickrZhDataset(
                flickr_cfg["root"], "test",
                flickr_cfg["embedding_cache"],
                zh_file=str(zh_abs),
                annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
            )
            zh_loader = torch.utils.data.DataLoader(
                zh_dataset,
                batch_size=cfg["training"]["batch_size"],
                shuffle=False,
                num_workers=cfg["training"]["num_workers"],
                collate_fn=collate_fn,
                pin_memory=True,
            )
            print(f"  Zh-Val: {len(zh_dataset):,} Chinese captions ({zh_abs.name})")

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
        queue = ContrastiveQueue(max_size=queue_max_size, device=device)
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

    # ── Chinese Flickr30k evaluation helper ────────────────
    @torch.no_grad()
    def _evaluate_zh(model, loader, device, temperature):
        """Evaluate on Chinese Flickr30k, return recall dict."""
        model.eval()
        loss_fn = SymmetricInfoNCE(temperature=temperature)
        all_img = []
        all_txt = []
        all_ids = []
        for batch in tqdm(loader, desc="Zh-Val", leave=False):
            img, input_ids, attn_mask, *rest = batch
            img_ids = rest[0] if rest else None
            img = img.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            txt = model(input_ids, attn_mask)
            all_img.append(img.cpu())
            all_txt.append(txt.cpu())
            if img_ids:
                all_ids.extend(img_ids)
        if not all_ids:
            return {}
        img_embs = torch.cat(all_img)
        txt_embs = torch.cat(all_txt)
        recall = compute_recall_metrics(img_embs, txt_embs, all_ids, ks=(1, 5, 10))
        recall["i2t_medR"] = _compute_median_rank(img_embs, txt_embs, all_ids, "i2t")
        recall["t2i_medR"] = _compute_median_rank(img_embs, txt_embs, all_ids, "t2i")
        return recall

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

    # CSV log for validation metrics
    val_csv_path = os.path.join(cfg["training"]["checkpoint_dir"], "val_log.csv")

    def _write_val_csv(row_dict):
        fieldnames = [
            "type", "step", "epoch", "val_loss",
            "i2t_R@1", "i2t_R@5", "i2t_R@10",
            "t2i_R@1", "t2i_R@5", "t2i_R@10",
            "i2t_medR", "t2i_medR",
            "ZH_i2t_R@1", "ZH_i2t_R@5", "ZH_i2t_R@10",
            "ZH_i2t_medR", "ZH_t2i_medR",
            "train_monitor_loss", "queue_loss", "uniformity",
            "i2t_loss", "t2i_loss",
            "lr", "best_t2i_R@1",
        ]
        write_header = not os.path.exists(val_csv_path)
        with open(val_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: v for k, v in row_dict.items() if k in fieldnames})

    global_step = trainer.global_step
    for epoch in range(start_epoch, epochs + 1):
        epoch_losses = []
        epoch_queue_losses = []
        epoch_uniform_vals = []
        epoch_i2t_losses = []
        epoch_t2i_losses = []
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
                epoch_i2t_losses.append(i2t_q)
                epoch_t2i_losses.append(t2i_q)
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

                csv_row = {
                    "type": "step", "step": global_step,
                    "val_loss": round(val_loss, 4),
                    "lr": lr, "best_t2i_R@1": best_t2i_r1,
                }
                for key in recall_keys:
                    if key in eval_results:
                        csv_row[key] = round(float(eval_results[key]), 2)
                for key in ("i2t_medR", "t2i_medR"):
                    if key in eval_results:
                        csv_row[key] = round(float(eval_results[key]), 1)
                _write_val_csv(csv_row)

        # ── Epoch-end validation & save ─────────────────────
        eval_results = trainer.evaluate()
        val_loss = eval_results["val_loss"]
        avg_monitor_loss = sum(epoch_losses) / len(epoch_losses)
        avg_queue_loss = sum(epoch_queue_losses) / len(epoch_queue_losses)
        lr = trainer.optimizer.param_groups[0]["lr"]

        # ── Chinese Flickr30k evaluation ──────────────────────
        zh_results = {}
        if zh_loader is not None:
            zh_results = _evaluate_zh(trainer.model, zh_loader, trainer.device,
                                      cfg["training"]["temperature"])

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
        # English Flickr recall
        for key in recall_keys:
            if key in eval_results:
                log_parts.append(f"{key}: {eval_results[key]:.2f}")
        # Chinese Flickr recall (if available)
        if zh_results:
            for key in recall_keys:
                v = zh_results.get(key)
                if v is not None:
                    log_parts.append(f"ZH_{key}: {v:.2f}")
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

        csv_row = {
            "type": "epoch", "epoch": epoch,
            "val_loss": round(val_loss, 4),
            "lr": lr, "best_t2i_R@1": best_t2i_r1,
            "train_monitor_loss": round(avg_monitor_loss, 4),
        }
        if trainer.queue is not None:
            csv_row["queue_loss"] = round(avg_queue_loss, 4)
        if epoch_uniform_vals:
            csv_row["uniformity"] = round(avg_uniform, 4)
        if epoch_i2t_losses:
            csv_row["i2t_loss"] = round(float(sum(epoch_i2t_losses) / len(epoch_i2t_losses)), 4)
            csv_row["t2i_loss"] = round(float(sum(epoch_t2i_losses) / len(epoch_t2i_losses)), 4)
        for key in recall_keys:
            if key in eval_results:
                csv_row[key] = round(float(eval_results[key]), 2)
        for key in ("i2t_medR", "t2i_medR"):
            if key in eval_results:
                csv_row[key] = round(float(eval_results[key]), 1)
        if zh_results:
            for key in recall_keys:
                v = zh_results.get(key)
                if v is not None:
                    csv_row[f"ZH_{key}"] = round(float(v), 2)
            for key in ("i2t_medR", "t2i_medR"):
                v = zh_results.get(key)
                if v is not None:
                    csv_row[f"ZH_{key}"] = round(float(v), 1)
        _write_val_csv(csv_row)

    print(f"Training complete. Best t2i_R@1: {best_t2i_r1:.2f}")


if __name__ == "__main__":
    main()
