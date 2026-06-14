"""
Joint training: LoRA BERT + TemporalTransformer on MSR-VTT with queue-based
contrastive learning.  Validate on MSVD every epoch.

Usage:
    # Phase 2 (after Phase 1 pretrain_temporal):
    python scripts/train_video_joint.py --config config/joint_train.yaml
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import BertEncoder, TemporalTransformer
from src.data import VideoDataset, video_collate_fn
from src.training import QueueInfoNCE, ContrastiveQueue, SymmetricInfoNCE
from src.training.evaluation import compute_recall_metrics

FLICKR_ROOT = Path("./dataset_annotation")


def _build_bert(cfg, device) -> BertEncoder:
    """Build BertEncoder with LoRA, load pretrained weights, return trainable."""
    bert_path = cfg["model"]["bert_model_path"]
    if not bert_path or not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        sys.exit(1)

    lora_cfg = cfg.get("lora", {})
    print("Building BERT encoder (with LoRA)...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
        initial_temperature=cfg["training"]["temperature"],
    ).to(device)

    # Load pretrained weights
    ckpt_path = cfg["checkpoint"]["bert_pretrain"]
    if not Path(ckpt_path).exists():
        print(f"BERT checkpoint not found: {ckpt_path}")
        sys.exit(1)
    print(f"Loading BERT from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    # Log trainable vs frozen
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  BERT: {total:,} total, {trainable:,} trainable (LoRA + proj)")
    return model


def _build_temporal(cfg, device) -> TemporalTransformer:
    """Build TemporalTransformer with fresh identity init (no Phase 1)."""
    t_cfg = cfg["temporal"]
    temporal = TemporalTransformer(
        d_model=t_cfg["d_model"],
        nhead=t_cfg["nhead"],
        num_layers=t_cfg["num_layers"],
        dim_feedforward=t_cfg["dim_feedforward"],
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg["max_frames"],
    ).to(device)

    n_temporal = sum(p.numel() for p in temporal.parameters())
    print(f"  TemporalTransformer: {n_temporal:,} trainable params (fresh identity init)")
    return temporal


def _build_dataloaders(cfg):
    """Build MSR-VTT train loader and MSVD evaluation loader."""
    msrvtt_cfg = cfg["data"]["msrvtt"]
    msrvtt_root = msrvtt_cfg["root"]
    train_ann = os.path.join(msrvtt_root, msrvtt_cfg["train_annotation"])
    frame_cache = msrvtt_cfg["frame_cache"]

    if not Path(frame_cache).is_dir():
        print(f"Frame cache not found: {frame_cache}")
        print("Run scripts/precompute_video_keyframes.py first.")
        sys.exit(1)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["bert_model_path"])

    # ── MSR-VTT train ──
    train_dataset = VideoDataset(train_ann, frame_cache)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=video_collate_fn(tokenizer),
        pin_memory=True,
    )
    print(f"  MSR-VTT train: {len(train_dataset):,} caption-video pairs")

    # ── MSVD evaluation ──
    msvd_cfg = cfg["data"]["msvd"]
    msvd_ann = msvd_cfg["annotation"]
    msvd_cache = msvd_cfg["frame_cache"]
    msvd_dataset = VideoDataset(msvd_ann, msvd_cache)
    msvd_loader = torch.utils.data.DataLoader(
        msvd_dataset,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=video_collate_fn(tokenizer),
        pin_memory=True,
    )
    print(f"  MSVD eval:   {len(msvd_dataset):,} caption-video pairs")

    return train_loader, msvd_loader, tokenizer


def _build_optimizer(bert_model, temporal, cfg):
    """Two-parameter-group optimizer: BERT LoRA + Temporal."""
    t_cfg = cfg["training"]
    lr = t_cfg["lr"]
    temporal_lr = cfg["temporal"]["lr"]
    wd = t_cfg["weight_decay"]

    # BERT trainable params (LoRA + projection head + logit_scale)
    bert_params = [p for p in bert_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr, "weight_decay": wd},
        {"params": temporal.parameters(), "lr": temporal_lr, "weight_decay": wd},
    ])
    print(f"  BERT LR: {lr:.2e}, Temporal LR: {temporal_lr:.2e}")
    return optimizer


def _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup → cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_msvd(bert_model, temporal, msvd_loader, cfg, device) -> dict:
    """Evaluate on MSVD: compute video retrieval recall."""
    bert_model.eval()
    temporal.eval()

    all_video_embs = []
    all_text_embs = []
    all_video_ids = []

    pbar = tqdm(msvd_loader, desc="MSVD eval", leave=False)
    for batch in pbar:
        frame_embs, padding_mask, input_ids, attention_mask, video_ids = batch
        frame_embs = frame_embs.to(device)
        padding_mask = padding_mask.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Encode video
        video_emb = temporal(frame_embs, padding_mask)
        # Encode text
        text_emb = bert_model(input_ids, attention_mask)

        all_video_embs.append(video_emb.cpu())
        all_text_embs.append(text_emb.cpu())
        all_video_ids.extend(video_ids)

    video_embs = torch.cat(all_video_embs, dim=0)
    text_embs = torch.cat(all_text_embs, dim=0)

    recall = compute_recall_metrics(video_embs, text_embs, all_video_ids)
    return recall


@torch.no_grad()
def evaluate_flickr30k(bert_model, tokenizer, device) -> dict:
    """Evaluate on Flickr30k test (1K): BERT text vs precomputed CLIP image embeddings."""
    bert_model.eval()

    ann_file = FLICKR_ROOT / "flickr_annotations_30k.csv"
    img_cache = Path("./data/flickr30k/clip_embeddings")

    with open(ann_file) as f:
        rows = list(csv.DictReader(f))

    pairs = []
    for r in rows:
        if r["split"].strip() != "test":
            continue
        stem = Path(r["filename"].strip()).stem
        for cap in json.loads(r["raw"]):
            pairs.append((stem, cap))

    unique_stems = list(dict.fromkeys(stem for stem, _ in pairs))
    stem_to_img = {}
    for stem in unique_stems:
        npy = img_cache / f"{stem}.npy"
        if npy.exists():
            stem_to_img[stem] = np.load(npy)

    all_image_embs = []
    all_text_embs = []
    all_image_ids = []
    cap_texts = []

    for stem, cap in pairs:
        img_emb = stem_to_img.get(stem)
        if img_emb is None:
            continue
        all_image_embs.append(img_emb)
        all_image_ids.append(f"flickr_{stem}")
        cap_texts.append(cap)

        if len(cap_texts) >= 128:
            tokens = tokenizer(cap_texts, padding=True, truncation=True,
                               max_length=77, return_tensors="pt")
            text_emb = bert_model(tokens["input_ids"].to(device),
                                  tokens["attention_mask"].to(device))
            all_text_embs.append(text_emb.cpu())
            cap_texts = []

    if cap_texts:
        tokens = tokenizer(cap_texts, padding=True, truncation=True,
                           max_length=77, return_tensors="pt")
        text_emb = bert_model(tokens["input_ids"].to(device),
                              tokens["attention_mask"].to(device))
        all_text_embs.append(text_emb.cpu())

    return compute_recall_metrics(
        torch.from_numpy(np.stack(all_image_embs)),
        torch.cat(all_text_embs, dim=0),
        all_image_ids,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Joint train LoRA BERT + Temporal on MSR-VTT."
    )
    parser.add_argument("--config", default="config/joint_train.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    t_cfg = cfg["training"]

    # ── Models ──
    bert_model = _build_bert(cfg, device)
    temporal = _build_temporal(cfg, device)

    # ── Data ──
    train_loader, msvd_loader, tokenizer = _build_dataloaders(cfg)

    # ── Queue & Loss ──
    q_cfg = cfg.get("queue", {})
    queue_max_size = q_cfg.get("max_size", 0)
    queue = None
    loss_fn = None
    if queue_max_size > 0:
        print(f"  ContrastiveQueue: max_size={queue_max_size:,}")
        queue = ContrastiveQueue(max_size=queue_max_size, device=device)
        loss_fn = QueueInfoNCE(
            temperature=t_cfg["temperature"],
            queue=queue,
            t2i_weight=t_cfg.get("t2i_weight", 0.5),
            uniform_weight=t_cfg.get("uniformity_weight", 0.0),
            mask_stale_texts=q_cfg.get("mask_stale_texts", True),
        )
    else:
        loss_fn = SymmetricInfoNCE(
            temperature=t_cfg["temperature"],
            t2i_weight=t_cfg.get("t2i_weight", 0.5),
        )

    # In-batch loss for monitoring
    eval_loss_fn = SymmetricInfoNCE(temperature=t_cfg["temperature"])

    # ── Optimizer & Scheduler ──
    optimizer = _build_optimizer(bert_model, temporal, cfg)
    total_steps = len(train_loader) * t_cfg["epochs"]
    scheduler = _get_cosine_schedule_with_warmup(
        optimizer, t_cfg["warmup_steps"], total_steps
    )
    scaler = torch.amp.GradScaler("cuda") if t_cfg["amp"] and torch.cuda.is_available() else None

    ckpt_dir = t_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── CSV log (same format as train_bert.py) ──
    val_csv_path = os.path.join(ckpt_dir, "val_log.csv")
    recall_keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]
    flickr_keys = [f"flickr_{k}" for k in recall_keys] + ["flickr_i2t_medR", "flickr_t2i_medR"]

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
        ] + flickr_keys
        write_header = not os.path.exists(val_csv_path)
        with open(val_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: v for k, v in row_dict.items() if k in fieldnames})

    # ── Resume ──
    start_epoch = 1
    global_step = 0
    best_msvd_t2i_r1 = -1.0
    best_flickr_t2i_r1 = -1.0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        bert_model.load_state_dict(ckpt["bert_state_dict"], strict=False)
        temporal.load_state_dict(ckpt["temporal_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if queue is not None and "queue_state" in ckpt:
            queue.load_state_dict(ckpt["queue_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_msvd_t2i_r1 = ckpt.get("best_msvd_t2i_r1", -1.0)
        best_flickr_t2i_r1 = ckpt.get("best_flickr_t2i_r1", -1.0)
        print(f"  Resumed at epoch {ckpt.get('epoch', 0)}, step {global_step}")

    # ── Training Loop ──
    epochs = t_cfg["epochs"]
    recall_keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]

    print(f"\nTraining: {epochs} epochs")
    print(f"  MSVD eval every epoch")
    print("-" * 60)

    for epoch in range(start_epoch, epochs + 1):
        bert_model.train()
        temporal.train()
        epoch_losses = []
        epoch_queue_losses = []
        epoch_i2t_losses = []
        epoch_t2i_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for batch in pbar:
            frame_embs, padding_mask, input_ids, attention_mask, video_ids = batch
            frame_embs = frame_embs.to(device)
            padding_mask = padding_mask.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # ── Forward ──
            if scaler:
                with torch.amp.autocast("cuda"):
                    video_emb = temporal(frame_embs, padding_mask)
                    text_emb = bert_model(input_ids, attention_mask)

                    if isinstance(loss_fn, QueueInfoNCE):
                        loss = loss_fn(
                            video_emb, text_emb, video_ids,
                            logit_scale=bert_model.logit_scale,
                        )
                        with torch.no_grad():
                            loss_inbatch = eval_loss_fn(
                                video_emb, text_emb,
                                logit_scale=bert_model.logit_scale,
                            )
                    else:
                        loss = loss_fn(
                            video_emb, text_emb,
                            logit_scale=bert_model.logit_scale,
                        )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(bert_model.parameters()) + list(temporal.parameters()),
                    t_cfg["max_grad_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                video_emb = temporal(frame_embs, padding_mask)
                text_emb = bert_model(input_ids, attention_mask)

                if isinstance(loss_fn, QueueInfoNCE):
                    loss = loss_fn(
                        video_emb, text_emb, video_ids,
                        logit_scale=bert_model.logit_scale,
                    )
                    with torch.no_grad():
                        loss_inbatch = eval_loss_fn(
                            video_emb, text_emb,
                            logit_scale=bert_model.logit_scale,
                        )
                else:
                    loss = loss_fn(
                        video_emb, text_emb,
                        logit_scale=bert_model.logit_scale,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(bert_model.parameters()) + list(temporal.parameters()),
                    t_cfg["max_grad_norm"],
                )
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Enqueue after loss
            if queue is not None and video_ids is not None:
                queue.enqueue(video_emb, text_emb, video_ids)

            # Logging
            monitor_loss = loss_inbatch.item() if isinstance(loss_fn, QueueInfoNCE) else loss.item()
            epoch_losses.append(monitor_loss)
            epoch_queue_losses.append(loss.item())
            i2t_q = getattr(loss_fn, "_last_i2t", None)
            t2i_q = getattr(loss_fn, "_last_t2i", None)
            if i2t_q is not None:
                epoch_i2t_losses.append(i2t_q.item())
                epoch_t2i_losses.append(t2i_q.item())
            postfix = {"loss": f"{monitor_loss:.4f}"}
            if isinstance(loss_fn, QueueInfoNCE):
                postfix["q"] = f"{loss.item():.4f}"
            if queue is not None:
                postfix["Q"] = len(queue)
            if i2t_q is not None:
                postfix["i2t"] = f"{i2t_q:.4f}"
                postfix["t2i"] = f"{t2i_q:.4f}"
            pbar.set_postfix(**postfix)

        # ── Epoch-end: MSVD + Flickr30k evaluation ──
        print(f"  Epoch {epoch:02d}/{epochs} — evaluating...")
        msvd_results = evaluate_msvd(bert_model, temporal, msvd_loader, cfg, device)
        torch.cuda.empty_cache()

        flickr_results = evaluate_flickr30k(bert_model, tokenizer, device)
        torch.cuda.empty_cache()

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        is_best = msvd_results.get("t2i_R@1", -1.0) > best_msvd_t2i_r1
        if is_best:
            best_msvd_t2i_r1 = msvd_results["t2i_R@1"]
        if flickr_results.get("t2i_R@1", -1.0) > best_flickr_t2i_r1:
            best_flickr_t2i_r1 = flickr_results["t2i_R@1"]

        # ── CSV log (same format as train_bert.py) ──
        avg_queue_loss = sum(epoch_queue_losses) / len(epoch_queue_losses) if epoch_queue_losses else 0.0
        lr_now = optimizer.param_groups[0]["lr"]
        csv_row = {
            "type": "epoch", "epoch": epoch, "val_loss": round(avg_loss, 4),
            "lr": lr_now, "best_t2i_R@1": best_msvd_t2i_r1,
            "train_monitor_loss": round(avg_loss, 4),
            "queue_loss": round(avg_queue_loss, 4),
        }
        if epoch_i2t_losses:
            csv_row["i2t_loss"] = round(float(sum(epoch_i2t_losses) / len(epoch_i2t_losses)), 4)
            csv_row["t2i_loss"] = round(float(sum(epoch_t2i_losses) / len(epoch_t2i_losses)), 4)
        for key in recall_keys:
            if key in msvd_results:
                csv_row[key] = round(float(msvd_results[key]), 2)
        for med_key in ("i2t_medR", "t2i_medR"):
            if med_key in msvd_results:
                csv_row[med_key] = round(float(msvd_results[med_key]), 1)
        for key in recall_keys:
            fk = f"flickr_{key}"
            if key in flickr_results:
                csv_row[fk] = round(float(flickr_results[key]), 2)
        for med_key in ("i2t_medR", "t2i_medR"):
            fm = f"flickr_{med_key}"
            if med_key in flickr_results:
                csv_row[fm] = round(float(flickr_results[med_key]), 1)
        _write_val_csv(csv_row)

        # Log
        log_parts = [
            f"Epoch {epoch:02d}/{epochs}",
            f"Train loss: {avg_loss:.4f}",
            f"MSVD t2i_R@1: {msvd_results.get('t2i_R@1', 0):.2f}",
            f"MSVD i2t_R@1: {msvd_results.get('i2t_R@1', 0):.2f}",
            f"Flickr t2i_R@1: {flickr_results.get('t2i_R@1', 0):.2f}",
            f"Flickr i2t_R@1: {flickr_results.get('i2t_R@1', 0):.2f}",
        ]
        if best_msvd_t2i_r1 > 0:
            log_parts.append(f"Best MSVD: {best_msvd_t2i_r1:.2f}")
        if best_flickr_t2i_r1 > 0:
            log_parts.append(f"Best Flickr: {best_flickr_t2i_r1:.2f}")
        lr_now = optimizer.param_groups[0]["lr"]
        log_parts.append(f"LR: {lr_now:.2e}")
        temp_now = bert_model.get_temperature()
        log_parts.append(f"Temp: {temp_now:.4f}")
        log_scale = bert_model.logit_scale.exp().item()
        log_parts.append(f"Scale: {log_scale:.2f}")

        # ── Save checkpoint ──
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "bert_state_dict": bert_model.state_dict(),
            "temporal_state_dict": temporal.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "msvd_results": msvd_results,
            "flickr_results": flickr_results,
            "best_msvd_t2i_r1": best_msvd_t2i_r1,
            "best_flickr_t2i_r1": best_flickr_t2i_r1,
        }
        if scaler:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        if queue is not None:
            qstate = queue.state_dict()
            if qstate is not None:
                ckpt["queue_state"] = qstate

        save_name = f"joint_e{epoch:02d}_msvd{msvd_results.get('t2i_R@1', 0):.1f}.pt"
        if is_best:
            save_name = f"joint_best_e{epoch:02d}_msvd{msvd_results.get('t2i_R@1', 0):.1f}.pt"
            log_parts.append("★ New best!")
        torch.save(ckpt, os.path.join(ckpt_dir, save_name))
        log_parts.append(f"→ {save_name}")

        print("  " + " | ".join(log_parts))
        print("-" * 60)

    print(f"Training complete. Best MSVD t2i_R@1: {best_msvd_t2i_r1:.2f}")


if __name__ == "__main__":
    main()
