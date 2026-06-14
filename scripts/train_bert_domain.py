# Stage 1: Domain-adapt BERT to MSR-VTT video captions.
# Video: frozen CLIP keyframes → mean pool (frozen, never stale in queue)
# Text:  BERT + LoRA (trainable)
# Queue: mask_stale_texts=True → t2i gets B+Q negatives (videos never stale)
#
# Eval every epoch: MSVD (video retrieval) + Flickr30k (image retrieval)
#
# Usage:
#   python scripts/train_bert_domain.py --config config/bert_domain.yaml

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

from src.models import BertEncoder
from src.data import VideoDataset, video_collate_fn
from src.training import QueueInfoNCE, ContrastiveQueue, SymmetricInfoNCE
from src.training.evaluation import compute_recall_metrics

FLICKR_ROOT = Path("./dataset_annotation")


def _build_bert(cfg, device) -> BertEncoder:
    bert_path = cfg["model"]["bert_model_path"]
    lora_cfg = cfg.get("lora", {})
    print("Building BERT encoder (with LoRA)...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
        initial_temperature=cfg["training"]["temperature"],
    ).to(device)

    ckpt_path = cfg["checkpoint"]["bert_pretrain"]
    print(f"Loading BERT from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  BERT: {total:,} total, {trainable:,} trainable (LoRA + proj)")
    return model


def _build_dataloaders(cfg):
    msrvtt_cfg = cfg["data"]["msrvtt"]
    frame_cache = msrvtt_cfg["frame_cache"]
    train_ann = os.path.join(msrvtt_cfg["root"], msrvtt_cfg["train_annotation"])

    msvd_cfg = cfg["data"]["msvd"]
    msvd_ann = msvd_cfg["annotation"]
    msvd_cache = msvd_cfg["frame_cache"]

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["bert_model_path"])

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


def mean_pool_video(frame_embs, padding_mask):
    """frame_embs: (B, T, 512), padding_mask: (B, T) True=padded → (B, 512) L2-norm"""
    mask = (~padding_mask).unsqueeze(-1).float()
    video_emb = (frame_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(video_emb, dim=-1)


@torch.no_grad()
def evaluate_msvd(bert_model, msvd_loader, device) -> dict:
    """Evaluate on MSVD: BERT text vs mean-pooled CLIP video."""
    bert_model.eval()
    all_video_embs = []
    all_text_embs = []
    all_video_ids = []

    for batch in tqdm(msvd_loader, desc="MSVD eval", leave=False):
        frame_embs, padding_mask, input_ids, attention_mask, video_ids = batch
        frame_embs = frame_embs.to(device)

        video_emb = mean_pool_video(frame_embs, padding_mask.to(device))
        text_emb = bert_model(input_ids.to(device), attention_mask.to(device))

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

    # Parse test annotations
    with open(ann_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pairs = []
    for r in rows:
        if r["split"].strip() != "test":
            continue
        stem = Path(r["filename"].strip()).stem
        for cap in json.loads(r["raw"]):
            pairs.append((stem, cap))

    # Load precomputed CLIP image embeddings
    unique_stems = list(dict.fromkeys(stem for stem, _ in pairs))
    stem_to_img = {}
    for stem in unique_stems:
        npy = img_cache / f"{stem}.npy"
        if npy.exists():
            stem_to_img[stem] = np.load(npy)

    # Build paired arrays
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

    image_embs = torch.from_numpy(np.stack(all_image_embs))
    text_embs = torch.cat(all_text_embs, dim=0)
    return compute_recall_metrics(image_embs, text_embs, all_image_ids)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Domain-adapt BERT on MSR-VTT")
    parser.add_argument("--config", default="config/bert_domain.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    t_cfg = cfg["training"]

    # ── Model ──
    bert_model = _build_bert(cfg, device)

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
            t2i_weight=t_cfg.get("t2i_weight", 0.75),
            mask_stale_texts=q_cfg.get("mask_stale_texts", True),
        )
    else:
        loss_fn = SymmetricInfoNCE(
            temperature=t_cfg["temperature"],
            t2i_weight=t_cfg.get("t2i_weight", 0.5),
        )

    eval_loss_fn = SymmetricInfoNCE(temperature=t_cfg["temperature"])

    # ── Optimizer ──
    bert_params = [p for p in bert_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        bert_params,
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    print(f"  BERT LR: {t_cfg['lr']:.2e}")

    total_steps = len(train_loader) * t_cfg["epochs"]
    warmup_steps = t_cfg.get("warmup_steps", 2000)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if t_cfg.get("amp", True) and torch.cuda.is_available() else None

    ckpt_dir = t_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume ──
    start_epoch = 1
    global_step = 0
    best_msvd_t2i_r1 = -1.0
    best_flickr_t2i_r1 = -1.0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        bert_model.load_state_dict(ckpt["model_state_dict"], strict=False)
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
    print(f"\nTraining: {epochs} epochs, eval each epoch: MSVD + Flickr30k")
    print("-" * 60)

    for epoch in range(start_epoch, epochs + 1):
        bert_model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for batch in pbar:
            frame_embs, padding_mask, input_ids, attention_mask, video_ids = batch
            frame_embs = frame_embs.to(device)
            padding_mask = padding_mask.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # ── Forward ──
            # Mean pool frozen CLIP video embeddings (no temporal)
            video_emb = mean_pool_video(frame_embs, padding_mask)
            text_emb = bert_model(input_ids, attention_mask)

            if scaler:
                with torch.amp.autocast("cuda"):
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
                torch.nn.utils.clip_grad_norm_(bert_params, t_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
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
                torch.nn.utils.clip_grad_norm_(bert_params, t_cfg["max_grad_norm"])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Enqueue after loss (video from frozen CLIP → never stale)
            if queue is not None and video_ids is not None:
                queue.enqueue(video_emb, text_emb, video_ids)

            # Logging
            monitor_loss = loss_inbatch.item() if isinstance(loss_fn, QueueInfoNCE) else loss.item()
            epoch_losses.append(monitor_loss)
            postfix = {"loss": f"{monitor_loss:.4f}"}
            if queue is not None:
                postfix["Q"] = len(queue)
            i2t_q = getattr(loss_fn, "_last_i2t", None)
            t2i_q = getattr(loss_fn, "_last_t2i", None)
            if i2t_q is not None:
                postfix["i2t"] = f"{i2t_q:.4f}"
                postfix["t2i"] = f"{t2i_q:.4f}"
            pbar.set_postfix(**postfix)

        # ── Epoch-end: MSVD + Flickr30k evaluation ──
        print(f"  Epoch {epoch:02d}/{epochs} — evaluating...")
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        msvd_results = evaluate_msvd(bert_model, msvd_loader, device)
        torch.cuda.empty_cache()

        flickr_results = evaluate_flickr30k(bert_model, tokenizer, device)
        torch.cuda.empty_cache()

        is_best = msvd_results.get("t2i_R@1", -1.0) > best_msvd_t2i_r1
        if is_best:
            best_msvd_t2i_r1 = msvd_results["t2i_R@1"]
        if flickr_results.get("t2i_R@1", -1.0) > best_flickr_t2i_r1:
            best_flickr_t2i_r1 = flickr_results["t2i_R@1"]

        log_parts = [
            f"Epoch {epoch:02d}/{epochs}",
            f"Loss: {avg_loss:.4f}",
            f"MSVD t2i_R@1: {msvd_results.get('t2i_R@1', 0):.2f}",
            f"MSVD i2t_R@1: {msvd_results.get('i2t_R@1', 0):.2f}",
            f"Flickr t2i_R@1: {flickr_results.get('t2i_R@1', 0):.2f}",
            f"Flickr i2t_R@1: {flickr_results.get('i2t_R@1', 0):.2f}",
        ]
        if best_msvd_t2i_r1 > 0:
            log_parts.append(f"Best(MSVD):{best_msvd_t2i_r1:.2f}")
        if best_flickr_t2i_r1 > 0:
            log_parts.append(f"Best(Flickr):{best_flickr_t2i_r1:.2f}")
        lr_now = optimizer.param_groups[0]["lr"]
        log_parts.append(f"LR:{lr_now:.2e}")
        temp_now = bert_model.get_temperature()
        log_parts.append(f"Temp:{temp_now:.4f}")

        # ── Save checkpoint ──
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": bert_model.state_dict(),
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

        save_name = f"bert_domain_e{epoch:02d}_msvd{msvd_results.get('t2i_R@1', 0):.1f}.pt"
        if is_best:
            save_name = f"bert_domain_best_e{epoch:02d}_msvd{msvd_results.get('t2i_R@1', 0):.1f}.pt"
            log_parts.append("★")
        torch.save(ckpt, os.path.join(ckpt_dir, save_name))
        log_parts.append(f"→ {save_name}")

        print("  " + " | ".join(log_parts))
        print("-" * 60)

    print(f"Training complete. Best MSVD t2i_R@1: {best_msvd_t2i_r1:.2f}, "
          f"Best Flickr t2i_R@1: {best_flickr_t2i_r1:.2f}")


if __name__ == "__main__":
    main()
