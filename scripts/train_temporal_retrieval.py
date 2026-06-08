"""
Phase-2 temporal fine-tuning: inject position awareness while preserving retrieval.

Loads a Phase-1 temporal checkpoint (already aligned with frozen BERT text space)
and fine-tunes on motion sequences with a small LR so that semantic alignment
(measured by Flickr30k single-frame recall) is not destroyed.

Loss: SymmetricInfoNCE + PositionPredictionReward + AnchorMSE (0.2)

Usage:
    python scripts/train_temporal_retrieval.py \\
        --temporal_checkpoint /path/to/phase1_temporal_best.pt \\
        --config config/default3_temporal.yaml \\
        --bert_checkpoint /path/to/bert_best.pt \\
        --device cuda:0

Evaluation:
    - Sequence val sNCE after each epoch
    - Flickr30k single-frame recall periodically (every ``eval_interval`` steps
      and at epoch end) to monitor semantic drift.
"""

import argparse
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
from src.data import (
    FlickrDataset,
    make_text_emb_collate_fn,
    get_mixed_dataloader,
)
from src.training import (
    SymmetricInfoNCE, PositionPredictionReward,
    compute_recall_metrics,
)
from src.training.trainer import _get_cosine_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_frozen_bert(cfg, device, bert_checkpoint):
    """Load BertEncoder, load pretrained, freeze ALL params."""
    bert_path = cfg["model"]["bert_model_path"]
    if not os.path.isdir(bert_path):
        print(f"BERT model not found at {bert_path}")
        sys.exit(1)

    lora_cfg = cfg.get("lora", {})
    print("Building BERT text encoder (frozen)...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
        initial_temperature=cfg["training"]["temperature"],
    ).to(device)

    if not bert_checkpoint or not Path(bert_checkpoint).exists():
        print(f"BERT checkpoint not found: {bert_checkpoint}")
        sys.exit(1)
    print(f"Loading pretrained BERT from: {bert_checkpoint}")
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  BERT frozen.")
    return model


def _load_temporal(t_cfg, device, checkpoint_path):
    """Build TemporalTransformer and load Phase-1 weights."""
    temporal = TemporalTransformer(
        d_model=t_cfg.get("d_model", 512),
        nhead=t_cfg.get("nhead", 8),
        num_layers=t_cfg.get("num_layers", 2),
        dim_feedforward=t_cfg.get("dim_feedforward", 1024),
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg.get("max_frames", 16),
    ).to(device)
    n = sum(p.numel() for p in temporal.parameters())
    print(f"  TemporalTransformer: {n:,} params")

    if checkpoint_path:
        print(f"  Loading Phase-1 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        temporal.load_state_dict(ckpt["temporal_state_dict"])
    else:
        print("  WARNING: no checkpoint — starting from random init!")
    return temporal


def _build_flickr_eval_loader(cfg, device):
    """Build a DataLoader for Flickr30k test with precomputed text embeddings.

    Returns (loader, None) on success or (None, error_msg) on failure.
    """
    flickr_cfg = cfg.get("flickr", {})
    if not flickr_cfg.get("root"):
        return None, "no Flickr root in config"

    cache = flickr_cfg.get("embedding_cache")
    if not cache or not Path(cache).is_dir():
        return None, f"Flickr embedding cache not found: {cache}"

    text_cache = cache + "_text"
    if not Path(text_cache).is_dir():
        return None, f"Flickr text cache not found: {text_cache} (run precompute_text_embeddings.py)"

    ann_file = flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv")
    dataset = FlickrDataset(
        flickr_cfg["root"], "test",
        cache,
        annotation_file=ann_file,
        text_cache_dir=text_cache,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=make_text_emb_collate_fn(),
        pin_memory=True,
    )
    print(f"  Flickr30k test: {len(dataset):,} captions (precomputed text)")
    return loader, None


def _project_single(image_emb, temporal, device):
    """Project single-frame image_emb [B, 512] through temporal."""
    B = image_emb.size(0)
    frame_embs = image_emb.unsqueeze(1)                      # [B, 1, 512]
    mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
    positions = torch.tensor([[0.5, 0.5, 1.0]], device=device).expand(B, 1, 3)
    return temporal(frame_embs, mask, positions=positions)    # [B, 512]


@torch.no_grad()
def evaluate_flickr(temporal, loader, device, logit_scale, temperature):
    """Evaluate single-frame Flickr30k recall. Returns dict with 'val_loss' and
    recall keys ('i2t_R@1', ..., 't2i_R@1', ...)."""
    temporal.eval()
    eval_loss_fn = SymmetricInfoNCE(temperature=temperature)
    total_loss = 0.0
    all_image_embs, all_text_embs, all_image_ids = [], [], []

    for batch in tqdm(loader, desc="Flickr eval", leave=False):
        image_emb, text_emb, image_ids = batch
        image_emb = image_emb.to(device)
        text_emb = text_emb.to(device)
        image_emb = _project_single(image_emb, temporal, device)

        loss = eval_loss_fn(image_emb, text_emb, logit_scale=logit_scale)
        total_loss += loss.item()
        all_image_embs.append(image_emb.cpu())
        all_text_embs.append(text_emb.cpu())
        all_image_ids.extend(image_ids)

    avg_loss = total_loss / max(len(loader), 1)
    results = {"val_loss": avg_loss}

    if all_image_ids:
        recall = compute_recall_metrics(
            torch.cat(all_image_embs),
            torch.cat(all_text_embs),
            all_image_ids,
        )
        results.update(recall)

    temporal.train()
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Sequence validation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_sequence(temporal, text_encoder, loader, sNCE_fn, pos_reward_fn,
                      device, use_amp):
    """Epoch-end sequence validation. Returns dict of averaged losses."""
    temporal.eval()
    text_encoder.eval()
    sNCE_total = pos_total = count = 0.0

    for batch in tqdm(loader, desc="Seq val", leave=False):
        frame_embs, frame_mask, corr_ids, corr_mask, positions, pos_mask = batch
        frame_embs = frame_embs.to(device)
        frame_mask = frame_mask.to(device)
        corr_ids = corr_ids.to(device)
        corr_mask = corr_mask.to(device)
        positions = positions.to(device)
        pos_mask = pos_mask.to(device)

        ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with ctx:
            video_emb, per_frame = temporal(
                frame_embs, frame_mask, positions=positions, return_per_frame=True,
            )
            text_emb = text_encoder(corr_ids, corr_mask)
            sNCE_total += sNCE_fn(video_emb, text_emb).item()
            pos_total += pos_reward_fn(per_frame, positions, pos_mask).item()
        count += 1

    temporal.train()
    return {"sNCE": sNCE_total / count, "pos": pos_total / count}


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase-2: fine-tune TemporalTransformer on motion sequences."
    )
    parser.add_argument("--config", default="config/default3_temporal.yaml")
    parser.add_argument("--temporal_checkpoint", required=True,
                        help="Phase-1 temporal checkpoint (from train_bert_temporal.py)")
    parser.add_argument("--bert_checkpoint", required=True,
                        help="Pretrained BERT checkpoint (for logit_scale + sequence text encoding)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint dir (default: from config)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Fine-tuning LR (default: 1e-5)")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="Flickr30k eval every N gradient steps (default: once per epoch)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    t_cfg = cfg.get("temporal", {})

    # ── Models ──────────────────────────────────────────────────
    text_encoder = _build_frozen_bert(cfg, device, args.bert_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["bert_model_path"],
                                              local_files_only=True)
    temporal = _load_temporal(t_cfg, device, args.temporal_checkpoint)

    # ── Sequence data ───────────────────────────────────────────
    motion_cache = cfg.get("motion", {}).get("cache_dir", "data/coco/mixed_sequences")
    print(f"Motion cache: {motion_cache}")
    if not Path(motion_cache, "train2017", "samples.json").exists():
        print(f"ERROR: no precomputed sequences at {motion_cache}.")
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
    print(f"  Seq train: {len(train_loader.dataset)} sequences")
    print(f"  Seq val:   {len(val_loader.dataset)} sequences")

    # ── Flickr30k eval loader ───────────────────────────────────
    flickr_loader, flickr_err = _build_flickr_eval_loader(cfg, device)
    if flickr_loader is None:
        print(f"  WARNING: Flickr30k eval unavailable ({flickr_err})")
    else:
        print(f"  Flickr eval: enabled")

    # ── Losses ──────────────────────────────────────────────────
    sNCE_fn = SymmetricInfoNCE(temperature=cfg["training"]["temperature"])
    pos_reward_fn = PositionPredictionReward(d_model=t_cfg.get("d_model", 512)).to(device)
    losses = {
        "sNCE_w": t_cfg.get("order_consistency_weight", 1.0),
        "pos_w": t_cfg.get("position_prediction_weight", 0.1),
        "anchor_w": t_cfg.get("anchor_mse_weight", 0.2),
    }
    print(f"  Losses: {losses['sNCE_w']}×sNCE + {losses['pos_w']}×pos + {losses['anchor_w']}×anchor")

    # ── Pos warmup ──────────────────────────────────────────────
    pw_cfg = t_cfg.get("pos_warmup", {})
    pw_start_epoch = pw_cfg.get("start_epoch", 0)
    pw_steps = pw_cfg.get("steps", 0)
    pw_rate = pw_cfg.get("rate", 1.0)
    steps_per_epoch = len(train_loader)
    pw_start_step = max(0, pw_start_epoch - 1) * steps_per_epoch
    pw_enabled = pw_start_epoch > 0 and pw_steps > 0
    base_pos_w = losses["pos_w"]

    def get_pos_w(global_step):
        if not pw_enabled:
            return base_pos_w
        elapsed = global_step - pw_start_step
        if elapsed < 0:
            return 0.0
        if elapsed >= pw_steps:
            return base_pos_w
        progress = (elapsed / pw_steps) ** pw_rate
        return progress * base_pos_w

    if pw_enabled:
        print(f"  Pos warmup: start_epoch={pw_start_epoch}, steps={pw_steps}, rate={pw_rate}")
    else:
        print(f"  Pos warmup: disabled")

    # ── Optimizer (temporal ONLY) ───────────────────────────────
    optimizer = torch.optim.AdamW(
        list(temporal.parameters()) + list(pos_reward_fn.parameters()),
        lr=args.lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    print(f"  Fine-tune LR: {args.lr}")

    # ── Scheduler ───────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = t_cfg.get("temporal_warmup", 500)
    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── AMP ─────────────────────────────────────────────────────
    use_amp = cfg["training"]["amp"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpoint dir ──────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir or cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Logit scale from frozen BERT ────────────────────────────
    logit_scale = getattr(text_encoder, "logit_scale", None)
    temperature = cfg["training"]["temperature"]

    # ── State ───────────────────────────────────────────────────
    global_step = 0
    start_epoch = 1
    best_t2i_r1 = -1.0  # track best Flickr30k t2i_R@1

    # ── Override eval_interval ──────────────────────────────────
    eval_interval = args.eval_interval or steps_per_epoch  # default: every epoch

    # ── Header ──────────────────────────────────────────────────
    print(f"\nPhase 2 fine-tuning: {epochs} epochs, {warmup_steps} warmup, cosine decay")
    print(f"  Flickr eval every {eval_interval} steps (+ epoch end)")
    print(f"  Save best by t2i_R@1 + every 5 epochs")
    print("-" * 60)

    # ══════════════════════════════════════════════════════════════
    #  Training Loop
    # ══════════════════════════════════════════════════════════════

    for epoch in range(start_epoch, epochs + 1):
        temporal.train()
        epoch_sNCE = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}", leave=False)

        for batch in pbar:
            # ── Forward / backward ─────────────────────────────
            frame_embs, frame_mask, corr_ids, corr_mask, positions, pos_mask = batch
            frame_embs = frame_embs.to(device)
            frame_mask = frame_mask.to(device)
            corr_ids = corr_ids.to(device)
            corr_mask = corr_mask.to(device)
            positions = positions.to(device)
            pos_mask = pos_mask.to(device)

            def _forward():
                video_emb, per_frame = temporal(
                    frame_embs, frame_mask,
                    positions=positions, return_per_frame=True,
                )
                text_emb = text_encoder(corr_ids, corr_mask)

                sNCE = sNCE_fn(video_emb, text_emb)
                pos_reward = pos_reward_fn(per_frame, positions, pos_mask)

                # AnchorMSE: keep video_emb near mean-pooled CLIP embeddings
                valid = (~frame_mask).unsqueeze(-1).float()
                frame_mean = (frame_embs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                frame_mean = F.normalize(frame_mean, dim=-1)
                anchor_mse = F.mse_loss(video_emb, frame_mean.detach())

                current_pos_w = get_pos_w(global_step)
                total = (losses["sNCE_w"] * sNCE
                         + current_pos_w * pos_reward
                         + losses["anchor_w"] * anchor_mse)
                return total, {"sNCE": sNCE, "pos": pos_reward,
                               "anchor": anchor_mse, "pos_w": current_pos_w}

            if scaler:
                with torch.amp.autocast("cuda"):
                    total, result = _forward()
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(temporal.parameters(),
                                               cfg["training"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                total, result = _forward()
                total.backward()
                torch.nn.utils.clip_grad_norm_(temporal.parameters(),
                                               cfg["training"]["max_grad_norm"])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            epoch_sNCE.append(result["sNCE"].item() if hasattr(result["sNCE"], "item") else result["sNCE"])
            pbar.set_postfix(
                loss=f"{total.item():.4f}",
                sNCE=f"{result['sNCE']:.4f}" if isinstance(result['sNCE'], torch.Tensor) else f"{result['sNCE']:.4f}",
                pos=f"{result['pos']:.4f}" if isinstance(result['pos'], torch.Tensor) else f"{result['pos']:.4f}",
                pos_w=f"{result['pos_w']:.3f}",
                a=f"{result['anchor']:.4f}" if isinstance(result['anchor'], torch.Tensor) else f"{result['anchor']:.4f}",
            )

            # ── Periodic Flickr30k eval ────────────────────────
            if flickr_loader is not None and global_step % eval_interval == 0:
                flickr_results = evaluate_flickr(
                    temporal, flickr_loader, device, logit_scale, temperature,
                )
                t2i = flickr_results.get("t2i_R@1", -1.0)
                i2t = flickr_results.get("i2t_R@1", -1.0)
                lr_val = optimizer.param_groups[0]["lr"]

                is_best = t2i > best_t2i_r1
                if is_best:
                    best_t2i_r1 = t2i

                log = (
                    f"  Step {global_step}  |  Flickr i2t_R@1: {i2t:.1f}  |  "
                    f"t2i_R@1: {t2i:.1f}  |  Best t2i: {best_t2i_r1:.1f}  |  "
                    f"Val loss: {flickr_results['val_loss']:.4f}  |  LR: {lr_val:.2e}"
                )
                if is_best:
                    _save_temporal(ckpt_dir, epoch, temporal, optimizer, scheduler, scaler,
                                   global_step, flickr_results)
                    log += "  ★ New best!"
                print(log)

        # ── End-of-epoch sequence validation ────────────────────
        val_metrics = evaluate_sequence(
            temporal, text_encoder, val_loader, sNCE_fn, pos_reward_fn,
            device, use_amp,
        )
        lr_val = optimizer.param_groups[0]["lr"]
        print(f"  Seq val: sNCE={val_metrics['sNCE']:.4f}  pos={val_metrics['pos']:.4f}  LR: {lr_val:.2e}")

        # ── End-of-epoch Flickr30k eval ─────────────────────────
        if flickr_loader is not None:
            flickr_results = evaluate_flickr(
                temporal, flickr_loader, device, logit_scale, temperature,
            )
            t2i = flickr_results.get("t2i_R@1", -1.0)
            i2t = flickr_results.get("i2t_R@1", -1.0)
            avg_sNCE = sum(epoch_sNCE) / len(epoch_sNCE) if epoch_sNCE else 0.0

            is_best = t2i > best_t2i_r1
            if is_best:
                best_t2i_r1 = t2i

            log = (
                f"Epoch {epoch:02d}/{epochs} done  |  "
                f"Train sNCE: {avg_sNCE:.4f}  |  "
                f"Flickr i2t_R@1: {i2t:.1f}  |  t2i_R@1: {t2i:.1f}  |  "
                f"Best t2i: {best_t2i_r1:.1f}  |  LR: {lr_val:.2e}"
            )

            if is_best or epoch % 5 == 0:
                path = _save_temporal(ckpt_dir, epoch, temporal, optimizer, scheduler,
                                       scaler, global_step, flickr_results)
                log += f"  → {Path(path).name}"
                if is_best:
                    log += "  ★ New best!"
        else:
            log = (
                f"Epoch {epoch:02d}/{epochs} done  |  "
                f"Seq val sNCE: {val_metrics['sNCE']:.4f}  |  LR: {lr_val:.2e}"
            )

        print("  " + log)
        print("-" * 60)

    print(f"Phase 2 complete. Best Flickr30k t2i_R@1: {best_t2i_r1:.1f}")


def _save_temporal(ckpt_dir, epoch, temporal, optimizer, scheduler, scaler,
                   global_step, flickr_results):
    """Save temporal checkpoint with Flickr recall in filename."""
    t2i = flickr_results.get("t2i_R@1", None)
    fname = f"temporal_epoch{epoch:02d}_t2i{t2i:.1f}.pt" if t2i is not None else f"temporal_epoch{epoch:02d}.pt"
    path = os.path.join(ckpt_dir, fname)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "temporal_state_dict": temporal.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "flickr_results": flickr_results,
    }
    if scaler:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)
    return path


if __name__ == "__main__":
    main()
