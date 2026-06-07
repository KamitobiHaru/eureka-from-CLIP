"""
Evaluate a trained BERT checkpoint on COCO val2017 + Flickr30k test.

Reports CLIP-style Recall@K and median rank for:
  - Combined COCO+Flickr evaluation (the primary metric)
  - COCO-only and Flickr-only breakdowns

Usage:
    python scripts/evaluate_checkpoint.py <checkpoint.pt> [--config config/default.yaml]
"""

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.eval_dataset import build_split_loaders, build_combined_val_loader
from src.models import BertEncoder
from src.training import SymmetricInfoNCE
from src.training.evaluation import compute_recall_metrics


@torch.no_grad()
def evaluate_loader(model, loader, device, temperature=0.07):
    """Run inference on a validation loader and return collected embeddings + ids."""
    model.eval()
    total_loss = 0.0
    all_image_embs = []
    all_text_embs = []
    all_image_ids = []
    loss_fn = SymmetricInfoNCE(temperature=temperature)

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in pbar:
        image_emb, input_ids, attention_mask, *rest = batch
        image_ids = rest[0] if rest else None
        image_emb = image_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_emb = model(input_ids, attention_mask)
        loss = loss_fn(image_emb, text_emb)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        all_image_embs.append(image_emb.cpu())
        all_text_embs.append(text_emb.cpu())
        if image_ids is not None:
            all_image_ids.extend(image_ids)

    avg_loss = total_loss / len(loader)

    # ── Compute recall metrics ──────────────────────────
    results = {"val_loss": avg_loss}
    if all_image_ids:
        image_embs = torch.cat(all_image_embs)
        text_embs = torch.cat(all_text_embs)
        recall = compute_recall_metrics(
            image_embs, text_embs, all_image_ids,
            ks=(1, 5, 10),
        )
        results.update(recall)
        # CLIP also reports median rank
        results["i2t_medR"] = _compute_median_rank(image_embs, text_embs, all_image_ids, direction="i2t")
        results["t2i_medR"] = _compute_median_rank(image_embs, text_embs, all_image_ids, direction="t2i")

    return results


@torch.no_grad()
def _compute_median_rank(image_embs, text_embs, image_ids, direction="i2t"):
    """Compute median rank (1-based). CLIP uses this as a key metric."""
    from collections import defaultdict

    # Dedup images
    seen = set()
    unique_indices = []
    unique_id_list = []
    for i, img_id in enumerate(image_ids):
        if img_id not in seen:
            seen.add(img_id)
            unique_indices.append(i)
            unique_id_list.append(img_id)

    unique_image_embs = image_embs[unique_indices]
    id_to_text_positions = defaultdict(set)
    for i, img_id in enumerate(image_ids):
        id_to_text_positions[img_id].add(i)

    id_to_uidx = {img_id: idx for idx, img_id in enumerate(unique_id_list)}

    if direction == "i2t":
        sim = unique_image_embs @ text_embs.T  # (U, N)
        ranks = []
        for u_idx, img_id in enumerate(unique_id_list):
            gt_set = id_to_text_positions[img_id]
            scores = sim[u_idx]
            # Mask false negatives
            for gt_idx in gt_set:
                scores[gt_idx] = float("-inf")
            # Find closest ground-truth
            gt_scores = sim[u_idx, list(gt_set)]
            max_gt_score = gt_scores.max()
            rank = (scores > max_gt_score).sum().item() + 1  # 1-based
            ranks.append(rank)
    else:
        sim = text_embs @ unique_image_embs.T  # (N, U)
        ranks = []
        for i, img_id in enumerate(image_ids):
            gt_uidx = id_to_uidx[img_id]
            scores = sim[i].clone()
            scores[gt_uidx] = float("-inf")
            rank = (scores > sim[i, gt_uidx]).sum().item() + 1
            ranks.append(rank)

    return statistics.median(ranks) if ranks else float("inf")


def format_results(results, prefix=""):
    """Pretty-print evaluation results."""
    parts = []
    if "val_loss" in results:
        parts.append(f"Loss: {results['val_loss']:.4f}")
    for key in ["i2t_R@1", "i2t_R@5", "i2t_R@10"]:
        if key in results:
            parts.append(f"{key}: {results[key]:.2f}")
    for key in ["t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        if key in results:
            parts.append(f"{key}: {results[key]:.2f}")
    if "i2t_medR" in results:
        parts.append(f"i2t_medR: {results['i2t_medR']:.0f}")
    if "t2i_medR" in results:
        parts.append(f"t2i_medR: {results['t2i_medR']:.0f}")
    label = f" [{prefix}]" if prefix else ""
    return " | ".join(parts) + label


def main():
    parser = argparse.ArgumentParser(description="Evaluate a BERT checkpoint on COCO+Flickr")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ────────────────────────────────────────────
    bert_path = cfg["model"]["bert_model_path"]
    lora_cfg = cfg.get("lora", {})
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)

    # ── Load checkpoint ──────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {Path(args.checkpoint).name} (epoch {epoch})")

    # ── Tokenizer ────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # ── CLIP-style per-dataset evaluation ─────────────────
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    temperature = cfg["training"]["temperature"]
    split_loaders = build_split_loaders(cfg, tokenizer, batch_size, num_workers)

    if "flickr" in split_loaders and split_loaders["flickr"] is not None:
        flickr_results = evaluate_loader(model, split_loaders["flickr"], device, temperature)
        print("  " + format_results(flickr_results, prefix="Flickr30k test (CLIP benchmark)"))

    coco_results = evaluate_loader(model, split_loaders["coco"], device, temperature)
    print("  " + format_results(coco_results, prefix="COCO val2017"))

    # ── Combined evaluation (COCO + Flickr) ──────────────
    print("\n--- Combined Evaluation ---")
    combined_loader = build_combined_val_loader(cfg, tokenizer, batch_size, num_workers)
    combined_results = evaluate_loader(model, combined_loader, device, temperature)
    print("  " + format_results(combined_results, prefix="Combined"))
    print("Done.")


if __name__ == "__main__":
    main()
