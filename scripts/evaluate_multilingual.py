"""
Evaluate a trained multilingual BERT checkpoint on English + Chinese Flickr30k.

Reports Recall@K for:
  - English Flickr30k test
  - Chinese Flickr30k test (translated via hy-mt-1.8b)
  - COCO val2017 (reference)

Usage:
    python scripts/evaluate_multilingual.py \\
        --checkpoint ../weights/bert_best.pt \\
        --config config/default.yaml \\
        --device cuda:0
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.coco_dataset import make_collate_fn
from src.data.eval_dataset import build_split_loaders
from src.data.flickr_dataset import FlickrDataset
from src.data.flickr_zh_dataset import FlickrZhDataset
from src.models import BertEncoder
from src.training import SymmetricInfoNCE
from src.training.evaluation import compute_recall_metrics


@torch.no_grad()
def _compute_median_rank(image_embs, text_embs, image_ids, direction="i2t"):
    """Compute median rank (1-based)."""
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
        sim = unique_image_embs @ text_embs.T
        ranks = []
        for u_idx, img_id in enumerate(unique_id_list):
            gt_set = id_to_text_positions[img_id]
            scores = sim[u_idx].clone()
            for gt_idx in gt_set:
                scores[gt_idx] = float("-inf")
            gt_scores = sim[u_idx, list(gt_set)]
            rank = (scores > gt_scores.max()).sum().item() + 1
            ranks.append(rank)
    else:
        sim = text_embs @ unique_image_embs.T
        ranks = []
        for i, img_id in enumerate(image_ids):
            gt_uidx = id_to_uidx[img_id]
            scores = sim[i].clone()
            scores[gt_uidx] = float("-inf")
            rank = (scores > sim[i, gt_uidx]).sum().item() + 1
            ranks.append(rank)
    return statistics.median(ranks) if ranks else float("inf")


@torch.no_grad()
def evaluate_loader(model, loader, device, temperature=0.07):
    """Run inference, return {val_loss, recall@K, medR}."""

    model.eval()
    total_loss = 0.0
    all_image_embs = []
    all_text_embs = []
    all_image_ids = []
    loss_fn = SymmetricInfoNCE(temperature=temperature)

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image_emb, input_ids, attention_mask, *rest = batch
        image_ids = rest[0] if rest else None
        image_emb = image_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_emb = model(input_ids, attention_mask)
        loss = loss_fn(image_emb, text_emb)
        total_loss += loss.item()

        all_image_embs.append(image_emb.cpu())
        all_text_embs.append(text_emb.cpu())
        if image_ids is not None:
            all_image_ids.extend(image_ids)

    avg_loss = total_loss / len(loader)
    results = {"val_loss": avg_loss}
    if all_image_ids:
        image_embs = torch.cat(all_image_embs)
        text_embs = torch.cat(all_text_embs)
        recall = compute_recall_metrics(image_embs, text_embs, all_image_ids, ks=(1, 5, 10))
        results.update(recall)
        results["i2t_medR"] = _compute_median_rank(image_embs, text_embs, all_image_ids, direction="i2t")
        results["t2i_medR"] = _compute_median_rank(image_embs, text_embs, all_image_ids, direction="t2i")
    return results


def main():
    parser = argparse.ArgumentParser(description="Bilingual evaluation of multilingual BERT")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ────────────────────────────────────────────
    te_cfg = cfg.get("text_encoder", {})
    bert_path = te_cfg.get("bert_model_path") or cfg["model"]["bert_model_path"]
    lora_cfg = cfg.get("lora", {})
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {Path(args.checkpoint).name}")

    # ── Tokenizer ───────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    collate_fn = make_collate_fn(tokenizer)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    temperature = cfg["training"]["temperature"]
    flickr_cfg = cfg.get("flickr", {})

    results = {}

    # ── English Flickr30k ───────────────────────────────
    print("\n--- English Flickr30k ---")
    en_dataset = FlickrDataset(
        flickr_cfg["root"], "test",
        flickr_cfg["embedding_cache"],
        flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
    )
    en_loader = torch.utils.data.DataLoader(
        en_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    en_results = evaluate_loader(model, en_loader, device, temperature)
    results["en"] = en_results
    print_en = " | ".join(
        f"{k}: {v:.2f}" for k, v in en_results.items()
        if k in ("i2t_R@1", "t2i_R@1", "i2t_R@5", "t2i_R@5", "i2t_R@10", "t2i_R@10")
    )
    print(f"  {print_en}")

    # ── Chinese Flickr30k ───────────────────────────────
    print("\n--- Chinese Flickr30k ---")
    zh_file = cfg.get("flickr_zh", {}).get("zh_file", "flickr_test_zh.json")
    zh_dataset = FlickrZhDataset(
        flickr_cfg["root"], "test",
        flickr_cfg["embedding_cache"],
        zh_file=zh_file,
        annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
    )
    print(f"  Samples: {len(zh_dataset)} ({len(zh_dataset)//5} images × 5 CN captions)")
    zh_loader = torch.utils.data.DataLoader(
        zh_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    zh_results = evaluate_loader(model, zh_loader, device, temperature)
    results["zh"] = zh_results
    print_zh = " | ".join(
        f"{k}: {v:.2f}" for k, v in zh_results.items()
        if k in ("i2t_R@1", "t2i_R@1", "i2t_R@5", "t2i_R@5", "i2t_R@10", "t2i_R@10")
    )
    print(f"  {print_zh}")

    # ── COCO val2017 (reference) ────────────────────────
    print("\n--- COCO val2017 ---")
    split_loaders = build_split_loaders(cfg, tokenizer, batch_size, num_workers)
    coco_results = evaluate_loader(model, split_loaders["coco"], device, temperature)
    results["coco"] = coco_results
    print_coco = " | ".join(
        f"{k}: {v:.2f}" for k, v in coco_results.items()
        if k in ("i2t_R@1", "t2i_R@1", "i2t_R@5", "t2i_R@5", "i2t_R@10", "t2i_R@10")
    )
    print(f"  {print_coco}")

    # ── Summary ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" Summary — Cross-lingual Flickr30k Recall")
    print("=" * 55)
    header = f"{'Metric':<12} {'English':<12} {'Chinese':<12} {'Diff':<12}"
    print(header)
    print("-" * 55)
    for metric in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        en_v = en_results.get(metric, 0)
        zh_v = zh_results.get(metric, 0)
        diff = en_v - zh_v
        print(f"{metric:<12} {en_v:<12.2f} {zh_v:<12.2f} {diff:<+12.2f}")
    print("=" * 55)
    print("Done.")


if __name__ == "__main__":
    main()
