"""
Evaluate a trained TemporalTransformer + BERT on Flickr30k test.

For each Flickr30k image, the single CLIP embedding is treated as a 1-frame
"video" sequence and passed through the TemporalTransformer. The resulting
video embedding is compared against the BERT text embedding using
CLIP-style Recall@K metrics.

Usage:
    python scripts/evaluate_temporal_flickr.py \\
        --temporal_checkpoint path/to/temporal_epochXX.pt \\
        --bert_checkpoint path/to/bert_epochXX.pt \\
        --temporal_config config/default3_temporal.yaml \\
        --bert_config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.flickr_dataset import FlickrDataset
from src.data.coco_dataset import make_collate_fn
from src.models import BertEncoder, TemporalTransformer
from src.training.evaluation import compute_recall_metrics
from scripts.evaluate_checkpoint import _compute_median_rank, format_results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TemporalTransformer + BERT on Flickr30k"
    )
    parser.add_argument("--temporal_checkpoint", required=True)
    parser.add_argument("--bert_checkpoint", required=True)
    parser.add_argument("--temporal_config", default="config/default3_temporal.yaml",
                        help="Config used for temporal training (architecture params)")
    parser.add_argument("--bert_config", default="config/default.yaml",
                        help="Config used for BERT training (data paths, etc.)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load configs ───────────────────────────────────────────
    with open(args.temporal_config) as f:
        t_cfg_full = yaml.safe_load(f)
    with open(args.bert_config) as f:
        b_cfg = yaml.safe_load(f)

    t_cfg = t_cfg_full.get("temporal", {})

    # ── Build TemporalTransformer ──────────────────────────────
    print("Building TemporalTransformer...")
    temporal = TemporalTransformer(
        d_model=t_cfg.get("d_model", 512),
        nhead=t_cfg.get("nhead", 8),
        num_layers=t_cfg.get("num_layers", 2),
        dim_feedforward=t_cfg.get("dim_feedforward", 1024),
        dropout=t_cfg.get("dropout", 0.1),
        max_frames=t_cfg.get("max_frames", 16),
    ).to(device)

    t_ckpt = torch.load(args.temporal_checkpoint, map_location=device, weights_only=True)
    temporal.load_state_dict(t_ckpt["temporal_state_dict"])
    temporal.eval()
    print(f"  Loaded: {Path(args.temporal_checkpoint).name}")

    # ── Build BERT text encoder ────────────────────────────────
    bert_path = (
        b_cfg.get("text_encoder", {}).get("bert_model_path")
        or b_cfg["model"].get("bert_model_path")
    )
    lora_cfg = b_cfg.get("lora", {})
    print(f"Building BERT text encoder from {bert_path}...")
    bert = BertEncoder(
        model_path=bert_path,
        embed_dim=b_cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)

    b_ckpt = torch.load(args.bert_checkpoint, map_location=device, weights_only=True)
    bert.load_state_dict(b_ckpt["model_state_dict"])
    bert.eval()
    print(f"  Loaded: {Path(args.bert_checkpoint).name}")

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # ── Load Flickr30k test set ────────────────────────────────
    flickr_cfg = b_cfg.get("flickr", {})
    flickr_root = flickr_cfg.get("root")
    if not flickr_root:
        print("ERROR: 'flickr.root' not found in config")
        sys.exit(1)
    flickr_cache = flickr_cfg.get("embedding_cache")
    ann_file = flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv")

    dataset = FlickrDataset(flickr_root, "test", flickr_cache, annotation_file=ann_file)
    print(f"Flickr30k test: {len(dataset):,} caption-image pairs")

    collate_fn = make_collate_fn(tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Evaluate ───────────────────────────────────────────────
    all_image_embs = []
    all_text_embs = []
    all_image_ids = []

    for batch in tqdm(loader, desc="Evaluating"):
        image_emb, input_ids, attention_mask, *rest = batch
        image_ids = rest[0] if rest else None

        # image_emb is [B, 512] — treat as 1-frame video [B, 1, 512]
        frame_embs = image_emb.unsqueeze(1).to(device)  # [B, 1, 512]
        mask = torch.zeros(frame_embs.size(0), 1, dtype=torch.bool, device=device)

        # For single-frame, positions are (0.5, 0.5, 1.0): center of frame, full scale
        positions = torch.tensor([[0.5, 0.5, 1.0]], device=device).unsqueeze(0).expand(
            frame_embs.size(0), 1, 3
        )

        video_emb = temporal(frame_embs, mask, positions=positions)  # [B, 512]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_emb = bert(input_ids, attention_mask)  # [B, 512]

        all_image_embs.append(video_emb.cpu())
        all_text_embs.append(text_emb.cpu())
        if image_ids is not None:
            all_image_ids.extend(image_ids)

    image_embs = torch.cat(all_image_embs)
    text_embs = torch.cat(all_text_embs)

    print(f"\n--- Flickr30k test (BERT + TemporalTransformer) ---")
    recall = compute_recall_metrics(image_embs, text_embs, all_image_ids, ks=(1, 5, 10))
    recall["i2t_medR"] = _compute_median_rank(image_embs, text_embs, all_image_ids, direction="i2t")
    recall["t2i_medR"] = _compute_median_rank(image_embs, text_embs, all_image_ids, direction="t2i")
    print("  " + format_results(recall, prefix="Flickr30k test"))

    # ── Also compute CLIP-zero-shot-style baseline: no temporal (direct image emb) ──
    # This tells us whether temporal is actively hurting or helping vs raw CLIP emb
    print("\n--- Flickr30k test (raw CLIP embedding, no temporal) ---")
    raw_clip = torch.cat([b.unsqueeze(0) for b in [x[0] for x in [dataset[i] for i in range(0, len(dataset), 5)]]])
    # Each image appears 5 times, we need to reconstruct
    # Actually, let's just use the input embeddings directly from the loader
    # Re-run with raw embs
    all_raw_embs = []
    all_raw_ids = []
    dataset2 = FlickrDataset(flickr_root, "test", flickr_cache, annotation_file=ann_file)
    loader2 = torch.utils.data.DataLoader(
        dataset2,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    for batch in tqdm(loader2, desc="Raw CLIP", leave=False):
        image_emb, input_ids, attention_mask, *rest = batch
        image_ids = rest[0] if rest else None
        all_raw_embs.append(image_emb)
        if image_ids:
            all_raw_ids.extend(image_ids)
    raw_embs = torch.cat(all_raw_embs)
    raw_recall = compute_recall_metrics(raw_embs, text_embs, all_raw_ids, ks=(1, 5, 10))
    raw_recall["i2t_medR"] = _compute_median_rank(raw_embs, text_embs, all_raw_ids, direction="i2t")
    raw_recall["t2i_medR"] = _compute_median_rank(raw_embs, text_embs, all_raw_ids, direction="t2i")
    print("  " + format_results(raw_recall, prefix="Flickr30k test (raw CLIP)"))

    print("\nDone.")


if __name__ == "__main__":
    main()
