"""
Precompute BERT text embeddings for all captions used in temporal training.

Since BERT is frozen during Phase-1 temporal training, we can run it once
offline and save the L2-normalized text embeddings.  This eliminates the
~110M-parameter BERT forward pass from every training step.

Output: single .pt file per split, shape (N, 512), saved alongside the
CLIP image embeddings under ``{embedding_cache}_text/``.

Usage:
    python scripts/precompute_text_embeddings.py \
        --config config/default.yaml \
        --bert_checkpoint /path/to/bert_best.pt \
        [--device cuda:0]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import BertEncoder


@torch.no_grad()
def precompute_split(bert_model, tokenizer, captions, batch_size, device):
    """Run BERT on a list of captions, return (N, 512) tensor."""
    all_embs = []
    pbar = tqdm(total=len(captions), desc="BERT enc", unit="cap", leave=False)
    for i in range(0, len(captions), batch_size):
        batch_caps = captions[i:i + batch_size]
        tokens = tokenizer(
            list(batch_caps),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # BertEncoder.forward == BERT → [CLS] → ProjectionHead → L2-norm
        embs = bert_model(input_ids, attention_mask)  # (B, 512), float32
        all_embs.append(embs.cpu())
        pbar.update(len(batch_caps))
    pbar.close()
    return torch.cat(all_embs, dim=0)


def get_coco_captions(coco_root, split, annotations_dir):
    ann_file = Path(coco_root) / annotations_dir / f"captions_{split}.json"
    with open(ann_file) as f:
        data = json.load(f)
    return [ann["caption"] for ann in data["annotations"]], split


def get_flickr_captions(flickr_root, split, annotation_file):
    import csv
    ann_path = Path(flickr_root) / annotation_file
    with open(ann_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    captions = []
    for r in rows:
        if r["split"].strip() != split:
            continue
        import json as _json
        for cap in _json.loads(r["raw"]):
            captions.append(cap)
    return captions, split


def main():
    parser = argparse.ArgumentParser(description="Precompute BERT text embeddings.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--bert_checkpoint", required=True,
                        help="Path to trained BERT checkpoint from train_bert.py")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Captions per batch during precomputation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # ── Load frozen BERT ──────────────────────────────────────────
    te_cfg = cfg.get("text_encoder", {})
    bert_path = te_cfg.get("bert_model_path") or cfg["model"].get("bert_model_path")
    lora_cfg = cfg.get("lora", {})

    print("Loading BERT model (frozen)...")
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
        initial_temperature=cfg["training"]["temperature"],
    ).to(device)

    if not Path(args.bert_checkpoint).exists():
        print(f"Checkpoint not found: {args.bert_checkpoint}")
        sys.exit(1)
    print(f"Loading checkpoint: {args.bert_checkpoint}")
    ckpt = torch.load(args.bert_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  BERT frozen.")

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    # ── Define splits to precompute ──────────────────────────────
    ann_dir = cfg["data"].get("annotations_dir", "annotations")
    coco_root = cfg["data"]["coco_root"]
    base_cache = cfg["data"]["embedding_cache"]
    text_cache_dir = Path(base_cache + "_text")
    text_cache_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("coco", "train2017", get_coco_captions, [coco_root, "train2017", ann_dir]),
        ("coco", "val2017", get_coco_captions, [coco_root, "val2017", ann_dir]),
    ]

    flickr_cfg = cfg.get("flickr", {})
    if flickr_cfg.get("root"):
        ann_file = flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv")
        for split_name in ["train", "test"]:
            splits.append(("flickr", split_name, get_flickr_captions,
                          [flickr_cfg["root"], split_name, ann_file]))

    # ── Precompute ───────────────────────────────────────────────
    for prefix, split_name, getter_fn, getter_args in splits:
        out_path = text_cache_dir / f"{prefix}_{split_name}.pt"
        if out_path.exists():
            existing = torch.load(out_path, weights_only=True)
            print(f"  [{prefix} {split_name}] already cached: {existing.shape[0]} embeds → skipping")
            continue

        print(f"  [{prefix} {split_name}] loading captions...")
        captions, _ = getter_fn(*getter_args)
        print(f"    {len(captions)} captions, precomputing...")

        embs = precompute_split(model, tokenizer, captions, args.batch_size, device)
        torch.save(embs, out_path)
        print(f"    → saved {out_path}  ({embs.shape[0]}×{embs.shape[1]})")

    print(f"\nAll done. Text embeddings in: {text_cache_dir}")


if __name__ == "__main__":
    main()
