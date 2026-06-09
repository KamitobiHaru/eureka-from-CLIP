"""
Comprehensive evaluation for the Eureka-from-CLIP report.

Evaluates:
  1. CLIP zero-shot (ViT-B/32, OpenAI) — actual CLIP text encoder
  2. BERT (uncased) English Flickr30k
  3. Multilingual BERT English + Chinese Flickr30k

Usage:
    python scripts/run_all_evaluations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import torch
import yaml
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.data.coco_dataset import make_collate_fn
from src.data.flickr_dataset import FlickrDataset
from src.data.flickr_zh_dataset import FlickrZhDataset
from src.models import BertEncoder
from src.training import SymmetricInfoNCE
from src.training.evaluation import compute_recall_metrics


def evaluate_clip_baseline(cfg, device="cuda"):
    """CLIP zero-shot: use CLIP's own text encoder to encode captions,
    load precomputed CLIP vision embeddings from cache."""
    print("\n" + "=" * 60)
    print("CLIP Zero-shot on Flickr30k test (1K)")
    print("=" * 60)

    # Use project's CLIPEncoder to load the correct checkpoint
    from clip_search.encoder import CLIPEncoder, _find_checkpoint
    clip_model_type = cfg.get("clip", {}).get("model", "openai")
    ckpt_path = _find_checkpoint(clip_model_type)
    encoder = CLIPEncoder(model_type=clip_model_type, device=device)
    clip_model = encoder.model
    clip_tokenizer = encoder.tokenizer

    flickr_cfg = cfg.get("flickr", {})
    flickr_root = Path(flickr_cfg["root"])
    ann_file = flickr_root / flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv")
    image_cache = Path(flickr_cfg["embedding_cache"])

    # Parse annotations
    with open(ann_file) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["split"].strip() == "test"]

    all_image_embs = []
    all_text_embs = []
    all_image_ids = []

    # Collect unique stems and their captions
    stem_to_captions = {}
    for r in rows:
        stem = Path(r["filename"].strip()).stem
        caps = json.loads(r["raw"])
        stem_to_captions[stem] = caps

    for stem, caps in tqdm(stem_to_captions.items(), desc="CLIP encoding"):
        npy_path = image_cache / f"{stem}.npy"
        if not npy_path.exists():
            continue
        img_emb = torch.from_numpy(np.load(npy_path)).float()
        text_tokens = clip_tokenizer(caps).to(device)
        with torch.no_grad():
            text_embs = clip_model.encode_text(text_tokens)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        for cap_idx in range(5):
            all_image_embs.append(img_emb)
            all_text_embs.append(text_embs[cap_idx].cpu())
            all_image_ids.append(f"flickr_{stem}")

    img_t = torch.stack(all_image_embs)
    txt_t = torch.stack(all_text_embs)
    recall = compute_recall_metrics(img_t, txt_t, all_image_ids, ks=(1, 5, 10))
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {k}: {recall[k]:.2f}")
    return recall


def evaluate_bert(model, loader, device, temperature=0.07):
    """Evaluate BERT on a given DataLoader, return recall dict."""
    model.eval()
    loss_fn = SymmetricInfoNCE(temperature=temperature)
    total_loss = 0.0
    all_img = []
    all_txt = []
    all_ids = []
    for batch in tqdm(loader, desc="BERT eval", leave=False):
        image_emb, input_ids, attention_mask, *rest = batch
        image_ids = rest[0] if rest else None
        image_emb = image_emb.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            text_emb = model(input_ids, attention_mask)
        loss = loss_fn(image_emb, text_emb)
        total_loss += loss.item()
        all_img.append(image_emb.cpu())
        all_txt.append(text_emb.cpu())
        if image_ids:
            all_ids.extend(image_ids)

    avg_loss = total_loss / len(loader)
    results = {"val_loss": avg_loss}
    if all_ids:
        results.update(compute_recall_metrics(
            torch.cat(all_img), torch.cat(all_txt), all_ids, ks=(1, 5, 10)
        ))
    return results


def load_bert_model(cfg, device):
    """Load BERT model from config."""
    te_cfg = cfg.get("text_encoder", {})
    bert_path = te_cfg.get("bert_model_path") or cfg["model"]["bert_model_path"]
    lora_cfg = cfg.get("lora", {})
    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)
    return model, bert_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}

    # ─── 1. CLIP baseline ──────────────────────────────────────────
    with open("config/default3.yaml") as f:
        cfg = yaml.safe_load(f)
    results["clip"] = evaluate_clip_baseline(cfg, device)

    # ─── 2. BERT uncased (best checkpoint: default3 config) ──────────
    print("\n" + "=" * 60)
    print("BERT uncased (default3) on Flickr30k test")
    print("=" * 60)
    cfg_uncased = cfg  # reuse default3
    bert_path = cfg_uncased["model"]["bert_model_path"]
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    collate_fn = make_collate_fn(tokenizer)

    flickr_cfg = cfg_uncased.get("flickr", {})
    en_dataset = FlickrDataset(
        flickr_cfg["root"], "test",
        flickr_cfg["embedding_cache"],
        flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
    )
    en_loader = DataLoader(
        en_dataset, batch_size=128, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True,
    )

    model, _ = load_bert_model(cfg_uncased, device)
    ckpt = torch.load("../weights/bert_epoch28_t2i59.9.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: bert_epoch28_t2i59.9.pt")
    results["bert_en"] = evaluate_bert(model, en_loader, device, cfg_uncased["training"]["temperature"])
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {k}: {results['bert_en'][k]:.2f}")

    # ─── 3. Multilingual BERT: English + Chinese Flickr30k ───────────
    print("\n" + "=" * 60)
    print("Multilingual BERT on Flickr30k (English + Chinese)")
    print("=" * 60)
    with open("config/default_multilingual.yaml") as f:
        cfg_multi = yaml.safe_load(f)

    bert_path_multi = cfg_multi["model"]["bert_model_path"]
    tokenizer_multi = BertTokenizer.from_pretrained(bert_path_multi, local_files_only=True)
    collate_fn_multi = make_collate_fn(tokenizer_multi)

    flickr_cfg_multi = cfg_multi.get("flickr", {})
    en_dataset_multi = FlickrDataset(
        flickr_cfg_multi["root"], "test",
        flickr_cfg_multi["embedding_cache"],
        flickr_cfg_multi.get("annotation_file", "flickr_annotations_30k.csv"),
    )
    en_loader_multi = DataLoader(
        en_dataset_multi, batch_size=128, shuffle=False, num_workers=4,
        collate_fn=collate_fn_multi, pin_memory=True,
    )

    # Chinese dataset (use absolute path to avoid root-relative resolution)
    zh_file_abs = str(Path("./data/flickr30k/flickr_test_zh.json").resolve())
    zh_dataset = FlickrZhDataset(
        flickr_cfg_multi["root"], "test",
        flickr_cfg_multi["embedding_cache"],
        zh_file=zh_file_abs,
        annotation_file=flickr_cfg_multi.get("annotation_file", "flickr_annotations_30k.csv"),
    )
    zh_loader = DataLoader(
        zh_dataset, batch_size=128, shuffle=False, num_workers=4,
        collate_fn=collate_fn_multi, pin_memory=True,
    )

    model_multi, _ = load_bert_model(cfg_multi, device)
    best_ckpt_path = "../weights/Multilingual_r8_weight0.75_uniformity2/bert_epoch28_t2i56.0.pt"
    ckpt_multi = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model_multi.load_state_dict(ckpt_multi["model_state_dict"])
    print(f"Loaded: {Path(best_ckpt_path).name}")

    temp = cfg_multi["training"]["temperature"]
    results["multi_en"] = evaluate_bert(model_multi, en_loader_multi, device, temp)
    results["multi_zh"] = evaluate_bert(model_multi, zh_loader, device, temp)

    print("\n--- Multilingual English ---")
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {k}: {results['multi_en'][k]:.2f}")
    print("--- Multilingual Chinese ---")
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {k}: {results['multi_zh'][k]:.2f}")

    # ─── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY — Flickr30k Recall@K")
    print("=" * 90)
    header = f"{'Method':<30} {'i2t_R@1':<10} {'i2t_R@5':<10} {'i2t_R@10':<10} {'t2i_R@1':<10} {'t2i_R@5':<10} {'t2i_R@10':<10}"
    print(header)
    print("-" * 90)
    rows_summary = [
        ("CLIP ViT-B/32 (zero-shot)", "clip"),
        ("BERT uncased (EN, Ours)", "bert_en"),
        ("Multilingual BERT (EN)", "multi_en"),
        ("Multilingual BERT (ZH)", "multi_zh"),
    ]
    for label, key in rows_summary:
        r = results[key]
        vals = " ".join(f"{r.get(m, 0):<10.2f}" for m in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"])
        print(f"{label:<30} {vals}")
    print("=" * 90)


if __name__ == "__main__":
    main()
