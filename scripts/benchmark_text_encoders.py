"""
Benchmark: compare CLIP text encoder vs. trained BERT on MSR-VTT, MSVD, Flickr30k.

Both use the same frozen CLIP vision encoder for video/image embeddings.
Only the text encoder changes — this isolates text-encoding quality.

Usage:
    python scripts/benchmark_text_encoders.py \
        --bert_checkpoint ./checkpoints/bert_epoch26_t2i60.5.pt
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import open_clip
from clip_search.encoder import CLIPEncoder, _find_checkpoint
from src.models import BertEncoder
from src.training.evaluation import compute_recall_metrics


# ── config ──────────────────────────────────────────────────────────────────

MSRVTT_ROOT = Path("./datasets/MSR-VTT")
MSRVTT_ANN = MSRVTT_ROOT / "msrvtt_test_1k.json"
MSRVTT_FRAMES = Path("./data/msrvtt/clip_keyframes")

MSVD_ANN = Path("./data/msvd/msvd_all.json")
MSVD_FRAMES = Path("./data/msvd/clip_keyframes")

FLICKR_ROOT = Path("./dataset_annotation")
FLICKR_ANN = FLICKR_ROOT / "flickr_annotations_30k.csv"

BERT_MODEL_PATH = Path("./models/bert-base-uncased")
NUM_FRAMES = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── helpers ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def mean_pool_video(frame_embs, frame_mask):
    """frame_embs: (V, T, 512), frame_mask: (V, T) True=padded → (V, 512) L2-norm"""
    mask = (~frame_mask).unsqueeze(-1).float()
    video_emb = (frame_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(video_emb, dim=-1)


def load_video_frames(frame_cache: Path, video_ids, max_frames=NUM_FRAMES):
    """load per-video .npy frame embeddings, pad to (V, max_frames, 512)."""
    V = len(video_ids)
    frame_embs = torch.zeros(V, max_frames, 512)
    frame_mask = torch.ones(V, max_frames, dtype=torch.bool)
    valid_counts = []
    for i, vid in enumerate(video_ids):
        npy = frame_cache / f"{vid}.npy"
        if not npy.exists():
            valid_counts.append(0)
            continue
        arr = np.load(npy)
        N = min(arr.shape[0], max_frames)
        frame_embs[i, :N] = torch.from_numpy(arr[:N])
        frame_mask[i, :N] = False
        valid_counts.append(N)
    return frame_embs, frame_mask, valid_counts


@torch.no_grad()
def encode_texts_clip(texts, clip_model, tokenizer, device, batch_size=128):
    """Encode list of strings with CLIP text encoder → (N, 512) L2-norm."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu())
        if i % (batch_size * 100) == 0:
            torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_texts_bert(texts, bert_model, tokenizer, device, batch_size=128):
    """Encode list of strings with BERT encoder → (N, 512) L2-norm."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=77,
                          return_tensors="pt").to(device)
        emb = bert_model(tokens["input_ids"], tokens["attention_mask"])
        all_embs.append(emb.cpu())
        if i % (batch_size * 100) == 0:
            torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


# ── dataset evaluators ──────────────────────────────────────────────────────

def evaluate_msrvtt(bert_model, clip_text_model, clip_tokenizer, device, method_key="BERT"):
    """MSR-VTT 1K-A test set: 1000 videos × 1 caption each."""
    print("\n" + "=" * 55)
    print("  MSR-VTT 1K-A test")
    print("=" * 55)

    with open(MSRVTT_ANN) as f:
        ann = json.load(f)
    video_ids = [e["video_id"] for e in ann]
    captions = [e["caption"] for e in ann]
    print(f"  Videos: {len(video_ids)}, Captions: {len(captions)}")

    # load video frames → mean pool → keep on CPU
    frame_embs, frame_mask, vc = load_video_frames(MSRVTT_FRAMES, video_ids)
    print(f"  Frames loaded: {sum(1 for c in vc if c > 0)}/{len(video_ids)} videos have frames")
    video_emb = mean_pool_video(frame_embs, frame_mask)

    # CLIP text
    text_clip = encode_texts_clip(captions, clip_text_model, clip_tokenizer, device)
    r_clip = compute_recall_metrics(video_emb, text_clip, video_ids)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  CLIP  {k}: {r_clip.get(k, 0):.2f}")

    # BERT/MLP text
    torch.cuda.empty_cache()
    text_bert = encode_texts_bert(captions, bert_model, BERT_TOKENIZER, device)
    r_bert = compute_recall_metrics(video_emb, text_bert, video_ids)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {method_key:<5} {k}: {r_bert.get(k, 0):.2f}")

    return {"CLIP": r_clip, method_key: r_bert}


def evaluate_msvd(bert_model, clip_text_model, clip_tokenizer, device, method_key="BERT"):
    """MSVD zero-shot: 1970 videos, each with ~20 captions."""
    print("\n" + "=" * 55)
    print("  MSVD")
    print("=" * 55)

    with open(MSVD_ANN) as f:
        ann = json.load(f)

    # Build (video_id, caption) pairs — keep on CPU
    pairs = []
    for entry in ann:
        vid = entry["video_id"]
        for cap in entry["captions"]:
            if cap and cap.strip():
                pairs.append((vid, cap.strip()))
    print(f"  Unique videos: {len(ann)}, total caption-pairs: {len(pairs)}")

    unique_ids = list(dict.fromkeys(vid for vid, _ in pairs))
    frame_embs, frame_mask, vc = load_video_frames(MSVD_FRAMES, unique_ids)
    print(f"  Frames loaded: {sum(1 for c in vc if c > 0)}/{len(unique_ids)} videos have frames")
    video_emb = mean_pool_video(frame_embs, frame_mask)  # (U, 512) CPU

    id_to_idx = {vid: i for i, vid in enumerate(unique_ids)}
    video_ids = [vid for vid, _ in pairs]
    all_captions = [cap for _, cap in pairs]

    # Build paired video embeddings on CPU (avoids large GPU allocation)
    vid_indices = torch.tensor([id_to_idx[vid] for vid, _ in pairs])
    pair_video_emb = video_emb[vid_indices]  # (N, 512) CPU

    # CLIP text
    print("  Encoding captions with CLIP text encoder...")
    text_clip = encode_texts_clip(all_captions, clip_text_model, clip_tokenizer,
                                  device, batch_size=128)
    torch.cuda.empty_cache()
    print("  Computing recall (CLIP)...")
    r_clip = compute_recall_metrics(pair_video_emb, text_clip, video_ids)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  CLIP  {k}: {r_clip.get(k, 0):.2f}")

    # BERT/MLP text
    print(f"  Encoding captions with {method_key} text encoder...")
    text_bert = encode_texts_bert(all_captions, bert_model, BERT_TOKENIZER,
                                  device, batch_size=128)
    torch.cuda.empty_cache()
    print(f"  Computing recall ({method_key})...")
    r_bert = compute_recall_metrics(pair_video_emb, text_bert, video_ids)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {method_key:<5} {k}: {r_bert.get(k, 0):.2f}")

    return {"CLIP": r_clip, method_key: r_bert}


def evaluate_flickr30k(bert_model, clip_model, clip_tokenizer, device, method_key="BERT"):
    """Flickr30k test (1K): 1000 images × 5 captions each.

    Uses precomputed CLIP image embeddings from data/flickr30k/clip_embeddings/
    and CLIP text embeddings from data/flickr30k/clip_embeddings_text/ for the
    CLIP baseline.
    """
    print("\n" + "=" * 55)
    print("  Flickr30k test (1K)")
    print("=" * 55)

    ann_file = FLICKR_ROOT / "flickr_annotations_30k.csv"
    import csv
    with open(ann_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pairs = []
    for r in rows:
        if r["split"].strip() != "test":
            continue
        stem = Path(r["filename"].strip()).stem
        captions = json.loads(r["raw"])
        for cap in captions:
            pairs.append((stem, cap))
    print(f"  Test pairs: {len(pairs)} ({len(pairs)//5} images × 5 captions)")

    # Load precomputed CLIP image embeddings
    img_cache = Path("./data/flickr30k/clip_embeddings")
    unique_stems = list(dict.fromkeys(stem for stem, _ in pairs))
    print(f"  Loading {len(unique_stems)} precomputed image embeddings...")
    stem_to_emb = {}
    for stem in tqdm(unique_stems, desc="Images"):
        npy = img_cache / f"{stem}.npy"
        if npy.exists():
            stem_to_emb[stem] = np.load(npy)
    print(f"  Loaded {len(stem_to_emb)}/{len(unique_stems)} images")

    # Build arrays
    all_image_embs = []
    all_text_clip = []
    all_text_bert = []
    all_image_ids = []

    # CLIP text: load precomputed from text cache
    text_cache_path = Path("./data/flickr30k/clip_embeddings_text/flickr_test.pt")
    if text_cache_path.exists():
        print("  Loading precomputed CLIP text embeddings...")
        cached = torch.load(text_cache_path, weights_only=True)
        print(f"  Cached type: {type(cached)}")
        if isinstance(cached, dict):
            print(f"  Cached keys: {list(cached.keys())[:3]}")
        elif isinstance(cached, torch.Tensor):
            print(f"  Cached shape: {cached.shape}")
    else:
        print("  No precomputed CLIP text embeddings found, encoding on the fly...")

    captions_batch = []
    for stem, cap in tqdm(pairs, desc="Processing"):
        img_emb = stem_to_emb.get(stem)
        if img_emb is None:
            continue
        all_image_embs.append(img_emb)
        all_image_ids.append(f"flickr_{stem}")
        captions_batch.append(cap)

        if len(captions_batch) >= 64:
            text_clip = encode_texts_clip(captions_batch, clip_model, clip_tokenizer, device)
            all_text_clip.append(text_clip.cpu())
            text_bert = encode_texts_bert(captions_batch, bert_model, BERT_TOKENIZER, device)
            all_text_bert.append(text_bert.cpu())
            captions_batch = []

    if captions_batch:
        text_clip = encode_texts_clip(captions_batch, clip_model, clip_tokenizer, device)
        all_text_clip.append(text_clip.cpu())
        text_bert = encode_texts_bert(captions_batch, bert_model, BERT_TOKENIZER, device)
        all_text_bert.append(text_bert.cpu())

    image_embs = torch.from_numpy(np.stack(all_image_embs))
    text_clip_all = torch.cat(all_text_clip, dim=0)
    text_bert_all = torch.cat(all_text_bert, dim=0)

    print(f"  Image embeddings: {image_embs.shape}")
    print(f"  CLIP text:        {text_clip_all.shape}")
    print(f"  {method_key} text:        {text_bert_all.shape}")

    r_clip = compute_recall_metrics(image_embs, text_clip_all, all_image_ids)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  CLIP  {k}: {r_clip.get(k, 0):.2f}")

    r_bert = compute_recall_metrics(image_embs, text_bert_all, all_image_ids)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {method_key:<5} {k}: {r_bert.get(k, 0):.2f}")

    return {"CLIP": r_clip, method_key: r_bert}


# ── global bert tokenizer ──────────────────────────────────────────────────
from transformers import BertTokenizer
BERT_TOKENIZER = BertTokenizer.from_pretrained(str(BERT_MODEL_PATH))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark CLIP vs BERT text encoder")
    parser.add_argument("--bert_checkpoint", default="./checkpoints/bert_epoch26_t2i60.5.pt")
    parser.add_argument("--encoder_type", default="bert", choices=["bert", "mlp"],
                        help="'bert' = BertEncoder+LoRA (default), 'mlp' = MLPBertEncoder (ablation)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", nargs="+", default=["msrvtt", "msvd", "flickr"],
                        choices=["msrvtt", "msvd", "flickr"])
    args = parser.parse_args()

    device = args.device or DEVICE
    print(f"Device: {device}")

    # ── Load CLIP (shared vision + text encoder) ──
    print("\n── Loading CLIP (openai ViT-B/32) ──")
    ckpt = _find_checkpoint("openai")
    if ckpt:
        print(f"  Checkpoint: {ckpt}")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt)
    clip_model = clip_model.to(device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ── Load BERT / MLP ──
    if args.encoder_type == "mlp":
        from src.models import MLPBertEncoder
        print(f"\n── Loading MLPBertEncoder ──")
        bert_model = MLPBertEncoder(
            model_path=str(BERT_MODEL_PATH),
            embed_dim=512,
            mlp_hidden=704,
        ).to(device)
    else:
        print("\n── Loading BERT ──")
        lora_cfg = {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.1,
                    "target_modules": ["key", "query", "value", "output.dense"]}
        bert_model = BertEncoder(
            model_path=str(BERT_MODEL_PATH),
            embed_dim=512,
            lora_cfg=lora_cfg,
        ).to(device)
    bert_model.eval()

    ckpt_path = args.bert_checkpoint
    print(f"  Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # If the checkpoint contains a full 'model_state_dict' (from Trainer.save_checkpoint), use it
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = bert_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print(f"  Model params: {sum(p.numel() for p in bert_model.parameters()):,}")

    # ── Run evaluations ──
    method_name = "MLP" if args.encoder_type == "mlp" else "BERT"
    all_results = {}

    if "msrvtt" in args.datasets:
        all_results["MSR-VTT"] = evaluate_msrvtt(bert_model, clip_model, clip_tokenizer, device, method_key=method_name)
        torch.cuda.empty_cache()
    if "msvd" in args.datasets:
        all_results["MSVD"] = evaluate_msvd(bert_model, clip_model, clip_tokenizer, device, method_key=method_name)
        torch.cuda.empty_cache()
    if "flickr" in args.datasets:
        all_results["Flickr30k"] = evaluate_flickr30k(bert_model, clip_model, clip_tokenizer, device, method_key=method_name)
        torch.cuda.empty_cache()

    # ── Summary table ──
    print("\n" + "=" * 60)
    print(f"  SUMMARY: CLIP text encoder vs {method_name}")
    print("=" * 60)
    header = f"  {'Dataset':<14} {'Method':<8} {'i2t_R@1':>8} {'i2t_R@5':>8} {'i2t_R@10':>8} {'t2i_R@1':>8} {'t2i_R@5':>8} {'t2i_R@10':>8}"
    print(header)
    print("  " + "-" * (len(header)-2))
    for ds_name, ds_results in all_results.items():
        if ds_results is None:
            continue
        for method in ["CLIP", method_name]:
            r = ds_results.get(method, {})
            parts = [
                f"{ds_name:<14}" if method == "CLIP" else "",
                f"{method:<8}",
                f'{r.get("i2t_R@1", 0):>8.1f}',
                f'{r.get("i2t_R@5", 0):>8.1f}',
                f'{r.get("i2t_R@10", 0):>8.1f}',
                f'{r.get("t2i_R@1", 0):>8.1f}',
                f'{r.get("t2i_R@5", 0):>8.1f}',
                f'{r.get("t2i_R@10", 0):>8.1f}',
            ]
            print("  " + " ".join(parts))
    print("=" * 60)


if __name__ == "__main__":
    main()
