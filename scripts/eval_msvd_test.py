"""
Evaluate MSVD test-set recall (filtered by test_list.txt only).

Three models:
  1) CLIP native (ViT-B/32 text encoder)
  2) Ours + Stack LoRA  (r=2, alpha=4, e25)
  3) Ours (No Stack)    (r=8, alpha=16, e28)

Usage:
    python scripts/eval_msvd_test.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import open_clip
from clip_search.encoder import _find_checkpoint
from src.models import BertEncoder
from src.training.evaluation import compute_recall_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_LIST = Path("./data/msvd/test_list.txt")
MSVD_ANN = Path("./data/msvd/msvd_all.json")
MSVD_FRAMES = Path("./data/msvd/clip_keyframes")
BERT_MODEL_PATH = Path("./models/bert-base-uncased")
NUM_FRAMES = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "Ours (Stack LoRA)": {
        "path": "/data2/zsy/weights/clip/bert_domain_best_e25_msvd17.9.pt",
        "lora_r": 2,
        "lora_alpha": 4,
    },
    "Ours (No Stack)": {
        "path": "/data2/zsy/weights/clip/bert_domain_best_e28_msvd17.5.pt",
        "lora_r": 8,
        "lora_alpha": 16,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_test_video_ids():
    with open(TEST_LIST) as f:
        return [l.strip() for l in f if l.strip()]


def filter_msvd_by_test(ann_path, test_ids):
    with open(ann_path) as f:
        ann = json.load(f)
    test_set = set(test_ids)
    filtered = [e for e in ann if e["video_id"] in test_set]
    print(f"  MSVD total: {len(ann)} videos, test set: {len(filtered)} videos")
    return filtered


def build_pairs(ann):
    pairs = []
    for entry in ann:
        vid = entry["video_id"]
        for cap in entry["captions"]:
            if cap and cap.strip():
                pairs.append((vid, cap.strip()))
    return pairs


def load_video_frames(frame_cache, video_ids, max_frames=NUM_FRAMES):
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


def mean_pool_video(frame_embs, frame_mask):
    mask = (~frame_mask).unsqueeze(-1).float()
    video_emb = (frame_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(video_emb, dim=-1)


@torch.no_grad()
def encode_texts_clip(texts, clip_model, tokenizer, device, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_texts_bert(texts, bert_model, tokenizer, device, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=77,
                          return_tensors="pt").to(device)
        emb = bert_model(tokens["input_ids"], tokens["attention_mask"])
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MSVD test-set recall")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or DEVICE
    print(f"Device: {device}\n")

    # ── Data ──
    print("── Loading MSVD test set ──")
    test_ids = load_test_video_ids()
    filtered_ann = filter_msvd_by_test(MSVD_ANN, test_ids)
    pairs = build_pairs(filtered_ann)
    print(f"  Caption pairs: {len(pairs)}")

    unique_ids = list(dict.fromkeys(vid for vid, _ in pairs))
    frame_embs, frame_mask, vc = load_video_frames(MSVD_FRAMES, unique_ids)
    n_valid = sum(1 for c in vc if c > 0)
    print(f"  Keyframes loaded: {n_valid}/{len(unique_ids)} videos")
    video_emb = mean_pool_video(frame_embs, frame_mask)  # (U, 512) CPU

    id_to_idx = {vid: i for i, vid in enumerate(unique_ids)}
    video_ids = [vid for vid, _ in pairs]
    vid_indices = torch.tensor([id_to_idx[vid] for vid, _ in pairs])
    pair_video_emb = video_emb[vid_indices]  # (N, 512) CPU

    captions = [cap for _, cap in pairs]

    # ── 1. CLIP baseline ──
    print("\n── Loading CLIP (openai ViT-B/32) ──")
    ckpt_path = _find_checkpoint("openai")
    print(f"  Checkpoint: {ckpt_path}")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt_path)
    clip_model = clip_model.to(device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    print("\n  ── CLIP baseline ──")
    text_clip = encode_texts_clip(captions, clip_model, clip_tokenizer, device)
    r_clip = compute_recall_metrics(pair_video_emb, text_clip, video_ids)
    clip_results = {k: r_clip.get(k, 0) for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10",
                                                    "t2i_R@1", "t2i_R@5", "t2i_R@10"]}
    for k, v in clip_results.items():
        print(f"  CLIP  {k}: {v:.2f}")
    torch.cuda.empty_cache()

    del clip_model

    # ── 2. BERT-based models ──
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(str(BERT_MODEL_PATH))

    all_results = {"CLIP": clip_results}

    for method_name, cfg in CHECKPOINTS.items():
        print(f"\n  ── {method_name} ──")
        print(f"  Checkpoint: {cfg['path']}")
        print(f"  LoRA r={cfg['lora_r']}, alpha={cfg['lora_alpha']}")

        lora_cfg = {
            "enabled": True,
            "r": cfg["lora_r"],
            "alpha": cfg["lora_alpha"],
            "dropout": 0.1,
            "target_modules": ["key", "query", "value", "output.dense"],
        }

        bert_model = BertEncoder(
            model_path=str(BERT_MODEL_PATH),
            embed_dim=512,
            lora_cfg=lora_cfg,
        ).to(device)
        bert_model.eval()

        ckpt = torch.load(cfg["path"], map_location=device, weights_only=True)
        missing, unexpected = bert_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        print(f"  Encoding captions...")
        text_bert = encode_texts_bert(captions, bert_model, bert_tokenizer, device)
        torch.cuda.empty_cache()

        print(f"  Computing recall...")
        r_bert = compute_recall_metrics(pair_video_emb, text_bert, video_ids)
        bert_results = {k: r_bert.get(k, 0) for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10",
                                                        "t2i_R@1", "t2i_R@5", "t2i_R@10"]}
        all_results[method_name] = bert_results

        for k, v in bert_results.items():
            print(f"  {method_name:<18} {k}: {v:.2f}")

        del bert_model
        torch.cuda.empty_cache()

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  MSVD Test Set Results (test_list.txt, {:d} videos)".format(len(unique_ids)))
    print("=" * 70)
    header = (
        f"  {'Method':<22} {'i2t_R@1':>8} {'i2t_R@5':>8} {'i2t_R@10':>8}"
        f" {'t2i_R@1':>8} {'t2i_R@5':>8} {'t2i_R@10':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in ["CLIP", "Ours (Stack LoRA)", "Ours (No Stack)"]:
        r = all_results.get(method, {})
        parts = [f"{method:<22}"]
        for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
            parts.append(f"{r.get(k, 0):>8.1f}")
        print("  " + " ".join(parts))
    print("=" * 70)


if __name__ == "__main__":
    main()
