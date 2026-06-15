"""
Evaluate CLIP zero-shot / Ours no-stack / Ours stack on VRIPT.

Usage:
    # CLIP baseline only
    python scripts/evaluate_vript.py --mode clip --vript_ann data/vript/annotations.json \
        --vript_frames data/vript/clip_keyframes --output results/vript_clip.json

    # Full three-way comparison (fair truncation — all to 77 tokens)
    python scripts/evaluate_vript.py --mode clip,nostack,stack \
        --vript_ann data/vript/annotations.json \
        --vript_frames data/vript/clip_keyframes \
        --truncation fair \
        --nostack_checkpoint /run/media/occccce/D/checkpoints/bert_domain_1e-5_batch128/bert_domain_best_e28_msvd17.5.pt \
        --stack_checkpoint /run/media/occccce/D/checkpoints/bert_domain_stack_lora_1e-5_batch128/bert_domain_best_e25_msvd17.9.pt \
        --pretrain_checkpoint /run/media/occccce/D/checkpoints/bert_epoch26_t2i60.5.pt \
        --output results/vript_ablation.json

    # Demonstrate CLIP's long-text weakness (BERT sees full caption)
    python scripts/evaluate_vript.py --mode clip,nostack,stack \
        --vript_ann data/vript/annotations.json \
        --vript_frames data/vript/clip_keyframes \
        --truncation clip-only \
        --nostack_checkpoint ... \
        --stack_checkpoint ... \
        --pretrain_checkpoint ... \
        --output results/vript_clip_only_trunc.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import open_clip
from clip_search.encoder import _find_checkpoint
from src.models import BertEncoder
from src.training.evaluation import compute_recall_metrics
from transformers import BertTokenizer

# ── constants ──────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 12
CLIP_MAX_LENGTH = 77          # ViT-B/32 context window
BERT_MAX_LENGTH = 512
TEXT_BATCH_SIZE = 128
ARGS = None


# ── video embedding helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def mean_pool_video(frame_embs: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """frame_embs: (V, T, 512), frame_mask: (V, T) True=padded → (V, 512) L2-norm"""
    mask = (~frame_mask).unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1)
    video_emb = (frame_embs * mask).sum(dim=1) / denom
    return F.normalize(video_emb, dim=-1)


def load_video_frames(frame_dir: Path, clip_ids: list, max_frames: int = NUM_FRAMES):
    """Load per-clip .npy frame embeddings, pad to (V, max_frames, 512).

    Returns:
        frame_embs: (V, T, 512) float32
        frame_mask: (V, T) bool, True = padded
        valid_counts: list of int, actual frame count per clip
    """
    V = len(clip_ids)
    frame_embs = torch.zeros(V, max_frames, 512)
    frame_mask = torch.ones(V, max_frames, dtype=torch.bool)
    valid_counts = []
    missing = 0
    for i, cid in enumerate(clip_ids):
        npy = frame_dir / f"{cid}.npy"
        if not npy.exists():
            missing += 1
            valid_counts.append(0)
            continue
        arr = np.load(npy)
        N = min(arr.shape[0], max_frames)
        frame_embs[i, :N] = torch.from_numpy(arr[:N])
        frame_mask[i, :N] = False
        valid_counts.append(N)
    if missing:
        print(f"  Warning: {missing}/{V} clips missing frame embeddings")
    return frame_embs, frame_mask, valid_counts


# ── text encoding helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts_clip(texts, clip_model, clip_tokenizer, device,
                      batch_size=TEXT_BATCH_SIZE, truncation="fair"):
    """Encode strings with CLIP text encoder → (N, 512) L2-norm.

    CLIP always truncates to 77 tokens (its max context length).
    The ``truncation`` parameter is accepted for interface consistency
    with the BERT encoder but has no effect here (CLIP always truncates).
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # open_clip's SimpleTokenizer does not accept HuggingFace-style kwargs
        tokens = clip_tokenizer(batch).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu())
        if i % (batch_size * 50) == 0:
            torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_texts_bert(texts, bert_model, tokenizer, device,
                      batch_size=TEXT_BATCH_SIZE,
                      bert_max_length=BERT_MAX_LENGTH):
    """Encode strings with BERT encoder → (N, 512) L2-norm."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=bert_max_length,
                           return_tensors="pt").to(device)
        emb = bert_model(tokens["input_ids"], tokens["attention_mask"])
        all_embs.append(emb.cpu())
        if i % (batch_size * 50) == 0:
            torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


# ── model loading ───────────────────────────────────────────────────────────────

def load_clip_model(device):
    """Load frozen CLIP ViT-B/32 (OpenAI)."""
    ckpt = _find_checkpoint("openai")
    print(f"  CLIP checkpoint: {ckpt or '(download)'}")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer


def load_nostack_model(bert_path, checkpoint_path, device):
    """Load BertEncoder with base LoRA (r=8) from no-stack checkpoint."""
    lora_cfg = {
        "enabled": True, "r": 8, "alpha": 16, "dropout": 0.1,
        "target_modules": ["key", "query", "value", "output.dense"],
    }
    bert_model = BertEncoder(
        model_path=bert_path, embed_dim=512, lora_cfg=lora_cfg,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = bert_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        for k in unexpected:
            if "lora" not in k:
                print(f"  Unexpected (non-LoRA) key: {k}")
    bert_model.eval()
    total = sum(p.numel() for p in bert_model.parameters())
    trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    print(f"  No-stack params: {total:,} total, {trainable:,} trainable")
    return bert_model


def load_stack_model(bert_path, pretrain_checkpoint, stack_checkpoint, device):
    """Load BertEncoder with Stack LoRA:
    1. Base LoRA (r=8) initialized from COCO pretrain
    2. Merge base LoRA, add stack LoRA (r=2)
    3. Overlay trained stack checkpoint weights
    """
    lora_cfg = {
        "enabled": True, "r": 8, "alpha": 16, "dropout": 0.1,
        "target_modules": ["key", "query", "value", "output.dense"],
    }
    stack_lora_cfg = {
        "r": 2, "alpha": 4, "dropout": 0.1,
        "target_modules": ["key", "query", "value", "output.dense"],
    }
    print(f"  Loading COCO pretrain: {pretrain_checkpoint}")
    bert_model = BertEncoder(
        model_path=bert_path, embed_dim=512, lora_cfg=lora_cfg,
    ).to(device)

    # Step 1: merge base LoRA (r=8 from pretrain) → add stack (r=2)
    bert_model.load_and_stack_lora(pretrain_checkpoint, stack_lora_cfg)

    # Step 2: overlay the trained stack checkpoint
    print(f"  Loading stack checkpoint: {stack_checkpoint}")
    ckpt = torch.load(stack_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = bert_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        for k in unexpected:
            if "lora" not in k:
                print(f"  Unexpected (non-LoRA) key: {k}")
    bert_model.eval()
    total = sum(p.numel() for p in bert_model.parameters())
    trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    print(f"  Stack params: {total:,} total, {trainable:,} trainable")
    return bert_model


# ── evaluation ──────────────────────────────────────────────────────────────────

def evaluate(annotations, clip_ids, video_emb, args):
    """Run specified text encoders and compute recall."""
    device = torch.device(DEVICE)
    results = {}

    # ── Load CLIP ────────────────────────────────────────────────
    if "clip" in args.mode:
        print("\n── CLIP native ──")
        clip_model, clip_tokenizer = load_clip_model(device)
        clip_text = encode_texts_clip(
            annotations, clip_model, clip_tokenizer, device,
            truncation=args.truncation,
        )
        r_clip = compute_recall_metrics(video_emb, clip_text, clip_ids)
        results["CLIP"] = r_clip
        for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
            print(f"  CLIP   {k}: {r_clip.get(k, 0):.2f}")
        del clip_model, clip_text
        torch.cuda.empty_cache()

    if not ({"nostack", "stack"} & set(args.mode)):
        return results

    # ── Load BERT tokenizer + check text length ──────────────────
    bert_path = args.bert_model
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

    # Determine BERT truncation length
    if args.truncation == "fair":
        bert_max_len = CLIP_MAX_LENGTH   # match CLIP's 77 tokens for fairness
        print(f"\n  Truncation: fair — BERT also truncated to {CLIP_MAX_LENGTH} tokens")
    else:
        bert_max_len = BERT_MAX_LENGTH   # BERT uses full caption (up to 512)
        print(f"\n  Truncation: clip-only — BERT uses full caption ({BERT_MAX_LENGTH} max)")

    # ── Ours: No-stack ────────────────────────────────────────────
    if "nostack" in args.mode:
        print("\n── Ours (no-stack) ──")
        nostack_model = load_nostack_model(bert_path, args.nostack_checkpoint, device)

        # Encode with same truncation as CLIP for fair comparison
        # For CLIP-only truncation, BERT still uses its own tokenizer with max_length
        nostack_text = encode_texts_bert(
            annotations, nostack_model, bert_tokenizer, device,
            bert_max_length=bert_max_len,
        )
        r_nostack = compute_recall_metrics(video_emb, nostack_text, clip_ids)
        results["NoStack"] = r_nostack
        for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
            print(f"  NoStack {k}: {r_nostack.get(k, 0):.2f}")
        del nostack_model, nostack_text
        torch.cuda.empty_cache()

    # ── Ours: Stack ───────────────────────────────────────────────
    if "stack" in args.mode:
        print("\n── Ours (stack) ──")
        stack_model = load_stack_model(
            bert_path, args.pretrain_checkpoint, args.stack_checkpoint, device,
        )
        stack_text = encode_texts_bert(
            annotations, stack_model, bert_tokenizer, device,
            bert_max_length=bert_max_len,
        )
        r_stack = compute_recall_metrics(video_emb, stack_text, clip_ids)
        results["Stack"] = r_stack
        for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
            print(f"  Stack  {k}: {r_stack.get(k, 0):.2f}")
        del stack_model, stack_text
        torch.cuda.empty_cache()

    return results


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP / no-stack / stack on VRIPT"
    )
    parser.add_argument("--vript_ann", default="/run/media/occccce/D/vript_processed/annotations.json",
                        help="VRIPT annotation JSON")
    parser.add_argument("--vript_frames", default="/run/media/occccce/D/vript_processed/clip_keyframes",
                        help="VRIPT CLIP frame embeddings directory")
    parser.add_argument("--mode", nargs="+",
                        default=["clip", "nostack", "stack"],
                        choices=["clip", "nostack", "stack"],
                        help="Which text encoders to evaluate")
    parser.add_argument("--truncation", default="fair",
                        choices=["fair", "clip-only"],
                        help="'fair': all truncated to 77 tokens; "
                             "'clip-only': CLIP truncated to 77, BERT sees full text")
    parser.add_argument("--nostack_checkpoint",
                        default="/run/media/occccce/D/checkpoints/bert_domain_1e-5_batch128/bert_domain_best_e28_msvd17.5.pt")
    parser.add_argument("--stack_checkpoint",
                        default="/run/media/occccce/D/checkpoints/bert_domain_stack_lora_1e-5_batch128/bert_domain_best_e25_msvd17.9.pt")
    parser.add_argument("--pretrain_checkpoint",
                        default="/run/media/occccce/D/checkpoints/bert_epoch26_t2i60.5.pt",
                        help="COCO-pretrained checkpoint for Stack LoRA initialization")
    parser.add_argument("--bert_model",
                        default="/run/media/occccce/D/models/bert-base-uncased",
                        help="BERT model path")
    parser.add_argument("--output", default="/run/media/occccce/D/results/vript_ablation.json",
                        help="Output JSON path for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples for quick testing")
    args = parser.parse_args()

    # Make output dir
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Truncation mode: {args.truncation}")
    print(f"Methods: {', '.join(args.mode)}")

    # ── Load annotations ─────────────────────────────────────────
    print(f"\n── Loading annotations from {args.vript_ann}")
    with open(args.vript_ann) as f:
        all_annotations = json.load(f)
    print(f"  Total clips: {len(all_annotations)}")

    # Use caption.content as the query text
    captions = [a["caption"] for a in all_annotations]
    clip_ids = [a["clip_id"] for a in all_annotations]

    if args.max_samples:
        captions = captions[:args.max_samples]
        clip_ids = clip_ids[:args.max_samples]
        print(f"  Limited to {args.max_samples} samples for testing")

    # ── Load video frames ────────────────────────────────────────
    frame_dir = Path(args.vript_frames)
    print(f"\n── Loading video frames from {frame_dir}")
    frame_embs, frame_mask, valid_counts = load_video_frames(frame_dir, clip_ids)
    video_emb = mean_pool_video(frame_embs, frame_mask)
    print(f"  Videos: {video_emb.shape[0]}, mean frames: {np.mean(valid_counts):.1f}")

    # Filter out clips with no frames
    has_frames = [i for i, c in enumerate(valid_counts) if c > 0]
    if len(has_frames) < len(clip_ids):
        print(f"  Filtering to {len(has_frames)} clips with valid frame embeddings")
        video_emb = video_emb[has_frames]
        clip_ids = [clip_ids[i] for i in has_frames]
        captions = [captions[i] for i in has_frames]

    # ── Run ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VRIPT Evaluation — {args.truncation} truncation")
    print(f"  {len(clip_ids)} clips, {len(captions)} captions")
    print(f"{'='*60}")

    t0 = time.time()
    results = evaluate(captions, clip_ids, video_emb, args)
    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY — VRIPT ({args.truncation} truncation)")
    print(f"{'='*60}")
    header = "  | {:<10} | {:>7} {:>7} {:>8} | {:>7} {:>7} {:>8} |".format(
        "", "i2tR1", "i2tR5", "i2tR10", "t2iR1", "t2iR5", "t2iR10")
    print(header)
    print("  |" + "-" * (len(header) - 4) + "|")
    for method_name in ["CLIP", "NoStack", "Stack"]:
        r = results.get(method_name)
        if r is None:
            continue
        print("  | {:<10} | {:>6.2f} {:>6.2f} {:>7.2f} | {:>6.2f} {:>6.2f} {:>7.2f} |".format(
            method_name,
            r.get("i2t_R@1", 0), r.get("i2t_R@5", 0), r.get("i2t_R@10", 0),
            r.get("t2i_R@1", 0), r.get("t2i_R@5", 0), r.get("t2i_R@10", 0),
        ))
    print(f"{'='*60}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "dataset": "VRIPT",
        "num_clips": len(clip_ids),
        "truncation": args.truncation,
        "elapsed_seconds": elapsed,
        "results": results,
        "config": {
            "mode": args.mode,
            "nostack_checkpoint": args.nostack_checkpoint,
            "stack_checkpoint": args.stack_checkpoint,
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
