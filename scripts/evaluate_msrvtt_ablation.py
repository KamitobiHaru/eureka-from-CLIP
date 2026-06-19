"""
MSR-VTT 1K-A ablation: compare Pure CLIP vs Pure BERT vs BERT+Temporal.

Loads precomputed CLIP frame embeddings and text embeddings (both CLIP and
BERT variants), then runs three retrieval evaluations under identical protocol.

Usage:
    python scripts/evaluate_msrvtt_ablation.py \
        --msrvtt_root ./datasets/MSR-VTT \
        --frame_cache ./data/msrvtt/clip_embeddings \
        --text_cache ./data/msrvtt/text_embeddings \
        --temporal_checkpoint /path/to/temporal_best.pt \
        --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import TemporalTransformer
from src.training.evaluation import compute_recall_metrics


def load_video_frames(frame_cache: Path, video_ids: list[str],
                      max_frames: int = 12):
    """Load per-video frame embeddings, pad / mask to uniform length.

    Returns:
        frame_embs:  (V, max_frames, 512)  — zero-padded
        frame_mask:  (V, max_frames)       — True = padded
        valid_counts: list[int]            — actual frame count per video
    """
    V = len(video_ids)
    D = 512
    frame_embs = torch.zeros(V, max_frames, D)
    frame_mask = torch.ones(V, max_frames, dtype=torch.bool)  # True = pad
    valid_counts = []

    for i, vid in enumerate(video_ids):
        npy_path = frame_cache / f"{vid}.npy"
        if not npy_path.exists():
            print(f"  WARNING: {npy_path} not found, using zeros")
            valid_counts.append(0)
            continue
        arr = np.load(npy_path)  # (N, 512)
        N = min(arr.shape[0], max_frames)
        frame_embs[i, :N] = torch.from_numpy(arr[:N])
        frame_mask[i, :N] = False  # valid
        valid_counts.append(N)

    return frame_embs, frame_mask, valid_counts


def mean_pool_video(frame_embs: torch.Tensor,
                    frame_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over valid frames, then L2 normalize.

    Args:
        frame_embs: (V, T, 512)
        frame_mask: (V, T)  True = padded
    Returns:
        (V, 512) L2-normalised video embeddings
    """
    mask = (~frame_mask).unsqueeze(-1).float()  # (V, T, 1), 1=valid
    video_emb = (frame_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(video_emb, dim=-1)


@torch.no_grad()
def temporal_encode(temporal: torch.nn.Module,
                    frame_embs: torch.Tensor,
                    frame_mask: torch.Tensor,
                    device: torch.device) -> torch.Tensor:
    """Encode videos through TemporalTransformer.

    Generates uniform positions (cx, cy, scale) = (0.5, 0.5, 1.0) for
    all frames (centered crop, full scale).

    Returns:
        (V, 512) L2-normalised video embeddings
    """
    frame_embs = frame_embs.to(device)
    frame_mask = frame_mask.to(device)
    B, T, _ = frame_embs.shape

    # Uniform positions: (cx=0.5, cy=0.5, scale=1.0) for every position
    positions = torch.full((B, T, 3), fill_value=0.5, device=device)
    positions[:, :, 2] = 1.0  # scale=1.0

    return temporal(frame_embs, frame_mask, positions=positions)


def print_table(results: dict[str, dict]):
    """Print three-method comparison table."""
    methods = ["Pure CLIP", "Pure BERT", "BERT+Temporal"]
    print()
    print("=" * 50)
    print("  MSR-VTT 1K-A Ablation".center(48))
    print("=" * 50)
    print(f"  {'Method':<16} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print("  " + "-" * 36)
    for m in methods:
        r = results.get(m, {})
        print(f"  {m:<16} {r.get('i2t_R@1', -1):>6.1f} {r.get('i2t_R@5', -1):>6.1f} {r.get('i2t_R@10', -1):>6.1f}")
    print()
    # t2i
    print(f"  {'Method':<16} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print("  " + "-" * 36)
    for m in methods:
        r = results.get(m, {})
        print(f"  {m:<16} {r.get('t2i_R@1', -1):>6.1f} {r.get('t2i_R@5', -1):>6.1f} {r.get('t2i_R@10', -1):>6.1f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="MSR-VTT ablation: CLIP vs BERT vs BERT+Temporal."
    )
    parser.add_argument("--msrvtt_root", default="./datasets/MSR-VTT")
    parser.add_argument("--frame_cache", default="./data/msrvtt/clip_embeddings",
                        help="Dir with per-video .npy frame embeddings")
    parser.add_argument("--text_cache", default="./data/msrvtt/text_embeddings",
                        help="Dir with clip_text.pt and bert_text.pt")
    parser.add_argument("--temporal_checkpoint",
                        help="Phase-1 temporal checkpoint (required for BERT+Temporal mode)")
    parser.add_argument("--num_frames", type=int, default=12,
                        help="Max frames per video (must match precomputation)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    frame_cache = Path(args.frame_cache)
    text_cache = Path(args.text_cache)

    # ── Load test annotations ─────────────────────────────────────
    ann_file = Path(args.msrvtt_root) / "msrvtt_test_1k.json"
    with open(ann_file) as f:
        annotations = json.load(f)
    video_ids = [e["video_id"] for e in annotations]
    print(f"Test videos: {len(video_ids)}")

    # ── Load video frame embeddings ───────────────────────────────
    print("Loading frame embeddings...")
    frame_embs, frame_mask, valid_counts = load_video_frames(
        frame_cache, video_ids, args.num_frames,
    )
    n_valid = sum(1 for c in valid_counts if c > 0)
    print(f"  Videos with frames: {n_valid}/{len(video_ids)}")
    v_ids_tensor = video_ids  # keep as list for compute_recall_metrics

    # ── Load text embeddings ──────────────────────────────────────
    clip_text = torch.load(text_cache / "clip_text.pt", weights_only=True)
    bert_text = torch.load(text_cache / "bert_text.pt", weights_only=True)
    print(f"  CLIP text: {clip_text.shape}")
    print(f"  BERT text: {bert_text.shape}")

    results = {}

    # ══════════════════════════════════════════════════════════════
    #  Mode 1: Pure CLIP
    # ══════════════════════════════════════════════════════════════
    print("\n─── Pure CLIP ───")
    clip_video = mean_pool_video(frame_embs.clone(), frame_mask.clone())
    clip_video = clip_video.to(clip_text.device)
    clip_recall = compute_recall_metrics(clip_video, clip_text.to(clip_video.device), video_ids)
    for k, v in clip_recall.items():
        print(f"  {k}: {v:.1f}")
    results["Pure CLIP"] = clip_recall

    # ══════════════════════════════════════════════════════════════
    #  Mode 2: Pure BERT
    # ══════════════════════════════════════════════════════════════
    print("\n─── Pure BERT ───")
    bert_video = mean_pool_video(frame_embs.clone(), frame_mask.clone())
    bert_video = bert_video.to(bert_text.device)
    bert_recall = compute_recall_metrics(bert_video, bert_text.to(bert_video.device), video_ids)
    for k, v in bert_recall.items():
        print(f"  {k}: {v:.1f}")
    results["Pure BERT"] = bert_recall

    # ══════════════════════════════════════════════════════════════
    #  Mode 3: BERT + Temporal
    # ══════════════════════════════════════════════════════════════
    print("\n─── BERT+Temporal ───")
    if not args.temporal_checkpoint:
        print("  SKIP: no --temporal_checkpoint provided")
    else:
        t_cfg = {"d_model": 512, "nhead": 8, "num_layers": 2,
                 "dim_feedforward": 1024, "dropout": 0.1, "max_frames": 16}
        temporal = TemporalTransformer(**t_cfg).to(device)
        ckpt = torch.load(args.temporal_checkpoint, map_location=device, weights_only=True)
        temporal.load_state_dict(ckpt["temporal_state_dict"])
        temporal.eval()
        print(f"  Loaded temporal checkpoint: {args.temporal_checkpoint}")

        temp_video = temporal_encode(temporal, frame_embs, frame_mask, device)
        temp_video = temp_video.cpu()
        temp_recall = compute_recall_metrics(temp_video, bert_text, video_ids)
        for k, v in temp_recall.items():
            print(f"  {k}: {v:.1f}")
        results["BERT+Temporal"] = temp_recall

    # ══════════════════════════════════════════════════════════════
    #  Results table
    # ══════════════════════════════════════════════════════════════
    print_table(results)


if __name__ == "__main__":
    main()
