"""
Diagnose MSR-VTT zero-shot performance.

Checks:
  1. Are BERT text embeddings in-distribution? (norms, similarity stats)
  2. Does mean-pool hurt? Try single middle frame too.
  3. Are correct pairs actually separated from random pairs?
  4. Verify on Flickr30k that the BERT model works correctly in this setup.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder
from src.models import BertEncoder, TemporalTransformer
from src.data import FlickrDataset, make_text_emb_collate_fn
from src.training.evaluation import compute_recall_metrics


def load_video_frames(frame_cache, video_ids, max_frames=12):
    V = len(video_ids)
    frame_embs = torch.zeros(V, max_frames, 512)
    frame_mask = torch.ones(V, max_frames, dtype=torch.bool)
    counts = []
    for i, vid in enumerate(video_ids):
        p = frame_cache / f"{vid}.npy"
        if not p.exists():
            counts.append(0)
            continue
        arr = np.load(p)
        N = min(arr.shape[0], max_frames)
        frame_embs[i, :N] = torch.from_numpy(arr[:N])
        frame_mask[i, :N] = False
        counts.append(N)
    return frame_embs, frame_mask, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msrvtt_root", default="./datasets/MSR-VTT")
    parser.add_argument("--frame_cache", default="./data/msrvtt/clip_embeddings")
    parser.add_argument("--text_cache", default="./data/msrvtt/text_embeddings")
    parser.add_argument("--config", default="config/default3_project.yaml")
    parser.add_argument("--bert_checkpoint", required=True)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load MSR-VTT test data ──────────────────────────────────
    with open(Path(args.msrvtt_root) / "msrvtt_test_1k.json") as f:
        annotations = json.load(f)
    video_ids = [e["video_id"] for e in annotations]
    captions = [e["caption"] for e in annotations]
    print(f"\nMSR-VTT test: {len(video_ids)} videos, {len(captions)} captions")

    frame_embs, frame_mask, counts = load_video_frames(
        Path(args.frame_cache), video_ids, args.num_frames)
    print(f"  Frames loaded: {sum(1 for c in counts if c > 0)}/{len(video_ids)} videos have data")

    # ── Load text embs ──────────────────────────────────────────
    clip_text = torch.load(Path(args.text_cache) / "clip_text.pt", weights_only=True)
    bert_text = torch.load(Path(args.text_cache) / "bert_text.pt", weights_only=True)

    # ══════════════════════════════════════════════════════════════
    #  Diagnostic 1: embedding statistics
    # ══════════════════════════════════════════════════════════════
    print("\n═══ Diagnostic 1: Embedding statistics ═══")
    for name, emb in [("CLIP text", clip_text), ("BERT text", bert_text),
                       ("CLIP frame", frame_embs.view(-1, 512))]:
        norms = emb.norm(dim=-1)
        print(f"  {name:20s}  shape={str(list(emb.shape)):20s}  "
              f"mean_norm={norms.mean():.4f}  std={norms.std():.4f}  "
              f"min={norms.min():.4f}  max={norms.max():.4f}")

    # ══════════════════════════════════════════════════════════════
    #  Diagnostic 2: similarity distribution
    # ══════════════════════════════════════════════════════════════
    print("\n═══ Diagnostic 2: Cosine similarity distributions ═══")

    for text_name, text_embs in [("CLIP text", clip_text), ("BERT text", bert_text)]:
        # Mean-pooled video
        mask = (~frame_mask).unsqueeze(-1).float()
        video_emb = (frame_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        video_emb = F.normalize(video_emb, dim=-1)

        sim = video_emb @ text_embs.T  # (1000, 1000)
        diag = sim.diag()  # correct pairs
        off_diag = sim[~torch.eye(1000, dtype=torch.bool)].view(-1)

        print(f"\n  {text_name} (mean-pooled frames):")
        print(f"    Correct pairs  — mean={diag.mean():.4f}  std={diag.std():.4f}  "
              f"median={diag.median():.4f}")
        print(f"    Random pairs   — mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  "
              f"median={off_diag.median():.4f}")

        # i2t R@1 from sim matrix directly
        V = sim.shape[0]
        i2t_hits = 0
        for i in range(V):
            topk = sim[i].topk(1).indices.item()
            if topk == i:
                i2t_hits += 1
        t2i_hits = 0
        for i in range(V):
            topk = sim[:, i].topk(1).indices.item()
            if topk == i:
                t2i_hits += 1
        print(f"    Direct i2t R@1: {i2t_hits/V*100:.1f}%")
        print(f"    Direct t2i R@1: {t2i_hits/V*100:.1f}%")

        # Try single middle frame
        mid_idx = args.num_frames // 2
        mid_emb = frame_embs[:, mid_idx, :]  # already L2-normed per frame
        sim_mid = mid_emb @ text_embs.T
        diag_mid = sim_mid.diag()
        off_mid = sim_mid[~torch.eye(1000, dtype=torch.bool)].view(-1)
        print(f"    Single middle frame:")
        print(f"      Correct pairs  — mean={diag_mid.mean():.4f}  median={diag_mid.median():.4f}")
        print(f"      Random pairs   — mean={off_mid.mean():.4f}  median={off_mid.median():.4f}")
        V = sim_mid.shape[0]
        i2t_hits = sum(1 for i in range(V) if sim_mid[i].topk(1).indices.item() == i)
        t2i_hits = sum(1 for i in range(V) if sim_mid[:, i].topk(1).indices.item() == i)
        print(f"      Direct i2t R@1: {i2t_hits/V*100:.1f}%")
        print(f"      Direct t2i R@1: {t2i_hits/V*100:.1f}%")

    # ══════════════════════════════════════════════════════════════
    #  Diagnostic 3: does caption style differ?
    # ══════════════════════════════════════════════════════════════
    print("\n═══ Diagnostic 3: Caption style check ═══")
    print("  First 5 MSR-VTT captions:")
    for c in captions[:5]:
        print(f"    • {c[:100]}")

    # ══════════════════════════════════════════════════════════════
    #  Diagnostic 4: run BERT on Flickr30k to verify model works
    # ══════════════════════════════════════════════════════════════
    print("\n═══ Diagnostic 4: BERT on Flickr30k verification ═══")
    flickr_cfg = cfg.get("flickr", {})
    flickr_cache = flickr_cfg.get("embedding_cache")
    flickr_text_cache = flickr_cache + "_text" if flickr_cache else None

    if flickr_cache and Path(flickr_cache).is_dir():
        # Load precomputed Flickr embs via dataset
        dataset = FlickrDataset(
            flickr_cfg["root"], "test",
            flickr_cache,
            annotation_file=flickr_cfg.get("annotation_file", "flickr_annotations_30k.csv"),
            text_cache_dir=flickr_text_cache,
        )
        # Use the same text embeddings as in the cache
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, num_workers=4,
            collate_fn=make_text_emb_collate_fn(), pin_memory=True,
        )

        all_img = []
        all_txt = []
        all_ids = []
        for batch in loader:
            img, txt, ids = batch
            all_img.append(img)
            all_txt.append(txt)
            all_ids.extend(ids)
        img_embs = torch.cat(all_img)
        txt_embs = torch.cat(all_txt)

        recall = compute_recall_metrics(img_embs, txt_embs, all_ids)
        print(f"  Flickr30k test with precomputed text embeddings:")
        for k, v in recall.items():
            print(f"    {k}: {v:.1f}")

        # Also try mean-pooled frames on flickr? No — Flickr is single images.
    else:
        print("  Skipping (no flickr cache available)")

    print("\n─── Done ───")


if __name__ == "__main__":
    main()
