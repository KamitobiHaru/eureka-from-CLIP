from collections import defaultdict
from typing import Dict, List, Tuple

import torch


def compute_recall_metrics(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    image_ids: List[str],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute Recall@K for image-to-text and text-to-image retrieval.

    Each image may have multiple ground-truth captions.  Deduplicates image
    embeddings so each unique image contributes once to image->text recall.

    Args:
        image_embs: (N, D) — may contain duplicates (same image, 5 captions).
        text_embs: (N, D) — one text embedding per caption.
        image_ids: Length N, image_id per (image_emb, text_emb) pair.
        ks: Recall cutoffs.

    Returns:
        Dict with keys like "i2t_R@1", "t2i_R@10", etc.  Values as
        percentages (0–100).
    """
    device = image_embs.device
    N = image_embs.size(0)

    # ── Deduplicate images (first occurrence per image_id) ──────
    seen: set = set()
    unique_indices: List[int] = []
    unique_id_list: List[str] = []
    for i, img_id in enumerate(image_ids):
        if img_id not in seen:
            seen.add(img_id)
            unique_indices.append(i)
            unique_id_list.append(img_id)

    unique_image_embs = image_embs[unique_indices]  # (U, D)
    U = len(unique_id_list)

    # ── Map image_id -> set of text positions ───────────────────
    id_to_text_positions: Dict[str, set] = defaultdict(set)
    for i, img_id in enumerate(image_ids):
        id_to_text_positions[img_id].add(i)

    max_k = max(ks)

    # ── Image-to-Text retrieval ─────────────────────────────────
    sim_i2t = unique_image_embs @ text_embs.T  # (U, N)
    i2t_k = min(max_k, sim_i2t.size(1))
    _, topk_indices_i2t = torch.topk(sim_i2t, i2t_k, dim=1, largest=True)

    i2t_hits = {k: 0 for k in ks}
    for u_idx, img_id in enumerate(unique_id_list):
        gt_set = id_to_text_positions[img_id]
        topk_list = topk_indices_i2t[u_idx].cpu().tolist()  # full top-k list
        for k in ks:
            if k <= i2t_k and set(topk_list[:k]) & gt_set:
                i2t_hits[k] += 1

    # ── Text-to-Image retrieval ─────────────────────────────────
    sim_t2i = text_embs @ unique_image_embs.T  # (N, U)
    t2i_k = min(max_k, sim_t2i.size(1))
    _, topk_indices_t2i = torch.topk(sim_t2i, t2i_k, dim=1, largest=True)

    id_to_uidx = {img_id: idx for idx, img_id in enumerate(unique_id_list)}

    t2i_hits = {k: 0 for k in ks}
    for i, img_id in enumerate(image_ids):
        gt_uidx = id_to_uidx[img_id]
        topk_list = topk_indices_t2i[i].cpu().tolist()
        for k in ks:
            if k <= t2i_k and gt_uidx in topk_list[:k]:
                t2i_hits[k] += 1

    # ── Aggregate ───────────────────────────────────────────────
    results: Dict[str, float] = {}
    for k in ks:
        i2t_denom = U
        if k > N:
            # Trivially 100% — fewer items than k
            results[f"i2t_R@{k}"] = 100.0
        else:
            results[f"i2t_R@{k}"] = i2t_hits[k] / i2t_denom * 100.0

        if k > U:
            results[f"t2i_R@{k}"] = 100.0
        else:
            results[f"t2i_R@{k}"] = t2i_hits[k] / N * 100.0
    return results
