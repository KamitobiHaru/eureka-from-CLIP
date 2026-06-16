"""
Evaluate VoP on Flickr30k zero-shot image retrieval.

Reuses VoPModel / ClipTokenizer from eval_vop_msvd.py.

Usage:
    python scripts/eval_vop_flickr30k.py
"""
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_vop_msvd import VoPModel, ClipTokenizer, DEVICE, VOP_DIR, CKPT_PATH, VOCAB_PATH
from src.training.evaluation import compute_recall_metrics

# ── Paths ──────────────────────────────────────────────────────────────
FLICKR_CSV = Path("/run/media/occccce/E/flickr30/flickr_annotations_30k.csv")
FLICKR_IMG_DIR = Path("/run/media/occccce/E/flickr30/f78a015e45ea38d4367367223b2bb63cec0d549d5da13c6211663e5439a93216 (2)/flickr30k-images")

# ── Image transform (CLIP standard) ────────────────────────────────────
image_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


def main():
    device = DEVICE
    print(f"Device: {device}\n")

    # ── Load model ──
    print("── Loading VoP checkpoint ──")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = VoPModel(ckpt["state_dict"]).to(device)
    model.eval()
    n_total = sum(p.numel() for p in model.parameters())
    n_prompts = sum(p.numel() for p in [model.visual_prompts, model.text_prompts])
    print(f"  Total params: {n_total:,}  (prompts: {n_prompts:,})")

    # ── Tokenizer ──
    tokenizer = ClipTokenizer(str(VOCAB_PATH))
    print(f"  Vocab size: {len(tokenizer.encoder)}")

    # ── Load Flickr30k test split ──
    print("\n── Loading Flickr30k test split ──")
    with open(FLICKR_CSV) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["split"] == "test"]
    print(f"  Test images: {len(rows)}")

    # Build: image_id → [captions], image_id → filename
    image_data = []
    for r in rows:
        image_id = r["img_id"]
        filename = r["filename"]
        captions = json.loads(r["raw"])
        image_data.append((image_id, filename, captions))

    unique_ids = [d[0] for d in image_data]
    print(f"  Unique images: {len(unique_ids)}")

    # ── Encode images ──
    print(f"\n── Encoding {len(unique_ids)} images ──")
    image_embs = {}
    for image_id, filename, _ in tqdm(image_data, desc="Image"):
        img_path = FLICKR_IMG_DIR / filename
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        img_t = image_transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        # VoP encode_video expects (B, T, 3, 224, 224) → T=1 for images
        with torch.no_grad():
            emb = model.encode_video(img_t.unsqueeze(0))  # (1, 1, 3, 224, 224)
        image_embs[image_id] = emb.cpu()

    print(f"  Encoded {len(image_embs)}/{len(unique_ids)} images")

    # ── Build pairs ──
    pairs = []
    for image_id, _, captions in image_data:
        if image_id not in image_embs:
            continue
        for cap in captions:
            if cap.strip():
                pairs.append((image_id, cap.strip()))
    print(f"  Caption pairs: {len(pairs):,}")

    # ── Encode texts ──
    print(f"\n── Encoding {len(pairs)} captions ──")
    all_image_embs, all_text_embs, all_image_ids = [], [], []
    cap_batch = []
    for img_id, cap in pairs:
        all_image_embs.append(image_embs[img_id])
        all_image_ids.append(img_id)
        cap_batch.append(cap)
        if len(cap_batch) >= 128:
            tokens = tokenizer(cap_batch).to(device)
            with torch.no_grad():
                embs = model.encode_text(tokens)
            all_text_embs.append(embs.cpu())
            cap_batch = []

    if cap_batch:
        tokens = tokenizer(cap_batch).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
        all_text_embs.append(embs.cpu())

    img_t = torch.cat(all_image_embs, dim=0)
    txt_t = torch.cat(all_text_embs, dim=0)
    print(f"  Image embs: {img_t.shape}   Text embs: {txt_t.shape}")

    # ── Recall ──
    print("\n── Computing Recall@K ──")
    recall = compute_recall_metrics(img_t, txt_t, all_image_ids, ks=(1, 5, 10))

    print("\n" + "=" * 50)
    print("  VoP on Flickr30k (zero-shot image retrieval)")
    print("=" * 50)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {k:<12}: {recall.get(k, 0):.2f}")
    print("=" * 50)

    # ── Compare ──
    print("\n  Flickr30k reference (t2i R@1):")
    print("    CLIP native text encoder: 57.90%")
    print("    Ours (Stage 1, BERT+LoRA): 60.48%")
    print("    Ours (Stage 2, after video adapt): ~56%")
    delta = recall.get("t2i_R@1", 0) - 57.90
    print(f"  VoP Δ vs CLIP: {delta:+.2f}")


if __name__ == "__main__":
    main()
