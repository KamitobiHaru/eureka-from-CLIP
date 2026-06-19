"""
Precompute CLIP and BERT text embeddings for MSR-VTT test captions.

Output:
    {cache_dir}/clip_text.pt   — (1000, 512) CLIP text embeddings
    {cache_dir}/bert_text.pt   — (1000, 512) BERT text embeddings

Usage:
    python scripts/precompute_msrvtt_text.py \
        --msrvtt_root ./datasets/MSR-VTT \
        --cache_dir ./data/msrvtt/text_embeddings \
        --config config/default3_project.yaml \
        --bert_checkpoint /path/to/bert_best.pt \
        --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip_search.encoder import CLIPEncoder
from src.models import BertEncoder


@torch.no_grad()
def encode_bert(bert_model, tokenizer, captions, batch_size, device):
    """Run BERT on a list of captions, return (N, 512) tensor."""
    all_embs = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=77, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        embs = bert_model(input_ids, attention_mask)  # (B, 512), L2-normed
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_clip(clip_encoder, captions, batch_size):
    """Run CLIP text encoder on captions, return (N, 512) tensor."""
    all_embs = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        tokens = clip_encoder.tokenizer(batch).to(clip_encoder.device)
        embs = clip_encoder.model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP and BERT text embeddings for MSR-VTT."
    )
    parser.add_argument("--msrvtt_root", default="./datasets/MSR-VTT",
                        help="Root dir containing annotation JSON files")
    parser.add_argument("--cache_dir", default="./data/msrvtt/text_embeddings",
                        help="Output directory for .pt files")
    parser.add_argument("--config", default="config/default3_project.yaml",
                        help="Config (reads lora + bert_model_path from it)")
    parser.add_argument("--bert_checkpoint", required=True,
                        help="Path to trained BERT checkpoint")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    lora_cfg = cfg.get("lora", {})
    bert_model_path = (
        cfg.get("text_encoder", {})
        .get("bert_model_path")
        or cfg["model"].get("bert_model_path")
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load captions ─────────────────────────────────────────────
    ann_file = Path(args.msrvtt_root) / "msrvtt_test_1k.json"
    if not ann_file.exists():
        print(f"Annotation file not found: {ann_file}")
        sys.exit(1)
    with open(ann_file) as f:
        data = json.load(f)
    captions = [entry["caption"] for entry in data]
    video_ids = [entry["video_id"] for entry in data]
    print(f"Loaded {len(captions)} test captions")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── CLIP text embeddings ──────────────────────────────────────
    clip_out = cache_dir / "clip_text.pt"
    if clip_out.exists():
        existing = torch.load(clip_out, weights_only=True)
        print(f"  CLIP text already cached: {existing.shape}")
    else:
        print("  Encoding CLIP text...")
        clip_enc = CLIPEncoder(model_type="openai", device=device)
        clip_embs = encode_clip(clip_enc, captions, args.batch_size)
        torch.save(clip_embs, clip_out)
        print(f"    → saved {clip_out}  ({clip_embs.shape})")

    # ── BERT text embeddings ──────────────────────────────────────
    bert_out = cache_dir / "bert_text.pt"
    if bert_out.exists():
        existing = torch.load(bert_out, weights_only=True)
        print(f"  BERT text already cached: {existing.shape}")
    else:
        print("  Encoding BERT text...")
        print(f"  BERT model path: {bert_model_path}")
        print(f"  LoRA config: {lora_cfg}")
        bert_model = BertEncoder(
            model_path=bert_model_path,
            embed_dim=512,
            lora_cfg=lora_cfg,
            initial_temperature=0.07,
        ).to(device)
        ckpt = torch.load(args.bert_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = bert_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        bert_model.eval()
        for p in bert_model.parameters():
            p.requires_grad = False

        tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)
        bert_embs = encode_bert(bert_model, tokenizer, captions, args.batch_size, device)
        torch.save(bert_embs, bert_out)
        print(f"    → saved {bert_out}  ({bert_embs.shape})")

    print(f"\nAll done. Files in {cache_dir}")


if __name__ == "__main__":
    main()
