"""
Translate Flickr30k test captions to Chinese using hy-mt-1.8b.

Output: data/flickr30k/flickr_test_zh.json
    {"<stem>": ["trans1", ..., "trans5"], ...}
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Translate Flickr30k test captions to Chinese")
    parser.add_argument("--ann_file", default="dataset_annotation/flickr_annotations_30k.csv")
    parser.add_argument("--output", default="data/flickr30k/flickr_test_zh.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)

    # ── Read test captions ────────────────────────────────────
    print(f"Reading: {args.ann_file}")
    pairs = []  # (stem, caption)
    with open(args.ann_file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"].strip() != "test":
                continue
            stem = Path(r["filename"].strip()).stem
            captions = json.loads(r["raw"])
            for cap in captions:
                pairs.append((stem, cap))
    print(f"  Test pairs: {len(pairs)} ({len(pairs)//5} unique images × 5 captions)")

    # ── Load model ────────────────────────────────────────────
    model_path = "models/hy-mt-1.8b"
    print(f"Loading hy-mt-1.8b on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Left padding needed for decoder-only batch generation
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    ).eval()
    print("  Model loaded.")

    # ── Translate ─────────────────────────────────────────────
    # Deduplicate captions to avoid translating the same text twice
    unique_texts = sorted(set(cap for _, cap in pairs))
    text_to_translation = {}

    batch_size = 8
    print(f"Translating {len(unique_texts)} unique captions...")
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Translating"):
        batch_texts = unique_texts[i:i + batch_size]
        messages = [[{"role": "user", "content": f"Translate to Chinese: {t}"}]
                     for t in batch_texts]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            padding=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for text, inp, out in zip(batch_texts, inputs, outputs):
            generated = out[len(inp):]
            translation = tokenizer.decode(generated, skip_special_tokens=True).strip()
            text_to_translation[text] = translation

        del inputs, outputs
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Build output ──────────────────────────────────────────
    result = {}
    for stem, cap in pairs:
        zh = text_to_translation.get(cap, cap)
        result.setdefault(stem, []).append(zh)

    # Validate: each stem should have exactly 5 captions
    for stem, caps in result.items():
        assert len(caps) == 5, f"{stem}: expected 5 captions, got {len(caps)}"

    # ── Save ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(result)} images × 5 captions → {output_path}")


if __name__ == "__main__":
    main()
