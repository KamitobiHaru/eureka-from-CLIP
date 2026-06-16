#!/usr/bin/env python3
"""CLI video search using natural language queries.

Usage:
    # Process videos on the fly:
    python search_cli.py /path/to/videos --model bert-coco \\
        --bert-checkpoint /path/to/checkpoint.pt \\
        -q "a squirrel eating nuts"

    # Use precomputed cache (skip video processing):
    python search_cli.py --cache ./data -q "a squirrel eating nuts"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from clip_search import SearchEngine, load_text_encoder


def main():
    parser = argparse.ArgumentParser(description="CLI video scene search")
    parser.add_argument("video_dir", nargs="?", default=None,
                        help="Path to folder containing video files")
    parser.add_argument("-q", "--query", required=True,
                        help="Natural language search query")
    parser.add_argument("--cache", default=None,
                        help="Path to precomputed cache directory (from cache_videos.py)")
    parser.add_argument("--model", default="clip",
                        choices=["clip", "bert-coco", "bert-stack-lora"])
    parser.add_argument("-b", "--bert-checkpoint", default=None,
                        help="BERT checkpoint .pt file")
    parser.add_argument("--config", default="config/default.yaml",
                        help="YAML config path")
    parser.add_argument("--device", default=None,
                        help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results (default: 10)")
    args = parser.parse_args()

    if not args.video_dir and not args.cache:
        parser.error("either video_dir or --cache must be provided")

    if args.model in ("bert-coco", "bert-stack-lora") and not args.bert_checkpoint:
        parser.error(f"--bert-checkpoint is required when --model={args.model}")

    # 1. Build engine
    if args.model != "clip":
        print(f"Loading text encoder: {args.model}")
        text_encoder = load_text_encoder(
            model=args.model,
            bert_checkpoint=args.bert_checkpoint,
            config_path=args.config,
            device=args.device,
        )
    else:
        text_encoder = None

    engine = SearchEngine(text_encoder=text_encoder, device=args.device)

    # 2. Load cache or process videos
    if args.cache:
        n = engine.load_cache(args.cache)
        if n == 0:
            print("Cache is empty.")
            return
        print(f"Loaded {n} videos from cache.")
    else:
        print(f"Processing videos from: {args.video_dir}")
        engine.process_folder(args.video_dir)
        if not engine.scenes:
            print("No videos processed.")
            sys.exit(1)
        print(f"Total videos indexed: {len(engine.scenes)}")

    # 3. Search
    print(f'\nSearching for: "{args.query}"')
    results = engine.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching results found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Top {len(results)} Results")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        print(f"\n  [{i + 1}] {r['video_id']}  (Score: {r['score']:.4f})")


if __name__ == "__main__":
    main()
