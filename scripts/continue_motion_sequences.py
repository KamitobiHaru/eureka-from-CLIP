"""
Continue generating pseudo-video sequences after precompute_motion_sequences.py.

Generates remaining 5000 connector sequences for train (continuing from the
45000 motion sequences already saved) and 2500 full sequences for val.

Usage:
    python scripts/continue_motion_sequences.py \\
        --config config/default3_temporal.yaml \\
        --device cuda:0 [--workers 4]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.precompute_motion_sequences import (
    generate_connector_sample,
    generate_motion_frames_only,
    _flush_motion_buffer,
    _load_coco,
)


def _verify_no_overlap(train_ids, val_ids, label_a="train2017", label_b="val2017"):
    """Explicitly verify two image-ID pools have zero intersection."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = train_set & val_set
    if overlap:
        print(f"\n⚠  CRITICAL: {len(overlap)} images appear in BOTH {label_a} and {label_b}!")
        print(f"  Overlapping IDs: {sorted(overlap)[:20]}{' ...' if len(overlap) > 20 else ''}")
        sys.exit(1)
    print(f"  ✓ No overlap between {label_a} ({len(train_ids)} imgs) and {label_b} ({len(val_ids)} imgs)")


def main():
    parser = argparse.ArgumentParser(description="Continue generating pseudo-video sequences.")
    parser.add_argument("--config", default="config/default3_temporal.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers for val motion generation (0 = sequential)")
    parser.add_argument("--train_connector", type=int, default=5000,
                        help="Remaining connector sequences for train")
    parser.add_argument("--val_total", type=int, default=2500,
                        help="Total sequences for val2017")
    parser.add_argument("--motion_ratio", type=float, default=0.9)
    parser.add_argument("--clip_batch_size", type=int, default=256)
    parser.add_argument("--visualize", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg = cfg.get("motion", {})
    t_cfg = cfg.get("temporal", {})
    coco_image_root = Path(m_cfg.get("coco_image_root", "/data2/zsy/datasets/coco2017_trainval"))
    ann_dir = Path(cfg["data"]["coco_root"]) / cfg["data"].get("annotations_dir", "annotations")
    emb_cache = Path(cfg["data"]["embedding_cache"])
    cache_dir = Path(m_cfg.get("cache_dir", "data/coco/mixed_sequences"))
    min_len = t_cfg.get("sequence_min_len", 3)
    max_len = t_cfg.get("sequence_max_len", 10)
    valid_classes = set(m_cfg.get("valid_classes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))
    min_area = m_cfg.get("min_object_area", 0.08)
    max_area = m_cfg.get("max_object_area", 0.5)
    conf_thresh = m_cfg.get("confidence_threshold", 0.5)
    yolo_path = m_cfg.get("yolo_model", "models/yolo26x-seg.pt")
    lama_path = m_cfg.get("lama_model", "models/lama")
    inpaint_size = m_cfg.get("inpaint_size", 256)

    # ── Load COCO annotations ─────────────────────────────────────
    for split in ("train2017", "val2017"):
        _, id_to_captions, id_to_file = _load_coco(split, ann_dir)
        if split == "train2017":
            train_captions, train_file = id_to_captions, id_to_file
        else:
            val_captions, val_file = id_to_captions, id_to_file

    cached_ids = set(int(p.stem) for p in emb_cache.glob("*.npy"))
    train_ids = [i for i in train_file if i in cached_ids]
    val_ids = [i for i in val_file if i in cached_ids]

    # ── Step 0: Verify NO overlap between train and val image pools ──
    print("=" * 60)
    print("Verifying train/val image separation")
    print("=" * 60)
    _verify_no_overlap(train_ids, val_ids)
    print(f"  Train images with embeddings: {len(train_ids)}")
    print(f"  Val images with embeddings:   {len(val_ids)}")

    print("\n" + "=" * 60)
    print("Step 1: Generate remaining connector sequences for train")
    print("=" * 60)

    # ── Step 1: Train connector sequences ─────────────────────────
    train_out = cache_dir / "train2017"
    train_out.mkdir(parents=True, exist_ok=True)

    # Find next available ID
    existing_ids = sorted(int(p.stem.split("_")[1]) for p in train_out.glob("seq_*.json"))
    next_id = max(existing_ids) + 1 if existing_ids else 0
    print(f"  Existing sequences: {len(existing_ids)}, next_id={next_id}")

    # samples.json from the original run was NOT saved (deadlock in cleanup),
    # so we rebuild it at the end by scanning all seq_*.json files.
    all_samples_train = []

    target_conn = args.train_connector
    conn_done = 0
    conn_attempts = 0
    max_attempts = target_conn * 5
    pbar = tqdm(total=target_conn, desc="Train connector")
    while conn_done < target_conn and conn_attempts < max_attempts:
        conn_attempts += 1
        try:
            result = generate_connector_sample(
                train_ids, train_captions, emb_cache, min_len, max_len,
            )
            if result is None:
                continue
            embs, corr_cap, shuf_cap, meta = result

            np.save(train_out / f"seq_{next_id:06d}.npy", embs.astype(np.float32))
            with open(train_out / f"seq_{next_id:06d}.json", "w") as f:
                json.dump({
                    "correct_caption": corr_cap,
                    "shuffled_caption": shuf_cap,
                    "type": "connector",
                    "img_ids": meta["img_ids"],
                    "K": meta["K"],
                }, f)

            all_samples_train.append({
                "id": next_id,
                "type": "connector",
                "K": meta["K"],
            })
            next_id += 1
            conn_done += 1
            pbar.update(1)
        except Exception:
            continue
    pbar.close()
    print(f"  Generated {conn_done} connector sequences for train")

    # Save updated samples.json for train — rebuild from ALL seq_*.json
    # (the original samples.json was lost due to the deadlock)
    all_samples_train = []
    for p in sorted(train_out.glob("seq_*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            sid = int(p.stem.split("_")[1])
            all_samples_train.append({
                "id": sid,
                "type": data["type"],
                "K": data["K"],
            })
        except Exception:
            continue
    random.shuffle(all_samples_train)
    with open(train_out / "samples.json", "w") as f:
        json.dump(all_samples_train, f, indent=2)
    print(f"  Rebuilt samples.json ({len(all_samples_train)} total train samples)")

    # ── Step 2: Load motion models for val generation ─────────────
    print("\n" + "=" * 60)
    print("Step 2: Generate val sequences (motion + connector)")
    print("=" * 60)

    val_out = cache_dir / "val2017"
    val_out.mkdir(parents=True, exist_ok=True)

    target_motion = int(args.val_total * args.motion_ratio)
    target_connector = args.val_total - target_motion
    print(f"  {target_motion} motion + {target_connector} connector = {args.val_total} total")

    from ultralytics import YOLO
    from modelscope.models.cv.image_inpainting import FFTInpainting

    print("  Loading YOLO...")
    yolo_model = YOLO(yolo_path)
    print("  Loading LaMa...")
    lama_model = FFTInpainting(model_dir=lama_path).to(device)
    lama_model.eval()
    print("  Loading CLIP encoder...")
    from clip_search.encoder import CLIPEncoder
    clip_model = cfg.get("clip", {}).get("model", "laion")
    clip_encoder = CLIPEncoder(model_type=clip_model, device=device)

    # ── Val motion sequences (sequential) ─────────────────────────
    val_next_id = 0
    all_samples_val = []
    motion_done = 0
    motion_buffer = []
    vis_count = 0
    vis_dir = None

    if target_motion > 0:
        pbar = tqdm(total=target_motion, desc="Val motion")
        attempts = 0
        max_attempts = target_motion * 10
        while motion_done < target_motion and attempts < max_attempts:
            attempts += 1
            img_id = random.choice(val_ids)
            img_path = coco_image_root / "val2017" / val_file[img_id]

            try:
                result = generate_motion_frames_only(
                    img_id, img_path, valid_classes, min_area, max_area,
                    conf_thresh, yolo_model, lama_model,
                    device, min_len, max_len, inpaint_size,
                )
                if result is None:
                    continue
                frame_imgs, corr_cap, shuf_cap, meta = result
                motion_buffer.append({
                    "frame_images": frame_imgs,
                    "correct_caption": corr_cap,
                    "shuffled_caption": shuf_cap,
                    "meta": meta,
                    "K": meta["K"],
                })
                motion_done += 1
                pbar.update(1)

                total_frames = sum(item["K"] for item in motion_buffer)
                if total_frames >= args.clip_batch_size:
                    val_next_id, vis_count = _flush_motion_buffer(
                        motion_buffer, clip_encoder, val_next_id, val_out,
                        all_samples_val, vis_count, args.visualize, vis_dir,
                    )
                    motion_buffer = []
            except Exception:
                continue

        # Flush remaining
        if motion_buffer:
            val_next_id, vis_count = _flush_motion_buffer(
                motion_buffer, clip_encoder, val_next_id, val_out,
                all_samples_val, vis_count, args.visualize, vis_dir,
            )
            motion_buffer = []
        pbar.close()
        if motion_done < target_motion:
            print(f"  ⚠  Only generated {motion_done}/{target_motion} val motion sequences")

    # ── Val connector sequences ───────────────────────────────────
    if target_connector > 0:
        pbar = tqdm(total=target_connector, desc="Val connector")
        conn_attempts = 0
        max_conn_attempts = target_connector * 5
        conn_done = 0
        while conn_done < target_connector and conn_attempts < max_conn_attempts:
            conn_attempts += 1
            try:
                result = generate_connector_sample(
                    val_ids, val_captions, emb_cache, min_len, max_len,
                )
                if result is None:
                    continue
                embs, corr_cap, shuf_cap, meta = result

                np.save(val_out / f"seq_{val_next_id:06d}.npy", embs.astype(np.float32))
                with open(val_out / f"seq_{val_next_id:06d}.json", "w") as f:
                    json.dump({
                        "correct_caption": corr_cap,
                        "shuffled_caption": shuf_cap,
                        "type": "connector",
                        "img_ids": meta["img_ids"],
                        "K": meta["K"],
                    }, f)
                all_samples_val.append({
                    "id": val_next_id,
                    "type": "connector",
                    "K": meta["K"],
                })
                val_next_id += 1
                conn_done += 1
                pbar.update(1)
            except Exception:
                continue
        pbar.close()

    # ── Save val samples.json ─────────────────────────────────────
    random.shuffle(all_samples_val)
    with open(val_out / "samples.json", "w") as f:
        json.dump(all_samples_val, f, indent=2)

    # ── Final verification: no image overlap in generated samples ──
    print("\n" + "=" * 60)
    print("Final verification")
    print("=" * 60)

    # Collect all image IDs used in train sequences.
    # samples.json may be missing (original process deadlocked before saving),
    # so we scan the actual JSON files directly.
    train_img_ids = set()
    n_train_seqs = 0
    for p in sorted(train_out.glob("seq_*.json")):
        n_train_seqs += 1
        try:
            with open(p) as f:
                data = json.load(f)
            if data["type"] == "motion":
                train_img_ids.add(data.get("img_id", -1))
            else:
                train_img_ids.update(data.get("img_ids", []))
        except Exception:
            continue

    val_img_ids = set()
    n_val_seqs = 0
    for p in sorted(val_out.glob("seq_*.json")):
        n_val_seqs += 1
        try:
            with open(p) as f:
                data = json.load(f)
            if data["type"] == "motion":
                val_img_ids.add(data.get("img_id", -1))
            else:
                val_img_ids.update(data.get("img_ids", []))
        except Exception:
            continue

    train_img_ids.discard(-1)
    val_img_ids.discard(-1)
    img_overlap = train_img_ids & val_img_ids
    if img_overlap:
        print(f"\n⚠  DATA LEAKAGE: {len(img_overlap)} images appear in BOTH train and val sequences!")
        print(f"  Overlapping IDs: {sorted(img_overlap)[:20]}")
        sys.exit(1)
    else:
        print(f"  ✓  Zero image-ID overlap between {n_train_seqs} train and {n_val_seqs} val sequences!")
    print(f"  Train unique images: {len(train_img_ids)}, Val unique images: {len(val_img_ids)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
