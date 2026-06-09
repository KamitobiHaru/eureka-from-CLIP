"""
Generate pseudo-video sequences for temporal-transformer training.

Combines two types of sequences:
  - **Motion**: YOLO seg + LaMa inpainting + object compositing + CLIP encoding
  - **Connector**: random COCO images with connector-word captions (e.g.
    "First, a dog. Then, a cat. Finally, a car.")

Both types are mixed at a configurable ratio and saved as a unified cache
that can be loaded by ``MixedSequenceDataset`` at training time.

Usage:
    python scripts/precompute_motion_sequences.py \\
        --config config/default.yaml \\
        --total_sequences 50000 --val_sequences 2000 \\
        --motion_ratio 0.5
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Note: clip_search.encoder and other heavy imports are done inside main()
# to keep --help fast and avoid cascading dependency errors.

# ═══════════════════════════════════════════════════════════════════════════
#  Connector vocabulary (mirrors src/data/sequence_dataset.py)
# ═══════════════════════════════════════════════════════════════════════════

INDEX_WORDS = [
    "First", "Second", "Third", "Fourth", "Fifth",
    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
]
LAST_WORDS = ["Last", "Finally", "At last"]
SEQUENTIAL_STYLES = [
    ["First", "Then", "Finally"],
    ["To begin with", "After that", "At last"],
    ["First", "After that", "Finally"],
    ["First", "Next", "Last"],
]
MIDDLE_WORDS = ["Then", "After that", "What's more", "Next"]

def _build_connectors(K: int, style: str):
    if K == 1:
        return [random.choice(["First", "To begin with"])]
    if style == "index":
        words = list(INDEX_WORDS[: K - 1])
        words.append(random.choice(LAST_WORDS))
        return words[:K]
    base = random.choice(SEQUENTIAL_STYLES)
    if K <= 3:
        return base[:K]
    mid = [random.choice(MIDDLE_WORDS) for _ in range(K - 2)]
    return [base[0]] + mid + [base[-1]]


def _format_caption(connectors, captions):
    parts = [f"{w}, {c}" for w, c in zip(connectors, captions)]
    return ". ".join(parts) + "."




# ═══════════════════════════════════════════════════════════════════════════
#  COCO annotation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_coco(split: str, ann_dir: str):
    """Load COCO caption annotations and image list.

    Returns (images, id_to_captions, id_to_file).
        images: list of {"id": int, "file_name": str, ...}
        id_to_captions: dict[int, list[str]]
        id_to_file: dict[int, str]
    """
    ann_file = Path(ann_dir) / f"captions_{split}.json"
    with open(ann_file) as f:
        data = json.load(f)

    images = data["images"]
    id_to_file = {img["id"]: img["file_name"] for img in images}
    id_to_captions: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return images, id_to_captions, id_to_file


# ═══════════════════════════════════════════════════════════════════════════
#  Motion-sequence helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pick_object(results, valid_classes, min_area_ratio, max_area_ratio, conf_threshold, img_size):
    """From YOLO results, pick the best object for motion generation.

    Returns (mask, bbox, class_id, confidence) or None.
        mask: binary np.ndarray [H, W]
        bbox: (x1, y1, x2, y2) tight bounding box
    """
    if results[0].boxes is None or results[0].masks is None:
        return None

    boxes = results[0].boxes
    masks = results[0].masks
    H, W = img_size
    img_area = H * W

    best = None
    best_area = 0

    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        if cls_id not in valid_classes or conf < conf_threshold:
            continue

        # Get tight bbox from mask (more accurate than box)
        mask_np = masks.data[i].cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        obj_area = (x2 - x1) * (y2 - y1)
        area_ratio = obj_area / img_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        if obj_area > best_area:
            best = (mask_np, (x1, y1, x2, y2), cls_id, conf)
            best_area = obj_area

    return best


def _inpaint_background(lama_model, image, mask, device, inpaint_size=256):
    """Remove object from image using LaMa inpainting.

    Resizes to ``inpaint_size`` before the forward pass (LaMa was trained
    on 256x256) then bilinearly upscales back to the original resolution.

    Args:
        image: np.ndarray [H, W, 3] RGB uint8
        mask: np.ndarray [H, W] uint8, 255 = object to remove
        device: torch device
        inpaint_size: target size for LaMa forward pass

    Returns np.ndarray [H, W, 3] RGB uint8.
    """
    H, W = image.shape[:2]
    target = inpaint_size

    need_resize = (H != target or W != target)
    if need_resize:
        img_inp = cv2.resize(image, (target, target), interpolation=cv2.INTER_LINEAR)
        mask_inp = cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)
    else:
        img_inp = image
        mask_inp = mask

    inp_H, inp_W = img_inp.shape[:2]

    # Pad to multiple of 8
    pad_h = (8 - inp_H % 8) % 8
    pad_w = (8 - inp_W % 8) % 8
    if pad_h or pad_w:
        img_pad = cv2.copyMakeBorder(img_inp, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_REFLECT)
        mask_pad = cv2.copyMakeBorder(mask_inp, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
    else:
        img_pad = img_inp
        mask_pad = mask_inp

    # Rough fill for the masked region (mean color of unmasked pixels)
    inpaint = img_pad.copy().astype(np.float32)
    mean_color = inpaint[mask_pad == 0].reshape(-1, 3).mean(axis=0)
    inpaint[mask_pad > 0] = mean_color

    # Prepare batch dict
    batch = {
        "image": torch.from_numpy(img_pad).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0,
        "mask": torch.from_numpy(mask_pad).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0,
        "inpaint_image": torch.from_numpy(inpaint).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0,
    }

    with torch.no_grad():
        out = lama_model(batch)

    # Extract inpainted result, crop padding, and resize back to original
    result = out["inpainted"][0].cpu().permute(1, 2, 0).numpy() * 255
    result = result.astype(np.uint8)
    result = result[:inp_H, :inp_W]
    if need_resize:
        result = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
    return result


def _generate_trajectory(K: int, traj_type: str, img_size, bbox):
    """Generate K (cx, cy, scale) tuples for object placement.

    Args:
        K: number of frames
        traj_type: "horizontal", "vertical", "diagonal", or "zoom"
        img_size: (H, W)
        bbox: (x1, y1, x2, y2) of original object

    Returns list of (cx, cy, scale).
    """
    H, W = img_size
    bx1, by1, bx2, by2 = bbox
    obj_cx = (bx1 + bx2) / 2
    obj_cy = (by1 + by2) / 2

    frames = []
    if traj_type == "horizontal":
        start_x = int(W * 0.15)
        end_x = int(W * 0.85)
        for i in range(K):
            t = i / (K - 1) if K > 1 else 0.5
            cx = int(start_x + t * (end_x - start_x))
            cx += random.randint(-int(W * 0.02), int(W * 0.02))
            cx = max(0, min(W, cx))
            cy = int(obj_cy)
            scale = 1.0 + random.uniform(-0.05, 0.05)
            frames.append((cx, cy, scale))
    elif traj_type == "diagonal":
        # Move from top-left (15%, 15%) to bottom-right (85%, 85%),
        # or from top-right to bottom-left (random direction).
        flip_x = random.choice([1, -1])
        start_x = int(W * (0.15 if flip_x == 1 else 0.85))
        end_x = int(W * (0.85 if flip_x == 1 else 0.15))
        for i in range(K):
            t = i / (K - 1) if K > 1 else 0.5
            cx = int(start_x + t * (end_x - start_x))
            cx += random.randint(-int(W * 0.02), int(W * 0.02))
            cx = max(0, min(W, cx))
            cy = int(obj_cy * (1 - t) + H * 0.85 * t)  # move downward
            cy += random.randint(-int(H * 0.02), int(H * 0.02))
            cy = max(0, min(H, cy))
            scale = 1.0 + random.uniform(-0.05, 0.05)
            frames.append((cx, cy, scale))
    elif traj_type == "zoom":
        min_s, max_s = 0.6, 1.4
        for i in range(K):
            t = i / (K - 1) if K > 1 else 0.5
            scale = min_s + t * (max_s - min_s)
            cx = int(obj_cx)
            cy = int(obj_cy)
            frames.append((cx, cy, scale))
    else:
        frames = [(int(obj_cx), int(obj_cy), 1.0)] * K

    return frames


def _composite_frame(background, object_img, object_mask, cx, cy, scale):
    """Place object onto background at (cx, cy) with given scale.

    Args:
        background: [H, W, 3] RGB uint8
        object_img: [obj_h, obj_w, 3] RGB uint8 (object only, bg zeroed)
        object_mask: [obj_h, obj_w] float32 [0, 1]
        cx, cy: target center position
        scale: scale factor

    Returns [H, W, 3] RGB uint8.
    """
    H, W = background.shape[:2]
    obj_h, obj_w = object_mask.shape

    # Resize object + mask
    new_w = max(1, int(obj_w * scale))
    new_h = max(1, int(obj_h * scale))
    if scale != 1.0:
        obj_resized = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(object_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        obj_resized = object_img
        mask_resized = object_mask

    # Ensure mask is float [0, 1]
    if mask_resized.dtype == np.uint8:
        mask_resized = mask_resized.astype(np.float32) / 255.0
    mask_resized = np.clip(mask_resized, 0, 1)

    # Top-left corner
    x1 = int(cx - new_w // 2)
    y1 = int(cy - new_h // 2)
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Clip to image boundaries
    clip_x1 = max(0, -x1)
    clip_y1 = max(0, -y1)
    clip_x2 = new_w - max(0, x2 - W)
    clip_y2 = new_h - max(0, y2 - H)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x1 >= x2 or y1 >= y2 or clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return background.copy()

    result = background.copy()
    obj_region = obj_resized[clip_y1:clip_y2, clip_x1:clip_x2]
    mask_region = mask_resized[clip_y1:clip_y2, clip_x1:clip_x2, np.newaxis]

    # Blend
    result[y1:y2, x1:x2] = (
        mask_region * obj_region + (1 - mask_region) * result[y1:y2, x1:x2]
    ).astype(np.uint8)

    return result


def _motion_phrase(traj_type: str, t: float, object_class: str, is_zoom_in: bool = True) -> str:
    """Generate a per-frame description at progress t ∈ [0, 1]."""
    if traj_type == "horizontal":
        buckets = [
            (0.0, 0.2, ["starting to move right", "beginning to move right", "appearing on the left"]),
            (0.2, 0.4, ["moving to the right", "moving rightward", "walking to the right"]),
            (0.4, 0.6, ["continuing rightward", "moving further right", "still moving to the right"]),
            (0.6, 0.8, ["moving to the far right", "approaching the right edge"]),
            (0.8, 1.1, ["reaching the right side", "exiting to the right", "arriving on the right"]),
        ]
    elif traj_type == "zoom":
        if is_zoom_in:
            buckets = [
                (0.0, 0.2, ["in the distance", "farther away", "more distant"]),
                (0.2, 0.4, ["coming closer", "moving closer", "approaching"]),
                (0.4, 0.6, ["getting nearer", "continuing to approach"]),
                (0.6, 0.8, ["getting quite close", "nearing the camera"]),
                (0.8, 1.1, ["up close", "very near", "close by"]),
            ]
        else:
            buckets = [
                (0.0, 0.2, ["up close", "very near", "close by"]),
                (0.2, 0.4, ["moving away", "going further"]),
                (0.4, 0.6, ["getting smaller", "becoming smaller", "getting farther"]),
                (0.6, 0.8, ["becoming smaller", "getting farther", "more distant"]),
                (0.8, 1.1, ["in the distance", "farther away", "more distant"]),
            ]
    elif traj_type == "diagonal":
        buckets = [
            (0.0, 0.2, ["starting at the top-left", "beginning from the top"]),
            (0.2, 0.4, ["moving diagonally downward", "moving down and right"]),
            (0.4, 0.6, ["continuing diagonally", "moving further diagonally"]),
            (0.6, 0.8, ["approaching the bottom", "nearing the bottom-right"]),
            (0.8, 1.1, ["reaching the bottom-right", "arriving at the bottom"]),
        ]
    else:
        return f"a {object_class} in view"

    for lo, hi, choices in buckets:
        if lo <= t < hi:
            return f"a {object_class} {random.choice(choices)}"
    return f"a {object_class} {buckets[-1][2][0]}"


_MOTION_CONTINUOUS = {
    "horizontal": [
        "keeps moving rightward",
        "continues moving to the right",
        "moves further right",
    ],
    "zoom": [
        "keeps getting closer",
        "continues approaching",
        "keeps coming closer",
    ],
    "diagonal": [
        "continues moving diagonally",
        "keeps moving diagonally",
        "moves further diagonally",
    ],
}


def _motion_continuous_phrase(traj_type: str, object_class: str) -> str:
    """Generate a continuous-action phrase for a multi-frame group.

    Returns something like "a dog keeps moving rightward".
    """
    phrases = _MOTION_CONTINUOUS.get(traj_type)
    if phrases is None:
        return f"a {object_class} stays in view"
    return f"a {object_class} {random.choice(phrases)}"


def _generate_motion_caption(object_class, traj_type, frames):
    """Generate grouped correct and shuffled captions.

    Consecutive same-direction frames are collapsed into a single
    descriptive phrase, so most sequences use 1-3 connectors rather
    than one per frame.  The grouping distribution is:

    - 35 %  single group   → "a dog keeps moving rightward"
    - 25 %  two groups     → "First, … Then, …"
    - 25 %  three groups   → "First, … Then, … Finally, …"
    - 15 %  per-frame      → one connector per frame (original)

    Returns (correct_caption, shuffled_caption).
    """
    K = len(frames)

    # ── Decide group count ────────────────────────────────────────
    r = random.random()
    if r < 0.35:
        n_groups = 1
    elif r < 0.60:
        n_groups = 2
    elif r < 0.85:
        n_groups = 3
    else:
        n_groups = K  # per-frame (original behaviour)

    # Clamp so we never have more groups than frames
    n_groups = min(n_groups, K)

    # ── Build phrase for each group ───────────────────────────────
    if n_groups == K:
        # Per-frame: unchanged original logic
        caption_parts = []
        for i in range(K):
            t = i / (K - 1) if K > 1 else 0.5
            caption_parts.append(_motion_phrase(traj_type, t, object_class))
    else:
        # Divide K frames into n_groups roughly-equal contiguous blocks
        boundaries = [0]
        for g in range(1, n_groups + 1):
            boundaries.append(int(round(g * K / n_groups)))
        boundaries[-1] = K  # avoid rounding past the end

        caption_parts = []
        for g in range(n_groups):
            start, end = boundaries[g], boundaries[g + 1]
            group_size = end - start
            if group_size == 1:
                t = start / (K - 1) if K > 1 else 0.5
                caption_parts.append(_motion_phrase(traj_type, t, object_class))
            elif n_groups == 1:
                # Entire sequence as one continuous motion
                caption_parts.append(_motion_continuous_phrase(traj_type, object_class))
            else:
                # Multi-frame group — decide: "start/middle/end" vs. continuous
                if g == 0:
                    # First group → starting phrase at the group midpoint
                    t = (start + end - 1) / 2 / (K - 1) if K > 1 else 0.5
                    caption_parts.append(_motion_phrase(traj_type, t, object_class))
                elif g == n_groups - 1:
                    # Last group → ending phrase
                    t = (start + end - 1) / 2 / (K - 1) if K > 1 else 0.5
                    caption_parts.append(_motion_phrase(traj_type, t, object_class))
                else:
                    # Middle group → continuous form
                    caption_parts.append(_motion_continuous_phrase(traj_type, object_class))

    # ── Build connectors and final captions ───────────────────────
    if len(caption_parts) <= 1:
        # Single continuous action — no connector needed
        correct_caption = caption_parts[0] if caption_parts else ""
        shuffled_caption = correct_caption
    else:
        style = random.choice(["index", "sequential"])
        connectors = _build_connectors(len(caption_parts), style)
        correct_caption = _format_caption(connectors, caption_parts)
        shuffled = list(range(len(caption_parts)))
        while shuffled == list(range(len(caption_parts))):
            random.shuffle(shuffled)
        shuffled_parts = [caption_parts[i] for i in shuffled]
        shuffled_caption = _format_caption(connectors, shuffled_parts)

    return correct_caption, shuffled_caption


def generate_motion_frames_only(img_id, image_path, valid_classes, min_area_ratio,
                                 max_area_ratio, conf_threshold, yolo_model,
                                 lama_model, device, min_len, max_len, inpaint_size=256):
    """Generate one motion pseudo-video without CLIP encoding.

    Returns (frame_images, correct_caption, shuffled_caption, meta) or None.

    CLIP encoding is deferred so that many sequences can share a single
    batched ``encode_frames`` call (see ``_flush_motion_buffer``).
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # 1. YOLO seg
    results = yolo_model(img_rgb, verbose=False)
    picked = _pick_object(results, valid_classes, min_area_ratio,
                          max_area_ratio, conf_threshold, (H, W))
    if picked is None:
        return None

    mask, bbox, cls_id, _ = picked
    class_name = yolo_model.names[cls_id]
    class_name = class_name.replace("_", " ").replace("-", " ")

    # 2. Extract object from mask
    bx1, by1, bx2, by2 = bbox
    margin = 10
    bx1 = max(0, bx1 - margin)
    by1 = max(0, by1 - margin)
    bx2 = min(W, bx2 + margin)
    by2 = min(H, by2 + margin)

    obj_mask = mask[by1:by2, bx1:bx2].astype(np.float32)
    obj_img = img_rgb[by1:by2, bx1:bx2].copy()

    # 3. Inpaint background
    inpaint_mask = np.zeros((H, W), dtype=np.uint8)
    inpaint_mask[by1:by2, bx1:bx2] = (mask[by1:by2, bx1:bx2] > 0).astype(np.uint8) * 255
    background = _inpaint_background(lama_model, img_rgb, inpaint_mask, device, inpaint_size)

    # 4. Trajectory
    K = random.randint(min_len, max_len)
    traj_type = random.choice(["horizontal", "diagonal", "zoom"])
    frames = _generate_trajectory(K, traj_type, (H, W), (bx1, by1, bx2, by2))

    # 5. Composite (no CLIP yet — deferred for batching)
    frame_images = []
    for cx, cy, scale in frames:
        frame = _composite_frame(background, obj_img, obj_mask, cx, cy, scale)
        frame_images.append(frame)

    # 6. Captions
    correct_caption, shuffled_caption = _generate_motion_caption(
        class_name, traj_type, frames
    )

    # Normalise pixel positions to [0, 1] so they are scale-invariant
    positions = [(cx / W, cy / H, scale) for cx, cy, scale in frames]

    meta = {
        "type": "motion",
        "img_id": img_id,
        "class_name": class_name,
        "trajectory": traj_type,
        "K": K,
        "positions": positions,
    }

    return frame_images, correct_caption, shuffled_caption, meta


# ═══════════════════════════════════════════════════════════════════════════
#  Connector-sequence helpers
# ═══════════════════════════════════════════════════════════════════════════

def generate_connector_sample(img_ids_pool, id_to_captions, emb_cache,
                              min_len, max_len):
    """Generate one connector pseudo-video from precomputed embeddings.

    Returns (embs, correct_caption, shuffled_caption, meta) or None.
    """
    K = random.randint(min_len, max_len)
    sampled = random.sample(img_ids_pool, K)

    # Load precomputed embeddings
    embs_list = []
    for sid in sampled:
        path = emb_cache / f"{sid}.npy"
        if not path.exists():
            return None
        embs_list.append(np.load(path).astype(np.float32))

    frame_embs = np.stack(embs_list)  # [K, 512]

    # Captions
    captions = [random.choice(id_to_captions[sid]) for sid in sampled]
    style = random.choice(["index", "sequential"])
    connectors = _build_connectors(K, style)
    correct_caption = _format_caption(connectors, captions)

    shuffled = list(range(K))
    while shuffled == list(range(K)):
        random.shuffle(shuffled)
    shuffled_captions = [captions[i] for i in shuffled]
    shuffled_caption = _format_caption(connectors, shuffled_captions)

    meta = {"type": "connector", "img_ids": sampled, "K": K}
    return frame_embs, correct_caption, shuffled_caption, meta


# ═══════════════════════════════════════════════════════════════════════════
#  Batched CLIP flush
# ═══════════════════════════════════════════════════════════════════════════

def _flush_motion_buffer(buffer, clip_encoder, next_id, out_dir,
                          all_samples, vis_count, vis_target, vis_dir):
    """Batch-CLIP-encode all frames in *buffer* and save to disk.

    Returns (new_next_id, new_vis_count).
    """
    if not buffer:
        return next_id, vis_count

    # Collect all frame images from every buffered sequence
    all_frames = []
    for item in buffer:
        all_frames.extend(item["frame_images"])

    # Single batched CLIP forward pass
    all_embs = clip_encoder.encode_frames(all_frames)  # [total_frames, 512]

    # Split embeddings back per sequence and save
    idx = 0
    for item in buffer:
        K = item["K"]
        embs = all_embs[idx: idx + K]
        idx += K
        sid = next_id
        next_id += 1

        np.save(out_dir / f"seq_{sid:06d}.npy", embs.astype(np.float32))
        with open(out_dir / f"seq_{sid:06d}.json", "w") as f:
            json.dump({
                "correct_caption": item["correct_caption"],
                "shuffled_caption": item["shuffled_caption"],
                "type": "motion",
                "img_id": item["meta"]["img_id"],
                "K": K,
                "positions": item["meta"]["positions"],
            }, f)

        all_samples.append({
            "id": sid,
            "type": "motion",
            "K": K,
        })

        # ── Visualization ──
        if vis_dir is not None and vis_count < vis_target:
            sample_dir = vis_dir / f"sample_{vis_count:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for fi, frame_rgb in enumerate(item["frame_images"]):
                cv2.imwrite(
                    str(sample_dir / f"frame_{fi:03d}.jpg"),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
            with open(sample_dir / "caption.txt", "w") as f:
                f.write(item["correct_caption"] + "\n")
            vis_count += 1

    return next_id, vis_count


# ═══════════════════════════════════════════════════════════════════════════
#  Multiprocessing workers for motion generation
# ═══════════════════════════════════════════════════════════════════════════

def _motion_worker(work_queue, result_queue, progress_queue,
                   cfg, device_str, worker_id):
    """Worker process: load YOLO + LaMa, generate sequences, push results.

    Reads ``(img_id, image_path)`` from *work_queue*, runs the full YOLO →
    LaMa → compositing → caption pipeline, and puts the result tuple into
    *result_queue*.  Exits when it reads ``None`` from the work queue.

    Pushes ``worker_id`` to *progress_queue* after each successful
    sequence so the main process can render per-worker tqdm bars.

    The *cfg* dict contains all configuration scalar/string parameters
    (pickled once per worker at startup).
    """
    import random as _rnd
    _rnd.seed(cfg["seed"] + worker_id * 1000)

    import numpy as np
    np.random.seed(cfg["seed"] + worker_id * 1000 + 1)

    import torch
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"] + worker_id * 1000 + 2)

    import cv2
    from pathlib import Path

    from ultralytics import YOLO
    from modelscope.models.cv.image_inpainting import FFTInpainting

    device = torch.device(device_str)
    yolo = YOLO(cfg["yolo_path"])
    lama = FFTInpainting(model_dir=cfg["lama_path"]).to(device)
    lama.eval()

    valid_classes = set(cfg["valid_classes"])
    sentinel = None  # poison pill

    while True:
        item = work_queue.get()
        if item is sentinel:
            break
        img_id, image_path = item

        try:
            result = generate_motion_frames_only(
                img_id, image_path,
                valid_classes, cfg["min_area"], cfg["max_area"],
                cfg["conf_thresh"], yolo, lama,
                device, cfg["min_len"], cfg["max_len"],
                cfg.get("inpaint_size", 256),
            )
            if result is not None:
                result_queue.put(result)
                if progress_queue is not None:
                    progress_queue.put(worker_id)
        except Exception as e:
            print(f"  [Worker {worker_id}] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue

    del yolo, lama
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _feed_worker(work_queue, ids_pool, image_root, id_to_file_map,
                 count, rng_seed):
    """Feed (img_id, image_path) items into the work queue (daemon thread)."""
    rng = random.Random(rng_seed)
    for _ in range(count):
        img_id = rng.choice(ids_pool)
        img_path = str(image_root / id_to_file_map[img_id])
        work_queue.put((img_id, img_path))


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Precompute pseudo-video sequences (motion + connector)."
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--total_sequences", type=int, default=50000,
                        help="Total sequences to generate (train)")
    parser.add_argument("--val_sequences", type=int, default=2500,
                        help="Total sequences to generate (val)")
    parser.add_argument("--motion_ratio", type=float, default=1,
                        help="Ratio of motion vs connector sequences (0-1)")
    parser.add_argument("--device", default=None,
                        help="Device for GPU models")
    parser.add_argument("--visualize", type=int, default=50,
                        help="Number of motion samples to save as images")
    parser.add_argument("--clip_batch_size", type=int, default=256,
                        help="Number of frames per batched CLIP encode call")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel worker processes for motion generation (0/1 = sequential)")
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

    # ── Load models ────────────────────────────────────────────────
    print(f"Device: {device}")

    import torch
    torch.set_num_threads(4)

    # With workers > 1, YOLO/LaMa run inside worker processes.
    # Only the main process needs the CLIP encoder for batched encoding.
    use_parallel = args.workers > 1

    if not use_parallel:
        print("Loading YOLO segmentation model...")
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_path)

        print("Loading LaMa inpainting model...")
        from modelscope.models.cv.image_inpainting import FFTInpainting
        lama_model = FFTInpainting(model_dir=lama_path).to(device)
        lama_model.eval()
    else:
        print("  (YOLO/LaMa loaded per worker process)")
        yolo_model = None
        lama_model = None

    print("Loading CLIP encoder...")
    from clip_search.encoder import CLIPEncoder
    clip_model = cfg.get("clip", {}).get("model", "laion")
    clip_encoder = CLIPEncoder(model_type=clip_model, device=device)

    # ── Load COCO annotations ─────────────────────────────────────
    for split in ("train2017", "val2017"):
        _, id_to_captions, id_to_file = _load_coco(split, ann_dir)
        if split == "train2017":
            train_captions, train_file = id_to_captions, id_to_file
        else:
            val_captions, val_file = id_to_captions, id_to_file

    # Images that have cached embeddings
    cached_ids = set(int(p.stem) for p in emb_cache.glob("*.npy"))
    train_ids = [i for i in train_file if i in cached_ids]
    val_ids = [i for i in val_file if i in cached_ids]
    print(f"Train images with embeddings: {len(train_ids)}")
    print(f"Val images with embeddings:   {len(val_ids)}")

    # Explicitly verify NO image overlap between train and val
    train_set, val_set = set(train_ids), set(val_ids)
    overlap = train_set & val_set
    if overlap:
        print(f"\n⚠  CRITICAL: {len(overlap)} images appear in BOTH train2017 and val2017!")
        print(f"  Overlapping IDs: {sorted(overlap)[:20]}{' ...' if len(overlap) > 20 else ''}")
        sys.exit(1)
    print("  ✓ No image overlap between train2017 and val2017")

    # ── Generate sequences for each split ─────────────────────────
    for split, total in [
        ("train2017", args.total_sequences),
        ("val2017", args.val_sequences),
    ]:
        if total == 0:
            continue
        print(f"\n{'='*60}")
        print(f"Generating {total} sequences for {split}")
        print(f"{'='*60}")

        out_dir = cache_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        id_to_captions = train_captions if split == "train2017" else val_captions
        id_to_file_map = train_file if split == "train2017" else val_file
        image_root = coco_image_root / split
        ids_pool = train_ids if split == "train2017" else val_ids

        target_motion = max(0, int(total * args.motion_ratio))
        target_connector = total - target_motion

        all_samples = []
        vis_count = 0
        next_id = 0
        motion_done = 0
        conn_done = 0

        # ── Motion sequences ──────────────────────────────────
        if target_motion > 0:
            print(f"Generating {target_motion} motion sequences... "
                  f"({args.workers} worker{'s' if args.workers > 1 else ''})")
            motion_buffer = []
            motion_needed = target_motion

            if args.workers > 1:
                # ── Parallel path ──────────────────────────────
                worker_cfg = {
                    "valid_classes": list(valid_classes),
                    "min_area": min_area,
                    "max_area": max_area,
                    "conf_thresh": conf_thresh,
                    "min_len": min_len,
                    "max_len": max_len,
                    "inpaint_size": inpaint_size,
                    "yolo_path": yolo_path,
                    "lama_path": lama_path,
                    "seed": 42,
                }
                ctx = mp.get_context("spawn")
                work_queue = ctx.Queue(maxsize=args.workers * 4)
                result_queue = ctx.Queue()
                progress_queue = ctx.Queue()

                workers = [
                    ctx.Process(target=_motion_worker,
                                args=(work_queue, result_queue, progress_queue,
                                      worker_cfg, device, i))
                    for i in range(args.workers)
                ]
                for w in workers:
                    w.start()

                # Feeder thread: generates work items on the fly
                feed_count = motion_needed * 5
                feeder = threading.Thread(
                    target=_feed_worker,
                    args=(work_queue, ids_pool, image_root, id_to_file_map,
                          feed_count, 43),
                    daemon=True,
                )
                feeder.start()

                # Per-worker progress bars (one line each, no overlap)
                worker_progress = [0] * args.workers
                worker_bars = [
                    tqdm(
                        desc=f"  Worker {i}",
                        position=i,
                        leave=False,
                        unit="seq",
                        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    )
                    for i in range(args.workers)
                ]
                overall_bar = tqdm(
                    desc="Total",
                    position=args.workers,
                    leave=True,
                    unit="seq",
                    bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
                overall_bar.reset(total=motion_needed)

                motion_done = 0

                import queue as _queue
                while motion_done < motion_needed:
                    # Drain progress updates so per-worker bars stay current
                    while True:
                        try:
                            wid = progress_queue.get_nowait()
                            worker_progress[wid] += 1
                            worker_bars[wid].update(1)
                        except _queue.Empty:
                            break

                    result = result_queue.get()
                    frame_imgs, corr_cap, shuf_cap, meta = result
                    motion_buffer.append({
                        "frame_images": frame_imgs,
                        "correct_caption": corr_cap,
                        "shuffled_caption": shuf_cap,
                        "meta": meta,
                        "K": meta["K"],
                    })
                    motion_done += 1
                    overall_bar.update(1)

                    total_frames = sum(item["K"] for item in motion_buffer)
                    if total_frames >= args.clip_batch_size:
                        vis_dir = cache_dir / "visualization" if split == "train2017" else None
                        next_id, vis_count = _flush_motion_buffer(
                            motion_buffer, clip_encoder, next_id, out_dir,
                            all_samples, vis_count, args.visualize, vis_dir,
                        )
                        motion_buffer = []

                # Flush remaining
                if motion_buffer:
                    vis_dir = cache_dir / "visualization" if split == "train2017" else None
                    next_id, vis_count = _flush_motion_buffer(
                        motion_buffer, clip_encoder, next_id, out_dir,
                        all_samples, vis_count, args.visualize, vis_dir,
                    )
                    motion_buffer = []

                # Drain any remaining progress updates before closing bars
                while True:
                    try:
                        wid = progress_queue.get_nowait()
                        worker_progress[wid] += 1
                        worker_bars[wid].update(1)
                    except _queue.Empty:
                        break

                # Cleanup workers
                # Drain remaining work items so poison pills can get through
                import queue as _q
                for _ in range(feed_count):
                    try:
                        work_queue.get_nowait()
                    except _q.Empty:
                        break
                for _ in workers:
                    work_queue.put(None)
                # Drain extra results workers produce before receiving poison,
                # otherwise the result_queue pipe fills up → deadlock on put()
                while any(w.is_alive() for w in workers):
                    try:
                        result_queue.get(timeout=0.5)
                    except _q.Empty:
                        pass
                for w in workers:
                    w.join()

                for bar in worker_bars:
                    bar.close()
                overall_bar.close()

            else:
                # ── Sequential path ────────────────────────────
                pbar = tqdm(total=motion_needed, desc="Motion")
                attempts = 0
                max_attempts = motion_needed * 10

                while motion_done < motion_needed and attempts < max_attempts:
                    attempts += 1
                    img_id = random.choice(ids_pool)
                    img_path = image_root / id_to_file_map[img_id]

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
                            vis_dir = cache_dir / "visualization" if split == "train2017" else None
                            next_id, vis_count = _flush_motion_buffer(
                                motion_buffer, clip_encoder, next_id, out_dir,
                                all_samples, vis_count, args.visualize, vis_dir,
                            )
                            motion_buffer = []

                    except Exception:
                        continue

                # Flush remaining
                if motion_buffer:
                    vis_dir = cache_dir / "visualization" if split == "train2017" else None
                    next_id, vis_count = _flush_motion_buffer(
                        motion_buffer, clip_encoder, next_id, out_dir,
                        all_samples, vis_count, args.visualize, vis_dir,
                    )
                    motion_buffer = []

                pbar.close()
                if motion_done < motion_needed:
                    print(f"  Warning: only generated {motion_done}/{motion_needed} motion sequences")

        # ── Connector sequences ───────────────────────────────
        if target_connector > 0:
            print(f"Generating {target_connector} connector sequences...")
            pbar = tqdm(total=target_connector, desc="Connector")
            conn_attempts = 0
            max_conn_attempts = target_connector * 5

            while conn_done < target_connector and conn_attempts < max_conn_attempts:
                conn_attempts += 1
                try:
                    result = generate_connector_sample(
                        ids_pool, id_to_captions, emb_cache, min_len, max_len,
                    )
                    if result is None:
                        continue
                    embs, corr_cap, shuf_cap, meta = result

                    np.save(out_dir / f"seq_{next_id:06d}.npy", embs.astype(np.float32))
                    with open(out_dir / f"seq_{next_id:06d}.json", "w") as f:
                        json.dump({
                            "correct_caption": corr_cap,
                            "shuffled_caption": shuf_cap,
                            "type": "connector",
                            "img_ids": meta["img_ids"],
                            "K": meta["K"],
                        }, f)

                    all_samples.append({
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

        # ── Shuffle and save index ────────────────────────────
        random.shuffle(all_samples)
        with open(out_dir / "samples.json", "w") as f:
            json.dump(all_samples, f, indent=2)
        print(f"  Saved {len(all_samples)} samples (motion={motion_done}, connector={conn_done}) to {out_dir}")
        print(f"  samples.json written.")

    # ── Summary ────────────────────────────────────────────────────
    vis_path = cache_dir / "visualization"
    if vis_path.exists():
        n_vis = len(list(vis_path.iterdir()))
        print(f"\nVisualization: {n_vis} motion samples saved to {vis_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
