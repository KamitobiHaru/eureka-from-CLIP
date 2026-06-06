# eureka-from-CLIP

**Text-to-Video Scene Retrieval with BERT-Aligned CLIP + Temporal Transformer**

Retrieve video scenes using natural language queries. Supports two pipelines:
- **Legacy**: CLIP encode → mean pool over frames → cosine similarity search
- **Temporal** (new): CLIP per-frame features → TemporalTransformer (learnable PE + attention) → BERT text encoder

---

## Pipeline Overview

```
Video → scene segmentation → T frames/scene → CLIP Vision (per frame) → TemporalTransformer → scene embedding
                                                                                                ↓
                                                                                         cosine similarity
                                                                                                ↑
Query text → BERT + Projection → query embedding
```

Two training stages:
1. **Pre-extract CLIP features** — offline, one-time
2. **Temporal training** — jointly train TemporalTransformer + BERT on synthetic sequences from COCO images

---

## Installation

```bash
git clone <repo>
cd eureka-from-CLIP
pip install -r requirements.txt
```

### Dependencies
- Python 3.9+, PyTorch 2.0+, CUDA-capable GPU recommended
- `open_clip_torch`, `transformers`, `opencv-python`, `scenedetect`, `pyyaml`, `numpy`, `tqdm`

### Download Pre-trained Models

```bash
# BERT-base (used for text encoding)
bash scripts/download_bert.sh

# CLIP ViT-B/32 (used for visual encoding)
python scripts/download_clip_model.py
```

---

## Data Preparation

### 1. Download MS COCO

You need the COCO 2017 train/val images and captions. Set `coco_root` in `config/default.yaml` to point to your COCO directory.

Expected structure:
```
{coco_root}/
├── train2017/             # 118K images
├── val2017/               # 5K images
└── annotations_trainval2017/annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

### 2. Pre-extract CLIP Features

**Required before any training. This is a one-time step.**

```bash
python scripts/precompute_embeddings.py
```

This encodes every COCO image through CLIP ViT-B/32 and saves 512-dim L2-normalized embeddings as `.npy` files to `data/coco/clip_embeddings/`. This takes ~30 minutes on a GPU.

---

## Training

### Stage 1: Train BERT Alignment (Legacy)

Trains BERT-base to align with CLIP's visual embedding space using standard image-caption pairs.

```bash
python scripts/train_bert.py --config config/default.yaml
```

Checkpoints saved to `checkpoints/bert_epoch*.pt`.

### Stage 2: Train TemporalTransformer + BERT (New)

Jointly trains the temporal transformer and BERT on **synthetic pseudo-video sequences** constructed from random COCO images.

**How synthetic sequences work:**
- Each sample groups K=3~10 random COCO images (uniform random)
- Picks one caption per image
- Builds a **correct** caption by joining captions with temporal connectors in image order
  - Index-based: *"First, a dog runs. Second, a car passes. Third, a bird flies. Last, sunset."*
  - Sequential: *"To begin with, a dog runs. After that, a car passes. Finally, sunset."*
- Builds a **shuffled** caption using the same captions in permuted order (negative sample for temporal loss)
  - *"First, sunset. Second, a dog runs. Third, a car passes. Last, a bird flies."* (wrong order)

**Two losses:**
- `SymmetricInfoNCE(video_emb, correct_text_emb)` — **semantic alignment**: scene embedding matches its correct description
- `OrderConsistencyLoss(video_emb, correct_text_emb, wrong_text_emb)` — **temporal order**: correct description must be closer than shuffled description (triplet margin)

```bash
python scripts/train_temporal.py --config config/default.yaml
```

Checkpoints saved to `checkpoints/temporal_epoch*.pt`.

### Key Training Parameters (in `config/default.yaml` under `temporal:`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 2 | Transformer encoder layers (1-3, trade-off speed vs modeling) |
| `nhead` | 8 | Attention heads (4 or 8; 8 is standard for 512-dim) |
| `dim_feedforward` | 1024 | FFN hidden dim (1024 is half of standard — lightweight) |
| `dropout` | 0.1 | Dropout rate (increase to 0.2 if overfitting) |
| `max_frames` | 16 | Max frames for learnable PE table |
| `sequence_min_len` | 3 | Min images per synthetic sequence |
| `sequence_max_len` | 10 | Max images per synthetic sequence |
| `order_consistency_margin` | 0.2 | Triplet margin for order loss (higher = stricter order penalty) |
| `order_consistency_weight` | 0.5 | λ weight balancing order loss vs semantic loss |
| `temporal_lr` | 1e-4 | Learning rate for TemporalTransformer (trained from scratch) |
| `bert_lr` | 3e-5 | Learning rate for BERT (fine-tuning) |
| `save_every_epoch` | true | Save checkpoint at every epoch; if false, save best only |
| `warmup_steps` | 6000 | Linear warmup before cosine decay (in `training:` section) |
| `batch_size` | 64 | Batch size (in `training:` section) |

### How to Tune

**If temporal sensitivity is weak** (shuffled vs correct captions give similar scores):
- Increase `order_consistency_weight` (e.g., 0.5 → 1.0)
- Increase `order_consistency_margin` (e.g., 0.2 → 0.3)
- Add more `num_layers` (e.g., 2 → 3)

**If overfitting** (val loss diverges from train loss):
- Increase `dropout` (e.g., 0.1 → 0.2)
- Decrease `temporal_lr` (e.g., 1e-4 → 5e-5)
- Decrease `sequence_max_len` (e.g., 10 → 8) for simpler sequences

**If training is unstable** (loss spikes):
- Increase `warmup_steps` (e.g., 6000 → 10000)
- Decrease `temporal_lr` (e.g., 1e-4 → 5e-5)

**If semantic alignment is poor** (search results don't match content):
- Increase `order_consistency_weight` may hurt semantic alignment — try decreasing it (e.g., 0.5 → 0.3) to focus more on InfoNCE
- Check BERT fine-tuning isn't too aggressive: keep `bert_lr` at 3e-5 or lower

---

## Inference

### Prerequisites

First, segment a video into scenes:

```bash
python scripts/segment_video.py demo.mp4 -o ./segments/demo
```

This saves scene thumbnails, frame images, and `metadata.json` to `./segments/demo/`.

### CLI Search

**Legacy mode** (CLIP mean-pool + CLIP text encoder):
```bash
python run.py ./segments/demo --query "a person walking"
```

**Legacy + BERT mode**:
```bash
python run.py ./segments/demo --query "a person walking" \
    --bert_checkpoint checkpoints/bert_epoch02_val1.5993.pt
```

**New temporal mode** (CLIP per-frame → TemporalTransformer + BERT):
```bash
python run.py ./segments/demo --query "a person walking then sitting" \
    --temporal_checkpoint checkpoints/temporal_epoch03_val1.2345.pt
```

### Gradio Web UI

```bash
python app.py
```

Open the browser URL. In the "Process Video" tab:
1. Upload a video
2. (Optional) Expand "Temporal Transformer", enable checkbox, enter checkpoint path
3. Click "Process Video"

Then switch to "Search Scenes" tab to query.

---

## Verification / Testing

### Test the Training Pipeline (Quick Check)

Run 1 epoch on a small subset to verify loss decreases:

```bash
# Override for fast test: reduce epochs, use fewer samples
python scripts/train_temporal.py --config config/default.yaml --checkpoint_dir /tmp/test_ckpt
```

Check that both `nce_loss` and `order_loss` decrease and are non-zero in the progress bar.

### Test Temporal Sensitivity

After training, verify the temporal model actually understands order:

```bash
python run.py ./segments/demo \
    --query "first a person walks in then sits down finally waves" \
    --temporal_checkpoint checkpoints/temporal_best.pt
```

Then compare with the same query in legacy mode — the temporal model should rank scenes where the action unfolds in that order higher.

### Regression Test

Ensure legacy mode still works identically:

```bash
python run.py ./segments/demo --query "a person" -o /tmp/legacy
python run.py ./segments/demo --query "a person" -t checkpoint.pt -o /tmp/temporal
```

Results should differ (different encoders) but both should produce plausible results.

---

## Project Structure

```
├── clip_search/                  # Inference package
│   ├── encoder.py                # CLIP encoder (encode_scene, encode_frames, encode_text)
│   ├── engine.py                 # SearchEngine (segment → encode → rank)
│   ├── segmenter.py              # PySceneDetect wrapper
│   └── temporal_pipeline.py      # Temporal inference factory
├── src/
│   ├── models/
│   │   ├── bert_encoder.py       # BERT-base + ProjectionHead
│   │   └── temporal_transformer.py  # Learnable PE + TransformerEncoder
│   ├── data/
│   │   ├── coco_dataset.py       # Standard COCO image-caption pairs
│   │   └── sequence_dataset.py   # Synthetic pseudo-video sequences
│   └── training/
│       ├── loss.py               # SymmetricInfoNCE + OrderConsistencyLoss
│       └── trainer.py            # Training loop (used by train_bert.py)
├── scripts/
│   ├── precompute_embeddings.py  # Extract CLIP features from COCO
│   ├── train_bert.py             # Train BERT alignment (legacy)
│   ├── train_temporal.py         # Joint temporal + BERT training
│   ├── segment_video.py          # CLI: video → scenes
│   ├── download_bert.sh          # Download BERT-base
│   └── download_clip_model.py    # Download CLIP ViT-B/32
├── config/default.yaml           # All configuration
├── run.py                        # CLI search entry point
└── app.py                        # Gradio web UI
```

## Citation

```bibtex
@software{eureka-from-CLIP,
  author = {ZhangSiYuan},
  title = {eureka-from-CLIP: Text-to-Video Retrieval with BERT-Aligned CLIP Embeddings},
  year = {2026},
}
```
