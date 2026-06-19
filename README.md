# eureka-from-CLIP

**Text-to-Video Scene Retrieval with BERT-Aligned CLIP**

Retrieve video scenes using natural language queries. Contrastive training aligns a text encoder (BERT or CLIP) with frozen CLIP vision embeddings. Video representations are obtained by mean-pooling per-frame CLIP embeddings.

---

## Pipeline

```
                              ┌──────────────────────┐
Video → PySceneDetect ──────→ │  T frames / scene     │
                              └──────────┬───────────┘
	                         ↓
                         ┌───────────────────────────────┐
                         │  CLIP ViT-B/32 Vision Encoder  │  (frozen)
                         │  (per-frame → L2-normed 512d) │
                         └──────────────────┬────────────┘
                                        mean pool
                                           ↓
                     ┌─────────────────────┴─────────────────────┐
                     │         scene_emb (L2-normed)             │
                     └─────────────────────┬─────────────────────┘
                                           ↓
                                   cosine similarity  ←───────  text embedding
                                                                 ↑
                                                       ┌─────────────────┐
                                                       │ Text Encoder    │
                                                       │ (BERT or CLIP)  │
                                                       └────────┬────────┘
                                                                │
                                                         "a person walking"
```

Two text encoder options, selected via `text_encoder.type` in config:

| Type | Encoder | Trainable | Best for |
|------|---------|-----------|----------|
| `bert` | BERT-base + ProjectionHead(768→512) | Full fine-tuning or LoRA | Higher recall when trained with queue + uniformity |
| `clip` | CLIP ViT-B/32 text encoder (open_clip) | LoRA only (base frozen) | Quick alignment, smaller checkpoints |

Scene encoding uses mean-pooled CLIP per-frame embeddings (L2-normed).

---

## Installation

```bash
git clone <repo>
cd eureka-from-CLIP
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+, PyTorch 2.0+, CUDA-capable GPU recommended
- `open_clip_torch`, `transformers`, `opencv-python`, `scenedetect`, `gradio`, `pyyaml`, `peft`, `numpy`, `tqdm`

### Download Models

**CLIP ViT-B/32** (OpenAI WIT-400M):

The model checkpoint is auto-detected from these paths (in order):
1. `models/clip-vit-base-patch32/open_clip_model.safetensors` — converted from HuggingFace (recommended)
2. `models/clip/openai_open_clip_model.safetensors` — downloaded via modelscope
3. `models/clip/openai_pytorch_model.bin` — legacy torch format

To convert from HuggingFace:

```bash
# Clone the HF model repo first, then:
python scripts/convert_hf_to_openclip.py
```

To download via modelscope (older method):

```bash
python scripts/download_openai_clip.py
```

**BERT-base** (for the BERT text encoder option):

```bash
# Download to models/bert-base-uncased/
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

---

## Data Preparation

### 1. COCO 2017

Download COCO 2017 train/val images and captions. Set paths in config:

```yaml
data:
  coco_root: ./dataset_annotation
  annotations_dir: annotations_trainval2017/annotations
```

Expected structure:

```
{data.coco_root}/
├── train2017/             # 118K images
├── val2017/               # 5K images
└── {data.annotations_dir}/
    ├── captions_train2017.json
    └── captions_val2017.json
```

### 2. Flickr30k (optional, for evaluation)

Download Flickr30k images and annotations. Expects:

```yaml
flickr:
  root: ./dataset_annotation
  annotation_file: flickr_annotations_30k.csv
  embedding_cache: ./data/flickr30k/clip_embeddings
```

The annotation CSV must have columns: `filename`, `split` (train/test), `raw` (JSON list of 5 captions).

### 3. Pre-extract CLIP Features

**Required before any training.** Encodes every image through CLIP ViT-B/32 vision encoder and saves 512-dim L2-normalized embeddings as `.npy` files.

```bash
# COCO
python scripts/precompute_embeddings.py

# Flickr30k (optional, needed for evaluation)
python scripts/precompute_flickr_embeddings.py
```

---

## Training

### Option A: BERT Alignment (Legacy, `text_encoder.type: bert`)

Trains BERT-base to align with CLIP's frozen visual embedding space using image-caption pairs from COCO (+ optionally Flickr30k). Supports LoRA fine-tuning, contrastive queue with false-negative masking, and text uniformity regularization.

```bash
python scripts/train_bert.py --config config/default.yaml
```

Key config parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.batch_size` | 128 | Batch size |
| `training.lr` | 3e-4 | Learning rate (AdamW) |
| `training.t2i_weight` | 0.75 | Weight for text→image direction (>0.5 = more weight on t2i) |
| `training.uniformity_weight` | 1.25 | Text uniformity regularizer (0 = disabled) |
| `lora.enabled` | true | LoRA for BERT (targets: key/query/value/output.dense) |
| `queue.max_size` | 49152 | Contrastive queue size (0 = disabled) |
| `queue.mask_stale_texts` | true | i2t uses in-batch negatives only (avoids stale queue texts) |

Checkpoints saved to `{training.checkpoint_dir}/bert_epoch{nn}_t2i{recall}.pt`.

### Option B: CLIP Text Encoder Alignment (`text_encoder.type: clip`)

Trains only LoRA adapters on CLIP's own text transformer (base frozen). The same SymmetricInfoNCE loss aligns text → frozen CLIP vision embeddings. Since both vision and text start from CLIP's pretrained space, this converges much faster.

```bash
python scripts/train_bert.py --config config/default_clip_text.yaml
```

Config differences from BERT mode:

```yaml
text_encoder:
  type: clip
  clip_lora:
    enabled: true
    r: 8
    alpha: 16
    target_modules: ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
```

Note: CLIP's transformer uses fused QKV (`in_proj_weight` in `nn.MultiheadAttention`), so LoRA is applied to output projections and MLP layers rather than query/value.


---

## Evaluation

### CLIP Zero-Shot Baseline

Evaluate CLIP's own text encoder on Flickr30k to get the ceiling performance:

```bash
python scripts/evaluate_clip_text_encoder.py --config config/default.yaml
```

### BERT Checkpoint Evaluation

Runs recall@K on COCO val2017 + Flickr30k test:

```bash
python scripts/evaluate_checkpoint.py <checkpoint.pt> --config config/default.yaml
```

### Multilingual Evaluation (Cross-Lingual Transfer)

If using multilingual BERT (`models/bert-base-multilingual-cased`), evaluate on Chinese Flickr30k:

```bash
python scripts/evaluate_multilingual.py \
    --checkpoint ../weights/bert_best.pt \
    --config config/default_multilingual.yaml \
    --device cuda:0
```

---

## Inference

### 1. Segment a Video

```bash
python scripts/segment_video.py demo.mp4 -o ./segments
```

Saves scene thumbnails, frames, and `metadata.json` to `./segments/demo/`.

### 2. Search

**Legacy mode** (CLIP mean-pool + CLIP text encoder):

```bash
python run.py ./segments/demo --query "a person walking"
```

**BERT** (trained text encoder):

```bash
python run.py ./segments/demo --query "a person walking" \
    --bert_checkpoint checkpoints/bert_epoch02_val1.5993.pt
```

### Gradio Web UI

```bash
python app.py
```

Opens a browser UI with two tabs:
1. **Process Video** — upload a video, detect scenes, extract frame embeddings
2. **Search Scenes** — query indexed scenes, view ranked results with thumbnails

---

## Project Structure

```
├── clip_search/                      # Inference package
│   ├── encoder.py                    # CLIP ViT-B/32: encode_scene, encode_frames, encode_text
│   ├── engine.py                     # SearchEngine: segment → encode → rank
│   ├── segmenter.py                  # PySceneDetect wrapper with short-scene merging
│   └── text_encoder_loader.py        # Shared text encoder loading (BERT / CLIP)
├── src/
│   ├── models/
│   │   ├── bert_encoder.py           # BERT-base + ProjectionHead (768→512) + LoRA + Stack LoRA
│   │   ├── clip_text_encoder.py      # CLIP text transformer (open_clip) + LoRA (differentiable forward)
│   │   └── mlp_encoder.py            # Frozen BERT + 3-layer MLP (ablation baseline)
│   ├── data/
│   │   ├── coco_dataset.py           # COCO image-caption pairs + collate functions
│   │   ├── flickr_dataset.py         # Flickr30k dataset (for evaluation)
│   │   ├── flickr_zh_dataset.py      # Chinese Flickr30k (cross-lingual eval)
│   │   ├── eval_dataset.py           # Combined COCO + Flickr val loader
│   │   └── video_dataset.py          # MSR-VTT / MSVD video frame embeddings
│   └── training/
│       ├── loss.py                   # SymmetricInfoNCE, QueueInfoNCE (contrastive losses)
│       ├── trainer.py                # Training loop with AMP, TQDM, checkpointing, queue management
│       ├── queue.py                  # GPU-resident FIFO contrastive queue (circular buffer)
│       └── evaluation.py             # Recall@K metrics (COCO-style)
├── scripts/
│   ├── precompute_embeddings.py      # Extract CLIP image embeddings from COCO
│   ├── precompute_flickr_embeddings.py  # Extract CLIP embeddings from Flickr30k
│   ├── train_bert.py                 # BERT text encoder training (COCO, LoRA, queue)
│   ├── train_bert_domain.py          # Domain adaptation on MSR-VTT (Stack LoRA)
│   ├── evaluate_checkpoint.py        # Evaluate trained checkpoint on COCO + Flickr
│   ├── evaluate_clip_text_encoder.py # CLIP zero-shot baseline on Flickr30k
│   ├── evaluate_multilingual.py      # Bilingual Flickr30k eval
│   ├── segment_video.py              # CLI: video → scene segmentation
│   ├── download_openai_clip.py       # Download OpenAI CLIP via modelscope + HF conversion
│   └── convert_hf_to_openclip.py     # Convert HF Transformers CLIP → open_clip safetensors
├── config/
│   ├── default_clip_text.yaml        # CLIP text encoder variant (LoRA on CLIP transformer)
│   ├── default_multilingual.yaml     # Multilingual BERT (bert-base-multilingual-cased)
│   ├── bert_domain*.yaml             # Domain adaptation on MSR-VTT
│   └── ablation/                     # Ablation studies (queue, masking, weight variants)
├── run.py                            # CLI search entry point
└── app.py                            # Gradio web UI
```

## Models Directory

```
models/
├── bert-base-uncased/                # BERT-base (English text encoder)
├── bert-base-multilingual-cased/     # Multilingual BERT (cross-lingual transfer)
├── clip-vit-base-patch32/            # OpenAI CLIP ViT-B/32 from HuggingFace (recommended)
│   └── open_clip_model.safetensors   #   → converted to open_clip format
├── clip/                             # Legacy OpenAI CLIP checkpoints
│   ├── openai_open_clip_model.safetensors
│   └── openai_pytorch_model.bin
└── hy-mt-1.8b/                       # Hy machine translation model (multilingual data pipeline)
```

## Config Files Reference

Each YAML config inherits the same base structure. Key differences between configs:

| Config | `text_encoder.type` | LoRA | Queue | Flicker train | Checkpoint dir |
|--------|---------------------|------|-------|---------------|----------------|
| `default_clip_text.yaml` | clip | r=8 (clip) | Yes (49K) | No | `weights/CLIP_r8_weight0.75_uniformity2` |
| `default_multilingual.yaml` | bert | r=8 (bert) | Yes (49K) | No | `weights/Multilingual_r8_weight0.75_uniformity2` |
| `config/ablation/ours.yaml` | bert | r=8 (bert) | Yes (49K) | No | `weights/ours` |

## Citation

```bibtex
@software{eureka-from-CLIP,
  author = {ZhangSiYuan},
  title = {eureka-from-CLIP: Text-to-Video Retrieval with BERT-Aligned CLIP Embeddings},
  year = {2026},
}
```
