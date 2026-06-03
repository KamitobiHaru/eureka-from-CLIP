# eureka-from-CLIP

**Text-to-Video Retrieval with BERT-Aligned CLIP Embeddings**

Eureka is a text-to-video retrieval system that finds relevant video scenes from natural language queries. It combines CLIP's visual understanding with BERT's linguistic capabilities to deliver accurate, scene-level search results.

## Motivation

CLIP (Contrastive Language-Image Pre-training) provides a powerful shared embedding space for images and text, but its text encoder is relatively shallow (a Transformer with only ~63M parameters) and struggles with nuanced language understanding. BERT, by contrast, offers deep bidirectional language understanding through its 110M-parameter architecture.

Our approach: **Keep CLIP's vision encoder (frozen), replace its text encoder with BERT, and train BERT to align with CLIP's visual embedding space.**

This gives us:
- **Better language understanding** — BERT captures semantics CLIP's text encoder misses
- **Rich visual representations** — CLIP's vision encoder remains intact
- **Practical efficiency** — training BERT on MS COCO is lightweight (~2.5 hours on a single GPU)

## Architecture

### Two-Stage Pipeline

**Stage 1 — BERT Alignment Training**

```
MS COCO (image + 5 captions)
  ├── CLIP Vision Encoder (frozen) → image_emb (512-dim)
  └── BERT-base + Projection Head  → text_emb  (512-dim)
       └── Symmetric InfoNCE loss aligns both spaces
```

We use the MS COCO dataset (118K images, 5 captions each) to train BERT to produce embeddings that are similar to CLIP's image embeddings for matched image-caption pairs, and dissimilar for unmatched pairs. The loss is symmetric InfoNCE (NT-Xent), the same contrastive objective used in CLIP's original training.

**Stage 2 — Video Search**

```
Video files → scene segmentation → per-scene frame sampling → CLIP Vision → mean pool
                                                                                   ↓
                                                                             FAISS index (scenes)
                                                                                   ↑
Text query → BERT + Projection → query_emb → FAISS search → top-K scenes
```

Rather than mean-pooling an entire multi-scene video into a single vector (which loses detail), we first segment each video into individual scenes using shot boundary detection. Each scene is encoded independently, enabling fine-grained retrieval at the scene level.

## Related Work

### CLIP-based Video Retrieval

**CLIP4Clip** (Luo et al., 2021) extended CLIP from image-text to video-text retrieval by aggregating frame-level CLIP embeddings via mean pooling, LSTM, or transformer. They showed that simple mean pooling is competitive with learned temporal aggregation — our frame aggregation strategy follows this finding.

**CLIPBERT** (Lei et al., CVPR 2021) introduced sparse sampling of 1–2 short clips per video during training, demonstrating that sparse temporal sampling outperforms dense feature extraction for video-language tasks.

### Replacing CLIP's Text Encoder

**LAVILA** (Hugging Face, 2023) replaced CLIP's text encoder with **DistilBERT** in a video-language pretraining setting to reduce memory usage while fitting a high-resolution video encoder. This validates the architectural choice of substituting BERT-family models for CLIP's native text encoder.

**InternVideo2** (2024) initializes its text encoder from the first 19 layers of **BERT-Large** for video-text alignment, using a two-stage training strategy: visual feature pretraining followed by video-text alignment.

### Distillation and Alignment

**DCLIP** (Algoverse AI / CMU, 2025) uses a cross-modal transformer teacher to distill enriched embeddings into CLIP, achieving &gt;20% R@1 improvement on MS COCO retrieval with a hybrid contrastive + cosine loss. Its use of InfoNCE on MS COCO directly parallels our training setup.

**CLIPS** (2024) uses synthetic captions from MLLMs with contrastive learning on MS COCO and Flickr30K, setting SOTA on zero-shot retrieval.

### Scene Segmentation for Video Search

**Video-RAG** (LearnOpenCV, 2025) demonstrated a complete video retrieval pipeline with CLIP-gated keyframe selection, FAISS indexing, and LLM-based verification — a similar architecture to our inference stage.

## Installation

```bash
git clone https://github.com/KamitobiHaru/eureka-from-CLIP.git
cd eureka-from-CLIP
pip install -e .
```

### Dependencies

- Python 3.9+
- PyTorch 2.0+
- `open_clip_torch` — CLIP model (ViT-B/32)
- `transformers` — BERT tokenizer and model
- `pycocotools` — MS COCO dataset loading
- `faiss-cpu` or `faiss-gpu` — similarity search
- `opencv-python` — video frame extraction
- `scenedetect` — scene segmentation (PySceneDetect)
- `pyyaml` — configuration

## Usage

### 1. Download MS COCO

```bash
bash scripts/download_mscoco.sh
```

This downloads the 2017 train/val splits (~19 GB) and caption annotations.

### 2. Train BERT Alignment

```bash
python -m src.training.train --config config/default.yaml
```

This trains BERT to align with CLIP's visual embedding space using MS COCO captions. The trained checkpoint is saved to `checkpoints/`.

**Training details:**
- Optimizer: AdamW (lr=5e-5, weight_decay=0.02)
- LR schedule: linear warmup → cosine decay
- Mixed precision (AMP) for 2× throughput
- Validation: R@1, R@5, R@10 on COCO val set
- Hardware: ~2.5 hours on RTX 3090 (20 epochs, batch 128)

### 3. Index Videos

```python
from src.search.pipeline import VideoSearchPipeline

pipeline = VideoSearchPipeline.from_config("config/default.yaml")
pipeline.process_videos("./data/videos")
```

This segments each video into scenes, encodes each scene (8 uniformly sampled frames → CLIP → mean pool), and builds a FAISS index.

### 4. Search

```python
results = pipeline.search("a dog running in a park")
for r in results:
    print(f"{r['video_id']} scene {r['scene_idx']} ({r['start_sec']:.1f}s-{r['end_sec']:.1f}s): {r['score']:.4f}")
```

Returns top-K matching scenes ranked by cosine similarity.

## Project Structure

```
eureka-from-CLIP/
├── config/           # YAML configuration (model, data, training, search)
├── src/
│   ├── models/       # BERT encoder, CLIP vision wrapper, video encoder
│   ├── data/         # MS COCO dataset, video frame loader
│   ├── training/     # Trainer, InfoNCE loss, evaluator (R@1/R@5/R@10)
│   ├── search/       # FAISS indexer, retriever, pipeline orchestrator
│   └── utils/        # Scene segmentation, frame extraction, embedding helpers
├── scripts/          # Download scripts for COCO and sample videos
├── notebooks/        # Interactive demo notebook
└── tests/            # Unit tests
```

## How It Works — Deep Dive

### Scene Segmentation

We use **PySceneDetect** (`ContentDetector` with adaptive thresholding) — the standard Python library for scene/shot boundary detection. Scenes shorter than 1 second are merged with adjacent scenes to avoid noise.

Each scene is described by metadata:
```
{video_id, scene_index, start_sec, end_sec, num_frames}
```

### Frame Sampling

From each scene, 8 frames are uniformly sampled (scenes are typically short and visually coherent, so 8 frames suffice). Frames are resized and center-cropped to 224×224 with CLIP's normalization.

### BERT Projection Head

BERT-base produces 768-dim [CLS] token representations. A linear projection layer maps these to 512-dim (matching CLIP's embedding space), followed by LayerNorm and L2 normalization.

```
BERT-base → [CLS] (768) → Linear(768, 512) → LayerNorm → L2Norm (512)
```

### Contrastive Loss

The symmetric InfoNCE loss operates on a B×B similarity matrix:

```python
loss = 0.5 * CE(sim / temp, labels) + 0.5 * CE(sim.T / temp, labels)
```

where `sim = img_emb @ txt_emb.T` and `temperature = 0.07`. The diagonal entries are positive pairs; all off-diagonal entries are negatives.

### FAISS Index

We index scenes (not whole videos) using `IndexFlatIP` with `IndexIDMap`. Since all embeddings are L2-normalized, inner product equals cosine similarity. For 100K scenes × 512 dims, the index uses ~200 MB and search completes in under 100 ms on CPU.

## Hardware Requirements

| Component | GPU | Time |
|-----------|-----|------|
| BERT training (20 epochs) | RTX 3060 (12GB) | ~4 hours |
| BERT training (20 epochs) | RTX 3090 (24GB) | ~2.5 hours |
| BERT training (20 epochs) | A100 (40GB) | ~1 hour |
| Scene segmentation | CPU | 0.1-0.3× video duration |
| Scene encoding | GPU | ~0.5 sec per scene |
| FAISS search (100K scenes) | CPU | < 100 ms |

## License

MIT

## Citation

```bibtex
@software{eureka-from-CLIP,
  author = {ZhangSiYuan},
  title = {eureka-from-CLIP: Text-to-Video Retrieval with BERT-Aligned CLIP Embeddings},
  year = {2026},
  url = {https://github.com/KamitobiHaru/eureka-from-CLIP}
}
```
