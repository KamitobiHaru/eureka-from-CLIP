# eureka-from-CLIP

**Text-to-Video Scene Retrieval with BERT-Aligned CLIP**

> ⚠️ **A resource-constrained engineering practice.** This project explores whether CLIP's text encoder can be replaced with BERT on **a single 8 GB laptop GPU** (GeForce RTX 4060). With only **0.63–1.35M trainable parameters** (~1% of total), it achieves competitive results through parameter-efficient alignment — no large-scale training infrastructure involved.

Retrieve video scenes using natural language queries. Contrastive training aligns a text encoder (BERT or CLIP) with frozen CLIP vision embeddings. Video representations are obtained by mean-pooling per-frame CLIP embeddings.

This framework replaces CLIP's native text encoder with BERT (or keeps the CLIP text encoder) for text-to-video retrieval. It supports multiple text encoder backends:

| Type | Model | Best for |
|------|-------|----------|
| `clip` | CLIP ViT-B/32 text encoder (open_clip) | Zero-shot baseline, no extra weights |
| `bert-coco` | BERT-base + ProjectionHead + LoRA | Higher recall (trained on COCO Captions) |
| `bert-stack-lora` | Stack LoRA BERT (domain-adapted) | Video domain adaptation (trained on MSR-VTT) |

**Key results:** Flickr30k 60.48% t2i R@1 (vs CLIP 57.90%), MSVD 28.13% t2i R@1 (vs CLIP 27.01%). All training done on a single 8 GB GPU with only ~1% of total parameters trainable.

---

## Architecture

```
Video → uniform frame sampling → CLIP ViT-B/32 (frozen) → per-frame 512d L2-normed embeddings
                                                                        ↓
                                                                   mean pool
                                                                        ↓
                                                                video embedding
                                                                        ↑
                                                            cosine similarity
                                                                        ↑
                                                         Text Encoder (CLIP / BERT + LoRA)
```

Scene encoding uses mean-pooled CLIP per-frame embeddings (L2-normed). The text encoder can be CLIP's native transformer, a COCO-trained BERT with LoRA adapters, or a Stack LoRA BERT adapted for video domain.

---

## Inference

### CLI

**CLIP** (default, mean-pooled frames + CLIP text encoder):

```bash
python run.py ./videos --query "a person walking"
```

**BERT** (trained text encoder):

```bash
python run.py ./videos --query "a person walking" \
    --model bert-coco --bert_checkpoint checkpoints/bert_epoch02.pt
```

**Stack LoRA BERT** (domain-adapted):

```bash
python run.py ./videos --query "a person walking" \
    --model bert-stack-lora --bert_checkpoint checkpoints/domain_adapted.pt \
    --config config/bert_domain.yaml
```

**With precomputed cache** (skip video processing):

```bash
python search_cli.py --cache ./data -q "a person walking"
```

### Gradio Web UI

```bash
python app.py
```

Opens a browser UI with two tabs:
1. **Process Videos** — select a folder of videos, extract frame embeddings
2. **Search Scenes** — query indexed scenes, view ranked results with video clips

---

## Project Structure

```
├── clip_search/                      # Inference package
│   ├── encoder.py                    # CLIP ViT-B/32: encode_scene, encode_frames, encode_text
│   ├── engine.py                     # SearchEngine: process videos → encode → search
│   ├── segmenter.py                  # Scene dataclass (frame container)
│   └── text_encoder_loader.py        # Shared text encoder loading (CLIP / BERT / Stack LoRA)
├── src/
│   ├── models/
│   │   ├── bert_encoder.py           # BERT-base + ProjectionHead (768→512) + LoRA + Stack LoRA
│   │   ├── clip_text_encoder.py      # CLIP text transformer (open_clip) + LoRA
│   │   └── mlp_encoder.py            # Frozen BERT + 3-layer MLP (ablation baseline)
│   ├── data/
│   │   ├── coco_dataset.py           # COCO image-caption pairs
│   │   ├── flickr_dataset.py         # Flickr30k dataset (evaluation)
│   │   ├── flickr_zh_dataset.py      # Chinese Flickr30k (cross-lingual eval)
│   │   ├── eval_dataset.py           # Combined COCO + Flickr val loader
│   │   └── video_dataset.py          # MSR-VTT / MSVD video frame embeddings
│   └── training/
│       ├── loss.py                   # SymmetricInfoNCE, QueueInfoNCE (contrastive losses)
│       ├── trainer.py                # Training loop with AMP, checkpointing, queue management
│       ├── queue.py                  # GPU-resident FIFO contrastive queue
│       └── evaluation.py             # Recall@K metrics
├── scripts/
│   ├── precompute_embeddings.py      # Extract CLIP image embeddings from COCO
│   ├── precompute_flickr_embeddings.py
│   ├── precompute_msrvtt_embeddings.py
│   ├── precompute_video_keyframes.py # Extract CLIP key-frame embeddings from videos
│   ├── train_bert.py                 # BERT alignment training (COCO, LoRA, queue)
│   ├── train_bert_domain.py          # Domain adaptation on MSR-VTT (Stack LoRA)
│   ├── evaluate_checkpoint.py        # Evaluate trained checkpoint
│   ├── evaluate_clip_text_encoder.py # CLIP zero-shot baseline on Flickr30k
│   ├── evaluate_multilingual.py      # Bilingual Flickr30k eval
│   ├── download_openai_clip.py       # Download OpenAI CLIP via modelscope
│   └── convert_hf_to_openclip.py     # Convert HF CLIP → open_clip safetensors
├── config/                           # YAML configuration files
│   ├── default_clip_text.yaml        # CLIP text encoder config
│   ├── default_multilingual.yaml     # Multilingual BERT config
│   ├── bert_domain*.yaml             # Domain adaptation configs
│   └── ablation/                     # Ablation study configs
├── run.py                            # CLI search entry point
├── search_cli.py                     # CLI search (supports precomputed cache)
├── app.py                            # Gradio web UI
└── cache_videos.py                   # Precompute video embeddings to cache
```
