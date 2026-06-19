# eureka-from-CLIP

**文本到视频场景检索 —— 基于 BERT 对齐的 CLIP**

> ⚠️ **一个有限资源下的工程实践。** 本项目探索在**单张 8 GB 笔记本电脑 GPU**（GeForce RTX 4060）上能否用 BERT 替换 CLIP 的文本编码器。仅使用 **0.63–1.35M 可训练参数**（约占总模型参数的 1%），通过参数高效对齐实现有竞争力的检索性能——不依赖任何大规模训练设施。

使用自然语言查询检索视频场景。通过对比学习将文本编码器（BERT 或 CLIP）与冻结的 CLIP 视觉嵌入对齐。视频表示通过对每帧 CLIP 嵌入进行均值池化得到。

本框架用 BERT 替换了 CLIP 的原生文本编码器（也可以保留 CLIP 文本编码器），实现文本到视频检索。支持多种文本编码器后端：

| 类型 | 模型 | 适用场景 |
|------|------|----------|
| `clip` | CLIP ViT-B/32 文本编码器（open_clip） | 零样本基线，无需额外权重 |
| `bert-coco` | BERT-base + ProjectionHead + LoRA | 更高召回率（在 COCO Captions 上训练） |
| `bert-stack-lora` | Stack LoRA BERT（领域自适应） | 视频领域自适应（在 MSR-VTT 上训练） |

**关键结果：** Flickr30k 上 t2i R@1 达 60.48%（超过 CLIP 的 57.90%），MSVD 上 t2i R@1 达 28.13%（超过 CLIP 的 27.01%）。全部训练在单张 8 GB GPU 上完成，仅约 1% 的参数可训练。

---

## 架构

```
视频 → 均匀帧采样 → CLIP ViT-B/32（冻结）→ 逐帧 512 维 L2 归一化嵌入
                                         ↓
                                     均值池化
                                         ↓
                                   视频嵌入
                                         ↑
                                  余弦相似度
                                         ↑
                             文本编码器（CLIP / BERT + LoRA）
```

场景编码使用均值池化的 CLIP 逐帧嵌入（L2 归一化）。文本编码器可以是 CLIP 的原生 Transformer、在 COCO 上训练的 BERT（带 LoRA 适配器），或经过视频领域自适应的 Stack LoRA BERT。

---

## 推理

### 命令行

**CLIP**（默认，均值池化帧 + CLIP 文本编码器）：

```bash
python run.py ./videos --query "一个人走路"
```

**BERT**（训练后的文本编码器）：

```bash
python run.py ./videos --query "一个人走路" \
    --model bert-coco --bert_checkpoint checkpoints/bert_epoch02.pt
```

**Stack LoRA BERT**（领域自适应）：

```bash
python run.py ./videos --query "一个人走路" \
    --model bert-stack-lora --bert_checkpoint checkpoints/domain_adapted.pt \
    --config config/bert_domain.yaml
```

**使用预计算缓存**（跳过视频处理）：

```bash
python search_cli.py --cache ./data -q "一个人走路"
```

### Gradio 网页界面

```bash
python app.py
```

打开一个浏览器界面，包含两个标签页：
1. **处理视频** — 选择视频文件夹，提取帧嵌入
2. **搜索场景** — 查询已索引的场景，查看带视频片段的排序结果

---

## 项目结构

```
├── clip_search/                      # 推理包
│   ├── encoder.py                    # CLIP ViT-B/32：编码场景、帧、文本
│   ├── engine.py                     # SearchEngine：处理视频 → 编码 → 搜索
│   ├── segmenter.py                  # Scene 数据类（帧容器）
│   └── text_encoder_loader.py        # 统一的文本编码器加载（CLIP / BERT / Stack LoRA）
├── src/
│   ├── models/
│   │   ├── bert_encoder.py           # BERT-base + ProjectionHead (768→512) + LoRA + Stack LoRA
│   │   ├── clip_text_encoder.py      # CLIP 文本 Transformer（open_clip）+ LoRA
│   │   └── mlp_encoder.py            # 冻结 BERT + 3 层 MLP（消融基线）
│   ├── data/
│   │   ├── coco_dataset.py           # COCO 图像-文本对数据集
│   │   ├── flickr_dataset.py         # Flickr30k 数据集（评估用）
│   │   ├── flickr_zh_dataset.py      # 中文 Flickr30k（跨语言评估）
│   │   ├── eval_dataset.py           # COCO + Flickr 验证集加载器
│   │   └── video_dataset.py          # MSR-VTT / MSVD 视频帧嵌入
│   └── training/
│       ├── loss.py                   # SymmetricInfoNCE、QueueInfoNCE（对比损失）
│       ├── trainer.py                # 训练循环：AMP、检查点保存、队列管理
│       ├── queue.py                  # GPU 上的 FIFO 对比队列
│       └── evaluation.py             # Recall@K 评估指标
├── scripts/
│   ├── precompute_embeddings.py      # 从 COCO 提取 CLIP 图像嵌入
│   ├── precompute_flickr_embeddings.py
│   ├── precompute_msrvtt_embeddings.py
│   ├── precompute_video_keyframes.py # 从视频提取 CLIP 关键帧嵌入
│   ├── train_bert.py                 # BERT 对齐训练（COCO、LoRA、队列）
│   ├── train_bert_domain.py          # MSR-VTT 领域自适应（Stack LoRA）
│   ├── evaluate_checkpoint.py        # 评估训练好的检查点
│   ├── evaluate_clip_text_encoder.py # CLIP 在 Flickr30k 上的零样本基线
│   ├── evaluate_multilingual.py      # 中英文 Flickr30k 评估
│   ├── download_openai_clip.py       # 通过 modelscope 下载 OpenAI CLIP
│   └── convert_hf_to_openclip.py     # 将 HF CLIP 转换为 open_clip safetensors
├── config/                           # YAML 配置文件
│   ├── default_clip_text.yaml        # CLIP 文本编码器配置
│   ├── default_multilingual.yaml     # 多语言 BERT 配置
│   ├── bert_domain*.yaml             # 领域自适应配置
│   └── ablation/                     # 消融实验配置
├── run.py                            # 命令行搜索入口
├── search_cli.py                     # 命令行搜索（支持预计算缓存）
├── app.py                            # Gradio 网页界面
└── cache_videos.py                   # 预计算视频嵌入到缓存
```
