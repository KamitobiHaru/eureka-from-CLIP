"""
Verify VoP and our Stack LoRA on MSR-VTT test 1k-A.

Usage:
    python scripts/eval_msrvtt_verify.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.evaluation import compute_recall_metrics

MSRVTT_ANN = Path("/run/media/occccce/E/MSR-VTT/msrvtt_test_1k.json")
MSRVTT_VIDEO = Path("/run/media/occccce/E/MSR-VTT/raw_data/MSRVTT_Videos")
FRAME_CACHE = Path("./data/msrvtt/clip_keyframes")       # uniform keyframes
FRAME_CACHE_SCENE = Path("./data/msrvtt/clip_keyframes_scenedetect")
NUM_FRAMES = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Image transform (CLIP standard) ────────────────────────────────────
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


def sample_frames(video_path, num_frames=12):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(image_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return torch.stack(frames) if frames else None


@torch.no_grad()
def eval_vop(test_data, device):
    from scripts.eval_vop_msvd import VoPModel, ClipTokenizer, CKPT_PATH, VOCAB_PATH

    print("  Loading VoP model...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = VoPModel(ckpt["state_dict"]).to(device)
    model.eval()
    tokenizer = ClipTokenizer(str(VOCAB_PATH))

    video_embs, text_embs, video_ids = [], [], []
    n_skipped = 0

    for item in tqdm(test_data, desc="VoP"):
        vid = item["video_id"]
        cap = item["caption"]

        # Video: extract 12 frames
        vpath = MSRVTT_VIDEO / f"{vid}.mp4"
        if not vpath.exists():
            n_skipped += 1
            continue
        frames = sample_frames(str(vpath), NUM_FRAMES)
        if frames is None or len(frames) < 1:
            n_skipped += 1
            continue

        # Text: tokenize
        tokens = tokenizer([cap]).to(device)

        enc_v = model.encode_video(frames.unsqueeze(0).to(device))
        enc_t = model.encode_text(tokens)

        video_embs.append(enc_v.cpu())
        text_embs.append(enc_t.cpu())
        video_ids.append(vid)

    print(f"  Skipped: {n_skipped}  Encoded: {len(video_embs)}")
    if len(video_embs) == 0:
        return {}

    img_t = torch.cat(video_embs, dim=0)
    txt_t = torch.cat(text_embs, dim=0)
    return compute_recall_metrics(img_t, txt_t, video_ids, ks=(1, 5, 10))


@torch.no_grad()
def eval_ours(test_data, checkpoint_path, frame_cache_path, lora_r, lora_alpha, device):
    from src.models import BertEncoder
    from transformers import BertTokenizer

    print(f"  Loading our model from {checkpoint_path}...")
    lora_cfg = {
        "enabled": True,
        "r": lora_r,
        "alpha": lora_alpha,
        "dropout": 0.1,
        "target_modules": ["key", "query", "value", "output.dense"],
    }
    model = BertEncoder(
        model_path="./models/bert-base-uncased",
        embed_dim=512,
        lora_cfg=lora_cfg,
    ).to(device)
    model.eval()

    # FIXED: use model_state_dict key from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"    {k}")

    tokenizer = BertTokenizer.from_pretrained("./models/bert-base-uncased")

    video_embs, text_embs, video_ids = [], [], []
    n_skipped = 0

    for item in tqdm(test_data, desc="Ours"):
        vid = item["video_id"]
        cap = item["caption"]

        # Load precomputed CLIP keyframes
        npy = frame_cache_path / f"{vid}.npy"
        if not npy.exists():
            n_skipped += 1
            continue
        arr = np.load(npy)
        n_frames = min(arr.shape[0], NUM_FRAMES)
        vemb = torch.from_numpy(arr[:n_frames]).mean(dim=0)  # (512,)
        vemb = F.normalize(vemb.unsqueeze(0), dim=-1)

        # Text
        tokens = tokenizer(cap, padding=True, truncation=True, max_length=77,
                          return_tensors="pt").to(device)
        temb = model(tokens["input_ids"], tokens["attention_mask"])

        video_embs.append(vemb.cpu())
        text_embs.append(temb.cpu())
        video_ids.append(vid)

    print(f"  Skipped: {n_skipped}  Encoded: {len(video_embs)}")
    if len(video_embs) == 0:
        return {}

    img_t = torch.cat(video_embs, dim=0)
    txt_t = torch.cat(text_embs, dim=0)
    return compute_recall_metrics(img_t, txt_t, video_ids, ks=(1, 5, 10))


@torch.no_grad()
def eval_clip(test_data, device):
    """Native CLIP ViT-B/32 on MSR-VTT videos (12 frames, mean pooled)."""
    import open_clip
    from clip_search.encoder import _find_checkpoint

    print("  Loading CLIP (ViT-B/32, openai)...")
    ckpt_path = _find_checkpoint("openai")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt_path)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    video_embs, text_embs, video_ids = [], [], []
    n_skipped = 0

    for item in tqdm(test_data, desc="CLIP"):
        vid = item["video_id"]
        cap = item["caption"]

        # Video: extract 12 frames, encode each with CLIP visual, mean pool
        vpath = MSRVTT_VIDEO / f"{vid}.mp4"
        if not vpath.exists():
            n_skipped += 1
            continue
        frames = sample_frames(str(vpath), NUM_FRAMES)
        if frames is None or len(frames) < 1:
            n_skipped += 1
            continue

        # CLIP encode_image expects (B, 3, 224, 224)
        frames = frames.to(device)                     # (T, 3, 224, 224)
        vemb = model.encode_image(frames)              # (T, 512)
        vemb = vemb.mean(dim=0, keepdim=True)          # (1, 512)
        vemb = F.normalize(vemb, dim=-1)

        # Text
        tokens = tokenizer([cap]).to(device)
        temb = model.encode_text(tokens)               # (1, 512)
        temb = F.normalize(temb, dim=-1)

        video_embs.append(vemb.cpu())
        text_embs.append(temb.cpu())
        video_ids.append(vid)

    print(f"  Skipped: {n_skipped}  Encoded: {len(video_embs)}")
    if len(video_embs) == 0:
        return {}

    img_t = torch.cat(video_embs, dim=0)
    txt_t = torch.cat(text_embs, dim=0)
    return compute_recall_metrics(img_t, txt_t, video_ids, ks=(1, 5, 10))


@torch.no_grad()
def eval_clip_from_cache(test_data, frame_cache_path, device):
    """CLIP baseline using precomputed CLIP keyframes (bypasses video decode)."""
    import open_clip
    from clip_search.encoder import _find_checkpoint

    print("  Loading CLIP text encoder...")
    ckpt_path = _find_checkpoint("openai")
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt_path)
    clip = clip.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    video_embs, text_embs, video_ids = [], [], []
    n_skipped = 0

    for item in tqdm(test_data, desc="CLIP-cache"):
        vid = item["video_id"]
        cap = item["caption"]
        npy = frame_cache_path / f"{vid}.npy"
        if not npy.exists():
            n_skipped += 1
            continue
        arr = np.load(npy)
        n_frames = min(arr.shape[0], NUM_FRAMES)
        vemb = torch.from_numpy(arr[:n_frames]).mean(dim=0)
        vemb = F.normalize(vemb.unsqueeze(0), dim=-1)

        tokens = tokenizer([cap]).to(device)
        temb = F.normalize(clip.encode_text(tokens), dim=-1)

        video_embs.append(vemb.cpu())
        text_embs.append(temb.cpu())
        video_ids.append(vid)

    print(f"  Skipped: {n_skipped}  Encoded: {len(video_embs)}")
    if len(video_embs) == 0:
        return {}
    img_t = torch.cat(video_embs, dim=0)
    txt_t = torch.cat(text_embs, dim=0)
    return compute_recall_metrics(img_t, txt_t, video_ids, ks=(1, 5, 10))


def main():
    device = DEVICE
    print(f"Device: {device}\n")

    # ── Load test data ──
    print("── Loading MSR-VTT test 1k-A ──")
    with open(MSRVTT_ANN) as f:
        test_data = json.load(f)
    print(f"  Test pairs: {len(test_data)}\n")

    results = {}

    # ── 1. CLIP native (on-the-fly video decoding) ──
    print("── CLIP ViT-B/32 on MSR-VTT (live decode) ──")
    r_clip = eval_clip(test_data, device)
    if r_clip:
        results["CLIP (live)"] = r_clip
        print(f"  CLIP t2i R@1: {r_clip.get('t2i_R@1', 0):.2f}  "
              f"t2i R@5: {r_clip.get('t2i_R@5', 0):.2f}  "
              f"t2i R@10: {r_clip.get('t2i_R@10', 0):.2f}")
        print(f"  CLIP i2t R@1: {r_clip.get('i2t_R@1', 0):.2f}  "
              f"i2t R@5: {r_clip.get('i2t_R@5', 0):.2f}  "
              f"i2t R@10: {r_clip.get('i2t_R@10', 0):.2f}\n")
    torch.cuda.empty_cache()

    # ── 1b. CLIP from precomputed frame cache ──
    print("── CLIP ViT-B/32 on MSR-VTT (from frame cache) ──")
    r_clip_cache = eval_clip_from_cache(test_data, FRAME_CACHE, device)
    if r_clip_cache:
        results["CLIP (cache)"] = r_clip_cache
        print(f"  CLIP t2i R@1: {r_clip_cache.get('t2i_R@1', 0):.2f}  "
              f"t2i R@5: {r_clip_cache.get('t2i_R@5', 0):.2f}  "
              f"t2i R@10: {r_clip_cache.get('t2i_R@10', 0):.2f}")
        print(f"  CLIP i2t R@1: {r_clip_cache.get('i2t_R@1', 0):.2f}  "
              f"i2t R@5: {r_clip_cache.get('i2t_R@5', 0):.2f}  "
              f"i2t R@10: {r_clip_cache.get('i2t_R@10', 0):.2f}\n")
    torch.cuda.empty_cache()

    # ── 2. VoP ──
    print("── VoP on MSR-VTT ──")
    r_vop = eval_vop(test_data, device)
    if r_vop:
        results["VoP"] = r_vop
        print(f"  VoP   t2i R@1: {r_vop.get('t2i_R@1', 0):.2f}  "
              f"t2i R@5: {r_vop.get('t2i_R@5', 0):.2f}  "
              f"t2i R@10: {r_vop.get('t2i_R@10', 0):.2f}")
        print(f"  VoP   i2t R@1: {r_vop.get('i2t_R@1', 0):.2f}  "
              f"i2t R@5: {r_vop.get('i2t_R@5', 0):.2f}  "
              f"i2t R@10: {r_vop.get('i2t_R@10', 0):.2f}\n")
    torch.cuda.empty_cache()

    # ── 3. Ours (Stack LoRA, uniform keyframes) ──
    print("── Ours Stack LoRA on MSR-VTT (uniform keyframes) ──")
    r_ours = eval_ours(
        test_data,
        checkpoint_path="/run/media/occccce/D/checkpoints/bert_domain_stack_lora_scenedetect/bert_domain_best_e27_msvd28.2.pt",
        frame_cache_path=FRAME_CACHE,
        lora_r=2, lora_alpha=4,
        device=device,
    )
    if r_ours:
        results["Ours (uniform)"] = r_ours
        print(f"  Ours  t2i R@1: {r_ours.get('t2i_R@1', 0):.2f}  "
              f"t2i R@5: {r_ours.get('t2i_R@5', 0):.2f}  "
              f"t2i R@10: {r_ours.get('t2i_R@10', 0):.2f}")
        print(f"  Ours  i2t R@1: {r_ours.get('i2t_R@1', 0):.2f}  "
              f"i2t R@5: {r_ours.get('i2t_R@5', 0):.2f}  "
              f"i2t R@10: {r_ours.get('i2t_R@10', 0):.2f}\n")
    torch.cuda.empty_cache()

    # ── 4. Ours (Stack LoRA, scenedetect keyframes) ──
    print("── Ours Stack LoRA on MSR-VTT (scenedetect keyframes) ──")
    r_ours_sd = eval_ours(
        test_data,
        checkpoint_path="/run/media/occccce/D/checkpoints/bert_domain_stack_lora_scenedetect/bert_domain_best_e27_msvd28.2.pt",
        frame_cache_path=FRAME_CACHE_SCENE,
        lora_r=2, lora_alpha=4,
        device=device,
    )
    if r_ours_sd:
        results["Ours (scenedetect)"] = r_ours_sd
        print(f"  Ours  t2i R@1: {r_ours_sd.get('t2i_R@1', 0):.2f}  "
              f"t2i R@5: {r_ours_sd.get('t2i_R@5', 0):.2f}  "
              f"t2i R@10: {r_ours_sd.get('t2i_R@10', 0):.2f}")
        print(f"  Ours  i2t R@1: {r_ours_sd.get('i2t_R@1', 0):.2f}  "
              f"i2t R@5: {r_ours_sd.get('i2t_R@5', 0):.2f}  "
              f"i2t R@10: {r_ours_sd.get('i2t_R@10', 0):.2f}\n")
    torch.cuda.empty_cache()

    # ── Summary table ──
    print("=" * 75)
    print("  MSR-VTT 1k-A Results Summary")
    print("=" * 75)
    header = (
        f"  {'Method':<20} {'t2i_R@1':>8} {'t2i_R@5':>8} {'t2i_R@10':>8}"
        f" {'i2t_R@1':>8} {'i2t_R@5':>8} {'i2t_R@10':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in ["CLIP (live)", "CLIP (cache)", "VoP", "Ours (uniform)", "Ours (scenedetect)"]:
        r = results.get(method, {})
        parts = [f"{method:<20}"]
        for k in ["t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10"]:
            parts.append(f"{r.get(k, 0):>8.1f}")
        print("  " + " ".join(parts))
    print("=" * 75)


if __name__ == "__main__":
    main()
