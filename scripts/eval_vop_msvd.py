"""
Evaluate VoP on MSVD zero-shot video retrieval.

Copies the VoP checkpoint weights into an open_clip model, then patches
each transformer block to inject visual/text prompts.

Usage:
    python scripts/eval_vop_msvd.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.evaluation import compute_recall_metrics

# ── Paths / constants ──────────────────────────────────────────────
VOP_DIR = Path("/run/media/occccce/E/VoP")
CKPT_PATH = VOP_DIR / "VoP_msrvtt9k.pth"
VOCAB_PATH = VOP_DIR / "bpe_simple_vocab_16e6.txt.gz"
MSVD_ANN = Path("./data/msvd/msvd_test_only.json")
NUM_FRAMES = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── CLIP BPE tokenizer (from open_clip source) ────────────────────
import gzip
import regex as re
from functools import lru_cache


@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for c in word[1:]:
        pairs.add((prev, c))
        prev = c
    return pairs


class ClipTokenizer:
    """OpenAI CLIP BPE tokenizer (from open_clip source)."""
    def __init__(self, bpe_path):
        self.byte_encoder = _bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:48895]  # skip header, take 48894 merges → vocab = 49408
        merges = [tuple(m.split()) for m in merges if m]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        vocab = list(self.byte_encoder.values()) + [v + "</w>" for v in self.byte_encoder.values()]
        for a, b in merges:
            vocab.append(a + b)
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.decoder = {i: v for v, i in self.encoder.items()}
        # Original CLIP regex — no [\x00-\x7f]{1,2} which mis-merges spaces
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w']+|[^\s\w<>]+""",
            re.IGNORECASE,
        )
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def bpe(self, token):
        """Byte-pair encode a single token string → space-separated subword tokens."""
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        return " ".join(word)

    def encode(self, text):
        """Tokenize text → list of token IDs."""
        text = text.lower().strip()
        text = re.sub(r"<\|startoftext\|>|<\|endoftext\|>", "", text)
        bpe_tokens = [self.sot]
        for match in re.findall(self.pat, text):
            token_uni = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
            for t in self.bpe(token_uni).split(" "):
                bpe_tokens.append(self.encoder.get(t, self.sot))
        bpe_tokens.append(self.eot)
        return bpe_tokens

    def __call__(self, texts, max_length=77):
        batch = []
        for t in texts:
            tok = self.encode(t)[:max_length]
            tok += [0] * (max_length - len(tok))
            batch.append(tok)
        return torch.tensor(batch, dtype=torch.long)


# ─── Image transforms (CLIP standard) ──────────────────────────────

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


# ─── Model ─────────────────────────────────────────────────────────

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualBlock(nn.Module):
    """Standard CLIP transformer block without prompt handling."""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x),
                          attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class VoPModel(nn.Module):
    """VoP: CLIP ViT-B/32 with per-layer visual and text prompts."""

    def __init__(self, state_dict):
        super().__init__()
        # ── Load prompts ──
        # visual_prompts: (12, 1, 8, 768) → squeeze dim 1 → (12, 8, 768)
        self.visual_prompts = nn.Parameter(
            state_dict["module.visual_prompt_learner.visual_prompts"].squeeze(1).clone())
        # text_prompts: (12, 8, 512)
        self.text_prompts = nn.Parameter(
            state_dict["module.text_prompt_learner.text_prompts"].clone())

        # ── Visual encoder (ViT-B/32) ──
        self.conv1 = nn.Conv2d(3, 768, 32, 32, bias=False)
        self.conv1.weight.data = state_dict["module.image_encoder.conv1.weight"]
        self.class_embedding = nn.Parameter(
            state_dict["module.image_encoder.class_embedding"].clone())
        self.positional_embedding = nn.Parameter(
            state_dict["module.image_encoder.positional_embedding"].clone())
        self.ln_pre = nn.LayerNorm(768)
        self.ln_pre.weight.data = state_dict["module.image_encoder.ln_pre.weight"]
        self.ln_pre.bias.data = state_dict["module.image_encoder.ln_pre.bias"]
        self.ln_post = nn.LayerNorm(768)
        self.ln_post.weight.data = state_dict["module.image_encoder.ln_post.weight"]
        self.ln_post.bias.data = state_dict["module.image_encoder.ln_post.bias"]
        self.visual_proj = nn.Parameter(
            state_dict["module.image_encoder.proj"].clone())

        self.visual_blocks = nn.ModuleList()
        for i in range(12):
            block = ResidualBlock(768, 12)
            p = f"module.image_encoder.transformer.resblocks.{i}"
            block.ln_1.weight.data = state_dict[f"{p}.ln_1.weight"]
            block.ln_1.bias.data = state_dict[f"{p}.ln_1.bias"]
            block.attn.in_proj_weight = nn.Parameter(
                state_dict[f"{p}.attn.in_proj_weight"].clone())
            block.attn.in_proj_bias = nn.Parameter(
                state_dict[f"{p}.attn.in_proj_bias"].clone())
            block.attn.out_proj.weight = nn.Parameter(
                state_dict[f"{p}.attn.out_proj.weight"].clone())
            block.attn.out_proj.bias = nn.Parameter(
                state_dict[f"{p}.attn.out_proj.bias"].clone())
            block.ln_2.weight.data = state_dict[f"{p}.ln_2.weight"]
            block.ln_2.bias.data = state_dict[f"{p}.ln_2.bias"]
            block.mlp[0].weight.data = state_dict[f"{p}.mlp.c_fc.weight"]
            block.mlp[0].bias.data = state_dict[f"{p}.mlp.c_fc.bias"]
            block.mlp[2].weight.data = state_dict[f"{p}.mlp.c_proj.weight"]
            block.mlp[2].bias.data = state_dict[f"{p}.mlp.c_proj.bias"]
            self.visual_blocks.append(block)

        # ── Text encoder ──
        self.token_embedding = nn.Embedding(49408, 512)
        self.token_embedding.weight.data = state_dict["module.text_encoder.token_embedding.weight"]
        self.text_positional_embedding = nn.Parameter(
            state_dict["module.text_encoder.positional_embedding"].clone())
        self.ln_final = nn.LayerNorm(512)
        self.ln_final.weight.data = state_dict["module.text_encoder.ln_final.weight"]
        self.ln_final.bias.data = state_dict["module.text_encoder.ln_final.bias"]
        self.text_projection = nn.Parameter(
            state_dict["module.clip.text_projection"].clone())

        self.text_blocks = nn.ModuleList()
        for i in range(12):
            block = ResidualBlock(512, 8)
            p = f"module.text_encoder.transformer.resblocks.{i}"
            block.ln_1.weight.data = state_dict[f"{p}.ln_1.weight"]
            block.ln_1.bias.data = state_dict[f"{p}.ln_1.bias"]
            block.attn.in_proj_weight = nn.Parameter(
                state_dict[f"{p}.attn.in_proj_weight"].clone())
            block.attn.in_proj_bias = nn.Parameter(
                state_dict[f"{p}.attn.in_proj_bias"].clone())
            block.attn.out_proj.weight = nn.Parameter(
                state_dict[f"{p}.attn.out_proj.weight"].clone())
            block.attn.out_proj.bias = nn.Parameter(
                state_dict[f"{p}.attn.out_proj.bias"].clone())
            block.ln_2.weight.data = state_dict[f"{p}.ln_2.weight"]
            block.ln_2.bias.data = state_dict[f"{p}.ln_2.bias"]
            block.mlp[0].weight.data = state_dict[f"{p}.mlp.c_fc.weight"]
            block.mlp[0].bias.data = state_dict[f"{p}.mlp.c_fc.bias"]
            block.mlp[2].weight.data = state_dict[f"{p}.mlp.c_proj.weight"]
            block.mlp[2].bias.data = state_dict[f"{p}.mlp.c_proj.bias"]
            self.text_blocks.append(block)

        # Causal mask for text (base 77 positions)
        self.register_buffer("_causal_mask",
                             torch.triu(torch.full((77, 77), float("-inf")), diagonal=1),
                             persistent=False)

        self.text_max_len = 77

    # ── Visual forward ───────────────────────────────────────────

    def encode_video(self, frames):
        """frames: (B, T, 3, 224, 224) or (T, 3, 224, 224)"""
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)  # (1, T, 3, 224, 224)
        B, T = frames.shape[:2]
        x = frames.view(B * T, 3, 224, 224)

        # Patch embedding
        x = self.conv1(x)  # (B*T, 768, 7, 7)
        x = x.flatten(2).transpose(1, 2)  # (B*T, 49, 768)

        # CLS + positional
        cls = self.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B*T, 50, 768)
        x = x + self.positional_embedding.unsqueeze(0)
        x = self.ln_pre(x)

        # Transformer blocks with per-layer visual prompts
        n_prompt = self.visual_prompts.shape[1]  # 8
        for i, block in enumerate(self.visual_blocks):
            prompts = self.visual_prompts[i].unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([prompts, x], dim=1)  # (B*T, 58, 768)
            x = block(x)  # attn + mlp on full seq
            x = x[:, n_prompt:]  # remove prompts → (B*T, 50, 768)

        # Pool CLS token
        x = self.ln_post(x)
        x = x[:, 0, :] @ self.visual_proj  # (B*T, 512)

        # Mean pool over T frames → (B, 512)
        x = x.view(B, T, 512).mean(dim=1)
        return F.normalize(x, dim=-1)

    # ── Text forward ─────────────────────────────────────────────

    def encode_text(self, text):
        """text: (B, L) token ids with L ≤ 77.

        Uses 4 prefix + 4 suffix prompts per the VoP paper config
        (tp_prefix_token_num=4, tp_suffix_token_num=4).
        """
        B, L = text.shape
        x = self.token_embedding(text)  # (B, L, 512)
        x = x + self.text_positional_embedding[:L].unsqueeze(0)

        n_prefix = 4
        n_suffix = 4

        for i, block in enumerate(self.text_blocks):
            prompts = self.text_prompts[i]  # (8, 512)
            prefix = prompts[:n_prefix].unsqueeze(0).expand(B, -1, -1)
            suffix = prompts[n_prefix:].unsqueeze(0).expand(B, -1, -1)

            # Concatenate: [prefix, text, suffix] → (B, L+8, 512)
            x = torch.cat([prefix, x, suffix], dim=1)

            # ── Attention mask ──
            # Total length = prefix(4) + text(L) + suffix(4)
            full_len = L + n_prefix + n_suffix
            mask = self._causal_mask.new_full((full_len, full_len), float("-inf"))
            # Causal: each row can attend to col ≤ row
            mask.triu_(1)

            # Prefix (0..n_prefix-1): can see all positions
            mask[:n_prefix, :] = 0
            # Text (n_prefix..n_prefix+L-1): can see prefix + previous text positions
            mask[n_prefix:n_prefix + L, :n_prefix] = 0  # text → prefix
            # text-text is already causal (triu)
            # Text cannot see suffix (future)
            # Suffix (n_prefix+L..): can see all positions
            mask[n_prefix + L:, :] = 0

            x = block(x, attn_mask=mask)
            # Remove prefix and suffix, keep only text → (B, L, 512)
            x = x[:, n_prefix:n_prefix + L]

        x = self.ln_final(x)

        # Take EOS token (last non-pad token = token id 49407)
        eot_pos = (text == 49407).int().argmax(dim=-1)
        x = x[torch.arange(B), eot_pos] @ self.text_projection
        return F.normalize(x, dim=-1)


# ── Video helpers ─────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate VoP on MSVD")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or DEVICE
    print(f"Device: {device}\n")

    # ── Load model ──
    print("── Loading VoP checkpoint ──")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")

    model = VoPModel(ckpt["state_dict"]).to(device)
    model.eval()
    n_total = sum(p.numel() for p in model.parameters())
    n_prompts = sum(p.numel() for p in [model.visual_prompts, model.text_prompts])
    print(f"  Total params: {n_total:,}  (prompts: {n_prompts:,})")

    # ── MSVD data ──
    print("\n── Loading MSVD test set ──")
    with open(MSVD_ANN) as f:
        ann = json.load(f)
    print(f"  Test videos: {len(ann)}")

    pairs = [(e["video_id"], c)
             for e in ann
             for c in e.get("captions", [])
             if c.strip()]
    print(f"  Caption pairs: {len(pairs):,}")

    unique_ids = list(dict.fromkeys(vid for vid, _ in pairs))
    print(f"  Unique videos: {len(unique_ids)}")

    # ── Tokenizer ──
    tokenizer = ClipTokenizer(str(VOCAB_PATH))
    print(f"  Vocab size: {len(tokenizer.encoder)}")

    # ── DIAGNOSTIC: test tokenizer and text encoder ──
    print("\n── DIAGNOSTIC: tokenizer + text encoder ──")
    test_texts = [
        "a man is playing a guitar",
        "a dog is running in the park",
        "a woman is cooking in the kitchen",
    ]
    for t in test_texts:
        tok = tokenizer([t])
        print(f"  '{t}' → {len(tok[0])} tokens, first 10: {tok[0][:10].tolist()}")
    test_tokens = tokenizer(test_texts).to(device)
    with torch.no_grad():
        test_embs = model.encode_text(test_tokens)
    print(f"  Text emb shape: {test_embs.shape}")
    sim = (test_embs @ test_embs.T).cpu()
    print(f"  Self-similarity matrix:\n{sim.numpy().round(3)}")
    print(f"  Norms: {test_embs.norm(dim=-1).cpu().tolist()}")

    # ── DIAGNOSTIC: test video encoder ──
    print("\n── DIAGNOSTIC: video encoder ──")
    video_dir = Path("/run/media/occccce/E/MSVD/OpenDataLab___MSVD/raw/MSVD/YouTubeClips")
    # Pick first 3 available videos
    test_vids = []
    for vid in unique_ids:
        if (video_dir / f"{vid}.avi").exists():
            test_vids.append(vid)
            if len(test_vids) >= 3:
                break
    test_vembs = []
    for vid in test_vids:
        vpath = video_dir / f"{vid}.avi"
        frames = sample_frames(str(vpath), NUM_FRAMES)
        if frames is not None:
            with torch.no_grad():
                emb = model.encode_video(frames.unsqueeze(0).to(device))
            test_vembs.append(emb.cpu())
            print(f"  {vid}: frames={len(frames)}, emb norm={emb.norm().item():.4f}")
    if len(test_vembs) >= 2:
        tv = torch.cat(test_vembs, dim=0)
        vsim = (tv @ tv.T).cpu()
        print(f"  Video self-similarity:\n{vsim.numpy().round(3)}")

    # ── DIAGNOSTIC: text-video alignment ──
    if len(test_vembs) >= 2:
        print("\n── DIAGNOSTIC: text↔video alignment (expected high on diag) ──")
        cross_sim = test_embs.cpu() @ tv.T
        print(f"  Cross-sim matrix (3 texts × {len(test_vids)} videos):\n{cross_sim.numpy().round(3)}")

    # ── Encode videos ──
    print(f"\n── Encoding {len(unique_ids)} videos (VoP, 12 frames, mean pool) ──")
    video_embs = {}
    for vid in tqdm(unique_ids, desc="Video"):
        vpath = video_dir / f"{vid}.avi"
        if not vpath.exists():
            continue
        frames = sample_frames(str(vpath), NUM_FRAMES)
        if frames is None or len(frames) == 0:
            continue
        with torch.no_grad():
            emb = model.encode_video(frames.unsqueeze(0).to(device))
        video_embs[vid] = emb.cpu()

    print(f"  Encoded {len(video_embs)}/{len(unique_ids)} videos")

    # ── Encode texts ──
    print(f"\n── Encoding {len(pairs)} captions ──")
    all_video_embs, all_text_embs, all_video_ids = [], [], []
    cap_batch = []
    for vid, cap in pairs:
        if vid not in video_embs:
            continue
        all_video_embs.append(video_embs[vid])
        all_video_ids.append(vid)
        cap_batch.append(cap)
        if len(cap_batch) >= 128:
            tokens = tokenizer(cap_batch).to(device)
            with torch.no_grad():
                embs = model.encode_text(tokens)
            all_text_embs.append(embs.cpu())
            cap_batch = []

    if cap_batch:
        tokens = tokenizer(cap_batch).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
        all_text_embs.append(embs.cpu())

    img_t = torch.cat(all_video_embs, dim=0)
    txt_t = torch.cat(all_text_embs, dim=0)

    # ── DIAGNOSTIC: embedding stats ──
    print(f"\n── DIAGNOSTIC: embedding statistics ──")
    print(f"  Video embs: {img_t.shape}  mean={img_t.mean():.4f}  std={img_t.std():.4f}")
    print(f"  Text embs:  {txt_t.shape}  mean={txt_t.mean():.4f}  std={txt_t.std():.4f}")
    print(f"  Video pairwise cos-sim mean: {(img_t @ img_t.T).mean().item():.4f}")
    print(f"  Text pairwise cos-sim mean:  {(txt_t @ txt_t.T).mean().item():.4f}")
    # Mean of diagonal (matching pairs) vs off-diagonal
    sim_matrix = img_t @ txt_t.T  # (N_vid, N_txt)
    n = sim_matrix.shape[0]
    diag_mean = sim_matrix.diag().mean().item()
    off_diag = (sim_matrix.sum() - sim_matrix.diag().sum()) / (n * n - n)
    print(f"  Mean sim (matching pairs): {diag_mean:.4f}")
    print(f"  Mean sim (non-matching):   {off_diag:.4f}")

    # ── Recall ──
    print("\n── Computing Recall@K ──")
    recall = compute_recall_metrics(img_t, txt_t, all_video_ids, ks=(1, 5, 10))

    print("\n" + "=" * 50)
    print("  VoP on MSVD (zero-shot)")
    print("=" * 50)
    for k in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"]:
        print(f"  {k:<12}: {recall.get(k, 0):.2f}")
    print("=" * 50)

    # ── Compare with CLIP baseline ──
    print("\n  CLIP baseline (from paper): 27.01 t2i R@1")
    delta = recall.get("t2i_R@1", 0) - 27.01
    print(f"  VoP Δ vs CLIP: {delta:+.2f}")


if __name__ == "__main__":
    main()
