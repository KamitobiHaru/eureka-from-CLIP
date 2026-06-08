import os
from PIL import Image
import numpy as np
import torch
import open_clip

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Checkpoint paths per model type ─────────────────────────────
_LAION_CKPT_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "models", "deprecated_laion_clip", "open_clip_model.safetensors"),
    os.path.join(_PROJECT_ROOT, "models", "deprecated_laion_clip", "open_clip_pytorch_model.bin"),
]

_OPENAI_CKPT_CANDIDATES = [
    # HF clone (converted) — preferred source
    os.path.join(_PROJECT_ROOT, "models", "clip-vit-base-patch32", "open_clip_model.safetensors"),
    # Legacy cached open-clip checkpoints (same weights)
    os.path.join(_PROJECT_ROOT, "models", "clip", "openai_open_clip_model.safetensors"),
    os.path.join(_PROJECT_ROOT, "models", "clip", "openai_pytorch_model.bin"),
]


def _find_checkpoint(model_type: str = "openai") -> str | None:
    """Return the first existing checkpoint path for the given model type."""
    candidates = _OPENAI_CKPT_CANDIDATES if model_type == "openai" else _LAION_CKPT_CANDIDATES
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class CLIPEncoder:
    """Wrapper around open_clip ViT-B/32 for encoding scenes and text queries."""

    def __init__(self, model_type: str = "openai", model_path: str = None, device: str = None):
        """Wrapper around open_clip ViT-B/32.

        Args:
            model_type: ``"laion"`` (default, LAION-2B trained) or ``"openai"`` (original WIT-400M).
            model_path: Optional explicit path to a checkpoint.  If omitted, auto-detected
                        from ``model_type``.
            device: Torch device (auto-detected if None).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = model_path or _find_checkpoint(model_type)
        self.model_type = model_type

        # ── Load model ───────────────────────────────────────────
        # open_clip supports passing a local file path directly as pretrained.
        # For "openai" without a local file, we fall back to the built-in
        # download (which fetches from GitHub Releases).
        if ckpt is None and model_type == "openai":
            print("No local OpenAI checkpoint found. Downloading from open_clip hub...")
            pretrained = "openai"
        else:
            pretrained = ckpt if ckpt else ""

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def encode_scene(self, frames: list) -> np.ndarray:
        """Encode a list of RGB frames (H,W,3 uint8) into a 512-dim normalized embedding.

        Each frame is preprocessed and passed through the vision encoder individually,
        then all frame embeddings are mean-pooled and L2-normalized.
        """
        if not frames:
            return np.zeros(512, dtype=np.float32)

        images = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        emb = self.model.encode_image(images)  # (N, 512)
        emb = emb.mean(dim=0, keepdim=True)     # (1, 512)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """Encode a list of PIL Images into (N, 512) L2-normalized embeddings."""
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        imgs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embs = self.model.encode_image(imgs)  # (N, 512)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_frames(self, frames: list) -> np.ndarray:
        """Encode a list of RGB frames into per-frame (N, 512) L2-normed embeddings.

        Unlike encode_scene(), this does NOT mean-pool. Returns all per-frame
        embeddings for downstream temporal processing.
        """
        if not frames:
            return np.zeros((0, 512), dtype=np.float32)
        images = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        embs = self.model.encode_image(images)  # (N, 512)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query into a 512-dim normalized embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)  # (1, 512)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)
