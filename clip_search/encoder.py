import os
from PIL import Image
import numpy as np
import torch
import open_clip

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Try safetensors first, fall back to .bin
_CKPT_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "models", "clip", "open_clip_model.safetensors"),
    os.path.join(_PROJECT_ROOT, "models", "clip", "open_clip_pytorch_model.bin"),
]


def _find_checkpoint() -> str | None:
    for p in _CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


class CLIPEncoder:
    """Wrapper around open_clip ViT-B/32 for encoding scenes and text queries."""

    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = model_path or _find_checkpoint()

        # open_clip supports passing a local file path directly as pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=ckpt if ckpt else "",
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
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query into a 512-dim normalized embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)  # (1, 512)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)
