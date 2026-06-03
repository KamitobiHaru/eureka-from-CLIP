"""Download CLIP ViT-B/32 pretrained weights with progress bar.

Usage:
    python scripts/download_clip_model.py
"""

import os
from huggingface_hub import snapshot_download

repo_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
allow_patterns = ["*.safetensors", "*.txt", "*.json"]

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"Downloading CLIP ViT-B/32 weights from {repo_id} (~600MB)...")
print()

snapshot_download(
    repo_id=repo_id,
    allow_patterns=allow_patterns,
    cache_dir=cache_dir,
    resume_download=True,
)

print("Done!")
