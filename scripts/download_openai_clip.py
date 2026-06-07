"""
Download OpenAI's original CLIP ViT-B/32 weights (WIT-400M) from modelscope
and convert to open_clip format.

Usage:
    python scripts/download_openai_clip.py

Saves to: models/clip/openai_open_clip_model.safetensors
"""

import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "models" / "clip" / "openai_open_clip_model.safetensors"

# Modelscope model ID for OpenAI's CLIP ViT-B/32 (HF Transformers format)
MODELSCOPE_ID = "thomas/clip-vit-base-patch32"


def convert_hf_to_openclip(hf_model):
    """Convert HF Transformers CLIPModel state dict to open_clip format.

    Key differences:
      - open_clip concatenates q/k/v projections (``attn.in_proj_weight``)
      - open_clip names are shorter (``visual.transformer.resblocks.0.attn.out_proj.weight``)
      - HF submodule names differ (``vision_model.encoder.layers.0.self_attn.out_proj.weight``)
    """
    hf = hf_model.state_dict()
    oc = {}

    # ── Helper: concat q/k/v into in_proj ─────────────────────
    def concat_qkv(prefix_in_oc, prefix_hf):
        """Concatenate q_proj, k_proj, v_proj into in_proj_weight/bias.
        Note: open_clip stores in_proj as a raw parameter (in_proj_weight),
        not as an nn.Linear module (in_proj.weight).
        """
        for suffix in [(".weight", "_weight"), (".bias", "_bias")]:
            q = hf.pop(f"{prefix_hf}self_attn.q_proj{suffix[0]}")
            k = hf.pop(f"{prefix_hf}self_attn.k_proj{suffix[0]}")
            v = hf.pop(f"{prefix_hf}self_attn.v_proj{suffix[0]}")
            oc[f"{prefix_in_oc}attn.in_proj{suffix[1]}"] = torch.cat([q, k, v])

    # ── Vision encoder ────────────────────────────────────────
    v_hf = "vision_model."
    v_oc = "visual."

    oc[v_oc + "class_embedding"] = hf.pop(v_hf + "embeddings.class_embedding")
    oc[v_oc + "positional_embedding"] = hf.pop(v_hf + "embeddings.position_embedding.weight")
    oc[v_oc + "conv1.weight"] = hf.pop(v_hf + "embeddings.patch_embedding.weight")
    oc[v_oc + "ln_pre.weight"] = hf.pop(v_hf + "pre_layrnorm.weight")
    oc[v_oc + "ln_pre.bias"] = hf.pop(v_hf + "pre_layrnorm.bias")

    # Transformer blocks
    n_layers = 12
    for i in range(n_layers):
        oc[v_oc + f"transformer.resblocks.{i}.ln_1.weight"] = hf.pop(
            v_hf + f"encoder.layers.{i}.layer_norm1.weight"
        )
        oc[v_oc + f"transformer.resblocks.{i}.ln_1.bias"] = hf.pop(
            v_hf + f"encoder.layers.{i}.layer_norm1.bias"
        )
        concat_qkv(
            v_oc + f"transformer.resblocks.{i}.",
            v_hf + f"encoder.layers.{i}.",
        )
        oc[v_oc + f"transformer.resblocks.{i}.attn.out_proj.weight"] = hf.pop(
            v_hf + f"encoder.layers.{i}.self_attn.out_proj.weight"
        )
        oc[v_oc + f"transformer.resblocks.{i}.attn.out_proj.bias"] = hf.pop(
            v_hf + f"encoder.layers.{i}.self_attn.out_proj.bias"
        )
        oc[v_oc + f"transformer.resblocks.{i}.ln_2.weight"] = hf.pop(
            v_hf + f"encoder.layers.{i}.layer_norm2.weight"
        )
        oc[v_oc + f"transformer.resblocks.{i}.ln_2.bias"] = hf.pop(
            v_hf + f"encoder.layers.{i}.layer_norm2.bias"
        )
        oc[v_oc + f"transformer.resblocks.{i}.mlp.c_fc.weight"] = hf.pop(
            v_hf + f"encoder.layers.{i}.mlp.fc1.weight"
        )
        oc[v_oc + f"transformer.resblocks.{i}.mlp.c_fc.bias"] = hf.pop(
            v_hf + f"encoder.layers.{i}.mlp.fc1.bias"
        )
        oc[v_oc + f"transformer.resblocks.{i}.mlp.c_proj.weight"] = hf.pop(
            v_hf + f"encoder.layers.{i}.mlp.fc2.weight"
        )
        oc[v_oc + f"transformer.resblocks.{i}.mlp.c_proj.bias"] = hf.pop(
            v_hf + f"encoder.layers.{i}.mlp.fc2.bias"
        )

    oc[v_oc + "ln_post.weight"] = hf.pop(v_hf + "post_layernorm.weight")
    oc[v_oc + "ln_post.bias"] = hf.pop(v_hf + "post_layernorm.bias")

    # ── Text encoder ──────────────────────────────────────────
    t_hf = "text_model."
    t_oc = "transformer."

    oc["token_embedding.weight"] = hf.pop(t_hf + "embeddings.token_embedding.weight")
    oc["positional_embedding"] = hf.pop(t_hf + "embeddings.position_embedding.weight")
    oc["ln_final.weight"] = hf.pop(t_hf + "final_layer_norm.weight")
    oc["ln_final.bias"] = hf.pop(t_hf + "final_layer_norm.bias")

    oc["text_projection"] = hf.pop("text_projection.weight").T.contiguous()  # HF Linear W.T → open_clip projection matrix

    # Transformer blocks — HF has separate q/k/v, open_clip has in_proj
    for i in range(n_layers):
        oc[t_oc + f"resblocks.{i}.ln_1.weight"] = hf.pop(
            t_hf + f"encoder.layers.{i}.layer_norm1.weight"
        )
        oc[t_oc + f"resblocks.{i}.ln_1.bias"] = hf.pop(
            t_hf + f"encoder.layers.{i}.layer_norm1.bias"
        )
        concat_qkv(
            t_oc + f"resblocks.{i}.",
            t_hf + f"encoder.layers.{i}.",
        )
        oc[t_oc + f"resblocks.{i}.attn.out_proj.weight"] = hf.pop(
            t_hf + f"encoder.layers.{i}.self_attn.out_proj.weight"
        )
        oc[t_oc + f"resblocks.{i}.attn.out_proj.bias"] = hf.pop(
            t_hf + f"encoder.layers.{i}.self_attn.out_proj.bias"
        )
        oc[t_oc + f"resblocks.{i}.ln_2.weight"] = hf.pop(
            t_hf + f"encoder.layers.{i}.layer_norm2.weight"
        )
        oc[t_oc + f"resblocks.{i}.ln_2.bias"] = hf.pop(
            t_hf + f"encoder.layers.{i}.layer_norm2.bias"
        )
        oc[t_oc + f"resblocks.{i}.mlp.c_fc.weight"] = hf.pop(
            t_hf + f"encoder.layers.{i}.mlp.fc1.weight"
        )
        oc[t_oc + f"resblocks.{i}.mlp.c_fc.bias"] = hf.pop(
            t_hf + f"encoder.layers.{i}.mlp.fc1.bias"
        )
        oc[t_oc + f"resblocks.{i}.mlp.c_proj.weight"] = hf.pop(
            t_hf + f"encoder.layers.{i}.mlp.fc2.weight"
        )
        oc[t_oc + f"resblocks.{i}.mlp.c_proj.bias"] = hf.pop(
            t_hf + f"encoder.layers.{i}.mlp.fc2.bias"
        )

    # ── Shared / misc ──────────────────────────────────────────
    oc["visual.proj"] = hf.pop("visual_projection.weight").T.contiguous()  # HF: [512,768] → open_clip: [768,512]
    oc["logit_scale"] = hf.pop("logit_scale")

    # Warn about any unconverted keys
    if hf:
        print(f"  Warning: {len(hf)} unconverted HF keys: {list(hf.keys())[:5]}...")

    return oc


def download_via_modelscope():
    """Download HF Transformers CLIP from modelscope and convert."""
    print(f"Downloading from modelscope: {MODELSCOPE_ID}")
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("modelscope not installed.  pip install modelscope")
        return None

    cache_dir = PROJECT_ROOT / "models" / "clip" / "modelscope_cache"
    model_dir = snapshot_download(MODELSCOPE_ID, cache_dir=cache_dir)
    print(f"  Downloaded to: {model_dir}")

    # Load with transformers
    print("  Loading with transformers...")
    try:
        from transformers import CLIPModel
    except ImportError:
        print("transformers not installed.  pip install transformers")
        return None

    hf_model = CLIPModel.from_pretrained(model_dir)
    print("  Converting to open_clip format...")
    oc_state = convert_hf_to_openclip(hf_model)
    return oc_state


def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_PATH.exists():
        print(f"Already exists: {SAVE_PATH}")
        return

    state = download_via_modelscope()
    if state is None:
        print("\nCould not download.")
        print("  Try: pip install modelscope transformers")
        sys.exit(1)

    # Save as safetensors
    try:
        from safetensors.torch import save_file as st_save

        st_save(state, str(SAVE_PATH))
        print(f"Saved to {SAVE_PATH}  ({os.path.getsize(SAVE_PATH) / 1024**3:.2f} GB)")
    except ImportError:
        import torch

        torch_save_path = SAVE_PATH.with_suffix(".bin")
        torch.save(state, str(torch_save_path))
        print(f"Saved to {torch_save_path}  ({os.path.getsize(torch_save_path) / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
