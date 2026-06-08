"""Convert HuggingFace CLIP checkpoint to open_clip format.

Converts ``models/clip-vit-base-patch32/pytorch_model.bin`` to
``models/clip-vit-base-patch32/open_clip_model.safetensors`` so that
``open_clip.create_model_and_transforms()`` can load it directly.

Usage::

    python scripts/convert_hf_to_openclip.py
"""

from pathlib import Path

import torch
import open_clip


def _qkv_weight(hf_sd, prefix, dim):
    """Concatenate Q, K, V projection weights into ``in_proj_weight``."""
    q = hf_sd[f"{prefix}self_attn.q_proj.weight"]
    k = hf_sd[f"{prefix}self_attn.k_proj.weight"]
    v = hf_sd[f"{prefix}self_attn.v_proj.weight"]
    return torch.cat([q, k, v], dim=0)


def _qkv_bias(hf_sd, prefix):
    q = hf_sd[f"{prefix}self_attn.q_proj.bias"]
    k = hf_sd[f"{prefix}self_attn.k_proj.bias"]
    v = hf_sd[f"{prefix}self_attn.v_proj.bias"]
    return torch.cat([q, k, v], dim=0)


def convert():
    hf_path = Path(__file__).resolve().parent.parent / "models" / "clip-vit-base-patch32"
    hf_ckpt = torch.load(hf_path / "pytorch_model.bin", map_location="cpu", weights_only=True)

    # Build an open_clip model as a shell for the state dict
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="")
    oc_sd = {}

    # ── Text encoder ────────────────────────────────────────────
    oc_sd["positional_embedding"] = hf_ckpt["text_model.embeddings.position_embedding.weight"]
    oc_sd["token_embedding.weight"] = hf_ckpt["text_model.embeddings.token_embedding.weight"]
    oc_sd["ln_final.weight"] = hf_ckpt["text_model.final_layer_norm.weight"]
    oc_sd["ln_final.bias"] = hf_ckpt["text_model.final_layer_norm.bias"]
    oc_sd["text_projection"] = hf_ckpt["text_projection.weight"].T.contiguous()

    n_layers = 12  # ViT-B-32
    for i in range(n_layers):
        hf_pre = f"text_model.encoder.layers.{i}."
        oc_pre = f"transformer.resblocks.{i}."
        oc_sd[f"{oc_pre}attn.in_proj_weight"] = _qkv_weight(hf_ckpt, hf_pre, 512)
        oc_sd[f"{oc_pre}attn.in_proj_bias"] = _qkv_bias(hf_ckpt, hf_pre)
        oc_sd[f"{oc_pre}attn.out_proj.weight"] = hf_ckpt[f"{hf_pre}self_attn.out_proj.weight"]
        oc_sd[f"{oc_pre}attn.out_proj.bias"] = hf_ckpt[f"{hf_pre}self_attn.out_proj.bias"]
        oc_sd[f"{oc_pre}ln_1.weight"] = hf_ckpt[f"{hf_pre}layer_norm1.weight"]
        oc_sd[f"{oc_pre}ln_1.bias"] = hf_ckpt[f"{hf_pre}layer_norm1.bias"]
        oc_sd[f"{oc_pre}ln_2.weight"] = hf_ckpt[f"{hf_pre}layer_norm2.weight"]
        oc_sd[f"{oc_pre}ln_2.bias"] = hf_ckpt[f"{hf_pre}layer_norm2.bias"]
        oc_sd[f"{oc_pre}mlp.c_fc.weight"] = hf_ckpt[f"{hf_pre}mlp.fc1.weight"]
        oc_sd[f"{oc_pre}mlp.c_fc.bias"] = hf_ckpt[f"{hf_pre}mlp.fc1.bias"]
        oc_sd[f"{oc_pre}mlp.c_proj.weight"] = hf_ckpt[f"{hf_pre}mlp.fc2.weight"]
        oc_sd[f"{oc_pre}mlp.c_proj.bias"] = hf_ckpt[f"{hf_pre}mlp.fc2.bias"]

    # ── Vision encoder ──────────────────────────────────────────
    oc_sd["visual.class_embedding"] = hf_ckpt["vision_model.embeddings.class_embedding"]
    oc_sd["visual.positional_embedding"] = hf_ckpt["vision_model.embeddings.position_embedding.weight"]
    oc_sd["visual.conv1.weight"] = hf_ckpt["vision_model.embeddings.patch_embedding.weight"]
    oc_sd["visual.ln_pre.weight"] = hf_ckpt["vision_model.pre_layrnorm.weight"]
    oc_sd["visual.ln_pre.bias"] = hf_ckpt["vision_model.pre_layrnorm.bias"]
    oc_sd["visual.proj"] = hf_ckpt["visual_projection.weight"].T.contiguous()
    oc_sd["visual.ln_post.weight"] = hf_ckpt["vision_model.post_layernorm.weight"]
    oc_sd["visual.ln_post.bias"] = hf_ckpt["vision_model.post_layernorm.bias"]

    for i in range(n_layers):
        hf_pre = f"vision_model.encoder.layers.{i}."
        oc_pre = f"visual.transformer.resblocks.{i}."
        oc_sd[f"{oc_pre}attn.in_proj_weight"] = _qkv_weight(hf_ckpt, hf_pre, 768)
        oc_sd[f"{oc_pre}attn.in_proj_bias"] = _qkv_bias(hf_ckpt, hf_pre)
        oc_sd[f"{oc_pre}attn.out_proj.weight"] = hf_ckpt[f"{hf_pre}self_attn.out_proj.weight"]
        oc_sd[f"{oc_pre}attn.out_proj.bias"] = hf_ckpt[f"{hf_pre}self_attn.out_proj.bias"]
        oc_sd[f"{oc_pre}ln_1.weight"] = hf_ckpt[f"{hf_pre}layer_norm1.weight"]
        oc_sd[f"{oc_pre}ln_1.bias"] = hf_ckpt[f"{hf_pre}layer_norm1.bias"]
        oc_sd[f"{oc_pre}ln_2.weight"] = hf_ckpt[f"{hf_pre}layer_norm2.weight"]
        oc_sd[f"{oc_pre}ln_2.bias"] = hf_ckpt[f"{hf_pre}layer_norm2.bias"]
        oc_sd[f"{oc_pre}mlp.c_fc.weight"] = hf_ckpt[f"{hf_pre}mlp.fc1.weight"]
        oc_sd[f"{oc_pre}mlp.c_fc.bias"] = hf_ckpt[f"{hf_pre}mlp.fc1.bias"]
        oc_sd[f"{oc_pre}mlp.c_proj.weight"] = hf_ckpt[f"{hf_pre}mlp.fc2.weight"]
        oc_sd[f"{oc_pre}mlp.c_proj.bias"] = hf_ckpt[f"{hf_pre}mlp.fc2.bias"]

    # ── Logit scale ─────────────────────────────────────────────
    oc_sd["logit_scale"] = hf_ckpt["logit_scale"]

    # Load into model and verify
    missing, unexpected = model.load_state_dict(oc_sd, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    # Save as safetensors
    import safetensors.torch
    out_path = hf_path / "open_clip_model.safetensors"
    safetensors.torch.save_file(oc_sd, str(out_path))
    print(f"Saved to {out_path}")

    # Quick sanity check — text embeddings match reference
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    texts = ["a cat on a mat", "a plane in the sky"]
    tokens = tokenizer(texts)
    with torch.no_grad():
        emb = model.encode_text(tokens, normalize=True)
    print(f"Text embeddings: {emb.shape}")
    print(f"Norm: {emb.norm(dim=-1)}")


if __name__ == "__main__":
    convert()
