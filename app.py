import argparse
import shutil
import subprocess
from pathlib import Path

import gradio as gr

from clip_search import SearchEngine, build_temporal_pipeline
from imageio_ffmpeg import get_ffmpeg_exe

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

FFMPEG = get_ffmpeg_exe()

# Set via --device CLI arg in __main__
DEVICE = None

NUM_RESULTS = 10


def _run_ffmpeg(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess | None:
    """Run ffmpeg with a timeout. Returns None on timeout/failure."""
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def extract_clip(video_path: str, start_sec: float, end_sec: float, output_path: str) -> str | None:
    """Extract a video segment using ffmpeg. Returns output path or None on failure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec
    cmd = [
        FFMPEG, "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",
        str(output_path),
    ]
    result = _run_ffmpeg(cmd)
    if result is None or result.returncode != 0:
        # Fallback: re-encode with libx264
        cmd = [
            FFMPEG, "-y",
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(output_path),
        ]
        result = _run_ffmpeg(cmd)
        if result is None or result.returncode != 0:
            return None
    return str(output_path)


def convert_to_mp4(input_path: str, output_path: str) -> str | None:
    """Convert any video to H.264 MP4 for browser compatibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = _run_ffmpeg(cmd)
    return str(output_path) if (result is not None and result.returncode == 0) else None


def load_bert_text_encoder(bert_checkpoint: str, device=None):
    """Load a trained BertEncoder checkpoint, return a text encoding function."""
    import yaml
    import numpy as np
    import torch
    from transformers import BertTokenizer
    from src.models import BertEncoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    bert_path = cfg["model"]["bert_model_path"]
    ckpt = torch.load(bert_checkpoint, map_location=device, weights_only=True)
    lora_cfg = ckpt.get("lora_config", None)

    model = BertEncoder(
        model_path=bert_path,
        embed_dim=cfg["model"]["embed_dim"],
        lora_cfg=lora_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    @torch.no_grad()
    def encode_text(text: str):
        tokens = tokenizer([text], padding=True, truncation=True, max_length=77, return_tensors="pt")
        emb = model(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
        return emb.cpu().numpy().flatten().astype(np.float32)

    return encode_text


def confirm_model(use_temporal: bool, temporal_checkpoint: str, bert_checkpoint: str):
    """Confirm model config and return (config_dict, display_name)."""
    config = {
        "use_temporal": use_temporal,
        "temporal_checkpoint": temporal_checkpoint,
        "bert_checkpoint": bert_checkpoint,
    }
    if use_temporal and temporal_checkpoint and bert_checkpoint:
        name = f"Temporal + BERT"
    elif bert_checkpoint:
        from pathlib import Path
        name = f"BERT ({Path(bert_checkpoint).name})"
    else:
        name = "CLIP (默认)"
    return config, f"✅ 当前模型: {name}"


def apply_text_encoder(engine: SearchEngine | None, model_config: dict) -> SearchEngine | None:
    """Hot-swap the engine's text encoder when model config changes.

    Temporal mode is skipped (requires re-processing).
    """
    if engine is None:
        return None

    use_temporal = model_config["use_temporal"]
    bert_ckpt = model_config["bert_checkpoint"]
    temporal_ckpt = model_config["temporal_checkpoint"]

    if use_temporal and temporal_ckpt and bert_ckpt:
        return engine  # Temporal changes scene_encoder too; can't hot-swap

    if bert_ckpt:
        engine.text_encoder = load_bert_text_encoder(bert_ckpt, device=DEVICE)
    else:
        engine.text_encoder = None  # fallback to CLIP's built-in text encoder

    return engine


def process_video(video_path: str, model_config: dict, progress=gr.Progress()):
    """Segment and encode a video. Returns (engine_state, gallery, status)."""
    if not video_path:
        raise gr.Error("Please upload a video file.")

    progress(0, desc="Preparing video...")

    # Use a stable persistent path for cache keying (not the Gradio temp path)
    video_stem = Path(video_path).stem
    persistent_video = Path("test_run") / video_stem / "input.mp4"
    persistent_video.parent.mkdir(parents=True, exist_ok=True)

    # Skip re-conversion if the persistent copy already exists — preserves
    # cache mtime/size and avoids slow ffmpeg re-encode on re-process.
    if persistent_video.exists() and persistent_video.stat().st_size > 0:
        video_path_to_use = str(persistent_video)
    else:
        converted = convert_to_mp4(video_path, str(persistent_video))
        if converted:
            video_path_to_use = converted
        else:
            shutil.copy2(video_path, str(persistent_video))
            video_path_to_use = str(persistent_video)

    progress(0, desc="Checking cache...")

    use_temporal = model_config["use_temporal"]
    temporal_ckpt = model_config["temporal_checkpoint"]
    bert_ckpt = model_config["bert_checkpoint"]

    if use_temporal and temporal_ckpt and bert_ckpt:
        pipeline = build_temporal_pipeline(
            temporal_ckpt, bert_ckpt, device=DEVICE,
        )
        engine = SearchEngine(scene_encoder=pipeline.scene_encoder,
                              text_encoder=pipeline.text_encoder)
    elif bert_ckpt:
        engine = SearchEngine(device=DEVICE)
        engine.text_encoder = load_bert_text_encoder(bert_ckpt, device=DEVICE)
    else:
        engine = SearchEngine(device=DEVICE)

    progress(0.2, desc="Detecting scenes...")
    scenes = engine.process_video(video_path_to_use)
    engine._video_path = video_path_to_use

    if not scenes:
        raise gr.Error("No scenes detected in the video.")

    progress(0.6, desc=f"Encoding {len(scenes)} scenes...")
    # Build gallery of scene thumbnails with captions
    gallery = []
    for s in scenes:
        gallery.append(
            (s.thumbnail, f"Scene {s.scene_idx}: {s.start_sec:.1f}s → {s.end_sec:.1f}s ({s.duration:.1f}s)")
        )

    progress(1.0, desc="Done!")
    return engine, gallery, f"✅ {len(scenes)} scenes indexed, ready to search."


def search_scenes(engine: SearchEngine, query: str, top_k: int):
    """Run a text query, extract video clips, return them as video components."""
    if engine is None or not engine.scenes:
        raise gr.Error("No video processed. Please upload and process a video first.")
    if not query.strip():
        raise gr.Error("Please enter a search query.")

    video_path = getattr(engine, '_video_path', None)
    if not video_path:
        raise gr.Error("Video path not found. Please re-process the video.")

    results = engine.search(query, top_k=top_k)
    if not results:
        return [None] * NUM_RESULTS + ["No matching scenes found."]

    # Extract video clips for top results
    base_dir = Path("test_run") / Path(video_path).stem / "clips"
    outputs = []
    for i, r in enumerate(results[:NUM_RESULTS]):
        clip_name = f"scene_{r['scene_idx']:03d}.mp4"
        clip_path = base_dir / clip_name
        clip_file = extract_clip(video_path, r['start_sec'], r['end_sec'], str(clip_path))
        label = f"Scene {r['scene_idx']}: {r['start_sec']:.1f}s → {r['end_sec']:.1f}s  Score: {r['score']:.4f}"
        if clip_file:
            outputs.append(gr.update(value=clip_file, label=label, visible=True))
        else:
            outputs.append(gr.update(value=None, label=f"{label} (clip failed)", visible=True))

    # Pad remaining slots with None (hidden)
    while len(outputs) < NUM_RESULTS:
        outputs.append(None)

    status = f"🎬 Top {len(results)} results for \"{query}\""
    outputs.append(status)
    return outputs


with gr.Blocks(title="CLIP Video Scene Search") as demo:
    gr.Markdown(
        "# 🎬 CLIP Video Scene Search\n"
        "Upload a video, then search for scenes using natural language queries."
    )

    engine_state = gr.State()
    model_config_state = gr.State({"use_temporal": False, "temporal_checkpoint": "", "bert_checkpoint": ""})

    with gr.Row(equal_height=False):
        # ── Left Column: Upload & Process ─────────────────────
        with gr.Column(scale=1, min_width=480):
            gr.Markdown("### Step 1: Upload & Process Video")
            video_input = gr.Video(label="Upload Video", height=280)
            with gr.Accordion("Model Settings (optional)", open=False):
                use_temporal = gr.Checkbox(label="Enable Temporal Transformer", value=False)
                temporal_checkpoint = gr.Textbox(
                    label="Temporal Checkpoint Path",
                    placeholder="checkpoints/temporal_best.pt",
                )
                bert_checkpoint = gr.Textbox(
                    label="BERT Checkpoint Path",
                    placeholder="checkpoints/bert_best.pt",
                )
                with gr.Row():
                    confirm_model_btn = gr.Button("✅ Confirm Model", variant="secondary", size="sm")
                    model_status = gr.Textbox(label="Model Status", value="CLIP (default)", interactive=False)
            process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
            status_text = gr.Textbox(label="Status", interactive=False)
            scene_gallery = gr.Gallery(
                label="Detected Scenes",
                columns=3,
                height=300,
                object_fit="contain",
            )

        # ── Right Column: Search ──────────────────────────────
        with gr.Column(scale=1, min_width=480):
            gr.Markdown("### Step 2: Search Scenes")
            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder='e.g., "a person walking", "car driving", "beautiful landscape"',
                    scale=3,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=NUM_RESULTS, value=5, step=1,
                    label="Top-K", scale=1,
                )
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg", scale=1)

            search_status = gr.Textbox(label="Search Status", interactive=False)

            # Pre-define result video components (initially hidden)
            result_videos = []
            for i in range(NUM_RESULTS):
                v = gr.Video(label=f"Result {i+1}", height=180, visible=False)
                result_videos.append(v)

    gr.Markdown(
        "---\n"
        "**How it works**: PySceneDetect splits the video into scenes → "
        "CLIP ViT-B/32 encodes each scene and your text query → "
        "cosine similarity ranks the scenes."
    )

    confirm_model_btn.click(
        fn=confirm_model,
        inputs=[use_temporal, temporal_checkpoint, bert_checkpoint],
        outputs=[model_config_state, model_status],
    ).then(
        fn=apply_text_encoder,
        inputs=[engine_state, model_config_state],
        outputs=[engine_state],
    )

    process_btn.click(
        fn=process_video,
        inputs=[video_input, model_config_state],
        outputs=[engine_state, scene_gallery, status_text],
    )

    search_btn.click(
        fn=search_scenes,
        inputs=[engine_state, query_input, top_k_slider],
        outputs=result_videos + [search_status],
    )


if __name__ == "__main__":
    import socket

    parser = argparse.ArgumentParser(description="Launch CLIP Video Scene Search UI")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    parser.add_argument("--no-share", action="store_true",
                        help="Disable public share link (only local access).")
    args = parser.parse_args()
    DEVICE = args.device

    def find_free_port(start: int, max_attempts: int = 10) -> int:
        for port in range(start, start + max_attempts):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return start

    port = find_free_port(7860)

    want_share = not args.no_share
    if want_share:
        try:
            import httpx
            r = httpx.get("https://api.gradio.sh/v0/tunnel", timeout=5)
            if r.status_code != 200:
                want_share = False
        except Exception:
            want_share = False

    print(f"\n  🌐 Local URL:      http://localhost:{port}")
    if want_share:
        print("  🌍 Generating public share link...")
    else:
        print("  (no public share link — use VS Code Ports tab to forward localhost:{port})")
    print()

    demo.launch(server_name="0.0.0.0", server_port=port,
                theme=gr.themes.Soft(), share=want_share)
