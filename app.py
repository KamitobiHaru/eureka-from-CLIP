import argparse
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np

from clip_search import SearchEngine, load_text_encoder, discover_videos
from imageio_ffmpeg import get_ffmpeg_exe

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

FFMPEG = get_ffmpeg_exe()

NUM_RESULTS = 10


def _run_ffmpeg(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess | None:
    """Run ffmpeg with a timeout. Returns None on timeout/failure."""
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def extract_clip(video_path: str, start_sec: float, end_sec: float, output_path: str) -> str | None:
    """Extract a video segment via ffmpeg stream copy (fast, no re-encode)."""
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
    return str(output_path) if (result is not None and result.returncode == 0) else None


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


def grab_frame(video_path: str, timestamp: float, output_path: str) -> str | None:
    """Extract a single frame from a video at the given timestamp as JPEG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path),
    ]
    result = _run_ffmpeg(cmd)
    return str(output_path) if (result is not None and result.returncode == 0) else None


def make_placeholder_thumb(width: int = 480, height: int = 270) -> "np.ndarray":
    """Create a solid light-gray placeholder image for missing thumbnails."""
    import numpy as np
    return np.full((height, width, 3), 220, dtype=np.uint8)


# ── Global encoder cache (avoids passing CUDA models through gr.State()) ──
_CLIP_ENCODER = None
_BERT_ENCODER = None  # tuple: (model_type, bert_ckpt, config_path, callable)

def _get_clip_encoder(device: str = None):
    """Lazily create and cache a single CLIPEncoder instance (module-level)."""
    global _CLIP_ENCODER
    if _CLIP_ENCODER is None:
        from clip_search.encoder import CLIPEncoder
        _CLIP_ENCODER = CLIPEncoder(device=device)
    return _CLIP_ENCODER

def _get_bert_encoder(model: str, bert_checkpoint: str, config_path: str,
                      device: str = None):
    """Lazily create and cache a BERT text encoder callable."""
    global _BERT_ENCODER
    key = (model, bert_checkpoint, config_path)
    if _BERT_ENCODER is not None and _BERT_ENCODER[0] == key:
        return _BERT_ENCODER[1]
    text_enc = load_text_encoder(
        model=model, bert_checkpoint=bert_checkpoint,
        config_path=config_path, device=device,
    )
    _BERT_ENCODER = (key, text_enc)
    return text_enc


def process_videos(folder_path: str, cache_path: str, model_config: dict, device: str,
                   progress=gr.Progress()):
    """Process all videos or load from cache. Returns (data_dict, gallery, status)."""
    model = model_config.get("model", "clip")
    bert_ckpt = model_config.get("bert_checkpoint", None)
    config_path = model_config.get("config_path", "config/default.yaml")

    # Ensure the global CLIP encoder is initialised (side effect: loads model once).
    _get_clip_encoder(device=device)

    if cache_path and cache_path.strip():
        progress(0, desc="Loading cache...")
        from clip_search import SearchEngine
        engine = SearchEngine(device=device)
        n = engine.load_cache(cache_path)
        if n == 0:
            raise gr.Error("Cache is empty.")

        # If a video folder was also provided, try to reconstruct video_map
        if folder_path and Path(folder_path).is_dir():
            progress(0.1, desc="Matching videos from folder...")
            for vp_str in discover_videos(folder_path):
                stem = Path(vp_str).stem
                existing = engine.video_map.get(stem)
                if existing and Path(existing).exists():
                    continue
                engine.video_map[stem] = vp_str

        gallery = []
        for s in engine.scenes:
            thumb = s.thumbnail if s.thumbnail is not None else make_placeholder_thumb()
            gallery.append((
                thumb,
                f"[{s.video_id}] {s.start_sec:.1f}s → {s.end_sec:.1f}s ({s.duration:.1f}s)"
            ))
        progress(1.0, desc="Done!")
        # Return only picklable data — no CUDA models.
        data = {
            "scenes": engine.scenes,
            "scene_embs": engine.scene_embs,
            "video_map": engine.video_map,
            "model": model,
            "bert_checkpoint": bert_ckpt,
            "config_path": config_path,
        }
        return data, gallery, f"✅ Loaded {len(engine.scenes)} videos from cache."

    if not folder_path:
        raise gr.Error("Please enter a video folder path.")

    progress(0, desc="Scanning folder for videos...")

    try:
        video_paths = discover_videos(folder_path)
    except (NotADirectoryError, FileNotFoundError) as e:
        raise gr.Error(str(e))

    if not video_paths:
        raise gr.Error(f"No video files found in {folder_path}")

    progress(0.05, desc=f"Found {len(video_paths)} videos")

    from clip_search import SearchEngine
    engine = SearchEngine(device=device)
    for i, vp in enumerate(video_paths):
        progress(0.05 + 0.6 * (i / len(video_paths)),
                 desc=f"Processing {Path(vp).name}...")
        video_stem = Path(vp).stem
        persistent_video = Path("test_run") / video_stem / "input.mp4"
        persistent_video.parent.mkdir(parents=True, exist_ok=True)
        if persistent_video.exists() and persistent_video.stat().st_size > 0:
            video_path_to_use = str(persistent_video)
        else:
            converted = convert_to_mp4(vp, str(persistent_video))
            video_path_to_use = converted if converted else vp
        engine.process_video(video_path_to_use, append=True)

    if not engine.scenes:
        raise gr.Error("No scenes detected in any video.")

    progress(0.8, desc="Building gallery...")
    gallery = []
    thumb_dir = Path("test_run") / "thumbnails"
    for s in engine.scenes:
        thumb = s.thumbnail
        if thumb is None:
            video_path = engine.video_map.get(s.video_id)
            if video_path and Path(video_path).exists():
                thumb_path = thumb_dir / f"{s.video_id}_scene_{s.scene_idx:03d}.jpg"
                thumb = grab_frame(video_path, s.start_sec, str(thumb_path))
        if thumb is None:
            thumb = make_placeholder_thumb()
        gallery.append((
            thumb,
            f"[{s.video_id}] {s.start_sec:.1f}s → {s.end_sec:.1f}s ({s.duration:.1f}s)"
        ))

    progress(1.0, desc="Done!")
    data = {
        "scenes": engine.scenes,
        "scene_embs": engine.scene_embs,
        "video_map": engine.video_map,
        "model": model,
        "bert_checkpoint": bert_ckpt,
        "config_path": config_path,
    }
    return data, gallery, f"✅ {len(video_paths)} videos processed."


def search_scenes(data: dict, query: str, top_k: int, device: str = None):
    """Run a text query, return ranked playable clips."""
    if data is None or not data.get("scenes"):
        raise gr.Error("No videos processed. Please process a video folder first.")
    if not query.strip():
        raise gr.Error("Please enter a search query.")

    scenes = data["scenes"]
    scene_embs = data["scene_embs"]
    video_map = data["video_map"]

    # Encode query using the global encoder (never stored in gr.State).
    model = data.get("model", "clip")
    bert_ckpt = data.get("bert_checkpoint")
    config_path = data.get("config_path", "config/default.yaml")

    if model != "clip" and bert_ckpt:
        text_encoder = _get_bert_encoder(model, bert_ckpt, config_path, device=device)
        query_emb = text_encoder(query)
    else:
        clip_enc = _get_clip_encoder(device=device)
        query_emb = clip_enc.encode_text(query)

    # Cosine similarity search.
    scores = scene_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]

    n_results = len(top_indices)
    if n_results == 0:
        outputs = [gr.update(value=None, visible=False) for _ in range(5)]
        return [*outputs, "No matching scenes found."]

    # Extract playable clips for search results (ffmpeg stream copy)
    base_dir = Path("test_run") / "folder_search" / "clips"
    video_outputs = []
    for i in range(min(5, n_results)):
        scene = scenes[top_indices[i]]
        score = scores[top_indices[i]]
        label = f"[{scene.video_id}] {scene.start_sec:.1f}s→{scene.end_sec:.1f}s  Score: {score:.4f}"
        video_path = video_map.get(scene.video_id)
        if video_path and Path(video_path).exists():
            clip_name = f"{scene.video_id}_scene_{scene.scene_idx:03d}.mp4"
            clip_file = extract_clip(video_path, scene.start_sec, scene.end_sec,
                                     str(base_dir / clip_name))
            video_outputs.append(gr.update(value=clip_file, label=label, visible=True) if clip_file
                                 else gr.update(value=None, label=label, visible=True))
        else:
            video_outputs.append(gr.update(value=None, label=f"{label} (video path missing)", visible=True))
    while len(video_outputs) < 5:
        video_outputs.append(gr.update(value=None, visible=False))

    status = f"Top {n_results} results for \"{query}\""
    return [*video_outputs, status]


def create_demo(device: str = None,
                model: str = "clip",
                bert_checkpoint: str = None,
                config_path: str = "config/default.yaml",
                video_dir: str = None):
    """Build and return the Gradio Blocks demo.

    Parameters are the CLI argument values used as UI defaults.
    """
    _model = model
    _bert_checkpoint = bert_checkpoint
    _config_path = config_path
    _device = device

    def _confirm_model(bert_ckpt: str, mdl: str, config_yaml: str):
        """Confirm model config and return (config_dict, display_name)."""
        resolved_bert = bert_ckpt or _bert_checkpoint
        resolved_model = mdl if mdl != "clip" else _model
        resolved_config = config_yaml or _config_path
        config = {
            "bert_checkpoint": resolved_bert,
            "model": resolved_model,
            "config_path": resolved_config,
            "stack_lora_cfg": None,
        }
        if resolved_bert:
            name = f"{resolved_model} ({Path(resolved_bert).name})"
        else:
            name = "CLIP (默认)"
        return config, f"✅ 当前模型: {name}"

    def _process_wrapper(folder_path, cache_path, model_config):
        return process_videos(folder_path, cache_path, model_config, device=_device)

    initial_model_status = (
        "CLIP (默认)" if _model == "clip"
        else f"{_model} ({Path(_bert_checkpoint).name if _bert_checkpoint else '?'})"
    )

    with gr.Blocks(title="Eureca from CLIP", css="") as demo:
        gr.Markdown(
            "# 🎬 Eureca from CLIP\n"
            "Select a folder of videos, then search for scenes using natural language queries."
        )

        engine_state = gr.State()
        model_config_state = gr.State({
            "bert_checkpoint": _bert_checkpoint,
            "model": _model,
            "config_path": _config_path,
            "stack_lora_cfg": None,
        })

        with gr.Row(equal_height=False):
            # ── Left Column ──────────────────────────────────────
            with gr.Column(scale=1, min_width=480):
                gr.Markdown("### Step 1: Select Folder & Process")
                folder_input = gr.Textbox(
                    label="Video Folder Path",
                    placeholder="/path/to/videos/",
                    value=video_dir or "",
                )
                cache_input = gr.Textbox(
                    label="Cache Folder Path (optional)",
                    placeholder="/path/to/cache/  — skips video processing",
                )
                with gr.Accordion("Model Settings (optional)", open=False):
                    model_dropdown = gr.Dropdown(
                        choices=["clip", "bert-coco", "bert-stack-lora"],
                        value=_model,
                        label="Model Type",
                    )
                    bert_checkpoint_box = gr.Textbox(
                        label="BERT Checkpoint Path",
                        placeholder="checkpoints/bert_best.pt",
                        value=_bert_checkpoint or "",
                    )
                    config_path_box = gr.Textbox(
                        label="Config YAML Path",
                        placeholder="config/default.yaml",
                        value=_config_path,
                    )
                    with gr.Row():
                        confirm_model_btn = gr.Button("✅ Confirm Model", variant="secondary", size="sm")
                        model_status = gr.Textbox(
                            label="Model Status",
                            value=initial_model_status,
                            interactive=False,
                        )
                process_btn = gr.Button("🚀 Process Folder", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)
                scene_gallery = gr.Gallery(
                    label="Video Library",
                    columns=3,
                    height=300,
                    object_fit="contain",
                )

            # ── Right Column ─────────────────────────────────────
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

                result_videos = []
                for i in range(5):
                    v = gr.Video(label="", height=180, visible=True)
                    result_videos.append(v)

        gr.Markdown(
            "---\n"
            "**How it works**: Uniformly sampled frames from each video → "
            "CLIP ViT-B/32 encodes and mean-pools into one video embedding → "
            "cosine similarity ranks the videos against your text query."
        )

        confirm_model_btn.click(
            fn=_confirm_model,
            inputs=[bert_checkpoint_box, model_dropdown, config_path_box],
            outputs=[model_config_state, model_status],
        )

        process_btn.click(
            fn=_process_wrapper,
            inputs=[folder_input, cache_input, model_config_state],
            outputs=[engine_state, scene_gallery, status_text],
        )

        search_btn.click(
            fn=search_scenes,
            inputs=[engine_state, query_input, top_k_slider],
            outputs=[*result_videos, search_status],
        )

    return demo


if __name__ == "__main__":
    import socket

    parser = argparse.ArgumentParser(description="Launch CLIP Video Scene Search UI")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect.")
    parser.add_argument("--no-share", action="store_true",
                        help="Disable public share link (only local access).")
    parser.add_argument("--model", default="clip",
                        choices=["clip", "bert-coco", "bert-stack-lora"],
                        help="Text encoder model type (default: clip).")
    parser.add_argument("--bert-checkpoint", "-b", default=None,
                        help="Path to BERT checkpoint .pt file (required for bert-coco and bert-stack-lora).")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config (for BERT model path and stack_lora parameters).")
    parser.add_argument("--video-dir", default=None,
                        help="Default video folder path (can be overridden in the UI).")
    args = parser.parse_args()

    if args.model in ("bert-coco", "bert-stack-lora") and not args.bert_checkpoint:
        parser.error(f"--bert-checkpoint is required when --model={args.model}")

    demo = create_demo(
        device=args.device,
        model=args.model,
        bert_checkpoint=args.bert_checkpoint,
        config_path=args.config,
        video_dir=args.video_dir,
    )

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
    print(f"  🎯 Model:          {args.model}")
    if args.bert_checkpoint:
        print(f"  📁 BERT checkpoint: {args.bert_checkpoint}")
    if args.video_dir:
        print(f"  📂 Video folder:    {args.video_dir}")
    if want_share:
        print("  🌍 Generating public share link...")
    else:
        print("  (no public share link — use VS Code Ports tab to forward localhost:{port})")
    print()

    demo.launch(server_name="0.0.0.0", server_port=port,
                theme=gr.themes.Soft(), share=want_share)
