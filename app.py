import argparse
import shutil
import subprocess
from pathlib import Path

import gradio as gr

from clip_search import SearchEngine, load_text_encoder, discover_videos
from imageio_ffmpeg import get_ffmpeg_exe

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

FFMPEG = get_ffmpeg_exe()

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


def process_videos(folder_path: str, model_config: dict, device: str,
                   progress=gr.Progress()):
    """Process all videos or load from cache. Returns (engine_state, gallery, status)."""
    model = model_config.get("model", "clip")
    bert_ckpt = model_config.get("bert_checkpoint", None)
    config_path = model_config.get("config_path", "config/default.yaml")
    stack_lora_cfg = model_config.get("stack_lora_cfg", None)

    engine = SearchEngine(device=device)
    if bert_ckpt:
        text_enc = load_text_encoder(
            model=model,
            bert_checkpoint=bert_ckpt,
            config_path=config_path,
            stack_lora_cfg=stack_lora_cfg,
            device=device,
        )
        engine.text_encoder = text_enc

    cache_path = model_config.get("cache_path")
    if cache_path:
        progress(0, desc="Loading cache...")
        n = engine.load_cache(cache_path)
        if n == 0:
            raise gr.Error("Cache is empty.")
        gallery = []
        for s in engine.scenes:
            gallery.append((
                s.thumbnail,
                f"[{s.video_id}] {s.start_sec:.1f}s → {s.end_sec:.1f}s ({s.duration:.1f}s)"
            ))
        progress(1.0, desc="Done!")
        return engine, gallery, f"✅ Loaded {n} videos from cache."

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
    for s in engine.scenes:
        gallery.append((
            s.thumbnail,
            f"[{s.video_id}] Scene {s.scene_idx}: {s.start_sec:.1f}s → {s.end_sec:.1f}s ({s.duration:.1f}s)"
        ))

    progress(1.0, desc="Done!")
    return engine, gallery, f"✅ {len(video_paths)} videos processed, {len(engine.scenes)} scenes indexed."


def search_scenes(engine: SearchEngine, query: str, top_k: int):
    """Run a text query, extract video clips, return them as video components."""
    if engine is None or not engine.scenes:
        raise gr.Error("No videos processed. Please process a video folder first.")
    if not query.strip():
        raise gr.Error("Please enter a search query.")

    results = engine.search(query, top_k=top_k)
    if not results:
        return [None] * NUM_RESULTS + ["No matching scenes found."]

    base_dir = Path("test_run") / "folder_search" / "clips"
    outputs = []
    for i, r in enumerate(results[:NUM_RESULTS]):
        video_path = engine.video_map.get(r["video_id"])
        clip_name = f"{r['video_id']}_scene_{r['scene_idx']:03d}.mp4"
        clip_path = base_dir / clip_name
        label = f"[{r['video_id']}] Scene {r['scene_idx']}: {r['start_sec']:.1f}s → {r['end_sec']:.1f}s  Score: {r['score']:.4f}"

        if video_path:
            clip_file = extract_clip(video_path, r['start_sec'], r['end_sec'], str(clip_path))
            if clip_file:
                outputs.append(gr.update(value=clip_file, label=label, visible=True))
            else:
                outputs.append(gr.update(value=None, label=f"{label} (clip failed)", visible=True))
        else:
            outputs.append(gr.update(value=None, label=f"{label} (video path missing)", visible=True))

    while len(outputs) < NUM_RESULTS:
        outputs.append(None)

    status = f"🎬 Top {len(results)} results for \"{query}\""
    outputs.append(status)
    return outputs


def create_demo(device: str = None,
                model: str = "clip",
                bert_checkpoint: str = None,
                config_path: str = "config/default.yaml",
                video_dir: str = None,
                cache_path: str = None):
    """Build and return the Gradio Blocks demo.

    Parameters are the CLI argument values used as UI defaults.
    """
    _model = model
    _bert_checkpoint = bert_checkpoint
    _config_path = config_path
    _device = device
    _cache_path = cache_path

    def _confirm_model(bert_ckpt: str, mdl: str):
        """Confirm model config and return (config_dict, display_name)."""
        resolved_bert = bert_ckpt or _bert_checkpoint
        resolved_model = mdl if mdl != "clip" else _model
        config = {
            "bert_checkpoint": resolved_bert,
            "model": resolved_model,
            "config_path": _config_path,
            "stack_lora_cfg": None,
            "cache_path": _cache_path,
        }
        if resolved_bert:
            name = f"{resolved_model} ({Path(resolved_bert).name})"
        else:
            name = "CLIP (默认)"
        return config, f"✅ 当前模型: {name}"

    def _process_wrapper(folder_path, model_config):
        return process_videos(folder_path, model_config, device=_device)

    initial_model_status = (
        "CLIP (默认)" if _model == "clip"
        else f"{_model} ({Path(_bert_checkpoint).name if _bert_checkpoint else '?'})"
    )

    with gr.Blocks(title="CLIP Video Scene Search", css="") as demo:
        gr.Markdown(
            "# 🎬 CLIP Video Scene Search\n"
            "Select a folder of videos, then search for scenes using natural language queries."
        )

        engine_state = gr.State()
        model_config_state = gr.State({
            "bert_checkpoint": _bert_checkpoint,
            "model": _model,
            "config_path": _config_path,
            "stack_lora_cfg": None,
            "cache_path": _cache_path,
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
                    label="Detected Scenes",
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
                for i in range(NUM_RESULTS):
                    v = gr.Video(label=f"Result {i+1}", height=180, visible=False)
                    result_videos.append(v)

        gr.Markdown(
            "---\n"
            "**How it works**: Uniformly sampled frames from each video → "
            "CLIP ViT-B/32 encodes and mean-pools into one video embedding → "
            "cosine similarity ranks the videos against your text query."
        )

        confirm_model_btn.click(
            fn=_confirm_model,
            inputs=[bert_checkpoint_box, model_dropdown],
            outputs=[model_config_state, model_status],
        )

        process_btn.click(
            fn=_process_wrapper,
            inputs=[folder_input, model_config_state],
            outputs=[engine_state, scene_gallery, status_text],
        )

        search_btn.click(
            fn=search_scenes,
            inputs=[engine_state, query_input, top_k_slider],
            outputs=result_videos + [search_status],
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
    parser.add_argument("--cache-path", default=None,
                        help="Path to precomputed cache directory (from cache_videos.py). "
                             "When set, video processing is skipped.")
    args = parser.parse_args()

    if args.model in ("bert-coco", "bert-stack-lora") and not args.bert_checkpoint:
        parser.error(f"--bert-checkpoint is required when --model={args.model}")

    demo = create_demo(
        device=args.device,
        model=args.model,
        bert_checkpoint=args.bert_checkpoint,
        config_path=args.config,
        video_dir=args.video_dir,
        cache_path=args.cache_path,
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
