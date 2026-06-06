import gradio as gr
from pathlib import Path

from clip_search import SearchEngine, build_temporal_pipeline

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def process_video(video_path: str, use_temporal: bool = False,
                  temporal_checkpoint: str = "", progress=gr.Progress()):
    """Segment and encode a video. Returns (engine_state, gallery, status)."""
    if not video_path:
        raise gr.Error("Please upload a video file.")

    progress(0, desc="Loading CLIP encoder...")

    if use_temporal and temporal_checkpoint:
        pipeline = build_temporal_pipeline(temporal_checkpoint)
        engine = SearchEngine(scene_encoder=pipeline.scene_encoder,
                              text_encoder=pipeline.text_encoder)
    else:
        engine = SearchEngine()

    progress(0.2, desc="Detecting scenes...")
    scenes = engine.process_video(video_path)

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
    """Run a text query against the indexed scenes."""
    if engine is None or not engine.scenes:
        raise gr.Error("No video processed. Please upload and process a video first.")
    if not query.strip():
        raise gr.Error("Please enter a search query.")

    results = engine.search(query, top_k=top_k)
    if not results:
        return [], "No matching scenes found."

    gallery = []
    for r in results:
        caption = (
            f"Scene {r['scene_idx']}  |  "
            f"{r['start_sec']:.1f}s → {r['end_sec']:.1f}s  |  "
            f"Score: {r['score']:.4f}"
        )
        gallery.append((r["thumbnail"], caption))

    return gallery, f"Top {len(results)} results for \"{query}\""


with gr.Blocks(title="CLIP Video Scene Search") as demo:
    gr.Markdown(
        "# 🎬 CLIP Video Scene Search\n"
        "Upload a video, then search for scenes using natural language queries."
    )

    engine_state = gr.State()

    with gr.Tabs():
        # ── Tab 1: Process Video ──────────────────────────────────
        with gr.Tab("1. Process Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video", height=300)
                    with gr.Accordion("Temporal Transformer (optional)", open=False):
                        use_temporal = gr.Checkbox(label="Enable Temporal Transformer", value=False)
                        temporal_checkpoint = gr.Textbox(
                            label="Checkpoint Path",
                            placeholder="checkpoints/temporal_best.pt",
                        )
                    process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    status_text = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    scene_gallery = gr.Gallery(
                        label="Detected Scenes",
                        columns=3,
                        height=400,
                        object_fit="contain",
                    )

            process_btn.click(
                fn=process_video,
                inputs=[video_input, use_temporal, temporal_checkpoint],
                outputs=[engine_state, scene_gallery, status_text],
            )

        # ── Tab 2: Search ─────────────────────────────────────────
        with gr.Tab("2. Search Scenes"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder='e.g., "a person walking", "car driving", "beautiful landscape"',
                    scale=3,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Top-K results", scale=1,
                )
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg", scale=1)

            search_status = gr.Textbox(label="Search Status", interactive=False)

            result_gallery = gr.Gallery(
                label="Search Results",
                columns=2,
                height=600,
                object_fit="contain",
            )

            search_btn.click(
                fn=search_scenes,
                inputs=[engine_state, query_input, top_k_slider],
                outputs=[result_gallery, search_status],
            )

    gr.Markdown(
        "---\n"
        "**How it works**: PySceneDetect splits the video into scenes → "
        "CLIP ViT-B/32 encodes each scene and your text query → "
        "cosine similarity ranks the scenes."
    )


if __name__ == "__main__":
    import socket

    def find_free_port(start: int, max_attempts: int = 10) -> int:
        for port in range(start, start + max_attempts):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return start

    port = find_free_port(7860)
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(), share=True)
