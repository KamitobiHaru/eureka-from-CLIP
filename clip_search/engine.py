from typing import Callable, List, Dict, Optional

import numpy as np

from .segmenter import detect_scenes, Scene
from .encoder import CLIPEncoder


class SearchEngine:
    """Orchestrator: segment video → encode scenes → rank by text query similarity."""

    def __init__(self, clip_encoder: CLIPEncoder = None,
                 text_encoder: Optional[Callable[[str], np.ndarray]] = None,
                 scene_encoder: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
                 device: str = None):
        self.encoder = clip_encoder or CLIPEncoder(device=device)
        self.text_encoder = text_encoder  # callable(str) -> 512-dim L2-normalized embedding
        self.scene_encoder = scene_encoder  # callable(frames) -> 512-dim embedding
        self.scenes: List[Scene] = []
        self.scene_embs: np.ndarray = None  # (N, 512)

    def process_video(self, video_path: str) -> List[Scene]:
        """Segment a video into scenes and encode each scene.

        Uses scene_encoder when available, otherwise falls back to
        CLIPEncoder.encode_scene() (mean pooling over frames).

        Returns the list of Scene objects (also stored internally).
        """
        self.scenes = detect_scenes(video_path)
        if not self.scenes:
            return []

        scene_embs = []
        for scene in self.scenes:
            if self.scene_encoder is not None:
                emb = self.scene_encoder(scene.frames)
            else:
                emb = self.encoder.encode_scene(scene.frames)
            scene_embs.append(emb)

        self.scene_embs = np.stack(scene_embs)  # (N, 512)
        return self.scenes

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search scenes by text query.

        Returns top_k results sorted by descending cosine similarity.
        Each result dict: {scene_idx, start_sec, end_sec, score, thumbnail, frames}
        """
        if not self.scenes or self.scene_embs is None:
            return []

        if self.text_encoder is not None:
            query_emb = self.text_encoder(query)  # (512,)
        else:
            query_emb = self.encoder.encode_text(query)  # (512,)

        scores = self.scene_embs @ query_emb  # (N,) cosine similarity (L2-normed)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            scene = self.scenes[idx]
            results.append({
                "scene_idx": scene.scene_idx,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "score": float(scores[idx]),
                "thumbnail": scene.thumbnail,
                "frames": scene.frames,
            })

        return results
