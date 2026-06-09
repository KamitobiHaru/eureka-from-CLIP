import hashlib
import json
import os
from pathlib import Path
from typing import Callable, List, Dict, Optional

import cv2
import numpy as np

from .segmenter import detect_scenes, Scene, FRAMES_PER_SCENE
from .encoder import CLIPEncoder

CACHE_BASE = Path("/tmp/clip_search_cache")


class SearchEngine:
    """Orchestrator: segment video → encode scenes → rank by text query similarity."""

    def __init__(self, clip_encoder: CLIPEncoder = None,
                 text_encoder: Optional[Callable[[str], np.ndarray]] = None,
                 scene_encoder: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
                 device: str = None):
        self.encoder = clip_encoder or CLIPEncoder(device=device)
        self.text_encoder = text_encoder
        self.scene_encoder = scene_encoder
        self.scenes: List[Scene] = []
        self.scene_embs: np.ndarray = None

    # ── Cache helpers ────────────────────────────────────────────────

    @staticmethod
    def _cache_dir(video_path: str) -> Path:
        p = Path(video_path).resolve()
        h = hashlib.md5(str(p).encode()).hexdigest()[:16]
        return CACHE_BASE / f"{p.stem}_{h}"

    def _try_load_cache(self, video_path: str) -> bool:
        """Restore scenes + embeddings from disk cache.  Returns True on success."""
        cache_dir = self._cache_dir(video_path)
        info_path = cache_dir / "info.json"
        meta_path = cache_dir / "metadata.json"
        emb_path = cache_dir / "embeddings.npy"

        if not all(p.exists() for p in [info_path, meta_path, emb_path]):
            return False

        # Validate against current file state
        st = os.stat(video_path)
        with open(info_path) as f:
            info = json.load(f)
        if info["mtime"] != int(st.st_mtime) or info["size"] != st.st_size:
            return False

        with open(meta_path) as f:
            metadata = json.load(f)

        video_stem = Path(video_path).stem
        self.scenes = [
            Scene(video_id=video_stem, scene_idx=m["scene_idx"],
                  start_sec=m["start_sec"], end_sec=m["end_sec"], frames=[])
            for m in metadata
        ]

        self._reconstruct_frames(video_path)
        self.scene_embs = np.load(str(emb_path))

        print(f"  [cache] ✓ loaded {len(self.scenes)} scenes from {cache_dir}")
        return True

    def _save_cache(self, video_path: str):
        """Write current scenes + embeddings to disk cache."""
        cache_dir = self._cache_dir(video_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        st = os.stat(video_path)
        with open(cache_dir / "info.json", "w") as f:
            json.dump({"mtime": int(st.st_mtime), "size": st.st_size}, f)

        metadata = [
            {"scene_idx": s.scene_idx, "start_sec": s.start_sec, "end_sec": s.end_sec}
            for s in self.scenes
        ]
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        np.save(str(cache_dir / "embeddings.npy"), self.scene_embs)
        print(f"  [cache] ✓ saved {len(self.scenes)} scenes to {cache_dir}")

    def _reconstruct_frames(self, video_path: str, num_frames: int = FRAMES_PER_SCENE):
        """Re-extract scene frames from video using cached timestamps (no scene detection)."""
        cap = cv2.VideoCapture(video_path)
        try:
            for scene in self.scenes:
                duration = scene.end_sec - scene.start_sec
                n = min(num_frames, max(1, int(duration * 30)))
                timestamps = np.linspace(scene.start_sec, scene.end_sec, n, endpoint=False)
                frames = []
                for t in timestamps:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                scene.frames = frames
        finally:
            cap.release()

    # ── Core API ─────────────────────────────────────────────────────

    def process_video(self, video_path: str) -> List[Scene]:
        """Segment a video into scenes and encode each scene.

        Caches scene metadata and embeddings to ``/tmp/clip_search_cache/``
        after the first run.  Subsequent calls with the same (unchanged)
        video file skip scene detection and re-encoding.
        """
        if self._try_load_cache(video_path):
            return self.scenes

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

        self.scene_embs = np.stack(scene_embs)
        self._save_cache(video_path)
        return self.scenes

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search scenes by text query.

        Returns top_k results sorted by descending cosine similarity.
        Each result dict: {scene_idx, start_sec, end_sec, score, thumbnail, frames}
        """
        if not self.scenes or self.scene_embs is None:
            return []

        if self.text_encoder is not None:
            query_emb = self.text_encoder(query)
        else:
            query_emb = self.encoder.encode_text(query)

        scores = self.scene_embs @ query_emb
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
