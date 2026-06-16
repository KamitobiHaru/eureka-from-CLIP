import hashlib
import json
import os
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple

import cv2
import numpy as np

from .segmenter import detect_scenes, Scene, FRAMES_PER_SCENE
from .encoder import CLIPEncoder

CACHE_BASE = Path("/tmp/clip_search_cache")

# Supported video extensions for folder processing.
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def discover_videos(folder_path: str) -> List[str]:
    """Return sorted list of absolute video file paths in *folder_path*."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")
    return sorted(
        str(p) for p in folder.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith('.')
    )


class SearchEngine:
    """Orchestrator: segment video(s) → encode scenes → rank by text query similarity.

    Supports single-video processing (legacy) and multi-video folder processing.
    Scenes from all videos are accumulated into a flat list for cross-video search.
    """

    def __init__(self, clip_encoder: CLIPEncoder = None,
                 text_encoder: Optional[Callable[[str], np.ndarray]] = None,
                 scene_encoder: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
                 device: str = None):
        self.encoder = clip_encoder or CLIPEncoder(device=device)
        self.text_encoder = text_encoder
        self.scene_encoder = scene_encoder
        self.scenes: List[Scene] = []
        self.scene_embs: np.ndarray = None

        # Mapping from video_id (stem) → absolute path of the source video.
        # Used by the app to look up which file to extract a clip from.
        self.video_map: Dict[str, str] = {}

    # ── Static cache helpers ─────────────────────────────────────────

    @staticmethod
    def _cache_dir(video_path: str) -> Path:
        p = Path(video_path).resolve()
        h = hashlib.md5(str(p).encode()).hexdigest()[:16]
        return CACHE_BASE / f"{p.stem}_{h}"

    @staticmethod
    def _load_video_cache(video_path: str) -> Optional[Tuple[List[Scene], np.ndarray]]:
        """Load cached scenes + embeddings for a single video.

        Returns ``(scenes, scene_embs)`` or ``None`` if the cache is missing
        or the source file has changed.
        """
        cache_dir = SearchEngine._cache_dir(video_path)
        info_path = cache_dir / "info.json"
        meta_path = cache_dir / "metadata.json"
        emb_path = cache_dir / "embeddings.npy"

        if not all(p.exists() for p in [info_path, meta_path, emb_path]):
            return None

        # Validate against current file state
        st = os.stat(video_path)
        with open(info_path) as f:
            info = json.load(f)
        if info["mtime"] != int(st.st_mtime) or info["size"] != st.st_size:
            return None

        with open(meta_path) as f:
            metadata = json.load(f)

        video_stem = Path(video_path).stem
        scenes = [
            Scene(
                video_id=m.get("video_id", video_stem),
                scene_idx=m["scene_idx"],
                start_sec=m["start_sec"],
                end_sec=m["end_sec"],
                frames=[],
            )
            for m in metadata
        ]

        scene_embs = np.load(str(emb_path))
        print(f"  [cache] ✓ loaded {len(scenes)} scenes from {cache_dir}")
        return scenes, scene_embs

    @staticmethod
    def _save_video_cache(video_path: str, scenes: List[Scene], scene_embs: np.ndarray):
        """Write scenes + embeddings to disk cache for a single video."""
        cache_dir = SearchEngine._cache_dir(video_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        st = os.stat(video_path)
        with open(cache_dir / "info.json", "w") as f:
            json.dump({"mtime": int(st.st_mtime), "size": st.st_size}, f)

        metadata = [
            {
                "video_id": s.video_id,
                "scene_idx": s.scene_idx,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
            }
            for s in scenes
        ]
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        np.save(str(cache_dir / "embeddings.npy"), scene_embs)
        print(f"  [cache] ✓ saved {len(scenes)} scenes to {cache_dir}")

    @staticmethod
    def _reconstruct_frames(video_path: str, scenes: List[Scene],
                            num_frames: int = FRAMES_PER_SCENE):
        """Re-extract scene frames from video using cached timestamps."""
        cap = cv2.VideoCapture(video_path)
        try:
            for scene in scenes:
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

    # ── Core processing ──────────────────────────────────────────────

    def _process_single_video(self, video_path: str) -> Tuple[List[Scene], np.ndarray]:
        """Process one video (cache-aware). Returns ``(scenes, embeddings)``.

        Does **not** touch ``self.scenes`` / ``self.scene_embs`` — those are
        managed by the public ``process_video`` / ``process_folder`` methods.
        """
        cached = self._load_video_cache(video_path)
        if cached is not None:
            scenes, scene_embs = cached
            self._reconstruct_frames(video_path, scenes)
            return scenes, scene_embs

        scenes = detect_scenes(video_path)
        if not scenes:
            return [], np.zeros((0, 512), dtype=np.float32)

        scene_embs_list = []
        for scene in scenes:
            if self.scene_encoder is not None:
                emb = self.scene_encoder(scene.frames)
            else:
                emb = self.encoder.encode_scene(scene.frames)
            scene_embs_list.append(emb)

        scene_embs = np.stack(scene_embs_list)
        self._save_video_cache(video_path, scenes, scene_embs)
        return scenes, scene_embs

    def process_video(self, video_path: str, append: bool = False) -> List[Scene]:
        """Segment and encode a video.

        Parameters
        ----------
        video_path:
            Path to the video file.
        append:
            If ``True`` the scenes are added to the existing pool; if ``False``
            (the default, legacy behaviour) any previously loaded scenes are
            cleared first.

        Returns the list of ``Scene`` objects for *this* video.
        """
        if not append:
            self.scenes = []
            self.scene_embs = None
            self.video_map = {}

        scenes, embs = self._process_single_video(video_path)
        video_id = Path(video_path).stem

        # Append to the global pool.
        self.scenes.extend(scenes)
        self.video_map[video_id] = str(Path(video_path).resolve())

        if self.scene_embs is None:
            self.scene_embs = embs
        elif len(scenes) > 0:
            self.scene_embs = np.concatenate([self.scene_embs, embs], axis=0)

        # Legacy accessor (kept for backwards compatibility).
        self._video_path = str(Path(video_path).resolve())

        return scenes

    def process_folder(self, folder_path: str) -> List[Scene]:
        """Process all videos in *folder_path* and accumulate scenes.

        Each video is processed independently with its own disk cache.  After
        this call, ``self.scenes`` contains scenes from every video and
        ``self.video_map`` maps each ``video_id`` to its source path.
        """
        video_paths = discover_videos(folder_path)

        self.scenes = []
        self.scene_embs = None
        self.video_map = {}

        print(f"Processing folder: {folder_path}  ({len(video_paths)} videos)")
        for vp in video_paths:
            print(f"  ── {Path(vp).name} ──")
            scenes, embs = self._process_single_video(vp)
            video_id = Path(vp).stem

            self.scenes.extend(scenes)
            self.video_map[video_id] = vp

            if self.scene_embs is None:
                self.scene_embs = embs
            elif len(scenes) > 0:
                self.scene_embs = np.concatenate([self.scene_embs, embs], axis=0)

            print(f"  → {len(scenes)} scenes from {Path(vp).name} "
                  f"(total: {len(self.scenes)} scenes)")

        return self.scenes

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search scenes by text query.

        Returns *top_k* results sorted by descending cosine similarity.
        Each result dict::

            {
                "scene_idx": int,
                "video_id": str,
                "start_sec": float,
                "end_sec": float,
                "score": float,
                "thumbnail": np.ndarray,
                "frames": List[np.ndarray],
            }
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
                "video_id": scene.video_id,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "score": float(scores[idx]),
                "thumbnail": scene.thumbnail,
                "frames": scene.frames,
            })

        return results
