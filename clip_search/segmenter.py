import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from scenedetect import open_video, ContentDetector, SceneManager, FrameTimecode

FRAMES_PER_SCENE = 8
MIN_SCENE_DURATION = 1.0  # seconds


@dataclass
class Scene:
    video_id: str
    scene_idx: int
    start_sec: float
    end_sec: float
    frames: List[np.ndarray] = field(default_factory=list)

    @property
    def thumbnail(self) -> np.ndarray:
        return self.frames[0] if self.frames else None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def detect_scenes(video_path: str, num_frames: int = FRAMES_PER_SCENE) -> List[Scene]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video = open_video(str(video_path))

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)

    raw_scenes = scene_manager.get_scene_list()

    # If no scene boundary detected, treat entire video as one scene
    if not raw_scenes:
        raw_scenes = [
            (FrameTimecode(0, video.frame_rate),
             FrameTimecode(int(video.duration.get_seconds() * video.frame_rate), video.frame_rate))
        ]

    merged = _merge_short_scenes(raw_scenes, float(video.frame_rate), MIN_SCENE_DURATION)

    scenes = []
    for idx, (start, end) in enumerate(merged):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        total_frames = end.frame_num - start.frame_num
        n_frames = min(num_frames, max(1, total_frames))

        # Uniformly sample frame indices within this scene
        frame_idxs = np.linspace(start.frame_num, end.frame_num - 1, n_frames, dtype=int)

        frames = _extract_frames(str(video_path), frame_idxs)

        scenes.append(Scene(
            video_id=video_path.stem,
            scene_idx=idx,
            start_sec=start_sec,
            end_sec=end_sec,
            frames=frames,
        ))

    return scenes


def _extract_frames(video_path: str, frame_idxs: np.ndarray) -> List[np.ndarray]:
    """Seek to specific frame indices and return decoded RGB frames."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    try:
        for fidx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _merge_short_scenes(
    scene_list: List,
    fps: float,
    min_duration: float,
) -> List:
    """Merge scenes shorter than min_duration into the previous scene."""
    if len(scene_list) <= 1:
        return scene_list

    merged = []
    min_frames = int(min_duration * fps)

    i = 0
    while i < len(scene_list):
        start, end = scene_list[i]
        scene_frames = end.frame_num - start.frame_num

        if scene_frames < min_frames and merged:
            # Merge into previous scene
            prev_start, prev_end = merged.pop()
            merged.append((prev_start, end))
        else:
            merged.append((start, end))
        i += 1

    return merged
