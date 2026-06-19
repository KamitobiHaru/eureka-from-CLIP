from dataclasses import dataclass, field
from typing import List

import numpy as np

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
