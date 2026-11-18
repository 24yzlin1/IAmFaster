"""MediaPipe-based hand detection and square hand cropping.

Replaces the fragile skin-color segmentation as the hand locator for both
data collection and inference. The classification pipeline still consumes
the cropped image unchanged, so training and inference stay consistent.
"""

import time
import urllib.request
from pathlib import Path
from typing import Final, NamedTuple

import cv2
import mediapipe
from cv2.typing import MatLike
from mediapipe.tasks.python import BaseOptions, vision

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY: Final[Path] = PROJECT_ROOT / "assets" / "models"
MODEL_FILE_NAME: Final[str] = "hand_landmarker.task"
MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
NUM_HANDS: Final[int] = 1
MIN_HAND_DETECTION_CONFIDENCE: Final[float] = 0.5
BBOX_MARGIN_RATIO: Final[float] = 0.25


class HandBox(NamedTuple):
    """A square pixel-coordinate region containing the detected hand."""

    left_x: int
    top_y: int
    size: int


def ensure_model_file() -> Path:
    """Return the local hand-landmarker model, downloading it if missing."""
    model_path: Path = MODEL_DIRECTORY / MODEL_FILE_NAME
    if model_path.is_file():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary name first so concurrent callers never see a
    # partially written model file.
    temporary_path: Path = model_path.with_suffix(".task.download")
    urllib.request.urlretrieve(MODEL_URL, temporary_path)
    temporary_path.replace(model_path)
    return model_path


class HandDetector:
    """Locate hands in camera frames with the MediaPipe HandLandmarker."""

    def __init__(self, running_mode: vision.RunningMode) -> None:
        options: vision.HandLandmarkerOptions = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model_file())),
            running_mode=running_mode,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        )
        self._landmarker: vision.HandLandmarker = (
            vision.HandLandmarker.create_from_options(options)
        )
        self._running_mode: vision.RunningMode = running_mode
        self._last_video_timestamp_ms: int = 0

    def detect_hand_box(self, frame: MatLike) -> HandBox | None:
        """Return the square hand box for *frame*, or None if no hand."""
        frame_height: int
        frame_width: int
        frame_height, frame_width = frame.shape[:2]
        rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB, data=rgb_array
        )
        if self._running_mode == vision.RunningMode.VIDEO:
            detection_result = self._detect_for_video(media_image)
        else:
            detection_result = self._landmarker.detect(media_image)
        hand_landmarks = detection_result.hand_landmarks
        if not hand_landmarks:
            return None
        landmarks = hand_landmarks[0]
        min_x = min(landmark.x for landmark in landmarks)
        max_x = max(landmark.x for landmark in landmarks)
        min_y = min(landmark.y for landmark in landmarks)
        max_y = max(landmark.y for landmark in landmarks)
        return self._square_hand_box(
            min_x, max_x, min_y, max_y, frame_width, frame_height
        )

    def crop_hand(self, frame: MatLike) -> MatLike | None:
        """Crop the detected hand region out of *frame*, or None if no hand."""
        hand_box: HandBox | None = self.detect_hand_box(frame)
        if hand_box is None:
            return None
        return frame[
            hand_box.top_y : hand_box.top_y + hand_box.size,
            hand_box.left_x : hand_box.left_x + hand_box.size,
        ].copy()

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        self._landmarker.close()

    def _detect_for_video(self, media_image: mediapipe.Image):
        timestamp_ms: int = time.monotonic_ns() // 1_000_000
        if timestamp_ms <= self._last_video_timestamp_ms:
            timestamp_ms = self._last_video_timestamp_ms + 1
        self._last_video_timestamp_ms = timestamp_ms
        return self._landmarker.detect_for_video(media_image, timestamp_ms)

    @staticmethod
    def _square_hand_box(
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        frame_width: int,
        frame_height: int,
    ) -> HandBox:
        """Expand the landmark bounding box into a clamped square region."""
        center_x: float = (min_x + max_x) / 2 * frame_width
        center_y: float = (min_y + max_y) / 2 * frame_height
        box_width: float = (max_x - min_x) * frame_width
        box_height: float = (max_y - min_y) * frame_height
        size: float = max(box_width, box_height) * (1.0 + 2.0 * BBOX_MARGIN_RATIO)
        half_size: float = size / 2
        left_x: int = max(int(center_x - half_size), 0)
        top_y: int = max(int(center_y - half_size), 0)
        box_size: int = min(
            int(size), frame_width - left_x, frame_height - top_y
        )
        if box_size <= 0:
            return HandBox(left_x=0, top_y=0, size=0)
        return HandBox(left_x=left_x, top_y=top_y, size=box_size)
