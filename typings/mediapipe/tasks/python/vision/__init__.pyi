"""Partial type stubs for the MediaPipe vision tasks used by this project."""

from enum import Enum

from mediapipe import Image
from mediapipe.tasks.python import BaseOptions


class RunningMode(Enum):
    """Execution modes for MediaPipe vision tasks."""

    IMAGE = ...
    VIDEO = ...
    LIVE_STREAM = ...


class NormalizedLandmark:
    """A landmark with coordinates normalized to the input image size."""

    x: float
    y: float
    z: float


class HandLandmarkerResult:
    """Detection output produced by the HandLandmarker task."""

    hand_landmarks: list[list[NormalizedLandmark]]


class HandLandmarkerOptions:
    """Configuration options for the HandLandmarker task."""

    def __init__(
        self,
        base_options: BaseOptions = ...,
        running_mode: RunningMode = ...,
        num_hands: int = ...,
        min_hand_detection_confidence: float = ...,
        min_hand_presence_confidence: float = ...,
        min_tracking_confidence: float = ...,
    ) -> None:
        ...


class HandLandmarker:
    """Detects hand landmarks in images or video frames."""

    @staticmethod
    def create_from_options(options: HandLandmarkerOptions) -> HandLandmarker:
        ...

    def detect(self, image: Image) -> HandLandmarkerResult:
        ...

    def detect_for_video(self, image: Image, timestamp_ms: int) -> HandLandmarkerResult:
        ...

    def close(self) -> None:
        ...
