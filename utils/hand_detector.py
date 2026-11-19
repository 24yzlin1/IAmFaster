"""MediaPipe-based hand detection and square hand cropping.

Replaces the fragile skin-color segmentation as the hand locator for both
data collection and inference. The classification pipeline still consumes
the cropped image unchanged, so training and inference stay consistent.
"""

import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import cv2
import mediapipe
import numpy
import numpy.typing
from cv2.typing import MatLike
from mediapipe.tasks.python import BaseOptions, vision

from utils.landmark import LandmarkPoint, landmarks_to_vector

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
# Fists and other closed-hand poses often defeat the landmarker when the
# hand fills the whole frame, so detection is retried once on a padded
# copy of the frame.
DETECTION_PADDING_RATIO: Final[float] = 0.5
PADDING_VALUE: Final[tuple[int, int, int]] = (114, 114, 114)


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

    def _detect_frame(self, frame: MatLike):
        """Run the landmarker on one frame and return the raw result."""
        rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB, data=rgb_array
        )
        if self._running_mode == vision.RunningMode.VIDEO:
            return self._detect_for_video(media_image)
        return self._landmarker.detect(media_image)

    def _detect_landmarks(
        self, frame: MatLike
    ) -> tuple[Sequence[LandmarkPoint], MatLike, int, int] | None:
        """Detect the first hand and return ``(landmarks, reference_frame,
        offset_x, offset_y)``.

        ``(offset_x, offset_y)`` is the position of *frame*'s top-left
        corner inside *reference_frame*; it is nonzero only when the
        padded retry succeeded. Returns None when no hand is detected.
        """
        first_hands: Sequence[Sequence[LandmarkPoint]] = (
            self._detect_frame(frame).hand_landmarks
        )
        if first_hands:
            return first_hands[0], frame, 0, 0
        padded_frame, offset_x, offset_y = self._pad_for_detection(frame)
        padded_hands: Sequence[Sequence[LandmarkPoint]] = self._detect_frame(
            padded_frame
        ).hand_landmarks
        if padded_hands:
            return padded_hands[0], padded_frame, offset_x, offset_y
        return None

    def detect_landmark_vector(self, frame: MatLike) -> numpy.typing.NDArray[numpy.float32] | None:
        """Return the normalized landmark feature vector for *frame*.

        Returns None when no hand is detected or the landmarks are
        degenerate (palm scale near zero). Closed-hand poses are given a
        second chance on a padded copy of the frame.
        """
        detected = self._detect_landmarks(frame)
        if detected is None:
            return None
        try:
            return landmarks_to_vector(detected[0])
        except ValueError:
            return None

    def detect_landmark_points(self, frame: MatLike) -> list[tuple[int, int]] | None:
        """Return the 21 landmark pixel coordinates on *frame*, or None.

        Used for drawing the hand skeleton on previews.
        """
        detected = self._detect_landmarks(frame)
        if detected is None:
            return None
        landmarks, reference_frame, offset_x, offset_y = detected
        reference_height: int
        reference_width: int
        reference_height, reference_width = reference_frame.shape[:2]
        return [
            (
                int(landmark.x * reference_width) - offset_x,
                int(landmark.y * reference_height) - offset_y,
            )
            for landmark in landmarks
        ]

    @staticmethod
    def box_from_points(
        points: Sequence[tuple[int, int]], frame_width: int, frame_height: int
    ) -> HandBox | None:
        """Return the square hand box covering landmark pixel *points*."""
        if not points:
            return None
        min_x = min(point_x for point_x, _ in points)
        max_x = max(point_x for point_x, _ in points)
        min_y = min(point_y for _, point_y in points)
        max_y = max(point_y for _, point_y in points)
        return HandDetector._square_hand_box(
            min_x / frame_width,
            max_x / frame_width,
            min_y / frame_height,
            max_y / frame_height,
            frame_width,
            frame_height,
        )

    @staticmethod
    def _pad_for_detection(frame: MatLike) -> tuple[MatLike, int, int]:
        """Center *frame* on a square gray canvas with a wide border.

        Returns ``(padded_frame, offset_x, offset_y)`` so padded-image
        coordinates can be mapped back onto the original frame. Padding
        is applied equally on both axes so the normalized landmark
        coordinates only change by a uniform affine transform, which the
        wrist-relative/palm-scale feature normalization cancels out.
        """
        frame_height: int
        frame_width: int
        frame_height, frame_width = frame.shape[:2]
        vertical_padding: int = max((frame_width - frame_height) // 2, 0)
        horizontal_padding: int = max((frame_height - frame_width) // 2, 0)
        square_frame: MatLike
        if frame_height < frame_width:
            square_frame = cv2.copyMakeBorder(
                frame,
                vertical_padding,
                frame_width - frame_height - vertical_padding,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=PADDING_VALUE,
            )
        else:
            square_frame = cv2.copyMakeBorder(
                frame,
                0,
                0,
                horizontal_padding,
                frame_height - frame_width - horizontal_padding,
                cv2.BORDER_CONSTANT,
                value=PADDING_VALUE,
            )
        side_length: int = max(frame_height, frame_width)
        border: int = int(side_length * DETECTION_PADDING_RATIO)
        padded_frame: MatLike = cv2.copyMakeBorder(
            square_frame,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=PADDING_VALUE,
        )
        offset_x: int = border + horizontal_padding
        offset_y: int = border + vertical_padding
        return padded_frame, offset_x, offset_y

    def crop_hand(self, frame: MatLike) -> MatLike | None:
        """Crop the detected hand region out of *frame*, or None if no hand."""
        frame_height: int
        frame_width: int
        frame_height, frame_width = frame.shape[:2]
        landmark_points: list[tuple[int, int]] | None = (
            self.detect_landmark_points(frame)
        )
        if landmark_points is None:
            return None
        hand_box: HandBox | None = self.box_from_points(
            landmark_points, frame_width, frame_height
        )
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
