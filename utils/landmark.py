"""Hand landmark feature extraction and augmentation.

The landmark route classifies gestures from the 21 normalized MediaPipe
hand keypoints instead of pixel data. Features are translation- and
scale-invariant by construction: the wrist is the origin and the
wrist-to-middle-finger-MCP distance is the unit of length.
"""

import math
import random
from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

import numpy
import numpy.typing
import torch

LANDMARK_COUNT: Final[int] = 21
# 21 keypoints × (x, y). The z component is too noisy to be useful.
FEATURE_DIMENSIONS: Final[int] = LANDMARK_COUNT * 2
WRIST_INDEX: Final[int] = 0
SCALE_LANDMARK_INDEX: Final[int] = 9  # Middle finger MCP joint.
MIN_SCALE: Final[float] = 1e-6
MAX_AUGMENTATION_DEGREES: Final[float] = 10.0
MIN_AUGMENTATION_SCALE: Final[float] = 0.9
MAX_AUGMENTATION_SCALE: Final[float] = 1.1
# Standard MediaPipe hand skeleton topology as (start, end) landmark index
# pairs, used for drawing the detected hand on previews.
HAND_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandNotDetectedError(RuntimeError):
    """Raised when no hand is found while extracting landmark features."""


@runtime_checkable
class LandmarkPoint(Protocol):
    """Structural type for a normalized landmark with x/y coordinates."""

    x: float
    y: float


def landmarks_to_vector(
    landmarks: Sequence[LandmarkPoint],
) -> numpy.typing.NDArray[numpy.float32]:
    """Convert normalized landmarks into a flat translation/scale-invariant
    feature vector of shape ``(FEATURE_DIMENSIONS,)``.

    Coordinates are expressed relative to the wrist and divided by the
    palm scale (wrist to middle finger MCP distance).
    """
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(
            "Expected {} landmarks, got {}.".format(LANDMARK_COUNT, len(landmarks))
        )
    points: numpy.typing.NDArray[numpy.float32] = numpy.array(
        [[landmark.x, landmark.y] for landmark in landmarks], dtype=numpy.float32
    )
    points -= points[WRIST_INDEX]
    scale: float = float(numpy.linalg.norm(points[SCALE_LANDMARK_INDEX]))  # type: ignore
    if scale < MIN_SCALE:
        raise ValueError("Degenerate hand landmarks: palm scale is nearly zero.")
    points /= scale
    return points.reshape(-1)


def augment_landmark_vector(vector_tensor: torch.Tensor) -> torch.Tensor:
    """Randomly rotate and scale a ``(FEATURE_DIMENSIONS,)`` feature tensor.

    Rotation stays within a small range because gestures such as
    "Left"/"Right" are direction-sensitive, and no translation is applied
    because the wrist is already the origin.
    """
    points: torch.Tensor = vector_tensor.reshape(LANDMARK_COUNT, 2)
    angle: float = math.radians(
        random.uniform(-MAX_AUGMENTATION_DEGREES, MAX_AUGMENTATION_DEGREES)
    )
    scale: float = random.uniform(MIN_AUGMENTATION_SCALE, MAX_AUGMENTATION_SCALE)
    cos_angle: float = math.cos(angle)
    sin_angle: float = math.sin(angle)
    rotation: torch.Tensor = torch.tensor(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        dtype=points.dtype,
    )
    return ((points @ rotation.T) * scale).reshape(-1)
