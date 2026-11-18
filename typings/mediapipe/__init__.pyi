"""Partial type stubs for the parts of mediapipe used by this project.

Only the top-level image API is declared here; vision-task APIs live in
``mediapipe.tasks.python``.
"""

from enum import Enum
from typing import Any


class ImageFormat(Enum):
    """Pixel formats accepted by mediapipe.Image."""

    SRGB = ...
    SRGBA = ...


class Image:
    """An image wrapper accepted by MediaPipe vision task APIs."""

    def __init__(
        self,
        image_format: ImageFormat = ...,
        data: Any = ...,
    ) -> None:
        ...
