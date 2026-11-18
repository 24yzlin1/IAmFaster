"""Partial type stubs for mediapipe.tasks.python."""

from . import vision


class BaseOptions:
    """Options shared by all MediaPipe task initializers."""

    def __init__(
        self,
        model_asset_path: str = ...,
        model_asset_buffer: bytes = ...,
    ) -> None:
        ...
