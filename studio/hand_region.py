"""Square region drawing helpers for camera previews."""

import cv2
from cv2.typing import MatLike

RECTANGLE_COLOR: tuple[int, int, int] = (0, 255, 0)
RECTANGLE_THICKNESS: int = 10


def draw_square_region(
    frame: MatLike, left_x: int, top_y: int, size: int
) -> MatLike:
    """Draw a square region outline of *size* onto *frame* and return it."""
    cv2.rectangle(
        frame,
        (left_x, top_y),
        (left_x + size - 1, top_y + size - 1),
        RECTANGLE_COLOR,
        RECTANGLE_THICKNESS,
    )
    return frame
