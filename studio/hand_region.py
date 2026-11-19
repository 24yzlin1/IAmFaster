"""Square region drawing helpers for camera previews."""

from collections.abc import Sequence

import cv2
from cv2.typing import MatLike

from utils.landmark import HAND_CONNECTIONS

RECTANGLE_COLOR: tuple[int, int, int] = (0, 255, 0)
RECTANGLE_THICKNESS: int = 10
LANDMARK_CONNECTION_COLOR: tuple[int, int, int] = (0, 200, 255)
LANDMARK_CONNECTION_THICKNESS: int = 2
LANDMARK_POINT_COLOR: tuple[int, int, int] = (255, 0, 255)
LANDMARK_POINT_RADIUS: int = 4


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


def draw_landmark_points(
    frame: MatLike, points: Sequence[tuple[int, int]]
) -> MatLike:
    """Draw the hand skeleton (connections then points) onto *frame*."""
    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            LANDMARK_CONNECTION_COLOR,
            LANDMARK_CONNECTION_THICKNESS,
        )
    for point in points:
        cv2.circle(frame, point, LANDMARK_POINT_RADIUS, LANDMARK_POINT_COLOR, -1)
    return frame
