"""Shared preprocessing helpers for training and inference."""

import random
from typing import Any, Callable, Final, Literal

import cv2
import numpy
import numpy.typing
import torch
from cv2.typing import MatLike
from PIL import Image
from torchvision.transforms import InterpolationMode, functional, transforms

from utils.segment import segment

IMG_SIZE: Final[int] = 128
EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
)
MASK_SCALE: Final[float] = 255.0
MASK_MARGIN_RATIO: Final[float] = 0.1
MAX_AUGMENTATION_DEGREES: Final[float] = 15.0
MIN_AUGMENTATION_SCALE: Final[float] = 0.85
MAX_AUGMENTATION_SCALE: Final[float] = 1.15
MAX_AUGMENTATION_TRANSLATION_RATIO: Final[float] = 0.1
Channels = Literal[1, 3]
SampleType = tuple[torch.Tensor, torch.Tensor]
Transform = Callable[..., torch.Tensor]
default_transform: Final[Transform] = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)


def center_hand_mask(segmentation_mask: numpy.typing.NDArray[Any]) -> numpy.typing.NDArray[Any]:
    """Crop the foreground bounding box and center it on a square canvas.

    Normalizing position and scale this way makes training and inference
    robust to where the hand appears inside the captured frame region.
    """
    if not numpy.any(segmentation_mask):
        # An all-zero mask has no foreground, so centering is meaningless.
        return segmentation_mask
    foreground_points: MatLike = cv2.findNonZero(segmentation_mask)
    bounding_x: int
    bounding_y: int
    bounding_width: int
    bounding_height: int
    bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(
        foreground_points
    )
    cropped_mask: numpy.typing.NDArray[Any] = numpy.asarray(
        segmentation_mask[
            bounding_y : bounding_y + bounding_height,
            bounding_x : bounding_x + bounding_width,
        ]
    )
    side_length: int = max(bounding_width, bounding_height)
    margin: int = int(side_length * MASK_MARGIN_RATIO)
    canvas_size: int = side_length + 2 * margin
    centered_mask: numpy.typing.NDArray[Any] = numpy.zeros(
        (canvas_size, canvas_size), dtype=segmentation_mask.dtype
    )
    offset_y: int = (canvas_size - bounding_height) // 2
    offset_x: int = (canvas_size - bounding_width) // 2
    centered_mask[
        offset_y : offset_y + bounding_height, offset_x : offset_x + bounding_width
    ] = cropped_mask
    return centered_mask


def augment_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    """Randomly rotate, scale, and translate a (1, H, W) mask tensor.

    Horizontal flip is deliberately not applied because gestures such as
    "Left"/"Right" are direction-sensitive.
    """
    height: int = int(mask_tensor.shape[-2])
    width: int = int(mask_tensor.shape[-1])
    angle: float = random.uniform(
        -MAX_AUGMENTATION_DEGREES, MAX_AUGMENTATION_DEGREES
    )
    scale: float = random.uniform(MIN_AUGMENTATION_SCALE, MAX_AUGMENTATION_SCALE)
    translate_x: float = (
        random.uniform(
            -MAX_AUGMENTATION_TRANSLATION_RATIO, MAX_AUGMENTATION_TRANSLATION_RATIO
        )
        * width
    )
    translate_y: float = (
        random.uniform(
            -MAX_AUGMENTATION_TRANSLATION_RATIO, MAX_AUGMENTATION_TRANSLATION_RATIO
        )
        * height
    )
    return functional.affine(
        mask_tensor,
        angle=angle,
        translate=[translate_x, translate_y],
        scale=scale,
        shear=0.0,
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    )


def get_device() -> torch.device:
    """Return the best available torch device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend: Any = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_mask(bgr_image: MatLike) -> torch.Tensor:
    """Convert a BGR image into a single-channel normalized mask tensor."""
    image_array: numpy.typing.NDArray[Any] = numpy.asarray(bgr_image)
    segmentation_mask: numpy.typing.NDArray[Any] = numpy.asarray(
        segment(image=image_array)
    )
    centered_mask: numpy.typing.NDArray[Any] = center_hand_mask(segmentation_mask)
    resized_mask: numpy.typing.NDArray[Any] = numpy.asarray(
        cv2.resize(
            centered_mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
    )
    normalized_mask: numpy.typing.NDArray[Any] = numpy.asarray(
        resized_mask, dtype=numpy.float32
    ) / MASK_SCALE
    if normalized_mask.ndim == 3:
        normalized_mask = normalized_mask[..., 0]
    contiguous_mask: numpy.typing.NDArray[Any] = numpy.ascontiguousarray(
        normalized_mask, dtype=numpy.float32
    )
    return torch.from_numpy(contiguous_mask).unsqueeze(0)  # type: ignore


def preprocess_rgb(
    bgr_image: MatLike,
    rgb_transform: Transform = default_transform,
) -> torch.Tensor:
    """Convert a BGR image into a normalized RGB tensor via *rgb_transform*."""
    image_array: numpy.typing.NDArray[Any] = numpy.asarray(bgr_image)
    rgb_array: numpy.typing.NDArray[Any] = numpy.asarray(
        cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    )
    pil_image: Image.Image = Image.fromarray(rgb_array)
    return rgb_transform(pil_image)


def preprocess(
    bgr_image: MatLike,
    channels: int,
    rgb_transform: Transform = default_transform,
) -> torch.Tensor:
    """Dispatch a BGR image to mask or RGB preprocessing by channel count."""
    if channels == 1:
        return preprocess_mask(bgr_image)
    if channels == 3:
        return preprocess_rgb(bgr_image, rgb_transform)
    raise ValueError(
        "Unsupported channel count {}; expected 1 (mask) or 3 (RGB).".format(channels)
    )
