from typing import Any, Callable, Literal
import cv2
import numpy
import numpy.typing
import torch
from cv2.typing import MatLike
from PIL import Image
from torchvision.transforms import transforms
from utils.segment import segment

IMG_SIZE = 128
EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
MASK_SCALE = 255.0
Channels = Literal[1, 3]
SampleType = tuple[torch.Tensor, torch.Tensor]
Transform = Callable[..., torch.Tensor]
default_transform: Transform = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_mask(bgr_image: MatLike) -> torch.Tensor:
    image_array: numpy.typing.NDArray[Any] = numpy.asarray(bgr_image)
    segmentation_mask = numpy.asarray(segment(image=image_array))
    resized_mask = numpy.asarray(
        cv2.resize(
            segmentation_mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
    )
    normalized_mask = numpy.asarray(resized_mask, dtype=numpy.float32) / MASK_SCALE
    if normalized_mask.ndim == 3:
        normalized_mask = normalized_mask[..., 0]
    contiguous_mask = numpy.ascontiguousarray(normalized_mask, dtype=numpy.float32)
    return torch.from_numpy(contiguous_mask).unsqueeze(0)  # type: ignore


def preprocess_rgb(
    bgr_image: MatLike,
    rgb_transform: Transform = default_transform,
) -> torch.Tensor:
    image_array: numpy.typing.NDArray[Any] = numpy.asarray(bgr_image)
    rgb_array = numpy.asarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    pil_image = Image.fromarray(rgb_array)
    return rgb_transform(pil_image)


def preprocess(
    bgr_image: MatLike,
    channels: int,
    rgb_transform: Transform = default_transform,
) -> torch.Tensor:
    if channels == 1:
        return preprocess_mask(bgr_image)
    if channels == 3:
        return preprocess_rgb(bgr_image, rgb_transform)
    raise ValueError(
        "Unsupported channel count {}; expected 1 (mask) or 3 (RGB).".format(channels)
    )
