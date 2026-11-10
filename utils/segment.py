from typing import Any
import cv2
import numpy
import numpy.typing
from cv2.typing import MatLike

TARGET_MEAN_INTENSITY = 128.0
MINIMUM_BRIGHTNESS_GAIN = 0.1
MAXIMUM_BRIGHTNESS_GAIN = 3.0
MAXIMUM_PIXEL_VALUE = 255
YCRCB_LOWER_BOUND = (0, 133, 77)
YCRCB_UPPER_BOUND = (255, 173, 127)
MORPHOLOGY_KERNEL_SIZE = (5, 5)
MASK_FOREGROUND_VALUE = 255


def normalize_light(image: MatLike) -> MatLike:
    image_array: numpy.typing.NDArray[Any] = numpy.asarray(image)
    channel_means = numpy.array(cv2.mean(image_array)[0:3], dtype=numpy.float32)
    brightness_gain = numpy.clip(
        TARGET_MEAN_INTENSITY / numpy.maximum(channel_means, 1.0),
        MINIMUM_BRIGHTNESS_GAIN,
        MAXIMUM_BRIGHTNESS_GAIN,
    )
    normalized_image = (
        numpy.clip(
            image_array.astype(numpy.float32) * brightness_gain,
            0,
            MAXIMUM_PIXEL_VALUE,
        )
        .round()
        .astype(numpy.uint8)
    )
    return normalized_image


def clean_mask(binary_mask: MatLike) -> MatLike:
    mask_array: numpy.typing.NDArray[Any] = numpy.asarray(binary_mask)
    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, MORPHOLOGY_KERNEL_SIZE
    )
    opened_mask: numpy.typing.NDArray[Any] = numpy.asarray(
        cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, morphology_kernel)
    )
    closed_mask: numpy.typing.NDArray[Any] = numpy.asarray(
        cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, morphology_kernel)
    )
    (
        component_count,
        raw_component_labels,
        raw_component_statistics,
        _,
    ) = cv2.connectedComponentsWithStats(closed_mask)
    component_labels: numpy.typing.NDArray[Any] = numpy.asarray(raw_component_labels)
    component_statistics: numpy.typing.NDArray[Any] = numpy.asarray(
        raw_component_statistics
    )
    if component_count > 1:
        largest_component_label = 1 + int(
            component_statistics[1:, cv2.CC_STAT_AREA].argmax()
        )
        cleaned_mask: numpy.typing.NDArray[Any] = numpy.asarray(
            (component_labels == largest_component_label) * MASK_FOREGROUND_VALUE,
            dtype=numpy.uint8,
        )
        return cleaned_mask
    return closed_mask


def segment(image: MatLike) -> MatLike:
    illumination_corrected_image = normalize_light(image)
    corrected_array: numpy.typing.NDArray[Any] = numpy.asarray(
        illumination_corrected_image
    )
    ycrcb_image = numpy.asarray(cv2.cvtColor(corrected_array, cv2.COLOR_BGR2YCrCb))
    binary_mask = cv2.inRange(ycrcb_image, YCRCB_LOWER_BOUND, YCRCB_UPPER_BOUND)
    return clean_mask(numpy.asarray(binary_mask))
