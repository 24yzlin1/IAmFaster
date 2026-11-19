"""Load a trained gesture recognition model and run inference.

Supports both feature routes transparently: mask CNN models consume the
raw image pixels, while landmark MLP models extract MediaPipe keypoints
from the input image first. The class table and metadata saved next to
the model file decide which route is used.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy
import numpy.typing
import torch
from cv2.typing import MatLike
from mediapipe.tasks.python import vision

from utils.common import IMG_SIZE, InputType, get_device, preprocess
from utils.hand_detector import HandDetector
from utils.landmark import FEATURE_DIMENSIONS, HandNotDetectedError


class GesturePrediction(NamedTuple):
    """Result of a single gesture prediction."""

    class_name: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class GesturePredictor:
    """Load a trained model from disk and classify gesture images."""

    def __init__(
        self,
        model_path: Path | str,
        device: torch.device | None = None,
    ) -> None:
        self.model_path: Path = Path(model_path)
        self.device: torch.device = device if device is not None else get_device()
        # Single-threaded inference is faster for this small network and
        # avoids oversubscribing CPU cores shared with the camera pipeline.
        torch.set_num_threads(1)
        self.classes: list[str] = self._load_classes()
        self.input_type: InputType
        self.in_channels: int
        self.img_size: int
        self.input_type, self.in_channels, self.img_size = self._load_metadata()
        self._hand_detector: HandDetector | None = (
            HandDetector(vision.RunningMode.IMAGE)
            if self.input_type == "landmark"
            else None
        )
        loaded_model: Any = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        if not isinstance(loaded_model, torch.nn.Module):
            raise TypeError(
                "Expected a torch.nn.Module in {}, got {}.".format(
                    self.model_path, type(loaded_model).__name__
                )
            )
        self.model: torch.nn.Module = loaded_model.to(self.device)
        self.model.eval()
        warmup_input: torch.Tensor
        if self.input_type == "landmark":
            warmup_input = torch.randn(1, self.in_channels)
        else:
            warmup_input = torch.randn(1, self.in_channels, self.img_size, self.img_size)
        warmup_input = warmup_input.to(self.device)
        with torch.inference_mode():
            self.model(warmup_input)

    def close(self) -> None:
        """Release the MediaPipe landmarker if this model uses one."""
        hand_detector: HandDetector | None = self._hand_detector
        if hand_detector is not None:
            hand_detector.close()
            self._hand_detector = None

    def _load_classes(self) -> list[str]:
        """Read the class table saved next to the model file."""
        classes_path: Path = self.model_path.with_suffix(".classes.json")
        if not classes_path.is_file():
            raise FileNotFoundError("Class table not found: {}".format(classes_path))
        parsed_classes: Any = json.loads(classes_path.read_text(encoding="utf-8"))
        if not isinstance(parsed_classes, list) or not parsed_classes:
            raise ValueError(
                "Class table must be a non-empty list of names: {}".format(classes_path)
            )
        return [str(class_name) for class_name in parsed_classes]  # type: ignore

    def _load_metadata(self) -> tuple[InputType, int, int]:
        """Read the preprocessing metadata saved next to the model file."""
        metadata_path: Path = self.model_path.with_suffix(".meta.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Model metadata not found: {}".format(metadata_path)
            )
        parsed_metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(parsed_metadata, dict):
            raise ValueError(
                "Model metadata must be a JSON object: {}".format(metadata_path)
            )
        raw_input_type: str = str(parsed_metadata.get("input_type", "mask"))  # type: ignore
        if raw_input_type not in ("mask", "landmark"):
            raise ValueError(
                "Unsupported input_type={} in {}; expected 'mask' or "
                "'landmark'.".format(raw_input_type, metadata_path)
            )
        input_type: InputType = raw_input_type
        in_channels: int = int(parsed_metadata.get("in_channels", 1))  # type: ignore
        image_size: int = int(parsed_metadata.get("img_size", IMG_SIZE))  # type: ignore
        if image_size != IMG_SIZE:
            raise ValueError(
                "Metadata img_size={} does not match the preprocessing size {}.".format(
                    image_size, IMG_SIZE
                )
            )
        if input_type == "landmark":
            # For landmark models in_channels stores the feature vector length.
            if in_channels != FEATURE_DIMENSIONS:
                raise ValueError(
                    "Metadata in_channels={} does not match the landmark "
                    "feature length {}.".format(in_channels, FEATURE_DIMENSIONS)
                )
        elif in_channels not in (1, 3):
            raise ValueError(
                "Unsupported in_channels={} in {}; expected 1 (mask) or 3 (RGB).".format(
                    in_channels, metadata_path
                )
            )
        return input_type, in_channels, image_size

    @torch.inference_mode()
    def predict(self, image: MatLike) -> GesturePrediction:
        """Classify one image and return the predicted class and probabilities.

        Raises HandNotDetectedError for landmark models when the input
        image contains no detectable hand.
        """
        input_tensor: torch.Tensor
        if self.input_type == "landmark":
            hand_detector: HandDetector | None = self._hand_detector
            if hand_detector is None:
                raise RuntimeError(
                    "Landmark model {} has no hand detector attached.".format(
                        self.model_path
                    )
                )
            feature_vector: numpy.typing.NDArray[numpy.float32] | None = (
                hand_detector.detect_landmark_vector(image)
            )
            if feature_vector is None:
                raise HandNotDetectedError(
                    "No hand detected in the input image."
                )
            contiguous_vector: numpy.typing.NDArray[numpy.float32] = (
                numpy.ascontiguousarray(feature_vector, dtype=numpy.float32)
            )
            input_tensor = torch.from_numpy(contiguous_vector)  # type: ignore
        else:
            input_tensor = preprocess(image, self.in_channels).to(self.device)
        batch_tensor: torch.Tensor = input_tensor.unsqueeze(0).to(self.device)
        logits: torch.Tensor = self.model(batch_tensor)
        probability_scores: torch.Tensor = torch.softmax(logits, dim=1)[0]
        prediction_index: int = int(probability_scores.argmax(dim=0).item())
        if not 0 <= prediction_index < len(self.classes):
            raise RuntimeError(
                "Predicted class index {} is out of range for {} classes.".format(
                    prediction_index, len(self.classes)
                )
            )
        confidence_value: float = float(probability_scores[prediction_index].item())
        probability_mapping: dict[str, float] = {
            class_name: float(probability_scores[class_index].item())
            for class_index, class_name in enumerate(self.classes)
        }
        return GesturePrediction(
            class_name=self.classes[prediction_index],
            class_index=prediction_index,
            confidence=confidence_value,
            probabilities=probability_mapping,
        )
