import json
from pathlib import Path
from typing import Any, NamedTuple
import torch
from cv2.typing import MatLike
from utils.common import IMG_SIZE, get_device, preprocess


class GesturePrediction(NamedTuple):
    class_name: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class GesturePredictor:
    def __init__(
        self,
        model_path: Path | str,
        device: torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device if device is not None else get_device()
        self.classes = self._load_classes()
        self.in_channels, self.img_size = self._load_metadata()
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
        warmup_input = torch.randn(
            1, self.in_channels, self.img_size, self.img_size
        ).to(self.device)
        with torch.inference_mode():
            self.model(warmup_input)

    def _load_classes(self) -> list[str]:
        classes_path = self.model_path.with_suffix(".classes.json")
        if not classes_path.is_file():
            raise FileNotFoundError("Class table not found: {}".format(classes_path))
        parsed_classes: Any = json.loads(classes_path.read_text(encoding="utf-8"))
        if not isinstance(parsed_classes, list) or not parsed_classes:
            raise ValueError(
                "Class table must be a non-empty list of names: {}".format(classes_path)
            )
        return [str(class_name) for class_name in parsed_classes]  # type: ignore

    def _load_metadata(self) -> tuple[int, int]:
        metadata_path = self.model_path.with_suffix(".meta.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Model metadata not found: {}".format(metadata_path)
            )
        parsed_metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(parsed_metadata, dict):
            raise ValueError(
                "Model metadata must be a JSON object: {}".format(metadata_path)
            )
        in_channels = int(parsed_metadata.get("in_channels", 1))  # type: ignore
        image_size = int(parsed_metadata.get("img_size", IMG_SIZE))  # type: ignore
        if in_channels not in (1, 3):
            raise ValueError(
                "Unsupported in_channels={} in {}; expected 1 (mask) or 3 (RGB).".format(
                    in_channels, metadata_path
                )
            )
        if image_size != IMG_SIZE:
            raise ValueError(
                "Metadata img_size={} does not match the preprocessing size {}.".format(
                    image_size, IMG_SIZE
                )
            )
        return in_channels, image_size

    @torch.inference_mode()
    def predict(self, image: MatLike) -> GesturePrediction:
        input_tensor = preprocess(image, self.in_channels)
        batch_tensor = input_tensor.unsqueeze(0).to(self.device)
        logits = self.model(batch_tensor)
        probability_scores = torch.softmax(logits, dim=1)[0]
        prediction_index = int(probability_scores.argmax(dim=0).item())
        if not 0 <= prediction_index < len(self.classes):
            raise RuntimeError(
                "Predicted class index {} is out of range for {} classes.".format(
                    prediction_index, len(self.classes)
                )
            )
        confidence_value = float(probability_scores[prediction_index].item())
        probability_mapping = {
            class_name: float(probability_scores[class_index].item())
            for class_index, class_name in enumerate(self.classes)
        }
        return GesturePrediction(
            class_name=self.classes[prediction_index],
            class_index=prediction_index,
            confidence=confidence_value,
            probabilities=probability_mapping,
        )
