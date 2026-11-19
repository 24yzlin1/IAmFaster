"""Train the gesture recognition model on a directory of images.

The dataset root must contain one subdirectory per gesture class, and
the images of each class must be stored directly inside that directory.
"""

import argparse
import json
import time
from collections.abc import Callable, Sized
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Sequence, TypedDict, cast

import cv2
import numpy
import numpy.typing
import torch
from cv2.typing import MatLike
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from utils.common import (
    EXTENSIONS,
    IMG_SIZE,
    Channels,
    InputType,
    SampleType,
    Transform,
    augment_mask,
    default_transform,
    get_device,
    preprocess,
)
from utils.landmark import FEATURE_DIMENSIONS, augment_landmark_vector

PROGRESS_BAR_WIDTH: Final[int] = 50


class EpochRecord(TypedDict):
    """Metrics recorded for a single training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


class GestureDataset(Dataset[SampleType]):
    """Image dataset that maps subdirectories under *root* to class labels."""

    def __init__(
        self,
        root: Path | str,
        channels: Channels = 1,
        transform: Transform = default_transform,
    ) -> None:
        super().__init__()
        self.root: Path = Path(root)
        self.channels: Channels = channels
        self.transform: Transform = transform
        self.classes: list[str] = []
        self.class_to_index: dict[str, int] = {}
        self.samples: list[tuple[Path, int]] = []
        self._load_data()
        if not self.samples:
            raise RuntimeError(
                "No images found under {}. Expected layout: root/class_name/*.jpg".format(
                    self.root
                )
            )

    def _load_data(self) -> None:
        self.classes = sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        self.class_to_index = {
            class_name: label for label, class_name in enumerate(self.classes)
        }
        for class_name, label in self.class_to_index.items():
            class_directory: Path = self.root / class_name
            for file_path in class_directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in EXTENSIONS:
                    self.samples.append((file_path, label))

    @property
    def in_channels(self) -> int:
        """Return the number of channels of the produced tensors."""
        return self.channels

    def __getitem__(self, index: int) -> SampleType:
        image_path, label = self.samples[index]
        bgr_image: MatLike | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise RuntimeError("Cannot read image: {}".format(image_path))
        tensor: torch.Tensor = preprocess(bgr_image, self.channels, self.transform)
        return (tensor, torch.tensor(label, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)


class LandmarkDataset(Dataset[SampleType]):
    """Feature dataset that maps subdirectories under *root* to class labels.

    Each sample is a ``.npy`` file holding one flat normalized landmark
    vector produced by ``utils.convert_dataset``.
    """

    def __init__(self, root: Path | str) -> None:
        super().__init__()
        self.root: Path = Path(root)
        self.classes: list[str] = []
        self.class_to_index: dict[str, int] = {}
        self.samples: list[tuple[Path, int]] = []
        self._load_data()
        if not self.samples:
            raise RuntimeError(
                "No landmark feature files found under {}. Expected layout: "
                "root/class_name/*.npy (run 'python -m utils.convert_dataset' "
                "to convert an image dataset).".format(self.root)
            )

    def _load_data(self) -> None:
        self.classes = sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        self.class_to_index = {
            class_name: label for label, class_name in enumerate(self.classes)
        }
        for class_name, label in self.class_to_index.items():
            class_directory: Path = self.root / class_name
            for file_path in class_directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".npy":
                    self.samples.append((file_path, label))

    @property
    def feature_dimensions(self) -> int:
        """Return the number of elements of the produced feature vectors."""
        return FEATURE_DIMENSIONS

    def __getitem__(self, index: int) -> SampleType:
        file_path, label = self.samples[index]
        vector: numpy.typing.NDArray[Any] = numpy.load(file_path)
        if vector.shape != (FEATURE_DIMENSIONS,):
            raise RuntimeError(
                "Feature file {} has shape {}, expected ({},).".format(
                    file_path, vector.shape, FEATURE_DIMENSIONS
                )
            )
        contiguous_vector: numpy.typing.NDArray[Any] = numpy.ascontiguousarray(
            vector, dtype=numpy.float32
        )
        tensor: torch.Tensor = torch.from_numpy(contiguous_vector)  # type: ignore
        return (tensor, torch.tensor(label, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)


def build_model(num_classes: int, in_channels: int = 1) -> torch.nn.Module:
    """Build the convolutional network used for gesture classification.

    Feature maps go 128 -> pool -> 64 -> pool -> 32 -> pool -> 16, so the
    flattened feature vector has 64 * 16 * 16 elements. The first 5x5
    kernel captures coarse hand shapes, and dropout curbs overfitting.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 16 * 16, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, num_classes),
    )


def build_landmark_model(num_classes: int) -> torch.nn.Module:
    """Build the MLP classifier over normalized landmark feature vectors.

    Landmarks are already translation- and scale-invariant semantic
    features, so a small fully connected network is sufficient and
    trains in seconds even on CPU.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIMENSIONS, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )


class AugmentedDataset(Dataset[SampleType]):
    """Wrap a dataset and apply an augmentation function to each sample."""

    def __init__(
        self,
        base_dataset: Dataset[SampleType],
        augment: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.base_dataset: Dataset[SampleType] = base_dataset
        self.augment: Callable[[torch.Tensor], torch.Tensor] = augment

    def __getitem__(self, index: int) -> SampleType:
        tensor, label = self.base_dataset[index]
        return (self.augment(tensor), label)

    def __len__(self) -> int:
        # torch's Dataset protocol does not declare __len__, but every
        # concrete dataset built by GestureDataset provides one.
        return len(cast(Sized, self.base_dataset))


class ModelTrainer:
    """Train a model on a dataset and record per-epoch metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset[SampleType],
        device: torch.device,
        epoch: int,
        *,
        classes: Sequence[str] | None = None,
        in_channels: int = 1,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_workers: int = 0,
        validate_split: float = 0.2,
        seed: int = 42,
        input_type: InputType = "mask",
    ) -> None:
        self.device: torch.device = device
        self.model: torch.nn.Module = model.to(device)
        self.dataset: Dataset[SampleType] = dataset
        self.epoch: int = epoch
        self.classes: list[str] = list(classes) if classes is not None else []
        self.in_channels: int = in_channels
        self.input_type: InputType = input_type
        self.history: list[EpochRecord] = []
        self.best_validation_accuracy: float = 0.0
        self.best_epoch: int = 0
        self._log_file: Path | None = None
        total_samples: int = len(dataset)  # type: ignore
        if not 0 <= validate_split < 1:
            raise ValueError("validate_split must be in [0, 1).")
        validation_samples: int = int(total_samples * validate_split)
        training_samples: int = total_samples - validation_samples
        self.training_sample_count: int = training_samples
        self.validation_sample_count: int = validation_samples
        if training_samples == 0 or validation_samples == 0:
            raise ValueError(
                "Too few samples ({}) to split by {:.0%} into train/validation sets.".format(
                    total_samples, validate_split
                )
            )
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        train_dataset, validation_dataset = random_split(
            dataset, [training_samples, validation_samples], generator=generator
        )
        if self.input_type == "landmark":
            train_dataset = AugmentedDataset(train_dataset, augment_landmark_vector)
        elif self.in_channels == 1:
            train_dataset = AugmentedDataset(train_dataset, augment_mask)
        pin_memory: bool = device.type == "cuda"
        self.train_loader: DataLoader[SampleType] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.validate_loader: DataLoader[SampleType] = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.criterion: torch.nn.Module = torch.nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

    def train(
        self,
        save_dir: Path | None = None,
        save_name_prefix: str = "model",
        save_frequency: int = 5,
    ) -> list[EpochRecord]:
        """Run the training loop and return the per-epoch history."""
        if save_frequency <= 0:
            raise ValueError("save_frequency must be a positive integer.")
        snapshot_directory: Path | None = (
            Path(save_dir) if save_dir is not None else None
        )
        if snapshot_directory is not None:
            snapshot_directory.mkdir(parents=True, exist_ok=True)
            self._log_file = snapshot_directory / "{}_training_log.jsonl".format(
                save_name_prefix
            )
            self._write_log_entry(
                {
                    "type": "header",
                    "input_type": self.input_type,
                    "in_channels": self.in_channels,
                    "classes": self.classes,
                    "training_samples": self.training_sample_count,
                    "validation_samples": self.validation_sample_count,
                    "epoch_count": self.epoch,
                    "batch_size": self.train_loader.batch_size,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "device": str(self.device),
                }
            )
        run_start_time: float = time.time()
        epoch_progress: tqdm[Any] = tqdm(
            total=self.epoch,
            ncols=PROGRESS_BAR_WIDTH,
            leave=True,
        )
        for epoch in range(1, self.epoch + 1):
            epoch_start_time: float = time.time()
            train_loss, train_accuracy = self._train_epoch(epoch)
            validation_loss, validation_accuracy = self._validate()
            epoch_progress.set_postfix(  # type: ignore
                accuracy="{:.2%}".format(validation_accuracy),
            )
            epoch_progress.update(1)
            epoch_record: EpochRecord = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            self.history.append(epoch_record)
            if self._log_file is not None:
                self._write_log_entry(
                    {
                        "type": "epoch",
                        "epoch": epoch,
                        "train_loss": round(train_loss, 6),
                        "train_accuracy": round(train_accuracy, 6),
                        "validation_loss": round(validation_loss, 6),
                        "validation_accuracy": round(validation_accuracy, 6),
                        "elapsed_seconds": round(time.time() - epoch_start_time, 1),
                    }
                )
            if snapshot_directory is not None and epoch % save_frequency == 0:
                snapshot_path: Path = snapshot_directory / "{}_{:02d}.pth".format(
                    save_name_prefix, epoch
                )
                self.save(snapshot_path)
                tqdm.write("Snapshot saved: {}".format(snapshot_path))
            if validation_accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = validation_accuracy
                self.best_epoch = epoch
                if snapshot_directory is not None:
                    best_checkpoint_path: Path = (
                        snapshot_directory / "{}_best.pth".format(save_name_prefix)
                    )
                    self.save(best_checkpoint_path)
                    tqdm.write("Best checkpoint saved: {}".format(best_checkpoint_path))
        epoch_progress.close()
        tqdm.write(
            "Best validation accuracy: {:.2%}".format(self.best_validation_accuracy)
        )
        if self._log_file is not None:
            self._write_log_entry(
                {
                    "type": "summary",
                    "best_epoch": self.best_epoch,
                    "best_validation_accuracy": round(
                        self.best_validation_accuracy, 6
                    ),
                    "final_validation_accuracy": round(
                        self.history[-1]["validation_accuracy"], 6
                    ),
                    "total_seconds": round(time.time() - run_start_time, 1),
                }
            )
        return self.history

    def _write_log_entry(self, entry: dict[str, object]) -> None:
        """Append one JSON line to the training log in the snapshot directory."""
        log_file: Path | None = self._log_file
        if log_file is None:
            return
        entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
        with log_file.open("a", encoding="utf-8") as log_stream:
            log_stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss: float = 0.0
        correct_predictions: int = 0
        total_predictions: int = 0
        batch_progress: tqdm[Any] = tqdm(
            self.train_loader,
            ncols=PROGRESS_BAR_WIDTH,
            leave=False,
        )
        for images, labels in batch_progress:
            images: torch.Tensor = images.to(self.device, non_blocking=True)
            labels: torch.Tensor = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits: torch.Tensor = self.model(images)
            loss: torch.Tensor = self.criterion(logits, labels)
            loss.backward()  # type: ignore
            self.optimizer.step()
            current_loss_value: float = loss.item()
            running_loss += current_loss_value * labels.size(0)
            correct_predictions += int((logits.argmax(dim=1) == labels).sum().item())
            total_predictions += labels.size(0)
            batch_progress.set_postfix(loss="{:.4f}".format(current_loss_value))  # type: ignore
        batch_progress.close()
        return (
            running_loss / max(total_predictions, 1),
            correct_predictions / max(total_predictions, 1),
        )

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss: float = 0.0
        correct_predictions: int = 0
        total_predictions: int = 0
        batch_progress: tqdm[Any] = tqdm(
            self.validate_loader,
            ncols=PROGRESS_BAR_WIDTH,
            leave=False,
        )
        for images, labels in batch_progress:
            images: torch.Tensor = images.to(self.device, non_blocking=True)
            labels: torch.Tensor = labels.to(self.device, non_blocking=True)
            logits: torch.Tensor = self.model(images)
            loss: torch.Tensor = self.criterion(logits, labels)
            current_loss_value: float = loss.item()
            running_loss += current_loss_value * labels.size(0)
            correct_predictions += int((logits.argmax(dim=1) == labels).sum().item())
            total_predictions += labels.size(0)
            batch_progress.set_postfix(loss="{:.4f}".format(current_loss_value))  # type: ignore
        batch_progress.close()
        return (
            running_loss / max(total_predictions, 1),
            correct_predictions / max(total_predictions, 1),
        )

    def save(self, path: Path | str, as_state_dict: bool = False) -> Path:
        """Save a full torch.nn.Module (torch.load(..., weights_only=False) on the
        predictor side), plus the class table and the preprocessing metadata.

        For landmark models the ``in_channels`` field stores the feature
        vector length instead of an image channel count.
        """
        model_path: Path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict() if as_state_dict else self.model, model_path)
        classes_path: Path = model_path.with_suffix(".classes.json")
        classes_path.write_text(
            json.dumps(self.classes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata_path: Path = model_path.with_suffix(".meta.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "input_type": self.input_type,
                    "in_channels": self.in_channels,
                    "img_size": IMG_SIZE,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tqdm.write(
            "Saved class table: {} | metadata: {}".format(classes_path, metadata_path)
        )
        return model_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train the gesture recognition model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("./dataset"),
        help="Dataset root containing one subdirectory per gesture class.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./gesture_model.pth"),
        help="Output path for the trained model.",
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--input-type",
        choices=("mask", "landmark"),
        default="mask",
        help="Feature route: 'mask' trains the skin-mask CNN, 'landmark' "
        "trains the landmark MLP on .npy feature files.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the model with the configured command-line arguments."""
    arguments: argparse.Namespace = parse_arguments()
    torch.manual_seed(42)  # type: ignore
    dataset_root: Path = arguments.dataset
    model_path: Path = arguments.model_path
    epoch_count: int = arguments.epoch_count
    input_type: InputType = arguments.input_type
    device: torch.device = get_device()
    dataset: GestureDataset | LandmarkDataset
    model: torch.nn.Module
    in_channels: int
    if input_type == "landmark":
        dataset = LandmarkDataset(dataset_root)
        model = build_landmark_model(num_classes=len(dataset.classes))
        in_channels = FEATURE_DIMENSIONS
    else:
        channels: Channels = 1
        dataset = GestureDataset(dataset_root, channels=channels)
        model = build_model(num_classes=len(dataset.classes), in_channels=channels)
        in_channels = channels
    print("Classes ({}): {}".format(len(dataset.classes), dataset.classes))
    print(
        "Samples: {} | input type: {} | device: {}".format(
            len(dataset), input_type, device
        )
    )
    trainer: ModelTrainer = ModelTrainer(
        model=model,
        dataset=dataset,
        device=device,
        epoch=epoch_count,
        classes=dataset.classes,
        in_channels=in_channels,
        batch_size=32,
        learning_rate=1e-3,
        validate_split=0.2,
        input_type=input_type,
    )
    trainer.train(
        save_dir=Path("./snapshot"), save_name_prefix="gesture", save_frequency=5
    )
    trainer.save(model_path)
    print("Model saved: {}".format(model_path))


if __name__ == "__main__":
    main()
