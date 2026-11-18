"""Train the gesture recognition model on a directory of images.

The dataset root must contain one subdirectory per gesture class, and
the images of each class must be stored directly inside that directory.
"""

import argparse
import json
from collections.abc import Sized
from pathlib import Path
from typing import Any, Final, Sequence, TypedDict, cast

import cv2
import torch
from cv2.typing import MatLike
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from utils.common import (
    EXTENSIONS,
    IMG_SIZE,
    Channels,
    SampleType,
    Transform,
    augment_mask,
    default_transform,
    get_device,
    preprocess,
)

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


class AugmentedMaskDataset(Dataset[SampleType]):
    """Wrap a dataset and apply random mask augmentation to each sample."""

    def __init__(self, base_dataset: Dataset[SampleType]) -> None:
        super().__init__()
        self.base_dataset: Dataset[SampleType] = base_dataset

    def __getitem__(self, index: int) -> SampleType:
        tensor, label = self.base_dataset[index]
        return (augment_mask(tensor), label)

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
        learning_rate: float = 1e-3,
        num_workers: int = 0,
        validate_split: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.device: torch.device = device
        self.model: torch.nn.Module = model.to(device)
        self.dataset: Dataset[SampleType] = dataset
        self.epoch: int = epoch
        self.classes: list[str] = list(classes) if classes is not None else []
        self.in_channels: int = in_channels
        self.history: list[EpochRecord] = []
        self.best_validation_accuracy: float = 0.0
        total_samples: int = len(dataset)  # type: ignore
        if not 0 <= validate_split < 1:
            raise ValueError("validate_split must be in [0, 1).")
        validation_samples: int = int(total_samples * validate_split)
        training_samples: int = total_samples - validation_samples
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
        if self.in_channels == 1:
            train_dataset = AugmentedMaskDataset(train_dataset)
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
        epoch_progress: tqdm[Any] = tqdm(
            total=self.epoch,
            ncols=PROGRESS_BAR_WIDTH,
            leave=True,
        )
        for epoch in range(1, self.epoch + 1):
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
            if snapshot_directory is not None and epoch % save_frequency == 0:
                snapshot_path: Path = snapshot_directory / "{}_{:02d}.pth".format(
                    save_name_prefix, epoch
                )
                self.save(snapshot_path)
                tqdm.write("Snapshot saved: {}".format(snapshot_path))
            if validation_accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = validation_accuracy
                if snapshot_directory is not None:
                    best_checkpoint_path: Path = snapshot_directory / "{}_best.pth".format(
                        save_name_prefix
                    )
                    self.save(best_checkpoint_path)
                    tqdm.write("Best checkpoint saved: {}".format(best_checkpoint_path))
        epoch_progress.close()
        tqdm.write(
            "Best validation accuracy: {:.2%}".format(self.best_validation_accuracy)
        )
        return self.history

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
        predictor side), plus the class table and the preprocessing metadata."""
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
                {"in_channels": self.in_channels, "img_size": IMG_SIZE},
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
    return parser.parse_args()


def main() -> None:
    """Train the model with the configured command-line arguments."""
    arguments: argparse.Namespace = parse_arguments()
    torch.manual_seed(42)  # type: ignore
    dataset_root: Path = arguments.dataset
    model_path: Path = arguments.model_path
    epoch_count: int = arguments.epoch_count
    channels: Channels = 1
    device: torch.device = get_device()
    dataset: GestureDataset = GestureDataset(dataset_root, channels=channels)
    print("Classes ({}): {}".format(len(dataset.classes), dataset.classes))
    print(
        "Samples: {} | channels: {} | device: {}".format(
            len(dataset), dataset.in_channels, device
        )
    )
    model: torch.nn.Module = build_model(
        num_classes=len(dataset.classes), in_channels=channels
    )
    trainer: ModelTrainer = ModelTrainer(
        model=model,
        dataset=dataset,
        device=device,
        epoch=epoch_count,
        classes=dataset.classes,
        in_channels=channels,
        batch_size=32,
        learning_rate=1e-3,
        validate_split=0.2,
    )
    trainer.train(
        save_dir=Path("./snapshot"), save_name_prefix="gesture", save_frequency=5
    )
    trainer.save(model_path)
    print("Model saved: {}".format(model_path))


if __name__ == "__main__":
    main()
