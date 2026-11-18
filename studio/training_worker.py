"""Background worker that trains the gesture model without blocking the UI."""

from pathlib import Path

import torch
from PySide6.QtCore import QObject, QThread, Signal

from utils.common import Channels, get_device
from utils.trainer import GestureDataset, ModelTrainer, build_model

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_VALIDATE_SPLIT: float = 0.2
SNAPSHOT_DIRECTORY_NAME: str = "snapshot"
SNAPSHOT_NAME_PREFIX: str = "gesture"
SNAPSHOT_SAVE_FREQUENCY: int = 5


class TrainingWorker(QThread):
    """Runs a complete training session on a background thread.

    The per-epoch metrics are appended to ``trainer.history``; the owning
    tab polls that list instead of receiving a signal per epoch, because
    the trainer logic stays untouched.
    """

    dataset_loaded = Signal(str, int, str)
    training_finished = Signal(str)
    training_failed = Signal(str)

    def __init__(
        self,
        dataset_root: Path,
        model_path: Path,
        epoch_count: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_root: Path = dataset_root
        self._model_path: Path = model_path
        self._epoch_count: int = epoch_count
        self.trainer: ModelTrainer | None = None

    def run(self) -> None:
        try:
            torch.manual_seed(42)  # type: ignore
            channels: Channels = 1
            device: torch.device = get_device()
            dataset: GestureDataset = GestureDataset(
                self._dataset_root, channels=channels
            )
            model: torch.nn.Module = build_model(
                num_classes=len(dataset.classes), in_channels=channels
            )
            trainer: ModelTrainer = ModelTrainer(
                model=model,
                dataset=dataset,
                device=device,
                epoch=self._epoch_count,
                classes=dataset.classes,
                in_channels=channels,
                batch_size=DEFAULT_BATCH_SIZE,
                learning_rate=DEFAULT_LEARNING_RATE,
                validate_split=DEFAULT_VALIDATE_SPLIT,
            )
            self.trainer = trainer
            self.dataset_loaded.emit(
                ", ".join(dataset.classes), len(dataset.samples), str(device)
            )
            trainer.train(
                save_dir=self._model_path.parent / SNAPSHOT_DIRECTORY_NAME,
                save_name_prefix=SNAPSHOT_NAME_PREFIX,
                save_frequency=SNAPSHOT_SAVE_FREQUENCY,
            )
            trainer.save(self._model_path)
            self.training_finished.emit(
                "Training finished. Best validation accuracy: {:.2%}. "
                "Model saved: {}".format(
                    trainer.best_validation_accuracy, self._model_path
                )
            )
        except Exception as error:
            self.training_failed.emit(
                "{}: {}".format(type(error).__name__, error)
            )
