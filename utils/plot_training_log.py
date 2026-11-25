"""Render training-curve charts from JSONL training logs.

The trainer writes one JSON object per line into
``snapshot/<prefix>_training_log.jsonl``: a ``header`` record describing the
run (input type, classes, learning rate, ...), one ``epoch`` record per epoch
and a final ``summary`` record. This module turns such a log into a PNG chart
with matplotlib, using an English monospace font chain (JetBrains Mono with
monospace fallbacks).

Command line usage::

    python -m utils.plot_training_log                     # newest snapshot log
    python -m utils.plot_training_log --log path.jsonl    # explicit log
    python -m utils.plot_training_log --output curve.png  # explicit output
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt

# matplotlib's public API is only partially typed; relax the member-level
# unknown-type rules for this module while keeping strict mode elsewhere.
# pyright: reportUnknownMemberType=false

SNAPSHOT_DIRECTORY: str = "snapshot"
LOG_SUFFIX: str = "_training_log.jsonl"
CHART_SUFFIX: str = "_curve.png"
CHART_DPI: int = 150

# Monospace font chain: JetBrains Mono first, well-known monospace fallbacks
# afterwards so the chart renders on machines without JetBrains Mono.
FONT_FAMILY: str = "monospace"
FONT_CHAIN: list[str] = ["JetBrains Mono", "Consolas", "DejaVu Sans Mono"]


@dataclass
class TrainingLogData:
    """Parsed contents of one JSONL training log."""

    input_type: str
    classes: list[str]
    learning_rate: float
    device: str
    epochs: list[dict[str, object]]
    best_epoch: int | None
    best_validation_accuracy: float | None


def _as_int(value: object) -> int:
    """Convert a JSON scalar to int, raising a clear error otherwise."""
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError("Expected a number, got: {!r}".format(value))


def _as_float(value: object) -> float:
    """Convert a JSON scalar to float, raising a clear error otherwise."""
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError("Expected a number, got: {!r}".format(value))


def _apply_font() -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.monospace"] = FONT_CHAIN


def read_training_log(log_path: Path) -> TrainingLogData:
    """Parse a JSONL training log into :class:`TrainingLogData`."""
    input_type: str = "unknown"
    classes: list[str] = []
    learning_rate: float = 0.0
    device: str = "unknown"
    epochs: list[dict[str, object]] = []
    best_epoch: int | None = None
    best_validation_accuracy: float | None = None
    with log_path.open("r", encoding="utf-8") as log_stream:
        for line in log_stream:
            entry: dict[str, object] = json.loads(line)
            entry_type: object = entry.get("type")
            if entry_type == "header":
                input_type = str(entry.get("input_type", "unknown"))
                classes_value: object = entry.get("classes", [])
                if isinstance(classes_value, list):
                    classes = [
                        str(name) for name in cast("list[object]", classes_value)
                    ]
                learning_rate = _as_float(entry.get("learning_rate", 0.0))
                device = str(entry.get("device", "unknown"))
            elif entry_type == "epoch":
                epochs.append(entry)
            elif entry_type == "summary":
                best_epoch_value = entry.get("best_epoch")
                if isinstance(best_epoch_value, int):
                    best_epoch = best_epoch_value
                accuracy_value = entry.get("best_validation_accuracy")
                if isinstance(accuracy_value, (int, float)):
                    best_validation_accuracy = float(accuracy_value)
    if not epochs:
        raise ValueError(
            "Training log {} contains no epoch records.".format(log_path)
        )
    return TrainingLogData(
        input_type=input_type,
        classes=classes,
        learning_rate=learning_rate,
        device=device,
        epochs=epochs,
        best_epoch=best_epoch,
        best_validation_accuracy=best_validation_accuracy,
    )


def render_training_log_chart(log_path: Path, output_path: Path) -> Path:
    """Render the training curves of ``log_path`` and save to ``output_path``."""
    _apply_font()
    log_data: TrainingLogData = read_training_log(log_path)
    epoch_numbers: list[int] = [
        _as_int(record["epoch"]) for record in log_data.epochs
    ]
    train_accuracy: list[float] = [
        _as_float(record["train_accuracy"]) for record in log_data.epochs
    ]
    validation_accuracy: list[float] = [
        _as_float(record["validation_accuracy"]) for record in log_data.epochs
    ]
    train_loss: list[float] = [
        _as_float(record["train_loss"]) for record in log_data.epochs
    ]
    validation_loss: list[float] = [
        _as_float(record["validation_loss"]) for record in log_data.epochs
    ]

    figure, (accuracy_axis, loss_axis) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True
    )

    accuracy_axis.plot(
        epoch_numbers, train_accuracy, marker="o", markersize=3, label="Train"
    )
    accuracy_axis.plot(
        epoch_numbers, validation_accuracy, marker="s", markersize=3, label="Validation"
    )
    if log_data.best_epoch is not None and log_data.best_epoch in epoch_numbers:
        accuracy_axis.axvline(
            log_data.best_epoch,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Best epoch ({})".format(log_data.best_epoch),
        )
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.05)
    accuracy_axis.grid(True, linestyle=":", linewidth=0.5)
    accuracy_axis.legend(loc="lower right")

    loss_axis.plot(epoch_numbers, train_loss, marker="o", markersize=3, label="Train")
    loss_axis.plot(
        epoch_numbers, validation_loss, marker="s", markersize=3, label="Validation"
    )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(True, linestyle=":", linewidth=0.5)
    loss_axis.legend(loc="upper right")

    title: str = "Training curves  |  mode: {}  |  lr: {}  |  device: {}".format(
        log_data.input_type, log_data.learning_rate, log_data.device
    )
    if log_data.classes:
        title += "  |  classes: {}".format(len(log_data.classes))
    figure.suptitle(title, fontfamily=FONT_FAMILY)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=CHART_DPI)
    plt.close(figure)
    return output_path


def latest_training_log(snapshot_directory: Path) -> Path:
    """Return the most recently modified training log under ``snapshot_directory``."""
    log_files: list[Path] = sorted(
        snapshot_directory.glob("*{}".format(LOG_SUFFIX)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        raise FileNotFoundError(
            "No training log (*{}) found under {}.".format(
                LOG_SUFFIX, snapshot_directory
            )
        )
    return log_files[0]


def main() -> None:
    """Entry point of the ``python -m utils.plot_training_log`` command."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a JSONL training log into a PNG chart of the "
            "training curves."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            "Path to the training log JSONL file. Defaults to the newest "
            "*{} under {}.".format(LOG_SUFFIX, SNAPSHOT_DIRECTORY)
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to '<log stem>{}' next to the log."
            .format(CHART_SUFFIX)
        ),
    )
    arguments = parser.parse_args()
    log_argument: object = arguments.log
    log_path: Path
    if isinstance(log_argument, Path):
        log_path = log_argument
    else:
        log_path = latest_training_log(Path(SNAPSHOT_DIRECTORY))
    if not log_path.is_file():
        parser.error("Training log not found: {}".format(log_path))
    output_argument: object = arguments.output
    output_path: Path
    if isinstance(output_argument, Path):
        output_path = output_argument
    else:
        stem: str = log_path.stem
        if stem.endswith("_training_log"):
            stem = stem[: -len("_training_log")] + "_curve"
        output_path = log_path.with_name("{}.png".format(stem))
    saved_path: Path = render_training_log_chart(log_path, output_path)
    print("Training curve chart saved: {}".format(saved_path))


if __name__ == "__main__":
    main()
