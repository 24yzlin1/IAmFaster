"""Training tab: configure and run gesture model training from the GUI."""

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studio.training_worker import TrainingWorker
from utils.trainer import EpochRecord

DEFAULT_DATASET_ROOT: str = "dataset"
DEFAULT_MODEL_PATH: str = "gesture_model/gesture_model.pth"
DEFAULT_EPOCH_COUNT: int = 20
POLL_INTERVAL_MILLISECONDS: int = 200
TABLE_COLUMN_COUNT: int = 5
TABLE_COLUMN_TITLES: tuple[str, ...] = (
    "Epoch",
    "Train Loss",
    "Train Accuracy",
    "Validation Loss",
    "Validation Accuracy",
)


class TrainingTab(QWidget):
    """Widget that launches the trainer in a background thread."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: TrainingWorker | None = None
        self._displayed_epoch_count: int = 0

        self._poll_timer: QTimer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MILLISECONDS)
        self._poll_timer.timeout.connect(self._poll_training_progress)

        self._build_ui()

    def _build_ui(self) -> None:
        settings_group: QGroupBox = QGroupBox("Settings")
        form_layout: QFormLayout = QFormLayout(settings_group)

        self.dataset_root_edit: QLabel = QLabel(DEFAULT_DATASET_ROOT)
        dataset_browse_button: QPushButton = QPushButton("Browse...")
        dataset_browse_button.clicked.connect(self._browse_dataset_root)
        dataset_row: QHBoxLayout = QHBoxLayout()
        dataset_row.addWidget(self.dataset_root_edit, stretch=1)
        dataset_row.addWidget(dataset_browse_button)
        form_layout.addRow("Dataset root:", dataset_row)

        self.model_path_edit: QLabel = QLabel(DEFAULT_MODEL_PATH)
        model_browse_button: QPushButton = QPushButton("Browse...")
        model_browse_button.clicked.connect(self._browse_model_path)
        model_row: QHBoxLayout = QHBoxLayout()
        model_row.addWidget(self.model_path_edit, stretch=1)
        model_row.addWidget(model_browse_button)
        form_layout.addRow("Model output:", model_row)

        self.epoch_spinbox: QSpinBox = QSpinBox()
        self.epoch_spinbox.setRange(1, 9999)
        self.epoch_spinbox.setValue(DEFAULT_EPOCH_COUNT)
        form_layout.addRow("Epoch count:", self.epoch_spinbox)

        self.start_button: QPushButton = QPushButton("Start Training")
        self.start_button.clicked.connect(self._start_training)

        self.status_label: QLabel = QLabel("Idle.")
        self.status_label.setWordWrap(True)

        self.history_table: QTableWidget = QTableWidget(0, TABLE_COLUMN_COUNT)
        self.history_table.setHorizontalHeaderLabels(list(TABLE_COLUMN_TITLES))
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        tab_layout: QHBoxLayout = QHBoxLayout(self)
        tab_layout.addWidget(self.history_table, stretch=2)

        sidebar_widget: QWidget = QWidget()
        sidebar_layout: QVBoxLayout = QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(settings_group)
        sidebar_layout.addWidget(self.start_button)
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addStretch()
        tab_layout.addWidget(sidebar_widget, stretch=1)

    def _browse_dataset_root(self) -> None:
        chosen_directory: str = QFileDialog.getExistingDirectory(
            self, "Select the dataset root directory", self.dataset_root_edit.text()
        )
        if chosen_directory:
            self.dataset_root_edit.setText(chosen_directory)

    def _browse_model_path(self) -> None:
        chosen_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select the model output path",
            self.model_path_edit.text(),
            "Model files (*.pth)",
        )
        if chosen_path:
            self.model_path_edit.setText(chosen_path)

    def _start_training(self) -> None:
        if self._worker is not None:
            return
        dataset_root: Path = Path(self.dataset_root_edit.text())
        model_path: Path = Path(self.model_path_edit.text())
        if not dataset_root.is_dir():
            QMessageBox.warning(
                self,
                "Invalid dataset root",
                "The dataset root directory does not exist: {}".format(dataset_root),
            )
            return
        self._displayed_epoch_count = 0
        self.history_table.setRowCount(0)
        worker: TrainingWorker = TrainingWorker(
            dataset_root, model_path, self.epoch_spinbox.value(), parent=self
        )
        worker.dataset_loaded.connect(self._on_dataset_loaded)
        worker.training_finished.connect(self._on_training_finished)
        worker.training_failed.connect(self._on_training_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._set_controls_enabled(False)
        self.status_label.setText("Loading the dataset...")
        worker.start()
        self._poll_timer.start()

    def _on_dataset_loaded(
        self, class_summary: str, sample_count: int, device_name: str
    ) -> None:
        self.status_label.setText(
            "Training on device {} | classes: {} | samples: {}".format(
                device_name, class_summary, sample_count
            )
        )

    def _poll_training_progress(self) -> None:
        worker: TrainingWorker | None = self._worker
        if worker is None or worker.trainer is None:
            return
        history: list[EpochRecord] = worker.trainer.history
        while self._displayed_epoch_count < len(history):
            self._append_epoch_record(history[self._displayed_epoch_count])
            self._displayed_epoch_count += 1

    def _append_epoch_record(self, epoch_record: EpochRecord) -> None:
        table: QTableWidget = self.history_table
        row_index: int = table.rowCount()
        table.insertRow(row_index)
        cell_values: tuple[str, ...] = (
            str(epoch_record["epoch"]),
            "{:.4f}".format(epoch_record["train_loss"]),
            "{:.2%}".format(epoch_record["train_accuracy"]),
            "{:.4f}".format(epoch_record["validation_loss"]),
            "{:.2%}".format(epoch_record["validation_accuracy"]),
        )
        for column_index, cell_value in enumerate(cell_values):
            table.setItem(row_index, column_index, QTableWidgetItem(cell_value))
        table.scrollToBottom()

    def _on_training_finished(self, summary_message: str) -> None:
        self._poll_training_progress()
        self._poll_timer.stop()
        self._set_controls_enabled(True)
        self._worker = None
        self.status_label.setText(summary_message)

    def _on_training_failed(self, error_message: str) -> None:
        self._poll_timer.stop()
        self._set_controls_enabled(True)
        self._worker = None
        QMessageBox.critical(self, "Training failed", error_message)

    def _on_worker_finished(self) -> None:
        # Ensure the UI always unlocks even when the thread ends abruptly.
        # Skip when a newer worker session is already running or the worker
        # was cleared by the normal finish/failure handlers.
        worker: TrainingWorker | None = self._worker
        if worker is None or worker.isRunning():
            return
        self._set_controls_enabled(True)
        self._worker = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.start_button.setEnabled(enabled)
        self.epoch_spinbox.setEnabled(enabled)
