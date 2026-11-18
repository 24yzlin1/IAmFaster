"""Inference tab: live camera preview with gesture prediction."""

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from studio.camera import (
    DEFAULT_CAMERA_ITEM,
    convert_frame_to_qimage,
    list_camera_names,
)
from studio.hand_region import draw_square_region
from studio.prediction_worker import PredictionResult, PredictionWorker

DEFAULT_MODEL_PATH: str = "gesture_model/gesture_model.pth"
RENDER_INTERVAL_MILLISECONDS: int = 33
SIDEBAR_WIDTH: int = 300
PREVIEW_LABEL_STYLE: str = "background: #202124; color: white; border-radius: 8px;"


class InferenceTab(QWidget):
    """Widget that shows the camera preview with live predictions."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: PredictionWorker | None = None

        self._render_timer: QTimer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MILLISECONDS)
        self._render_timer.timeout.connect(self._render_latest_result)

        self._build_ui()

    def _build_ui(self) -> None:
        settings_group: QGroupBox = QGroupBox("Settings")
        form_layout: QFormLayout = QFormLayout(settings_group)

        self.model_path_edit: QLabel = QLabel(DEFAULT_MODEL_PATH)
        model_browse_button: QPushButton = QPushButton("Browse...")
        model_browse_button.clicked.connect(self._browse_model_path)
        model_row: QHBoxLayout = QHBoxLayout()
        model_row.addWidget(self.model_path_edit, stretch=1)
        model_row.addWidget(model_browse_button)
        form_layout.addRow("Model path:", model_row)

        self.camera_combobox: QComboBox = QComboBox()
        self.camera_combobox.addItem(DEFAULT_CAMERA_ITEM)
        for camera_name in list_camera_names():
            self.camera_combobox.addItem(camera_name)
        form_layout.addRow("Camera:", self.camera_combobox)

        self.toggle_button: QPushButton = QPushButton("Start Inference")
        self.toggle_button.clicked.connect(self._toggle_inference)

        self.preview_label: QLabel = QLabel("Inference stopped.")
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_label.setStyleSheet(PREVIEW_LABEL_STYLE)

        self.result_label: QLabel = QLabel("No prediction yet.")
        self.result_label.setWordWrap(True)

        tab_layout: QHBoxLayout = QHBoxLayout(self)
        tab_layout.addWidget(self.preview_label, stretch=2)

        sidebar_widget: QWidget = QWidget()
        sidebar_widget.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout: QVBoxLayout = QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(settings_group)
        sidebar_layout.addWidget(self.toggle_button)
        sidebar_layout.addWidget(self.result_label)
        sidebar_layout.addStretch()
        tab_layout.addWidget(sidebar_widget, stretch=1)

    def _browse_model_path(self) -> None:
        chosen_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select the trained model file",
            self.model_path_edit.text(),
            "Model files (*.pth)",
        )
        if chosen_path:
            self.model_path_edit.setText(chosen_path)

    def _toggle_inference(self) -> None:
        if self._worker is not None:
            self._stop_inference()
            return
        model_path: Path = Path(self.model_path_edit.text())
        if not model_path.is_file():
            QMessageBox.warning(
                self,
                "Invalid model path",
                "The model file does not exist: {}".format(model_path),
            )
            return
        camera_name: str | None = None
        if self.camera_combobox.currentText() != DEFAULT_CAMERA_ITEM:
            camera_name = self.camera_combobox.currentText()
        worker: PredictionWorker = PredictionWorker(
            model_path, camera_name, parent=self
        )
        worker.worker_failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._set_controls_enabled(False)
        self.preview_label.setText("Loading the model...")
        worker.start()
        self._render_timer.start()

    def _stop_inference(self) -> None:
        worker: PredictionWorker | None = self._worker
        if worker is None:
            return
        self._render_timer.stop()
        worker.stop()
        worker.wait()
        self._worker = None
        self.preview_label.setText("Inference stopped.")
        self.result_label.setText("No prediction yet.")
        self._set_controls_enabled(True)

    def _render_latest_result(self) -> None:
        worker: PredictionWorker | None = self._worker
        if worker is None:
            return
        result: PredictionResult | None = worker.take_latest_result()
        if result is None:
            return
        if result.hand_box is not None:
            draw_square_region(
                result.frame,
                result.hand_box.left_x,
                result.hand_box.top_y,
                result.hand_box.size,
            )
        qt_image: QImage = convert_frame_to_qimage(result.frame)
        self.preview_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if result.prediction is None:
            self.result_label.setText("No hand detected.")
            return
        probability_items: list[tuple[str, float]] = sorted(
            result.prediction.probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        probability_summary: str = " | ".join(
            "{} {:.1%}".format(class_name, probability)
            for class_name, probability in probability_items
        )
        self.result_label.setText(
            "Prediction: {} ({:.1%})\n{}".format(
                result.prediction.class_name,
                result.prediction.confidence,
                probability_summary,
            )
        )

    def _on_worker_failed(self, error_message: str) -> None:
        self._render_timer.stop()
        self._worker = None
        self._set_controls_enabled(True)
        self.preview_label.setText("Inference stopped.")
        QMessageBox.critical(self, "Inference failed", error_message)

    def _on_worker_finished(self) -> None:
        # Unlock the UI when the worker exits on its own, for example after
        # an unrecoverable prediction failure.
        worker: PredictionWorker | None = self._worker
        if worker is None:
            return
        self._render_timer.stop()
        self._worker = None
        self.preview_label.setText("Inference stopped.")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.toggle_button.setText(
            "Stop Inference" if not enabled else "Start Inference"
        )
        self.model_path_edit.setEnabled(enabled)
        self.camera_combobox.setEnabled(enabled)

    def shutdown(self) -> None:
        """Release the worker and camera when the application closes."""
        if self._worker is not None:
            self._stop_inference()
