"""Data collection tab: capture labelled hand crops for training."""

import time
from datetime import datetime
from pathlib import Path

import cv2
from cv2.typing import MatLike
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
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
    CameraFrameReader,
    convert_frame_to_qimage,
    create_camera_frame_reader,
    list_camera_names,
)
from studio.hand_region import draw_square_region
from mediapipe.tasks.python import vision
from utils.hand_detector import HandBox, HandDetector

LABELS: list[str] = ["Fist", "Palm", "Up", "Down", "Left", "Right"]
CAPTURE_INTERVAL: float = 0.5
DEFAULT_DATASET_ROOT: str = "dataset"
RENDER_INTERVAL_MILLISECONDS: int = 33
DETECTION_EVERY_N_FRAMES: int = 5
SIDEBAR_WIDTH: int = 300
PREVIEW_LABEL_STYLE: str = "background: #202124; color: white; border-radius: 8px;"


class DataCollectionTab(QWidget):
    """Widget that records labelled hand region crops into the dataset."""

    def __init__(self) -> None:
        super().__init__()
        self._camera_reader: CameraFrameReader | None = None
        self._hand_detector: HandDetector | None = None
        self._detected_hand_box: HandBox | None = None
        self._render_frame_count: int = 0
        self._is_recording: bool = False
        self._last_capture_monotonic_time: float = 0.0
        self._saved_frame_count: int = 0
        self._current_frame: MatLike | None = None

        self._render_timer: QTimer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MILLISECONDS)
        self._render_timer.timeout.connect(self._render_frame)

        self._build_ui()

    def _build_ui(self) -> None:
        self.preview_label: QLabel = QLabel("Camera stopped.")
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_label.setStyleSheet(PREVIEW_LABEL_STYLE)

        settings_group: QGroupBox = QGroupBox("Settings")
        form_layout: QFormLayout = QFormLayout(settings_group)

        self.camera_combobox: QComboBox = QComboBox()
        self.camera_combobox.addItem(DEFAULT_CAMERA_ITEM)
        for camera_name in list_camera_names():
            self.camera_combobox.addItem(camera_name)
        form_layout.addRow("Camera:", self.camera_combobox)

        self.label_combobox: QComboBox = QComboBox()
        self.label_combobox.addItems(LABELS)
        form_layout.addRow("Gesture label:", self.label_combobox)

        self.dataset_root_edit: QLabel = QLabel(DEFAULT_DATASET_ROOT)
        dataset_browse_button: QPushButton = QPushButton("Browse...")
        dataset_browse_button.clicked.connect(self._browse_dataset_root)
        dataset_row: QHBoxLayout = QHBoxLayout()
        dataset_row.addWidget(self.dataset_root_edit, stretch=1)
        dataset_row.addWidget(dataset_browse_button)
        form_layout.addRow("Dataset root:", dataset_row)

        self.camera_button: QPushButton = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self._toggle_camera)

        self.record_button: QPushButton = QPushButton("Start Recording")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._toggle_recording)

        self.count_label: QLabel = QLabel(
            "{} frames have been saved.".format(self._saved_frame_count)
        )
        self.count_label.setWordWrap(True)

        tab_layout: QHBoxLayout = QHBoxLayout(self)
        tab_layout.addWidget(self.preview_label, stretch=2)

        sidebar_widget: QWidget = QWidget()
        sidebar_widget.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout: QVBoxLayout = QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(settings_group)
        sidebar_layout.addWidget(self.camera_button)
        sidebar_layout.addWidget(self.record_button)
        sidebar_layout.addWidget(self.count_label)
        sidebar_layout.addStretch()
        tab_layout.addWidget(sidebar_widget, stretch=1)

        self.record_shortcut: QShortcut = QShortcut(QKeySequence("R"), self)
        self.record_shortcut.activated.connect(self._toggle_recording)

    def _browse_dataset_root(self) -> None:
        chosen_directory: str = QFileDialog.getExistingDirectory(
            self, "Select the dataset root directory", self.dataset_root_edit.text()
        )
        if chosen_directory:
            self.dataset_root_edit.setText(chosen_directory)

    def _toggle_camera(self) -> None:
        if self._camera_reader is not None:
            self._stop_camera()
            return
        camera_name: str | None = None
        if self.camera_combobox.currentText() != DEFAULT_CAMERA_ITEM:
            camera_name = self.camera_combobox.currentText()
        try:
            self._camera_reader = create_camera_frame_reader(camera_name)
            self._hand_detector = HandDetector(vision.RunningMode.IMAGE)
        except Exception as error:
            self._camera_reader = None
            self._hand_detector = None
            QMessageBox.critical(
                self,
                "Camera error",
                "Cannot start the camera or hand detector: {}".format(error),
            )
            return
        self._render_timer.start()
        self._render_frame_count = 0
        self._detected_hand_box = None
        self.camera_button.setText("Stop Camera")
        self.camera_combobox.setEnabled(False)
        self.record_button.setEnabled(True)
        self.preview_label.setText("")

    def _stop_camera(self) -> None:
        self._render_timer.stop()
        camera_reader: CameraFrameReader | None = self._camera_reader
        hand_detector: HandDetector | None = self._hand_detector
        self._camera_reader = None
        self._hand_detector = None
        self._detected_hand_box = None
        self._is_recording = False
        self._current_frame = None
        if camera_reader is not None:
            camera_reader.release()
        if hand_detector is not None:
            hand_detector.close()
        self.camera_button.setText("Start Camera")
        self.camera_combobox.setEnabled(True)
        self.record_button.setEnabled(False)
        self.record_button.setText("Start Recording")
        self.preview_label.setText("Camera stopped.")
        self.preview_label.setPixmap(QPixmap())

    def _toggle_recording(self) -> None:
        if self._is_recording:
            self._is_recording = False
            self.record_button.setText("Start Recording")
        elif self._camera_reader is not None:
            self._is_recording = True
            self._last_capture_monotonic_time = 0.0
            self.record_button.setText("Stop Recording")

    def _refresh_detected_hand_box(self) -> None:
        """Re-run MediaPipe detection every few render frames so the preview
        box tracks the hand without slowing rendering down."""
        hand_detector: HandDetector | None = self._hand_detector
        current_frame: MatLike | None = self._current_frame
        if hand_detector is None or current_frame is None:
            self._detected_hand_box = None
            return
        self._render_frame_count += 1
        if self._render_frame_count % DETECTION_EVERY_N_FRAMES != 0:
            return
        self._detected_hand_box = hand_detector.detect_hand_box(current_frame)

    def _render_frame(self) -> None:
        camera_reader: CameraFrameReader | None = self._camera_reader
        if camera_reader is None:
            return
        frame = camera_reader.read_frame()
        if frame is None:
            return
        # Copy so later pipeline overwrites do not mutate the saved image.
        self._current_frame = frame.copy()
        display_frame: MatLike = self._current_frame.copy()
        self._refresh_detected_hand_box()
        if self._detected_hand_box is not None:
            # Show the box actually used for saving.
            draw_square_region(
                display_frame,
                self._detected_hand_box.left_x,
                self._detected_hand_box.top_y,
                self._detected_hand_box.size,
            )
        qt_image = convert_frame_to_qimage(display_frame)
        self.preview_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if (
            self._is_recording
            and time.monotonic() - self._last_capture_monotonic_time
            >= CAPTURE_INTERVAL
        ):
            self._save_current_crop()
            self._last_capture_monotonic_time = time.monotonic()

    def _save_current_crop(self) -> None:
        current_frame: MatLike | None = self._current_frame
        if current_frame is None:
            return
        detected_crop: MatLike | None = (
            self._hand_detector.crop_hand(current_frame)
            if self._hand_detector is not None
            else None
        )
        if detected_crop is None:
            # No hand detected: skip this capture entirely instead of saving
            # a fallback crop that may not even contain the hand.
            return
        try:
            label_name: str = self.label_combobox.currentText()
            label_directory: Path = Path(self.dataset_root_edit.text()) / label_name
            label_directory.mkdir(parents=True, exist_ok=True)
            filename: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            cv2.imwrite(str(label_directory / filename), detected_crop)
        except (OSError, cv2.error, IndexError) as error:
            self._is_recording = False
            self.record_button.setText("Start Recording")
            QMessageBox.critical(self, "Saving failed", str(error))
            return
        self._saved_frame_count += 1
        self.count_label.setText(
            "{} frames have been saved.".format(self._saved_frame_count)
        )

    def shutdown(self) -> None:
        """Release the camera when the application closes."""
        if self._camera_reader is not None:
            self._stop_camera()
