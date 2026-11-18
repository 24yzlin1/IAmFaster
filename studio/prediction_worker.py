"""Background worker that runs camera frames through the gesture model."""

import threading
from pathlib import Path
from typing import NamedTuple

from cv2.typing import MatLike
from PySide6.QtCore import QObject, QThread, Signal

from studio.camera import create_camera_frame_reader
from mediapipe.tasks.python import vision
from utils.hand_detector import HandBox, HandDetector
from utils.predictor import GesturePrediction, GesturePredictor

IDLE_SLEEP_MILLISECONDS: int = 5


class PredictionResult(NamedTuple):
    """A camera frame together with the prediction computed from it.

    *prediction* is None when no hand was detected in the frame.
    """

    frame: MatLike
    prediction: GesturePrediction | None
    hand_box: HandBox | None


class PredictionWorker(QThread):
    """Reads camera frames, predicts gestures, and stores the latest result."""

    worker_failed = Signal(str)

    def __init__(
        self,
        model_path: Path,
        camera_name: str | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_path: Path = model_path
        self._camera_name: str | None = camera_name
        self._result_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_result: PredictionResult | None = None

    def run(self) -> None:
        try:
            predictor: GesturePredictor = GesturePredictor(self._model_path)
        except Exception as error:
            self.worker_failed.emit("{}: {}".format(type(error).__name__, error))
            return
        try:
            camera_reader = create_camera_frame_reader(self._camera_name)
        except Exception as error:
            self.worker_failed.emit("{}: {}".format(type(error).__name__, error))
            return
        try:
            hand_detector = HandDetector(vision.RunningMode.VIDEO)
        except Exception as error:
            camera_reader.release()
            self.worker_failed.emit("{}: {}".format(type(error).__name__, error))
            return
        while not self._stop_event.is_set():
            frame = camera_reader.read_frame()
            if frame is None:
                self.msleep(IDLE_SLEEP_MILLISECONDS)
                continue
            try:
                # Track the hand with MediaPipe so the model is fed a tight,
                # position-independent crop. Without a hand the frame is
                # published without a prediction, keeping the preview live
                # without showing stale or unreliable results.
                hand_box: HandBox | None = hand_detector.detect_hand_box(frame)
                prediction: GesturePrediction | None = None
                if hand_box is not None:
                    hand_crop: MatLike = frame[
                        hand_box.top_y : hand_box.top_y + hand_box.size,
                        hand_box.left_x : hand_box.left_x + hand_box.size,
                    ].copy()
                    prediction = predictor.predict(hand_crop)
                result: PredictionResult = PredictionResult(
                    frame=frame, prediction=prediction, hand_box=hand_box
                )
            except Exception as error:
                self.worker_failed.emit(
                    "{}: {}".format(type(error).__name__, error)
                )
                break
            with self._result_lock:
                self._latest_result = result
        camera_reader.release()
        hand_detector.close()

    def stop(self) -> None:
        """Ask the worker loop to stop; call wait() afterwards."""
        self._stop_event.set()

    def take_latest_result(self) -> PredictionResult | None:
        """Return and clear the newest frame/prediction pair, if any."""
        with self._result_lock:
            latest_result = self._latest_result
            self._latest_result = None
            return latest_result
