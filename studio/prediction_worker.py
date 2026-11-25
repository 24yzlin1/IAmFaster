"""Background worker that runs camera frames through the gesture model."""

import threading
from pathlib import Path
from typing import NamedTuple

from cv2.typing import MatLike
from PySide6.QtCore import QObject, QThread, Signal

from studio.camera import create_camera_frame_reader
from mediapipe.tasks.python import vision
from utils.hand_detector import HandBox, HandDetection, HandDetector
from utils.predictor import GesturePrediction, GesturePredictor

IDLE_SLEEP_MILLISECONDS: int = 5


class PredictionResult(NamedTuple):
    """A camera frame together with the prediction computed from it.

    *prediction* is None when no hand was detected in the frame;
    *landmark_points* holds the drawn hand skeleton, if a hand is visible.
    """

    frame: MatLike
    prediction: GesturePrediction | None
    hand_box: HandBox | None
    landmark_points: list[tuple[int, int]] | None


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
                # Track the hand with MediaPipe so mask models are fed a
                # tight, position-independent crop. Without a hand the frame
                # is published without a prediction, keeping the preview live
                # without showing stale or unreliable results. The landmark
                # points are also kept for drawing the hand skeleton.
                frame_height: int
                frame_width: int
                frame_height, frame_width = frame.shape[:2]
                hand_detection: HandDetection | None = hand_detector.detect_hand(
                    frame
                )
                landmark_points: list[tuple[int, int]] | None = (
                    hand_detection.landmark_points
                    if hand_detection is not None
                    else None
                )
                hand_box: HandBox | None = (
                    HandDetector.box_from_points(
                        landmark_points or [], frame_width, frame_height
                    )
                )
                prediction: GesturePrediction | None = None
                if predictor.input_type == "landmark":
                    # Landmark models consume the normalized skeleton. Reuse
                    # the detection above: re-detecting inside the tight crop
                    # lets a closed hand fill the crop and fail detection,
                    # which the preview would report as a missing hand.
                    if hand_detection is not None:
                        prediction = predictor.predict_from_landmarks(
                            hand_detection.feature_vector
                        )
                elif hand_box is not None:
                    hand_crop: MatLike = frame[
                        hand_box.top_y : hand_box.top_y + hand_box.size,
                        hand_box.left_x : hand_box.left_x + hand_box.size,
                    ].copy()
                    prediction = predictor.predict(hand_crop)
                result: PredictionResult = PredictionResult(
                    frame=frame,
                    prediction=prediction,
                    hand_box=hand_box,
                    landmark_points=landmark_points,
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
