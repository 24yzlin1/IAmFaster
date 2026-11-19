"""Camera-driven gesture input with prediction debouncing.

Turns raw per-frame model predictions into clean gesture events:
a prediction only becomes a confirmed gesture after several consecutive
frames agree on the same class with sufficient confidence.
"""

import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Final, NamedTuple

from cv2.typing import MatLike

from game.input_backends.actions import GameAction
from game.input_backends.camera import CameraFrameReader, create_camera_frame_reader
from utils.landmark import HandNotDetectedError
from utils.predictor import GesturePrediction, GesturePredictor

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.8
DEFAULT_STABLE_PREDICTION_COUNT: Final[int] = 3
DEFAULT_WORKER_INTERVAL_SECONDS: Final[float] = 0.005
DEFAULT_REPEAT_INTERVAL_SECONDS: Final[float] = 0.25


class GestureEvent(NamedTuple):
    class_name: str
    confidence: float


class GestureInputSource:
    """Reads camera frames and emits debounced gesture events.

    A background worker thread performs the camera grabs and model
    predictions so the game loop only performs cheap event collection.
    Trigger gestures (held without repetition support) fire exactly once
    per gesture hold; repeatable gestures fire again every
    ``repeat_interval_seconds`` while the gesture stays recognized.
    """

    def __init__(
        self,
        model_path: Path | str,
        repeatable_class_names: frozenset[str] = frozenset(),
        camera_name: str | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        stable_prediction_count: int = DEFAULT_STABLE_PREDICTION_COUNT,
        repeat_interval_seconds: float = DEFAULT_REPEAT_INTERVAL_SECONDS,
        required_class_names: frozenset[str] = frozenset(),
    ) -> None:
        model_file_path: Path = Path(model_path)
        if not model_file_path.is_file():
            raise FileNotFoundError(
                "Gesture model not found: {}; train it first with "
                "'python -m utils.trainer'.".format(model_file_path)
            )
        if stable_prediction_count < 1:
            raise ValueError(
                "stable_prediction_count must be a positive integer, got {}.".format(
                    stable_prediction_count
                )
            )
        self._repeatable_class_names = repeatable_class_names
        self._confidence_threshold = confidence_threshold
        self._stable_prediction_count = stable_prediction_count
        self._repeat_interval_seconds = repeat_interval_seconds
        self._predictor = GesturePredictor(model_file_path)
        missing: list[str] = sorted(required_class_names.difference(self._predictor.classes))
        if missing:
            raise ValueError(f"gesture model is missing required classes: {', '.join(missing)}")
        self._release_count: int = 0
        self._camera_reader: CameraFrameReader = create_camera_frame_reader(camera_name)
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._run_worker, daemon=True, name="gesture-prediction-worker"
        )
        self._pending_class_name: str | None = None
        self._pending_count: int = 0
        self._confirmed_class_name: str | None = None
        self._emitted_class_name: str | None = None
        self._latest_confidence: float = 0.0
        self._last_emit_monotonic_time: float = 0.0
        try:
            self._worker_thread.start()
        except RuntimeError:
            self._camera_reader.release()
            raise

    def poll(self) -> GestureEvent | None:
        """Collect one gesture event without touching the camera or model."""
        with self._state_lock:
            return self._collect_event(monotonic())

    def close(self) -> None:
        """Stop the worker thread and release the camera resource."""
        self._stop_event.set()
        self._worker_thread.join(timeout=2.0)
        self._camera_reader.release()

    def _run_worker(self) -> None:
        """Grab frames, predict, and update confirmation until stopped."""
        while not self._stop_event.is_set():
            frame = self._read_frame()
            if frame is not None:
                try:
                    self._update_confirmation(self._predictor.predict(frame))
                except HandNotDetectedError:
                    # Landmark models report a missing hand as an exception;
                    # treat the frame like a zero-confidence prediction so
                    # the debouncer releases any confirmed gesture.
                    self._update_confirmation(
                        GesturePrediction(
                            class_name="",
                            class_index=-1,
                            confidence=0.0,
                            probabilities={},
                        )
                    )
            sleep(DEFAULT_WORKER_INTERVAL_SECONDS)

    def _update_confirmation(self, prediction: GesturePrediction) -> None:
        with self._state_lock:
            self._latest_confidence = prediction.confidence
            if prediction.confidence < self._confidence_threshold:
                self._release_count += 1
                self._pending_count = 0
                if self._release_count >= self._stable_prediction_count:
                    self._reset_pending()
                return
            self._release_count = 0
            if prediction.class_name == self._pending_class_name:
                self._pending_count += 1
            else:
                self._pending_class_name = prediction.class_name
                self._pending_count = 1
            if self._pending_count >= self._stable_prediction_count:
                self._confirmed_class_name = self._pending_class_name

    def _collect_event(self, current_monotonic_time: float) -> GestureEvent | None:
        confirmed_class_name = self._confirmed_class_name
        if confirmed_class_name is None:
            return None
        if confirmed_class_name != self._emitted_class_name:
            self._emitted_class_name = confirmed_class_name
            self._last_emit_monotonic_time = current_monotonic_time
            return GestureEvent(
                class_name=confirmed_class_name,
                confidence=self._latest_confidence,
            )
        if confirmed_class_name in self._repeatable_class_names:
            elapsed_seconds = (
                current_monotonic_time - self._last_emit_monotonic_time
            )
            if elapsed_seconds >= self._repeat_interval_seconds:
                self._last_emit_monotonic_time = current_monotonic_time
                return GestureEvent(
                    class_name=confirmed_class_name,
                    confidence=self._latest_confidence,
                )
        return None

    def _reset_pending(self) -> None:
        self._pending_class_name = None
        self._pending_count = 0
        self._confirmed_class_name = None
        self._emitted_class_name = None

    def _read_frame(self) -> MatLike | None:
        return self._camera_reader.read_frame()


class GestureInputBackend:
    """Gesture backend that maps debounced gesture events to game actions."""

    def __init__(
        self,
        model_path: Path | str,
        gesture_actions: dict[str, GameAction],
        repeatable_class_names: frozenset[str] = frozenset(),
        camera_name: str | None = None,
        required_class_names: frozenset[str] = frozenset(),
    ) -> None:
        self._source: GestureInputSource = GestureInputSource(
            model_path,
            repeatable_class_names=repeatable_class_names,
            camera_name=camera_name,
            required_class_names=required_class_names,
        )
        self._gesture_actions: dict[str, GameAction] = gesture_actions

    def poll(self) -> list[GameAction]:
        gesture_event = self._source.poll()
        if gesture_event is None:
            return []
        print(f"[gesture] {gesture_event.class_name} ({gesture_event.confidence:.2f})")
        gesture_action = self._gesture_actions.get(gesture_event.class_name)
        if gesture_action is None:
            return []
        return [gesture_action]

    def close(self) -> None:
        self._source.close()
