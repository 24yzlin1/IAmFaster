"""Camera frame readers with a DirectShow backend for Windows.

``cv2.VideoCapture`` can hang indefinitely with DirectShow-only cameras on
Windows, so the Windows path builds a pygrabber DirectShow filter graph
instead, matching the working data collection tool startup sequence.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy
from cv2.typing import MatLike

import cv2

if TYPE_CHECKING:
    from pygrabber.dshow_graph import Mat

DEFAULT_CAMERA_INDEX: Final[int] = 0
FRAME_DTYPE: Final = numpy.uint8
NATIVE_PREVIEW_HIDE_DELAY_SECONDS: Final[float] = 1.0
NATIVE_PREVIEW_WINDOW_TITLE: Final[str] = "ActiveMovie Window"


class CameraFrameReader(Protocol):
    """A camera connection that delivers the newest frame on demand."""

    def read_frame(self) -> MatLike | None:
        """Return the newest frame, or None when no frame is available."""
        ...

    def release(self) -> None:
        """Stop the camera and release its resources."""
        ...


class OpenCVCameraFrameReader:
    """Frame reader built on cv2.VideoCapture for non-Windows platforms."""

    def __init__(self, camera_index: int) -> None:
        self._capture: cv2.VideoCapture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                "Cannot open camera with index {}.".format(camera_index)
            )

    def read_frame(self) -> MatLike | None:
        capture_success, frame = self._capture.read()
        if not capture_success:
            return None
        return frame.astype(FRAME_DTYPE)

    def release(self) -> None:
        self._capture.release()


class DirectShowCameraFrameReader:
    """Frame reader built on a pygrabber DirectShow filter graph (Windows)."""

    def __init__(self, camera_name: str | None) -> None:
        self._graph: Any = self._build_running_graph(camera_name)
        self._latest_frame: Mat | None = None
        hide_preview_timer = threading.Timer(
            NATIVE_PREVIEW_HIDE_DELAY_SECONDS, self._hide_native_preview_window
        )
        hide_preview_timer.daemon = True
        hide_preview_timer.start()

    def read_frame(self) -> MatLike | None:
        self._graph.grab_frame()
        latest_frame = self._latest_frame
        if latest_frame is None:
            return None
        # astype copies the frame so later pipeline writes cannot mutate it,
        # and normalizes the pygrabber dtype to a plain uint8 image.
        return latest_frame.astype(FRAME_DTYPE)

    def release(self) -> None:
        try:
            self._graph.stop()
        except Exception:
            print("Failed to stop the DirectShow graph.")

    def _build_running_graph(self, camera_name: str | None) -> Any:
        from pygrabber.dshow_graph import FilterGraph

        filter_graph = FilterGraph()
        device_names: list[str] = list(filter_graph.get_input_devices())
        print("Available cameras: {}".format(device_names))
        device_index: int = 0
        if camera_name is not None:
            if camera_name in device_names:
                device_index = device_names.index(camera_name)
            else:
                print(
                    "WARN: camera '{}' not found; using the first device.".format(
                        camera_name
                    )
                )
        filter_graph.add_video_input_device(device_index)
        filter_graph.add_sample_grabber(self._on_sample_grabbed)
        filter_graph.add_default_render()
        filter_graph.prepare_preview_graph()
        filter_graph.run()
        return filter_graph

    def _on_sample_grabbed(self, image: Mat) -> None:
        self._latest_frame = image

    @staticmethod
    def _hide_native_preview_window() -> None:
        try:
            user32_library: Any = ctypes.windll.user32
            window_handle: int = int(
                user32_library.FindWindowW(None, NATIVE_PREVIEW_WINDOW_TITLE)
            )
            if window_handle:
                user32_library.ShowWindow(window_handle, 0)
        except OSError:
            print("Failed to hide the native preview window.")


def create_camera_frame_reader(camera_name: str | None) -> CameraFrameReader:
    """Create the platform-appropriate camera frame reader.

    On Windows a DirectShow reader is used because ``cv2.VideoCapture``
    hangs with DirectShow-only cameras; other platforms use OpenCV.
    """
    if sys.platform == "win32":
        return DirectShowCameraFrameReader(camera_name)
    return OpenCVCameraFrameReader(DEFAULT_CAMERA_INDEX)
