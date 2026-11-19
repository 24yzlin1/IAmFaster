"""Background worker that converts an image dataset to landmark features."""

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from utils.convert_dataset import convert_dataset


class ConvertWorker(QThread):
    """Runs dataset conversion off the UI thread and reports the outcome."""

    convert_finished = Signal(int)
    convert_failed = Signal(str)

    def __init__(
        self, input_root: Path, output_root: Path, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._input_root: Path = input_root
        self._output_root: Path = output_root

    def run(self) -> None:
        try:
            total_saved: int = convert_dataset(self._input_root, self._output_root)
        except Exception as error:
            self.convert_failed.emit(
                "{}: {}".format(type(error).__name__, error)
            )
            return
        self.convert_finished.emit(total_saved)
