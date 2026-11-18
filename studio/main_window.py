"""Main window hosting the gesture data collection, training, and inference tabs."""

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QTabWidget

from studio.data_collection_tab import DataCollectionTab
from studio.inference_tab import InferenceTab
from studio.training_tab import TrainingTab

WINDOW_TITLE: str = "Gesture Studio"
WINDOW_WIDTH: int = 1080
WINDOW_HEIGHT: int = 720


class MainWindow(QMainWindow):
    """Main window hosting the data collection, training, and inference tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self._data_collection_tab: DataCollectionTab = DataCollectionTab()
        self._training_tab: TrainingTab = TrainingTab()
        self._inference_tab: InferenceTab = InferenceTab()
        tab_widget: QTabWidget = QTabWidget()
        tab_widget.addTab(self._data_collection_tab, "Data Collection")
        tab_widget.addTab(self._training_tab, "Training")
        tab_widget.addTab(self._inference_tab, "Inference")
        self.setCentralWidget(tab_widget)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._data_collection_tab.shutdown()
        self._inference_tab.shutdown()
        super().closeEvent(event)
