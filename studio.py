"""GUI entry point for gesture model data collection, training, and inference."""

import sys

from PySide6.QtWidgets import QApplication

from studio.main_window import MainWindow


def main() -> None:
    application: QApplication = QApplication(sys.argv)
    main_window: MainWindow = MainWindow()
    main_window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
