from pathlib import Path
from typing import Final

import pygame

from app.input_backends import (
    BACKEND_CHOICES,
    BACKEND_NAMES_ENABLING_GESTURE,
    BACKEND_NAMES_ENABLING_KEYBOARD,
    GameAction,
    GestureInputBackend,
    InputBackend,
    KeyboardInputBackend,
)
from app.models.counter import CounterModel
from app.presenters.counter import CounterPresenter
from app.views.counter import CounterView, PygameCounterView

WINDOW_WIDTH: Final[int] = 390
WINDOW_HEIGHT: Final[int] = 844
WINDOW_SIZE: Final[tuple[int, int]] = (WINDOW_WIDTH, WINDOW_HEIGHT)
WINDOW_TITLE: Final[str] = "Counter"
FRAMES_PER_SECOND: Final[int] = 60
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH: Final[Path] = PROJECT_ROOT / "gesture_model" / "gesture_model.pth"
INCREMENT_GESTURE_NAME: Final[str] = "Palm"
RESET_GESTURE_NAME: Final[str] = "Fist"
REPEATABLE_GESTURE_CLASS_NAMES: Final[frozenset[str]] = frozenset(
    {"Up", "Down", "Left", "Right"}
)
GESTURE_ACTIONS: Final[dict[str, GameAction]] = {
    INCREMENT_GESTURE_NAME: GameAction.INCREMENT,
    RESET_GESTURE_NAME: GameAction.RESET,
}


class CounterApplication:
    def __init__(
        self, model_path: Path, backend_name: str, camera_name: str | None
    ) -> None:
        pygame.init()
        self._display_surface: pygame.Surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(WINDOW_TITLE)
        self._frame_clock: pygame.time.Clock = pygame.time.Clock()
        self._input_backends: list[InputBackend] = self._build_input_backends(
            model_path, backend_name, camera_name
        )
        self._model: CounterModel = CounterModel()
        self._view: CounterView = PygameCounterView()
        self._presenter: CounterPresenter = CounterPresenter(self._model)

    def run(self) -> None:
        try:
            while self._model.is_running:
                self._poll_input_backends()
                self._view.render_frame(self._display_surface, self._model)
                pygame.display.flip()
                self._frame_clock.tick(FRAMES_PER_SECOND)
        finally:
            self._close_input_backends()
            pygame.quit()

    def _build_input_backends(
        self, model_path: Path, backend_name: str, camera_name: str | None
    ) -> list[InputBackend]:
        if backend_name not in (
            BACKEND_NAMES_ENABLING_KEYBOARD | BACKEND_NAMES_ENABLING_GESTURE
        ):
            raise ValueError(
                "Unknown backend name {}; expected one of {}.".format(
                    backend_name, ", ".join(BACKEND_CHOICES)
                )
            )
        input_backends: list[InputBackend] = []
        if backend_name in BACKEND_NAMES_ENABLING_KEYBOARD:
            input_backends.append(KeyboardInputBackend())
        if backend_name in BACKEND_NAMES_ENABLING_GESTURE:
            input_backends.append(
                GestureInputBackend(
                    model_path,
                    GESTURE_ACTIONS,
                    repeatable_class_names=REPEATABLE_GESTURE_CLASS_NAMES,
                    camera_name=camera_name,
                )
            )
        return input_backends

    def _poll_input_backends(self) -> None:
        for input_backend in self._input_backends:
            for action in input_backend.poll():
                self._presenter.handle_action(action)

    def _close_input_backends(self) -> None:
        for input_backend in self._input_backends:
            input_backend.close()


def run(model_path: Path, backend_name: str, camera_name: str | None) -> None:
    CounterApplication(model_path, backend_name, camera_name).run()
