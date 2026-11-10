from typing import Final
import pygame
from app.models.counter import CounterModel
from app.presenters.counter import CounterPresenter
from app.views.counter import CounterView, PygameCounterView

WINDOW_WIDTH: Final[int] = 390
WINDOW_HEIGHT: Final[int] = 844
WINDOW_SIZE: Final[tuple[int, int]] = (WINDOW_WIDTH, WINDOW_HEIGHT)
WINDOW_TITLE: Final[str] = "Counter"
FRAMES_PER_SECOND: Final[int] = 60


def collect_window_events() -> list[pygame.event.Event]:
    return list(pygame.event.get())


class CounterApplication:
    def __init__(self) -> None:
        pygame.init()
        self._display_surface: pygame.Surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(WINDOW_TITLE)
        self._frame_clock: pygame.time.Clock = pygame.time.Clock()
        self._model: CounterModel = CounterModel()
        self._view: CounterView = PygameCounterView()
        self._presenter: CounterPresenter = CounterPresenter(self._model)

    def run(self) -> None:
        try:
            while self._model.is_running:
                events: list[pygame.event.Event] = collect_window_events()
                self._presenter.handle_events(events)
                self._view.render_frame(self._display_surface, self._model)
                pygame.display.flip()
                self._frame_clock.tick(FRAMES_PER_SECOND)
        finally:
            pygame.quit()


def run() -> None:
    CounterApplication().run()
