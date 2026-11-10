from typing import Final
import pygame
from app.models.counter import CounterModel

PRIMARY_MOUSE_BUTTON: Final[int] = 1


def read_integer_attribute(
    event: pygame.event.Event, attribute_name: str
) -> int | None:
    raw_value: object = getattr(event, attribute_name, None)
    if isinstance(raw_value, int):
        return raw_value
    return None


class CounterPresenter:
    def __init__(self, model: CounterModel) -> None:
        self._model: CounterModel = model

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self._model.is_running = False
            elif event.type == pygame.KEYDOWN:
                pressed_key: int | None = read_integer_attribute(event, "key")
                if pressed_key == pygame.K_ESCAPE:
                    self._model.is_running = False
                elif pressed_key == pygame.K_r:
                    self._model.count = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pressed_button: int | None = read_integer_attribute(event, "button")
                if pressed_button == PRIMARY_MOUSE_BUTTON:
                    self._model.count += 1
