from typing import Final, Protocol
import pygame
from app.models.counter import CounterModel

COUNT_FONT_SIZE: Final[int] = 72
HINT_FONT_SIZE: Final[int] = 20
HINT_BOTTOM_MARGIN: Final[int] = 32
ANTIALIAS_ENABLED: Final[bool] = True
BACKGROUND_COLOR: Final[pygame.Color] = pygame.Color(0x0F, 0x14, 0x17)
COUNT_COLOR: Final[pygame.Color] = pygame.Color(0xDF, 0xE3, 0xE7)
HINT_COLOR: Final[pygame.Color] = pygame.Color(0xC0, 0xC8, 0xCD)
HINT_TEXT: Final[str] = "Click +1   R reset   ESC quit"


class CounterView(Protocol):
    def render_frame(
        self, display_surface: pygame.Surface, model: CounterModel
    ) -> None: ...
class PygameCounterView:
    def __init__(self) -> None:
        self._count_font: pygame.font.Font = pygame.font.Font(None, COUNT_FONT_SIZE)
        self._hint_font: pygame.font.Font = pygame.font.Font(None, HINT_FONT_SIZE)

    def render_frame(
        self, display_surface: pygame.Surface, model: CounterModel
    ) -> None:
        display_surface.fill(BACKGROUND_COLOR)
        surface_width: int = display_surface.get_width()
        surface_height: int = display_surface.get_height()
        count_surface: pygame.Surface = self._count_font.render(
            str(model.count), ANTIALIAS_ENABLED, COUNT_COLOR
        )
        count_left: int = (surface_width - count_surface.get_width()) // 2
        count_top: int = (surface_height - count_surface.get_height()) // 2
        display_surface.blit(count_surface, (count_left, count_top))
        hint_surface: pygame.Surface = self._hint_font.render(
            HINT_TEXT, ANTIALIAS_ENABLED, HINT_COLOR
        )
        hint_left: int = (surface_width - hint_surface.get_width()) // 2
        hint_top: int = surface_height - hint_surface.get_height() - HINT_BOTTOM_MARGIN
        display_surface.blit(hint_surface, (hint_left, hint_top))
