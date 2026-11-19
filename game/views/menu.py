"""Render the main menu as a 2D overlay on the OpenGL window."""

from typing import Final

import moderngl
import pygame

from game.presenters.menu import MenuPresenter
from game.views.runner import WINDOW_SIZE
from game.views.ui import DEFAULT_TEXT_COLOR, HIGHLIGHT_TEXT_COLOR, Color, UiLayer

TITLE_TEXT: Final[str] = "I Am Faster"
FOOTER_TEXT: Final[str] = "Up/Down: select    R / Fist: confirm    Esc: back"


class MenuView:
    """Draws the title, the selectable entries, and a control hint footer."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context: moderngl.Context = context
        self.ui: UiLayer = UiLayer(context, WINDOW_SIZE)

    def render(self, presenter: MenuPresenter) -> None:
        """Draw the whole menu frame and present it."""
        self.context.clear(0.07, 0.08, 0.12, 1.0)
        self.ui.draw_text(TITLE_TEXT, (0, 110, WINDOW_SIZE[0], 90), font_size=58, color=HIGHLIGHT_TEXT_COLOR)
        item_height: int = 56
        panel_height: int = item_height * len(presenter.items) + 16
        panel_top: int = min(270, 550 - panel_height)
        self.ui.draw_panel((300, panel_top, 360, panel_height), (24, 26, 34), alpha=0.85)
        index: int
        for index, item in enumerate(presenter.items):
            selected: bool = index == presenter.selected_index
            label: str = f">  {item}" if selected else f"    {item}"
            color: Color = HIGHLIGHT_TEXT_COLOR if selected else DEFAULT_TEXT_COLOR
            self.ui.draw_text(
                label, (300, panel_top + 8 + index * item_height, 360, item_height),
                font_size=30, color=color,
            )
        self.ui.draw_text(FOOTER_TEXT, (0, 560, WINDOW_SIZE[0], 40), font_size=18)
        pygame.display.flip()
