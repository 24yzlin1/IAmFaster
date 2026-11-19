"""Render the environment selection screen."""

from typing import Final

import moderngl
import pygame

from game.presenters.menu import MenuPresenter
from game.views.runner import WINDOW_SIZE
from game.views.ui import DEFAULT_TEXT_COLOR, HIGHLIGHT_TEXT_COLOR, Color, UiLayer

TITLE_TEXT: Final[str] = "Environment"
CURRENT_MARKER: Final[str] = " (current)"
FOOTER_TEXT: Final[str] = "Up/Down: select    Enter / Fist: confirm    Esc: back"


class EnvironmentSelectView:
    """Draws the day and night presets and marks the currently active one."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context: moderngl.Context = context
        self.ui: UiLayer = UiLayer(context, WINDOW_SIZE)

    def render(self, presenter: MenuPresenter, current_environment: str) -> None:
        """Draw the whole selection frame and present it."""
        self.context.clear(0.07, 0.08, 0.12, 1.0)
        self.ui.draw_text(TITLE_TEXT, (0, 150, WINDOW_SIZE[0], 80), font_size=50, color=HIGHLIGHT_TEXT_COLOR)
        item_height: int = 56
        panel_height: int = item_height * len(presenter.items) + 16
        panel_top: int = min(280, 550 - panel_height)
        self.ui.draw_panel((300, panel_top, 360, panel_height), (24, 26, 34), alpha=0.85)
        index: int
        for index, item in enumerate(presenter.items):
            selected: bool = index == presenter.selected_index
            is_current: bool = item == current_environment
            marker: str = CURRENT_MARKER if is_current else ""
            label: str = f">  {item}{marker}" if selected else f"    {item}{marker}"
            color: Color = HIGHLIGHT_TEXT_COLOR if selected else DEFAULT_TEXT_COLOR
            self.ui.draw_text(
                label, (300, panel_top + 8 + index * item_height, 360, item_height),
                font_size=28, color=color,
            )
        self.ui.draw_text(FOOTER_TEXT, (0, 570, WINDOW_SIZE[0], 40), font_size=18)
        pygame.display.flip()
