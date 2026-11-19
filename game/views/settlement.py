"""Render the run settlement screen after a crash."""

from typing import Final

import moderngl
import pygame

from game.presenters.menu import MenuPresenter
from game.views.runner import WINDOW_SIZE
from game.views.ui import DEFAULT_TEXT_COLOR, HIGHLIGHT_TEXT_COLOR, Color, UiLayer

TITLE_TEXT: Final[str] = "Run Over"
SCORE_TEXT: Final[str] = "Score"
BEST_TEXT: Final[str] = "Best"
NEW_RECORD_TEXT: Final[str] = "New record!"
FOOTER_TEXT: Final[str] = "Up/Down: select    R / Fist: confirm    Esc: menu"


class SettlementView:
    """Draws the final score, the best score, and the follow-up choices."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context: moderngl.Context = context
        self.ui: UiLayer = UiLayer(context, WINDOW_SIZE)

    def render(self, score: int, best_score: int, is_new_best: bool, presenter: MenuPresenter) -> None:
        """Draw the whole settlement frame and present it."""
        self.context.clear(0.07, 0.08, 0.12, 1.0)
        self.ui.draw_text(TITLE_TEXT, (0, 90, WINDOW_SIZE[0], 80), font_size=50, color=HIGHLIGHT_TEXT_COLOR)
        self.ui.draw_panel((260, 200, 440, 150), (24, 26, 34), alpha=0.85)
        self.ui.draw_text(SCORE_TEXT, (260, 215, 440, 50), font_size=30, align="left")
        self.ui.draw_text(f"{score}", (700, 215, 0, 50), font_size=30)
        self.ui.draw_text(BEST_TEXT, (260, 290, 440, 50), font_size=30, align="left")
        self.ui.draw_text(f"{best_score}", (700, 290, 0, 50), font_size=30)
        if is_new_best:
            self.ui.draw_text(NEW_RECORD_TEXT, (260, 215, 440, 50), font_size=30, color=HIGHLIGHT_TEXT_COLOR)
        item_height: int = 52
        panel_top: int = 400
        panel_height: int = item_height * len(presenter.items) + 16
        self.ui.draw_panel((300, panel_top, 360, panel_height), (24, 26, 34), alpha=0.85)
        index: int
        for index, item in enumerate(presenter.items):
            selected: bool = index == presenter.selected_index
            label: str = f">  {item}" if selected else f"    {item}"
            color: Color = HIGHLIGHT_TEXT_COLOR if selected else DEFAULT_TEXT_COLOR
            self.ui.draw_text(
                label, (300, panel_top + 8 + index * item_height, 360, item_height),
                font_size=28, color=color,
            )
        self.ui.draw_text(FOOTER_TEXT, (0, 570, WINDOW_SIZE[0], 40), font_size=18)
        pygame.display.flip()
