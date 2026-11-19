"""Render the run history screen."""

from typing import Final

import moderngl
import pygame

from game.models.profile import ProfileStore
from game.views.runner import WINDOW_SIZE
from game.views.ui import HIGHLIGHT_TEXT_COLOR, UiLayer

TITLE_TEXT: Final[str] = "Run History"
FOOTER_TEXT: Final[str] = "Esc / Enter / Fist: back to menu"
ROW_LIMIT: Final[int] = 10
ROW_HEIGHT: Final[int] = 38

# Column rectangles: time, score, duration, character.
TIME_COLUMN: Final[tuple[int, int, int, int]] = (90, 0, 230, ROW_HEIGHT)
SCORE_COLUMN: Final[tuple[int, int, int, int]] = (340, 0, 110, ROW_HEIGHT)
DURATION_COLUMN: Final[tuple[int, int, int, int]] = (460, 0, 120, ROW_HEIGHT)
CHARACTER_COLUMN: Final[tuple[int, int, int, int]] = (590, 0, 280, ROW_HEIGHT)


class RunHistoryView:
    """Draws the best score and a table of the most recent runs."""

    def __init__(self, context: moderngl.Context) -> None:
        self.context: moderngl.Context = context
        self.ui: UiLayer = UiLayer(context, WINDOW_SIZE)

    def render(self, store: ProfileStore) -> None:
        """Draw the whole history frame and present it."""
        self.context.clear(0.07, 0.08, 0.12, 1.0)
        self.ui.draw_text(TITLE_TEXT, (0, 70, WINDOW_SIZE[0], 70), font_size=46, color=HIGHLIGHT_TEXT_COLOR)
        self.ui.draw_text(f"Best  {store.best_score} m", (0, 150, WINDOW_SIZE[0], 40), font_size=26)
        rows: list[list[str]] = [
            [record.time, f"{record.score} m", f"{record.duration_s:.1f} s", record.character.removesuffix(".glb")]
            for record in reversed(store.runs[-ROW_LIMIT:])
        ]
        self.ui.draw_panel((80, 220, 800, ROW_HEIGHT * max(len(rows), 1) + 58), (24, 26, 34), alpha=0.85)
        columns: tuple[tuple[tuple[int, int, int, int], str], ...] = (
            (TIME_COLUMN, "Time"), (SCORE_COLUMN, "Score"),
            (DURATION_COLUMN, "Duration"), (CHARACTER_COLUMN, "Character"),
        )
        header_y: int = 230
        column_rect: tuple[int, int, int, int]
        header: str
        for column_rect, header in columns:
            self.ui.draw_text(header, _offset_row(column_rect, header_y), font_size=20, align="left", color=HIGHLIGHT_TEXT_COLOR)
        row_y: int = header_y + ROW_HEIGHT + 6
        row: list[str]
        for row in rows:
            column_index: int
            for column_index, value in enumerate(row):
                self.ui.draw_text(value, _offset_row(columns[column_index][0], row_y), font_size=20, align="left")
            row_y += ROW_HEIGHT
        if not rows:
            self.ui.draw_text("No runs yet. Start playing to set a record!", (80, 220, 800, 58), font_size=24)
        self.ui.draw_text(FOOTER_TEXT, (0, 570, WINDOW_SIZE[0], 40), font_size=18)
        pygame.display.flip()


def _offset_row(column: tuple[int, int, int, int], row_y: int) -> tuple[int, int, int, int]:
    """Return the column rectangle moved down to the given row position."""
    return (column[0], row_y, column[2], column[3])
