"""Menu state: selectable entries and the highlighted index."""

from game.input_backends.actions import GameAction


class MenuPresenter:
    """Tracks which menu entry is highlighted and whether it was confirmed."""

    def __init__(self, items: tuple[str, ...]) -> None:
        self.items: tuple[str, ...] = items
        self.selected_index: int = 0
        self.confirmed: bool = False

    @property
    def selected_item(self) -> str:
        """The label of the highlighted entry."""
        return self.items[self.selected_index]

    def handle_action(self, action: GameAction) -> None:
        """Move the highlight or confirm the selected entry."""
        if action in (GameAction.JUMP, GameAction.LEFT):
            self.selected_index = (self.selected_index - 1) % len(self.items)
        elif action in (GameAction.SLIDE, GameAction.RIGHT):
            self.selected_index = (self.selected_index + 1) % len(self.items)
        elif action is GameAction.RESET:
            self.confirmed = True
