"""The settlement screen shown when a run ends."""

from typing import Final

from game.input_backends.actions import GameAction
from game.presenters.menu import MenuPresenter
from game.scenes.base import Scene, SceneHost
from game.views.settlement import SettlementView

SETTLEMENT_ITEMS: Final[tuple[str, ...]] = ("Retry", "Back to Menu")
RETRY_ITEM_INDEX: Final[int] = 0
MENU_ITEM_INDEX: Final[int] = 1


class SettlementScene(Scene):
    """Shows the finished run's score and lets the player replay or leave."""

    def __init__(self, host: SceneHost, score: int, is_new_best: bool) -> None:
        super().__init__(host)
        self.score: int = score
        self.best_score: int = host.profile_store.best_score
        self.is_new_best: bool = is_new_best
        self.presenter: MenuPresenter = MenuPresenter(SETTLEMENT_ITEMS)
        self._view: SettlementView | None = None

    def handle_action(self, action: GameAction) -> None:
        if action is GameAction.QUIT:
            super().handle_action(action)
            return
        if action is GameAction.BACK:
            self._go_to_menu()
            return
        self.presenter.handle_action(action)
        if self.presenter.confirmed:
            self.presenter.confirmed = False
            if self.presenter.selected_index == RETRY_ITEM_INDEX:
                from game.scenes.playing import PlayingScene

                self._host.switch_scene(PlayingScene(self._host))
            elif self.presenter.selected_index == MENU_ITEM_INDEX:
                self._go_to_menu()

    def _go_to_menu(self) -> None:
        """Switch back to the main menu, importing it lazily to break a cycle."""
        from game.scenes.menu import MenuScene

        self._host.switch_scene(MenuScene(self._host))

    def render(self) -> None:
        if self._view is None:
            self._view = SettlementView(self._host.context)
        self._view.render(self.score, self.best_score, self.is_new_best, self.presenter)
