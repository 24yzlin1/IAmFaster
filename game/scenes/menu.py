"""The main menu screen."""

from typing import Final

from game.input_backends.actions import GameAction
from game.presenters.menu import MenuPresenter
from game.scenes.base import Scene, SceneHost
from game.views.menu import MenuView

MENU_ITEMS: Final[tuple[str, ...]] = ("Start Game", "Characters", "Environment", "History", "Quit")
START_ITEM_INDEX: Final[int] = 0
CHARACTER_ITEM_INDEX: Final[int] = 1
ENVIRONMENT_ITEM_INDEX: Final[int] = 2
HISTORY_ITEM_INDEX: Final[int] = 3
QUIT_ITEM_INDEX: Final[int] = 4


class MenuScene(Scene):
    """Shows the entry menu and routes the confirmed item to a scene."""

    def __init__(self, host: SceneHost) -> None:
        super().__init__(host)
        self.presenter: MenuPresenter = MenuPresenter(MENU_ITEMS)
        self._view: MenuView | None = None

    def handle_action(self, action: GameAction) -> None:
        if action is GameAction.QUIT:
            super().handle_action(action)
            return
        if action is GameAction.BACK:
            return  # The menu is the top-level screen; there is nothing to leave.
        self.presenter.handle_action(action)
        if self.presenter.confirmed:
            self.presenter.confirmed = False
            # Scene modules import each other back, so import targets lazily.
            from game.scenes.character_select import CharacterSelectScene
            from game.scenes.environment_select import EnvironmentSelectScene
            from game.scenes.history import RunHistoryScene
            from game.scenes.playing import PlayingScene

            if self.presenter.selected_index == START_ITEM_INDEX:
                self._host.switch_scene(PlayingScene(self._host))
            elif self.presenter.selected_index == CHARACTER_ITEM_INDEX:
                self._host.switch_scene(CharacterSelectScene(self._host))
            elif self.presenter.selected_index == ENVIRONMENT_ITEM_INDEX:
                self._host.switch_scene(EnvironmentSelectScene(self._host))
            elif self.presenter.selected_index == HISTORY_ITEM_INDEX:
                self._host.switch_scene(RunHistoryScene(self._host))
            elif self.presenter.selected_index == QUIT_ITEM_INDEX:
                self._host.quit()

    def render(self) -> None:
        if self._view is None:
            self._view = MenuView(self._host.context)
        self._view.render(self.presenter)
