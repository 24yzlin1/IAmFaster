"""The run history screen listing past runs and the best score."""

from game.input_backends.actions import GameAction
from game.scenes.base import Scene, SceneHost
from game.views.history import RunHistoryView


class RunHistoryScene(Scene):
    """Shows the recorded runs until the player goes back to the menu."""

    def __init__(self, host: SceneHost) -> None:
        super().__init__(host)
        self._view: RunHistoryView | None = None

    def handle_action(self, action: GameAction) -> None:
        if action is GameAction.QUIT:
            super().handle_action(action)
        elif action in (GameAction.BACK, GameAction.RESET):
            from game.scenes.menu import MenuScene

            self._host.switch_scene(MenuScene(self._host))

    def render(self) -> None:
        if self._view is None:
            self._view = RunHistoryView(self._host.context)
        self._view.render(self._host.profile_store)
