"""The runner gameplay screen pairing the runner presenter with its view."""

from game.input_backends.actions import GameAction
from game.presenters.runner import RunnerPresenter
from game.scenes.base import Scene, SceneHost
from game.views.runner import RunnerView


class PlayingScene(Scene):
    """Runs the three-lane runner until the player crashes or leaves."""

    def __init__(self, host: SceneHost) -> None:
        super().__init__(host)
        self.presenter: RunnerPresenter = RunnerPresenter()
        self.run_time: float = 0.0
        self._view: RunnerView | None = None

    def handle_action(self, action: GameAction) -> None:
        if action is GameAction.QUIT:
            super().handle_action(action)
        elif action is GameAction.BACK:
            # Scene modules import each other back, so import targets lazily.
            from game.scenes.menu import MenuScene

            self._host.switch_scene(MenuScene(self._host))
        elif action is GameAction.RESET and self.presenter.model.crashed:
            self._finish_run()
        else:
            self.presenter.handle_action(action)

    def advance(self, seconds: float) -> None:
        if not self.presenter.model.crashed:
            self.run_time += seconds
        self.presenter.model.advance(seconds)

    def _finish_run(self) -> None:
        """Record the crashed run and move to the settlement screen."""
        from game.scenes.settlement import SettlementScene

        score: int = int(self.presenter.model.distance)
        character: str = self._host.profile_store.settings.character
        is_new_best: bool = self._host.profile_store.record_run(
            score, character, self.run_time
        )
        self._host.switch_scene(SettlementScene(self._host, score, is_new_best))

    def render(self) -> None:
        if self._view is None:
            self._view = self._host.runner_view()
        self._view.render(self.presenter.model)
