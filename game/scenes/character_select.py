"""The character selection screen."""

from game.input_backends.actions import GameAction
from game.models.characters import character_names
from game.presenters.menu import MenuPresenter
from game.scenes.base import Scene, SceneHost
from game.views.character_select import CharacterSelectView


class CharacterSelectScene(Scene):
    """Lets the player pick a character and saves it to the profile."""

    def __init__(self, host: SceneHost) -> None:
        super().__init__(host)
        names: tuple[str, ...] = character_names()
        if not names:
            raise FileNotFoundError("no character models are available to select")
        self.presenter: MenuPresenter = MenuPresenter(names)
        current: str = host.profile_store.settings.character
        if current in names:
            self.presenter.selected_index = names.index(current)
        self._view: CharacterSelectView | None = None

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
            self._host.profile_store.update_settings(character=self.presenter.selected_item)
            self._go_to_menu()

    def _go_to_menu(self) -> None:
        """Switch back to the main menu, importing it lazily to break a cycle."""
        from game.scenes.menu import MenuScene

        self._host.switch_scene(MenuScene(self._host))

    def render(self) -> None:
        if self._view is None:
            self._view = CharacterSelectView(self._host.context)
        self._view.render(self.presenter, self._host.profile_store.settings.character)
