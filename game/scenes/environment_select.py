"""The environment selection screen choosing between day and night."""

from typing import Final

from game.input_backends.actions import GameAction
from game.models.environment import (
    PRESET_CHOICES, EnvironmentPreset, environment_display_name, resolve_environment,
)
from game.presenters.menu import MenuPresenter
from game.scenes.base import Scene, SceneHost
from game.views.environment_select import EnvironmentSelectView

ITEM_LABELS: Final[tuple[str, ...]] = tuple(
    environment_display_name(preset) for preset in PRESET_CHOICES
)


class EnvironmentSelectScene(Scene):
    """Lets the player pick the track environment and saves it to the profile."""

    def __init__(self, host: SceneHost) -> None:
        super().__init__(host)
        self.presenter: MenuPresenter = MenuPresenter(ITEM_LABELS)
        current: str = host.profile_store.settings.environment
        self._presets: dict[str, EnvironmentPreset] = {
            environment_display_name(preset): preset for preset in PRESET_CHOICES
        }
        label: str
        for label in ITEM_LABELS:
            if self._presets[label].name == current:
                self.presenter.selected_index = ITEM_LABELS.index(label)
        self._view: EnvironmentSelectView | None = None

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
            preset: EnvironmentPreset = self._presets[self.presenter.selected_item]
            self._host.profile_store.update_settings(environment=preset.name)
            self._go_to_menu()

    def _go_to_menu(self) -> None:
        """Switch back to the main menu, importing it lazily to break a cycle."""
        from game.scenes.menu import MenuScene

        self._host.switch_scene(MenuScene(self._host))

    def render(self) -> None:
        if self._view is None:
            self._view = EnvironmentSelectView(self._host.context)
        current: EnvironmentPreset = resolve_environment(
            self._host.profile_store.settings.environment
        )
        self._view.render(self.presenter, environment_display_name(current))
