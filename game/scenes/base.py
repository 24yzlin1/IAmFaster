"""Scene base class and host protocol for the application's screen flow."""

from typing import Protocol

import moderngl

from game.input_backends.actions import GameAction
from game.models.profile import ProfileStore
from game.views.runner import RunnerView


class SceneHost(Protocol):
    """Callbacks and shared resources a scene uses via the application."""

    context: moderngl.Context
    """The OpenGL context shared by every view."""

    profile_store: ProfileStore
    """The persistent player profile shared by every scene."""

    def switch_scene(self, scene: "Scene") -> None:
        """Replace the currently active scene."""
        ...

    def quit(self) -> None:
        """Leave the main loop and close the application."""
        ...

    def runner_view(self) -> RunnerView:
        """Return the cached runner game view, creating it on first use."""
        ...


class Scene:
    """Base class for one interactive screen of the game.

    Subclasses pair a presenter (state) with a view (rendering) and
    dispatch polled input actions.  Scenes switch by calling the host.
    """

    def __init__(self, host: SceneHost) -> None:
        self._host: SceneHost = host

    def handle_action(self, action: GameAction) -> None:
        """Dispatch a polled input action; quitting ends the application."""
        if action is GameAction.QUIT:
            self._host.quit()

    def advance(self, seconds: float) -> None:
        """Advance the scene simulation by one fixed sub-step."""

    def render(self) -> None:
        """Draw the current scene state to the window."""
