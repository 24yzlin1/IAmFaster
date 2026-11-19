"""Application lifecycle: window setup, input backends, and the scene loop."""

from contextlib import ExitStack
from pathlib import Path
from typing import Final

import moderngl
import pygame

from game.input_backends.actions import BACKEND_CHOICES, BACKEND_NAMES_ENABLING_GESTURE, GameAction, InputBackend
from game.input_backends.keyboard import KeyboardInputBackend
from game.models.characters import resolve_character
from game.models.environment import resolve_environment
from game.models.profile import ProfileStore
from game.scenes.base import Scene
from game.scenes.menu import MenuScene
from game.views.runner import WINDOW_SIZE, RunnerView

DEFAULT_MODEL_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "gesture_model/gesture_model.pth"
PROFILE_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "data/profile.json"
GESTURE_ACTIONS: Final[dict[str, GameAction]] = {
    "Left": GameAction.LEFT, "Right": GameAction.RIGHT,
    "Up": GameAction.JUMP, "Down": GameAction.SLIDE, "Fist": GameAction.RESET,
}


class GameApp:
    """Owns the window, input backends, and the active scene."""

    def __init__(self, model_path: Path, backend_name: str, camera_name: str | None) -> None:
        if backend_name not in BACKEND_CHOICES:
            raise ValueError(f"unknown input backend: {backend_name!r}")
        self._model_path: Path = model_path
        self._backend_name: str = backend_name
        self._camera_name: str | None = camera_name
        self._current_scene: Scene | None = None
        self._running: bool = False
        self.context: moderngl.Context
        self._runner_view: RunnerView | None = None
        self._runner_view_character: str | None = None
        self._runner_view_environment: str | None = None
        self.profile_store: ProfileStore = ProfileStore(PROFILE_PATH)

    def run(self) -> None:
        """Open the window and loop until the active scene quits."""
        with ExitStack() as resources:
            backends: list[InputBackend] = [KeyboardInputBackend(enable_game_keys=self._backend_name != "gesture")]
            if self._backend_name in BACKEND_NAMES_ENABLING_GESTURE:
                if not self._model_path.is_file():
                    raise FileNotFoundError(f"gesture model not found: {self._model_path}")
                from game.input_backends.gesture import GestureInputBackend

                gestures: GestureInputBackend = GestureInputBackend(
                    self._model_path, GESTURE_ACTIONS, camera_name=self._camera_name,
                    required_class_names=frozenset(("Left", "Right", "Up", "Down")),
                )
                resources.callback(gestures.close)
                backends.append(gestures)
            pygame.display.init()
            resources.callback(pygame.quit)
            pygame.font.init()
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.set_mode(WINDOW_SIZE, pygame.DOUBLEBUF | pygame.OPENGL)
            pygame.display.set_caption("I Am Faster")
            context: moderngl.Context = moderngl.create_context()
            resources.callback(context.release)
            self.context = context
            self._current_scene = MenuScene(self)
            self._running = True
            clock: pygame.time.Clock = pygame.time.Clock()
            while self._running:
                elapsed: float = min(clock.tick(60) / 1000.0, 0.1)
                backend: InputBackend
                action: GameAction
                for backend in backends:
                    for action in backend.poll():
                        self._current_scene.handle_action(action)
                if not self._running:
                    break
                while elapsed > 0:
                    step: float = min(elapsed, 1 / 120)
                    self._current_scene.advance(step)
                    elapsed -= step
                self._current_scene.render()

    def switch_scene(self, scene: Scene) -> None:
        """Replace the currently active scene."""
        self._current_scene = scene

    def runner_view(self) -> RunnerView:
        """Return the cached runner game view for the selected character and environment."""
        character: str = self.profile_store.settings.character
        environment: str = self.profile_store.settings.environment
        if (
            self._runner_view is None
            or self._runner_view_character != character
            or self._runner_view_environment != environment
        ):
            self._runner_view = RunnerView(
                self.context, resolve_character(character), resolve_environment(environment)
            )
            self._runner_view_character = character
            self._runner_view_environment = environment
        return self._runner_view

    def quit(self) -> None:
        """Leave the main loop and close the application."""
        self._running = False


def run(model_path: Path, backend_name: str, camera_name: str | None) -> None:
    """Create a GameApp and run it to completion."""
    GameApp(model_path, backend_name, camera_name).run()
