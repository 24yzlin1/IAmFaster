from contextlib import ExitStack
from pathlib import Path
from typing import Final

import moderngl
import pygame

from game.input_backends.actions import BACKEND_CHOICES, BACKEND_NAMES_ENABLING_GESTURE, GameAction, InputBackend
from game.input_backends.keyboard import KeyboardInputBackend
from game.presenters.runner import RunnerPresenter
from game.views.runner import WINDOW_SIZE, RunnerView

DEFAULT_MODEL_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "gesture_model/gesture_model.pth"
GESTURE_ACTIONS: Final[dict[str, GameAction]] = {
    "Left": GameAction.LEFT, "Right": GameAction.RIGHT,
    "Up": GameAction.JUMP, "Down": GameAction.SLIDE, "Fist": GameAction.RESET,
}


def run(model_path: Path, backend_name: str, camera_name: str | None) -> None:
    if backend_name not in BACKEND_CHOICES:
        raise ValueError(f"unknown input backend: {backend_name!r}")
    with ExitStack() as resources:
        backends: list[InputBackend] = [KeyboardInputBackend(enable_game_keys=backend_name != "gesture")]
        if backend_name in BACKEND_NAMES_ENABLING_GESTURE:
            if not model_path.is_file():
                raise FileNotFoundError(f"gesture model not found: {model_path}")
            from game.input_backends.gesture import GestureInputBackend

            gestures: GestureInputBackend = GestureInputBackend(
                model_path, GESTURE_ACTIONS, camera_name=camera_name,
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
        view: RunnerView = RunnerView(context)
        presenter: RunnerPresenter = RunnerPresenter()
        clock: pygame.time.Clock = pygame.time.Clock()
        while presenter.running:
            elapsed: float = min(clock.tick(60) / 1000.0, 0.1)
            backend: InputBackend
            action: GameAction
            for backend in backends:
                for action in backend.poll():
                    presenter.handle_action(action)
            if not presenter.running:
                break
            while elapsed > 0:
                step: float = min(elapsed, 1 / 120)
                presenter.model.advance(step)
                elapsed -= step
            view.render(presenter.model)
