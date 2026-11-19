"""Keyboard and mouse input backend built on pygame events."""

from typing import Final

import pygame

from game.input_backends.actions import GameAction

QUIT_KEY: Final[int] = pygame.K_ESCAPE
CONFIRM_KEY: Final[int] = pygame.K_RETURN
INCREMENT_MOUSE_BUTTON: Final[int] = 1


class KeyboardInputBackend:
    """Translates pygame keyboard and mouse events into game actions."""

    def __init__(self, enable_game_keys: bool = True) -> None:
        # The escape key means "go back one screen"; only closing the
        # window produces QUIT, which ends the application.  Enter is the
        # keyboard confirm key, matching the fist gesture.
        self._key_actions: dict[int, GameAction] = {
            QUIT_KEY: GameAction.BACK, CONFIRM_KEY: GameAction.RESET,
        }
        if enable_game_keys:
            self._key_actions.update({
                pygame.K_LEFT: GameAction.LEFT,
                pygame.K_RIGHT: GameAction.RIGHT, pygame.K_UP: GameAction.JUMP,
                pygame.K_DOWN: GameAction.SLIDE,
            })

    def poll(self) -> list[GameAction]:
        actions: list[GameAction] = []
        event: pygame.event.Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append(GameAction.QUIT)
            elif event.type == pygame.KEYDOWN:
                key_action: GameAction | None = self._key_actions.get(event.key)
                if key_action is not None:
                    actions.append(key_action)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == INCREMENT_MOUSE_BUTTON:
                    actions.append(GameAction.INCREMENT)
        return actions

    def close(self) -> None:
        return None
