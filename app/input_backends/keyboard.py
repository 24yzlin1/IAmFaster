"""Keyboard and mouse input backend built on pygame events."""

from typing import Final

import pygame

from app.input_backends.actions import GameAction

QUIT_KEY: Final[int] = pygame.K_ESCAPE
RESET_KEY: Final[int] = pygame.K_r
INCREMENT_MOUSE_BUTTON: Final[int] = 1


class KeyboardInputBackend:
    """Translates pygame keyboard and mouse events into game actions."""

    def __init__(self) -> None:
        self._key_actions: dict[int, GameAction] = {
            QUIT_KEY: GameAction.QUIT,
            RESET_KEY: GameAction.RESET,
        }

    def poll(self) -> list[GameAction]:
        actions: list[GameAction] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append(GameAction.QUIT)
            elif event.type == pygame.KEYDOWN:
                key_action = self._key_actions.get(event.key)
                if key_action is not None:
                    actions.append(key_action)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == INCREMENT_MOUSE_BUTTON:
                    actions.append(GameAction.INCREMENT)
        return actions

    def close(self) -> None:
        return None
