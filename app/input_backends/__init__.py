"""Input backend package translating device events into game actions.

Public API:
- GameAction / InputBackend: the device-independent action vocabulary.
- KeyboardInputBackend / GestureInputBackend: concrete input backends.
- Backend name constants used by startup arguments.
"""

from app.input_backends.actions import (
    BACKEND_CHOICES,
    BACKEND_NAMES_ENABLING_GESTURE,
    BACKEND_NAMES_ENABLING_KEYBOARD,
    DEFAULT_BACKEND_NAME,
    GameAction,
    InputBackend,
)
from app.input_backends.gesture import GestureInputBackend
from app.input_backends.keyboard import KeyboardInputBackend

__all__: list[str] = [
    "BACKEND_CHOICES",
    "BACKEND_NAMES_ENABLING_GESTURE",
    "BACKEND_NAMES_ENABLING_KEYBOARD",
    "DEFAULT_BACKEND_NAME",
    "GameAction",
    "GestureInputBackend",
    "InputBackend",
    "KeyboardInputBackend",
]
