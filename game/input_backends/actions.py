"""Shared game actions, backend names, and the input backend protocol."""

from enum import Enum, auto
from typing import Final, Protocol

BACKEND_CHOICES: Final[tuple[str, ...]] = ("keyboard", "gesture", "both")
DEFAULT_BACKEND_NAME: Final[str] = "keyboard"
BACKEND_NAMES_ENABLING_KEYBOARD: Final[frozenset[str]] = frozenset(
    {"keyboard", "both"}
)
BACKEND_NAMES_ENABLING_GESTURE: Final[frozenset[str]] = frozenset(
    {"gesture", "both"}
)


class GameAction(Enum):
    """Device-independent game actions dispatched to the presenter."""

    LEFT = auto()
    RIGHT = auto()
    JUMP = auto()
    SLIDE = auto()
    INCREMENT = auto()
    RESET = auto()
    BACK = auto()
    QUIT = auto()


class InputBackend(Protocol):
    """A source of game actions polled once per frame."""

    def poll(self) -> list[GameAction]:
        """Return the actions produced since the previous poll."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...
