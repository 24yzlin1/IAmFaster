"""Locate the playable character models shipped with the game."""

from pathlib import Path
from typing import Final

CHARACTER_DIRECTORY: Final[Path] = (
    Path(__file__).resolve().parents[2] / "assets/KayKit_Adventurers_2.0_FREE/Characters/gltf"
)
DEFAULT_CHARACTER: Final[str] = "Knight.glb"


def character_names() -> tuple[str, ...]:
    """Return the sorted file names of all available character models."""
    return tuple(sorted(path.name for path in CHARACTER_DIRECTORY.glob("*.glb")))


def resolve_character(name: str) -> Path:
    """Return a usable character model path, falling back to a default.

    Stale settings may name a model that no longer exists; in that case the
    default character is used, or the first available one if absent.
    """
    candidate: Path = CHARACTER_DIRECTORY / name
    if candidate.is_file():
        return candidate
    names: tuple[str, ...] = character_names()
    if not names:
        raise FileNotFoundError(f"no character models found in {CHARACTER_DIRECTORY}")
    if DEFAULT_CHARACTER in names:
        return CHARACTER_DIRECTORY / DEFAULT_CHARACTER
    return CHARACTER_DIRECTORY / names[0]
