"""Day and night visual presets for the runner track."""

from dataclasses import dataclass
from typing import Final

type Triple = tuple[float, float, float]


@dataclass(frozen=True)
class EnvironmentPreset:
    """Colors used for the sky, the track surface, and the lane markers."""

    name: str
    sky: Triple
    ground: Triple
    marker: Triple


DAY: Final[EnvironmentPreset] = EnvironmentPreset(
    name="day", sky=(0.54, 0.75, 0.84), ground=(0.24, 0.32, 0.36), marker=(0.85, 0.84, 0.65),
)
NIGHT: Final[EnvironmentPreset] = EnvironmentPreset(
    name="night", sky=(0.05, 0.06, 0.12), ground=(0.10, 0.11, 0.15), marker=(0.55, 0.55, 0.45),
)
PRESET_CHOICES: Final[tuple[EnvironmentPreset, ...]] = (DAY, NIGHT)
PRESETS: Final[dict[str, EnvironmentPreset]] = {preset.name: preset for preset in PRESET_CHOICES}


def resolve_environment(name: str) -> EnvironmentPreset:
    """Return the named preset, falling back to the day preset."""
    return PRESETS.get(name, DAY)


def environment_display_name(preset: EnvironmentPreset) -> str:
    """Return the label shown in menus for a preset."""
    return "Day" if preset.name == "day" else "Night"
