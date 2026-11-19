"""Persistent player profile: best score, run history, and settings.

The profile lives in ``data/profile.json`` next to the project root and is
reloaded on startup; every mutation is written back immediately.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, TypedDict, cast

DEFAULT_CHARACTER: Final[str] = "Knight.glb"
DEFAULT_ENVIRONMENT: Final[str] = "day"
TIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


class RunPayload(TypedDict):
    """JSON shape of one stored run record."""

    time: str
    score: int
    character: str
    duration_s: float


class ProfilePayload(TypedDict):
    """JSON shape of the whole profile file."""

    best_score: int
    runs: list[RunPayload]
    settings: dict[str, str]


@dataclass(frozen=True)
class RunRecord:
    """One finished run as shown in the history screen."""

    time: str
    score: int
    character: str
    duration_s: float


@dataclass(frozen=True)
class ProfileSettings:
    """Persisted user preferences."""

    character: str = DEFAULT_CHARACTER
    environment: str = DEFAULT_ENVIRONMENT


class ProfileStore:
    """Loads, mutates, and saves the player profile."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.best_score: int = 0
        self.runs: list[RunRecord] = []
        self.settings: ProfileSettings = ProfileSettings()
        if path.is_file():
            self._load()

    def record_run(self, score: int, character: str, duration_s: float) -> bool:
        """Append a finished run, persist, and return whether it is a new best."""
        record: RunRecord = RunRecord(
            time=datetime.now().strftime(TIME_FORMAT),
            score=score,
            character=character,
            duration_s=round(duration_s, 1),
        )
        self.runs.append(record)
        is_new_best: bool = score > self.best_score
        if is_new_best:
            self.best_score = score
        self._save()
        return is_new_best

    def update_settings(
        self, *, character: str | None = None, environment: str | None = None
    ) -> None:
        """Replace individual settings and persist the profile."""
        self.settings = ProfileSettings(
            character=self.settings.character if character is None else character,
            environment=self.settings.environment if environment is None else environment,
        )
        self._save()

    def _load(self) -> None:
        """Read the profile file, falling back to defaults on any error."""
        try:
            payload: ProfilePayload = cast(
                ProfilePayload, json.loads(self.path.read_text(encoding="utf-8"))
            )
            self.best_score = int(payload["best_score"])
            self.runs = [
                RunRecord(
                    time=str(item["time"]),
                    score=int(item["score"]),
                    character=str(item["character"]),
                    duration_s=float(item["duration_s"]),
                )
                for item in payload["runs"]
            ]
            settings: dict[str, str] = payload["settings"]
            self.settings = ProfileSettings(
                character=settings.get("character", DEFAULT_CHARACTER),
                environment=settings.get("environment", DEFAULT_ENVIRONMENT),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.best_score = 0
            self.runs = []
            self.settings = ProfileSettings()

    def _save(self) -> None:
        """Write the whole profile to disk as pretty-printed JSON."""
        payload: ProfilePayload = {
            "best_score": self.best_score,
            "runs": [cast(RunPayload, asdict(record)) for record in self.runs],
            "settings": asdict(self.settings),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
