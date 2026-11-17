"""State and collision rules for the three-lane runner."""

from dataclasses import dataclass, field
from enum import Enum, auto
from random import Random
from typing import Final

LANE_WIDTH: Final[float] = 2.4
ACTION_DURATION: Final[float] = 0.85


class Motion(Enum):
    RUN = auto()
    JUMP = auto()
    SLIDE = auto()


class Hazard(Enum):
    HURDLE = auto()
    OVERHEAD_BAR = auto()
    WALL = auto()


OBSTACLE_STYLES: Final[dict[Hazard, tuple[str, ...]]] = {
    Hazard.HURDLE: ("railing_straight_padded", "railing_straight_double", "barrier_2x1x1"),
    Hazard.OVERHEAD_BAR: ("arch_wide", "arch", "arch_tall"),
    Hazard.WALL: ("barrier_2x1x4", "platform_decorative_2x2x2", "barrier_2x1x2"),
}


@dataclass(frozen=True)
class Obstacle:
    lane: int
    distance: float
    kind: Hazard
    style_index: int = 0


@dataclass
class RunnerModel:
    random: Random = field(default_factory=Random)
    lane: int = 0
    lateral_position: float = 0.0
    distance: float = 0.0
    motion: Motion = Motion.RUN
    motion_time: float = 0.0
    crashed: bool = False
    obstacles: list[Obstacle] = field(default_factory=list[Obstacle])
    next_row: float = 24.0

    def __post_init__(self) -> None:
        self._extend_track()

    @property
    def speed(self) -> float:
        return min(19.0, 8.0 + self.distance / 150.0)

    @property
    def height(self) -> float:
        if self.motion is Motion.JUMP:
            fraction: float = self.motion_time / ACTION_DURATION
            return 6.0 * fraction * (1.0 - fraction)
        return 0.0

    def change_lane(self, direction: int) -> None:
        if not self.crashed:
            self.lane = max(-1, min(1, self.lane + direction))

    def begin_motion(self, motion: Motion) -> None:
        if not self.crashed and self.motion is Motion.RUN:
            self.motion = motion
            self.motion_time = 0.0

    def advance(self, seconds: float) -> None:
        if self.crashed:
            return
        self.distance += self.speed * seconds
        self.motion_time += seconds
        if self.motion is not Motion.RUN and self.motion_time >= ACTION_DURATION:
            self.motion = Motion.RUN
            self.motion_time = 0.0
        difference: float = self.lane * LANE_WIDTH - self.lateral_position
        maximum_step: float = 10.0 * seconds
        self.lateral_position += max(-maximum_step, min(maximum_step, difference))
        obstacle: Obstacle
        for obstacle in self.obstacles:
            if abs(obstacle.distance - self.distance) > 0.55:
                continue
            if abs(self.lateral_position - obstacle.lane * LANE_WIDTH) > 1.1:
                continue
            if obstacle.kind is Hazard.HURDLE and self.height > 0.55:
                continue
            if obstacle.kind is Hazard.OVERHEAD_BAR and self.motion is Motion.SLIDE:
                continue
            self.crashed = True
            return
        self.obstacles = [item for item in self.obstacles if item.distance >= self.distance - 5]
        self._extend_track()

    def _extend_track(self) -> None:
        while self.next_row < self.distance + 90.0:
            lane: int
            for lane in self.random.sample((-1, 0, 1), self.random.randint(1, 2)):
                kind: Hazard = self.random.choice(tuple(Hazard))
                style_index: int = self.random.randrange(len(OBSTACLE_STYLES[kind]))
                self.obstacles.append(Obstacle(lane, self.next_row, kind, style_index))
            self.next_row += 22.0
