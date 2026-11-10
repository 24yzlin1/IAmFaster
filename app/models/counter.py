from dataclasses import dataclass


@dataclass
class CounterModel:
    is_running: bool = True
    count: int = 0
