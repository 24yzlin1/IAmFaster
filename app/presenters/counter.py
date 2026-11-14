"""Presenter that executes game actions against the counter model."""

from app.input_backends import GameAction
from app.models.counter import CounterModel


class CounterPresenter:
    def __init__(self, model: CounterModel) -> None:
        self._model: CounterModel = model

    def handle_action(self, action: GameAction) -> None:
        if action == GameAction.INCREMENT:
            self._model.count += 1
        elif action == GameAction.RESET:
            self._model.count = 0
        elif action == GameAction.QUIT:
            self._model.is_running = False
