from app.input_backends.actions import GameAction
from app.models.runner import Motion, RunnerModel


class RunnerPresenter:
    def __init__(self) -> None:
        self.model: RunnerModel = RunnerModel()
        self.running: bool = True

    def handle_action(self, action: GameAction) -> None:
        if action is GameAction.QUIT:
            self.running = False
        elif action is GameAction.RESET and self.model.crashed:
            self.model = RunnerModel()
        elif action is GameAction.LEFT:
            self.model.change_lane(-1)
        elif action is GameAction.RIGHT:
            self.model.change_lane(1)
        elif action is GameAction.JUMP:
            self.model.begin_motion(Motion.JUMP)
        elif action is GameAction.SLIDE:
            self.model.begin_motion(Motion.SLIDE)
