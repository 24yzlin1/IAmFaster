import argparse
from pathlib import Path

from app.app import DEFAULT_MODEL_PATH, run
from app.input_backends import BACKEND_CHOICES, DEFAULT_BACKEND_NAME


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Run the gesture-controlled counter game."
    )
    argument_parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained gesture model.",
    )
    argument_parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=DEFAULT_BACKEND_NAME,
        help="Input backend selection.",
    )
    return argument_parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    model_path: Path = arguments.model_path
    backend_name: str = arguments.backend
    run(model_path, backend_name)


if __name__ == "__main__":
    main()
