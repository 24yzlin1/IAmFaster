import argparse
from pathlib import Path

from app.app import DEFAULT_MODEL_PATH, run
from app.input_backends import BACKEND_CHOICES, DEFAULT_BACKEND_NAME


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Run the three-lane runner with keyboard or gesture controls."
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
    argument_parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help="Camera device name for the gesture backend; "
        "defaults to the first available device.",
    )
    return argument_parser.parse_args()


def main() -> None:
    arguments: argparse.Namespace = parse_arguments()
    model_path: Path = arguments.model_path
    backend_name: str = arguments.backend
    camera_name: str | None = arguments.camera_name
    run(model_path, backend_name, camera_name)


if __name__ == "__main__":
    main()
