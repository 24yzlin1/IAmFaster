"""Convert an image-based gesture dataset into landmark feature files.

Every image in ``<input>/<class_name>/*.jpg`` is run through the MediaPipe
hand landmarker, and the resulting normalized landmark vector is saved as
``<output>/<class_name>/<stem>.npy``. Images without a detectable hand are
skipped and reported, so existing collections can be reused without
re-recording anything.
"""

import argparse
from pathlib import Path

import cv2
import numpy
from mediapipe.tasks.python import vision

from utils.common import EXTENSIONS
from utils.hand_detector import HandDetector


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the conversion script."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert gesture images to landmark feature files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset"),
        help="Dataset root containing one subdirectory per gesture class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_landmark"),
        help="Output root that receives one .npy feature file per image.",
    )
    return parser.parse_args()


def convert_dataset(input_root: Path, output_root: Path) -> int:
    """Convert every class directory under *input_root* into feature files.

    Returns the total number of written feature files.
    """
    detector: HandDetector = HandDetector(vision.RunningMode.IMAGE)
    total_saved: int = 0
    class_directories: list[Path] = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not class_directories:
        raise RuntimeError(
            "No class directories found under {}. Expected layout: "
            "root/class_name/*.jpg".format(input_root)
        )
    for class_directory in class_directories:
        saved_count: int = 0
        failed_count: int = 0
        output_directory: Path = output_root / class_directory.name
        output_directory.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(class_directory.iterdir()):
            if not (file_path.is_file() and file_path.suffix.lower() in EXTENSIONS):
                continue
            image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if image is None:
                print("Cannot read image, skipped: {}".format(file_path))
                failed_count += 1
                continue
            feature_vector = detector.detect_landmark_vector(image)
            if feature_vector is None:
                print("No hand detected, skipped: {}".format(file_path))
                failed_count += 1
                continue
            numpy.save(output_directory / (file_path.stem + ".npy"), feature_vector)
            saved_count += 1
        print(
            "Class '{}': {} converted, {} skipped".format(
                class_directory.name, saved_count, failed_count
            )
        )
        total_saved += saved_count
    detector.close()
    return total_saved


def main() -> None:
    """Run the dataset conversion with the configured arguments."""
    arguments: argparse.Namespace = parse_arguments()
    total_saved: int = convert_dataset(arguments.input, arguments.output)
    print(
        "Landmark dataset written to {} ({} vectors).".format(
            arguments.output, total_saved
        )
    )


if __name__ == "__main__":
    main()
