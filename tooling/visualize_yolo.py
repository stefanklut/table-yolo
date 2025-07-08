import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization of prediction/GT of model")

    yolo_args = parser.add_argument_group("YOLO")
    yolo_args.add_argument("--yolo_model", type=str, required=True, help="Path to YOLO model")

    io_args = parser.add_argument_group("IO")
    # io_args.add_argument("-t", "--train", help="Train input folder/file",
    #                         nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-o", "--output", help="Output folder", type=str)

    parser.add_argument("--save", nargs="?", const="all", default=None, help="Save images instead of displaying")

    args = parser.parse_args()

    return args


def main(args):
    # Load a model
    model = YOLO(args.yolo_model)

    # Run batched inference on a list of images
    input_paths = get_file_paths(args.input, SUPPORTED_IMAGE_FORMATS)

    output_dir = None
    if args.output is not None:
        output_dir = Path(args.output)
        if args.save is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

    # Process results list
    for path in input_paths:
        result = model(path)[0]
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        if output_dir is not None:
            filepath = output_dir.joinpath(f"{Path(result.path).stem}_result.jpg")
            result.save(filename=filepath)
        else:
            result.show()  # display to screen


if __name__ == "__main__":
    args = get_arguments()
    main(args)
