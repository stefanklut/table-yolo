import argparse
from pathlib import Path

from ultralytics import YOLO

from data.convert_pubtabnet_to_yolo import PubTabNetToYOLO
from utils.tempdir import OptionalTemporaryDirectory


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PubTabNet to YOLO format")
    parser.add_argument("--pubtabnet_path", type=str, required=True, help="Path to PubTabNet JSONL file")
    parser.add_argument("--yolo_base", type=str, default="yolo11n.pt", help="YOLO base model")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default="0", help="Device")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-i", "--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()


def main(args: argparse.Namespace):
    with OptionalTemporaryDirectory() as tmp_dir:
        pubtabnet_to_yolo = PubTabNetToYOLO(args.pubtabnet_path, tmp_dir)
        pubtabnet_to_yolo.convert()

        yolo_data_path = Path(tmp_dir).joinpath("yolo.yaml")

        model = YOLO(args.yolo_base)
        model.train(
            data=str(yolo_data_path),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
        )


if __name__ == "__main__":
    args = get_arguments()
    main(args)
