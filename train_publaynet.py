import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import (
    CFG_BOOL_KEYS,
    CFG_FLOAT_KEYS,
    CFG_FRACTION_KEYS,
    CFG_INT_KEYS,
    DEFAULT_CFG_DICT,
)

from data.convert_publaynet_to_yolo import PubLayNetToYOLO
from utils.tempdir import OptionalTemporaryDirectory


def bool_arg(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False


def fractional_arg(v):
    if "/" in v:
        num, den = v.split("/")
        try:
            num = int(num)
            den = int(den)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{v} is not a fraction")
        if den == 0:
            raise argparse.ArgumentTypeError("Denominator cannot be zero")
        return num / den
    try:
        v = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v} is not a float")

    if 0 <= v <= 1:
        return v
    else:
        raise argparse.ArgumentTypeError(f"{v} is not a fraction between 0 and 1")


def add_cfg_arguments(parser: argparse.ArgumentParser):
    parser.add_argument_group("YOLO Configuration")
    for key in CFG_BOOL_KEYS:
        parser.add_argument(f"--{key}", type=bool_arg, help=f"Set {key} to true/false")
    for key in CFG_INT_KEYS:
        parser.add_argument(f"--{key}", type=int, help=f"Set {key} to an integer")
    for key in CFG_FLOAT_KEYS:
        if key == "batch":
            parser.add_argument(f"--{key}", type=int, help=f"Set {key} to an integer")
            continue
        parser.add_argument(f"--{key}", type=float, help=f"Set {key} to a float")
    for key in CFG_FRACTION_KEYS:
        parser.add_argument(f"--{key}", type=fractional_arg, help=f"Set {key} to a fraction between 0 and 1")
    for key, value in DEFAULT_CFG_DICT.items():
        if key in CFG_BOOL_KEYS or key in CFG_INT_KEYS or key in CFG_FLOAT_KEYS or key in CFG_FRACTION_KEYS:
            continue
        if value is None:
            parser.add_argument(f"--{key}", type=str, help=f"Set {key} to a string")
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=bool_arg, help=f"Set {key} to true/false")
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, help=f"Set {key} to an integer")
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, help=f"Set {key} to a float")
        elif isinstance(value, str):
            parser.add_argument(f"--{key}", type=str, help=f"Set {key} to a string")
        else:
            raise ValueError(f"Unsupported type for {key}: {type(value)}")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PubTabNet to YOLO format")
    parser.add_argument("--publaynet_path", type=str, required=True, help="Path to PubTabNet JSONL file")
    parser.add_argument("--yolo_base", type=str, default="yolo11n.pt", help="YOLO base model")

    add_cfg_arguments(parser)

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    with OptionalTemporaryDirectory() as tmp_dir:
        publaynet_to_yolo = PubLayNetToYOLO(args.pubtabnet_path, tmp_dir)
        publaynet_to_yolo.convert()

        yolo_data_path = Path(tmp_dir).joinpath("yolo.yaml")

        model = YOLO(args.yolo_base)

        kwargs = vars(args)
        # Remove non-YOLO arguments
        kwargs.pop("pubtabnet_path")
        kwargs.pop("yolo_base")

        kwargs["data"] = str(yolo_data_path)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        model.train(**kwargs)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
