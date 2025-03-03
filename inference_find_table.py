import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from tqdm import tqdm
from ultralytics import YOLO

from utils.image_utils import load_image_array_from_path
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.logging_utils import get_logger_name


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file to inference using the model found in the config file")

    yolo_args = parser.add_argument_group("YOLO")
    yolo_args.add_argument("--yolo", help="yolo model", type=str, default="yolo11n.pt")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input folder",
        type=str,
        action="extend",
        required=True,
    )
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)

    dataloader_args = parser.add_argument_group("Dataloader")
    dataloader_args.add_argument("--num_workers", help="Number of workers to use", type=int, default=4)

    args = parser.parse_args()

    return args


class Predictor:
    """
    Predictor runs the model specified in the config, on call the image is processed and the results dict is output
    """

    def __init__(self, yolo_model: str = "yolo11n.pt"):
        """
        Predictor runs the model specified in the config, on call the image is processed and the results dict is output

        Args:
            cfg (CfgNode): config
        """
        self.model = YOLO(yolo_model)
        self.model.eval()

    def __call__(self, image):
        """
        Run the model on the image

        Args:
            image (np.ndarray): image to run the model on

        Returns:
            dict: results of the model
        """
        with torch.no_grad():
            return self.model(image)


class LoadingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        # TODO Move resize and load to this part of the dataloader
        data = load_image_array_from_path(path)
        if data is None:
            return None, None, path
        image = data["image"]
        dpi = data["dpi"]
        return image, dpi, path


def collate_numpy(batch):
    collate_map = default_collate_fn_map

    def new_map(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
    ):
        return batch

    collate_map.update({np.ndarray: new_map, type(None): new_map})
    return collate(batch, collate_fn_map=collate_map)


class SavePredictor(Predictor):
    """
    Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML
    """

    def __init__(
        self,
        yolo_model: str,
        input_paths: str | Path | Sequence[str | Path],
        output_dir: str | Path,
        num_workers: int = 4,
    ):
        """
        Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML

        Args:
            cfg (CfgNode): config
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images
            output_dir (str | Path): path to output dir
            output_page (OutputPageXML): output pageXML object
            num_workers (int): number of workers to use

        """
        super().__init__(yolo_model=yolo_model)

        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        self.num_workers = num_workers

    def set_input_paths(
        self,
        input_paths: str | Path | Sequence[str | Path],
    ) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        self.input_paths = get_file_paths(input_paths, SUPPORTED_IMAGE_FORMATS)

    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Setter for the output dir

        Args:
            output_dir (str | Path): path to output dir
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            self.logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()

    # def save_prediction(self, input_path: Path | str):
    def save_prediction(self, image: np.ndarray, dpi: int, input_path: Path):
        """
        Run the model on the image and save the results as pageXML

        Args:
            image (np.ndarray): image to run the model on
            dpi (int): dpi of the image
            input_path (Path): path to the image

        Raises:
            TypeError: no input dir is specified
        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        if image is None:
            self.logger.warning(f"Image at {input_path} has not loaded correctly, ignoring for now")
            return

        outputs = self.__call__(image)

        yolo_output = outputs[0]

        classes = yolo_output.names
        bboxes = {name: [] for name in classes}

        boxes = yolo_output.boxes
        for i in range(boxes.shape[0]):
            xyxy = boxes.xyxy[i]
            xyxy = xyxy.cpu().numpy().round().astype(np.int32)

            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = classes[class_id]

            bboxes[class_name].append(xyxy)

        with self.output_dir.joinpath(f"{input_path.stem}.json").open("w") as f:
            json.dump(bboxes, f)

    def process(self):
        """
        Run the model on all images within the input dir

        Raises:
            TypeError: no input dir is specified
            TypeError: no output dir is specified
        """
        if self.input_paths is None:
            raise TypeError("Cannot run when the input_paths is None")
        if self.output_dir is None:
            raise TypeError("Cannot run when the output_dir is None")

        dataset = LoadingDataset(self.input_paths)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_numpy,
        )
        for inputs in tqdm(dataloader, desc="Predicting PageXML"):
            self.save_prediction(inputs[0], inputs[1], inputs[2])


def main(args: argparse.Namespace) -> None:
    predictor = SavePredictor(
        yolo_model=args.yolo,
        input_paths=args.input,
        output_dir=args.output,
        num_workers=args.num_workers,
    )
    predictor.process()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
