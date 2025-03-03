import json
import sys
from collections import defaultdict
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy_utils import copy_mode


class PubLayNetToYOLO:
    def __init__(self, publaynet_path, output_dir):
        self.publaynet_path = Path(publaynet_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.name_to_id = {}
        self.id_to_name = {}

    def convert_json(self, json_path: Path):
        with open(json_path, "r") as f:
            publaynet_json = json.load(f)

        split = json_path.stem

        data = defaultdict(dict)

        if not self.name_to_id or not self.id_to_name:
            for category in publaynet_json["categories"]:
                name = category["name"]
                _id = category["id"]
                self.name_to_id[name] = _id
                self.id_to_name[_id] = name
        else:
            for category in publaynet_json["categories"]:
                name = category["name"]
                _id = category["id"]
                if name not in self.name_to_id:
                    raise ValueError(f"Category name {name} not found in name_to_id")
                if _id not in self.id_to_name:
                    raise ValueError(f"Category id {_id} not found in id_to_name")

        for image in publaynet_json["images"]:
            # HACK: There are some images that contain annotations remove them for now
            if len(image.keys()) > 4:
                image.pop("annotations")
                image.pop("corrected")

            image_id = image["id"]
            image.pop("id")
            data[image_id].update(image)
            data[image_id]["annotations"] = []
            data[image_id]["split"] = split

        for annotation in publaynet_json["annotations"]:
            # HACK: There are some annotations that contain modified remove them for now
            if len(annotation.keys()) > 7:
                annotation.pop("modified")
            if annotation["image_id"] not in data:
                raise ValueError(f"Image id {annotation['image_id']} not found in data")

            image_id = annotation["image_id"]
            annotation.pop("image_id")

            if annotation["category_id"] not in self.id_to_name:
                raise ValueError(f"Category id {annotation['category_id']} not found in id_to_name")

            bbox = annotation["bbox"]
            height = data[image_id]["height"]
            width = data[image_id]["width"]

            # bbox = [x, y, w, h]
            x, y, w, h = bbox
            if not (0 <= x < width and 0 <= y < height and x + w <= width and y + h <= height):
                continue
                images_input_path = self.publaynet_path.joinpath(split, data[image_id]["file_name"])
                raise ValueError(
                    f"Bounding box {bbox} is out of image bounds for image id {image_id}: {images_input_path}, height {height}, width {width}"
                )

            data[image_id]["annotations"].append(annotation)

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.convert_single_image, data.values()),
                    desc=f"Converting {split} JSON to YOLO",
                    total=len(data),
                )
            )

    @staticmethod
    def _normalize_coords(bbox: list[int | float], size: tuple[int, int]) -> list[float]:
        """
        Normalize coordinates to a new size

        Args:
            coords (np.ndarray): the coordinates to normalize in the format [x, y, w, h]
            size (tuple[int, int]): the size of the output image

        Returns:
            list[float]: the normalized coordinates, with the format [center_x, center_y, width, height]
        """
        assert len(bbox) == 4, f"Invalid bounding box: {bbox}"
        assert len(size) == 2, f"Invalid size: {size}"

        height, width = size
        x, y, w, h = bbox

        new_h = h / height
        new_w = w / width
        center = (x + w / 2, y + h / 2)
        new_x = center[0] / width
        new_y = center[1] / height

        if not (0 <= new_x < 1 and 0 <= new_y < 1 and 0 <= new_w < 1 and 0 <= new_h < 1):
            raise ValueError(f"Bounding box {bbox} is out of image bounds for image size {size}")

        return [new_x, new_y, new_w, new_h]

    def convert_single_image(self, data: dict):
        filename = data["file_name"]
        split = data["split"]

        self.output_dir.joinpath("images", split).mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath("labels", split).mkdir(parents=True, exist_ok=True)

        images_input_path = self.publaynet_path.joinpath(split, filename)

        # Get image size
        height = data["height"]
        width = data["width"]

        # Write to YOLO format
        images_path = self.output_dir.joinpath("images", split, filename)
        labels_path = self.output_dir.joinpath("labels", split, images_path.stem + ".txt")

        copy_mode(path=images_input_path, destination=images_path, mode="symlink")

        with open(labels_path, "w") as f:
            for annotation in data["annotations"]:
                category_id = annotation["category_id"] - 1  # YOLO categories are 0-indexed

                bbox = annotation["bbox"]
                bbox_normalized = self._normalize_coords(bbox, (height, width))

                f.write(f"{category_id} {' '.join(map(str, bbox_normalized))}\n")

    def convert(self):
        val_path = self.publaynet_path.joinpath("val.json")
        self.convert_json(val_path)

        train_path = self.publaynet_path.joinpath("train.json")
        self.convert_json(train_path)

        info_yaml = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": list(self.id_to_name.keys()),
        }

        with open(self.output_dir.joinpath("yolo.yaml"), "w") as f:
            yaml.dump(info_yaml, f)


if __name__ == "__main__":
    publaynet_json_path = Path("/home/stefan/Documents/datasets/publaynet")
    output_dir = Path("/tmp/publaynet_yolo")
    publaynet_to_yolo = PubLayNetToYOLO(publaynet_json_path, output_dir)
    publaynet_to_yolo.convert()
