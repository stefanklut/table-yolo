import json
import sys
from pathlib import Path

import imagesize
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy_utils import copy_mode


class PubTabNetToYOLO:
    def __init__(self, pubtabnet_jsonl_path, output_dir):
        self.pubtabnet_jsonl_path = Path(pubtabnet_jsonl_path).absolute()
        assert self.pubtabnet_jsonl_path.exists(), f"File not found: {self.pubtabnet_jsonl_path}"
        assert self.pubtabnet_jsonl_path.suffix == ".jsonl", f"Invalid file format: {self.pubtabnet_jsonl_path}"

        self.pubtabnet_dir = self.pubtabnet_jsonl_path.parent
        assert self.pubtabnet_dir.joinpath("train").exists(), f"Directory not found: {self.pubtabnet_dir.joinpath('train')}"
        assert self.pubtabnet_dir.joinpath("val").exists(), f"Directory not found: {self.pubtabnet_dir.joinpath('val')}"

        self.output_dir = Path(output_dir)

    def convert_single_line(self, line):
        data = json.loads(line)
        image_id = data["imgid"]
        filename = data["filename"]
        split = data["split"]

        cells = data["html"]["cells"]
        bbox_list = []
        for cell in cells:
            bbox = cell.get("bbox")
            if bbox is None:
                continue
            bbox_list.append(bbox)

        return {"image_id": image_id, "filename": filename, "split": split, "bbox_list": bbox_list}

    @staticmethod
    def _bounding_box_center(bbox: list[int | float]) -> list[float]:
        assert len(bbox) == 4, f"Invalid bounding box: {bbox}"

        min_x = bbox[0]
        min_y = bbox[1]
        max_x = bbox[2]
        max_y = bbox[3]

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y

        bbox_output = np.asarray([center_x, center_y, width, height]).astype(np.float32).tolist()
        return bbox_output

    @staticmethod
    def _normalize_coords(bbox: list[int], size: tuple[int, int]) -> list[float]:
        """
        Normalize coordinates to a new size

        Args:
            coords (np.ndarray): the coordinates to normalize
            size (tuple[int, int]): the size of the output image

        Returns:
            list[float]: the normalized coordinates
        """
        assert len(bbox) == 4, f"Invalid bounding box: {bbox}"
        assert len(size) == 2, f"Invalid size: {size}"

        height, width = size
        min_x = bbox[0] / (width - 1)
        min_y = bbox[1] / (height - 1)
        max_x = bbox[2] / (width - 1)
        max_y = bbox[3] / (height - 1)

        return [min_x, min_y, max_x, max_y]

    def read_jsonl(self):
        with open(self.pubtabnet_jsonl_path, "r") as f:
            for line in f:
                yield self.convert_single_line(line)

    def convert(self):
        for data in self.read_jsonl():
            image_id = data["image_id"]
            filename = data["filename"]
            split = data["split"]
            bbox_list = data["bbox_list"]

            self.output_dir.joinpath("images", split).mkdir(parents=True, exist_ok=True)
            self.output_dir.joinpath("labels", split).mkdir(parents=True, exist_ok=True)

            images_input_path = self.pubtabnet_dir.joinpath(split, filename)

            # Get image size
            width, height = imagesize.get(images_input_path)

            # Write to YOLO format
            images_path = self.output_dir.joinpath("images", split, filename)
            labels_path = self.output_dir.joinpath("labels", split, images_path.stem + ".txt")

            copy_mode(path=images_input_path, destination=images_path, mode="symlink")

            with open(labels_path, "w") as f:
                for bbox in bbox_list:
                    bbox = self._normalize_coords(bbox, (height, width))
                    bbox_output = self._bounding_box_center(bbox)
                    bbox_output = " ".join(map(str, bbox_output))
                    f.write(f"0 {bbox_output}\n")

        info_yaml = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": ["table_cell"],
        }

        with open(self.output_dir.joinpath("yolo.yaml"), "w") as f:
            yaml.dump(info_yaml, f)


if __name__ == "__main__":
    pubtabnet_jsonl_path = Path("/home/stefan/Documents/datasets/pubtabnet/PubTabNet_2.0.0.jsonl")
    output_dir = Path("/tmp/pubtabnet_yolo")

    converter = PubTabNetToYOLO(pubtabnet_jsonl_path=pubtabnet_jsonl_path, output_dir=output_dir)
    converter.convert()
