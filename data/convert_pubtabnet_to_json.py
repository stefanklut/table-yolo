import json
import logging
import re
import sys
from collections import Counter, defaultdict

# from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional, TypedDict, override

import imagesize
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy_utils import copy_mode
from utils.logging_utils import get_logger_name


class PubTabNetToJSON:
    def __init__(self, pubtabnet_path, output_dir):
        self.pubtabnet_path = Path(pubtabnet_path)

        self.pubtabnet_jsonl_path = self.pubtabnet_path.joinpath("PubTabNet_2.0.0.jsonl")
        assert self.pubtabnet_jsonl_path.exists(), f"File not found: {self.pubtabnet_jsonl_path}"
        assert self.pubtabnet_jsonl_path.suffix == ".jsonl", f"Invalid file format: {self.pubtabnet_jsonl_path}"

        assert self.pubtabnet_path.joinpath("train").exists(), f"Directory not found: {self.pubtabnet_path.joinpath('train')}"
        assert self.pubtabnet_path.joinpath("val").exists(), f"Directory not found: {self.pubtabnet_path.joinpath('val')}"

        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(get_logger_name())

    def single_line_to_data(self, line):
        data = json.loads(line)
        image_id = data["imgid"]
        filename = data["filename"]
        split = data["split"]

        cells = data["html"]["cells"]

        cell_i = 0
        rows_i = 0
        cols_i = 0
        structure = iter(data["html"]["structure"]["tokens"])
        CellEntry = TypedDict(
            "CellEntry", {"bbox": Optional[list[int]], "text": Optional[list[str]], "columns": set[int], "rows": set[int]}
        )
        cell_data: list[CellEntry] = []

        seen_cells = set()

        cell_cols = None
        cell_rows = None
        for token in structure:
            if token == "<td":
                colspan_match = None
                rowspan_match = None
                while (next_token := next(structure)) != ">":
                    colspan_match = re.match(r" colspan=\"(\d+)\"", next_token)
                    rowspan_match = re.match(r" rowspan=\"(\d+)\"", next_token)
                    if not (colspan_match or rowspan_match):
                        raise ValueError(f"Invalid <td> tag: {next_token}")

                if colspan_match:
                    colspan = int(colspan_match.group(1))
                else:
                    colspan = 1

                if rowspan_match:
                    rowspan = int(rowspan_match.group(1))
                else:
                    rowspan = 1

                cell_cols = set(range(cols_i, cols_i + colspan))
                cell_rows = set(range(rows_i, rows_i + rowspan))

                for row in cell_rows:
                    for col in cell_cols:
                        seen_cells.add((row, col))

                cols_i += colspan

            if token == "<td>":
                while (rows_i, cols_i) in seen_cells:
                    cols_i += 1
                cell_cols = set([cols_i])
                cell_rows = set([rows_i])

                cols_i += 1

            if token == "</td>":
                if not cell_cols or not cell_rows:
                    raise ValueError(f"Invalid cell: {cell_cols}, {cell_rows}")

                cell = cells[cell_i]

                bbox = cell.get("bbox", None)
                if not (bbox is None or isinstance(bbox, list) and len(bbox) == 4):
                    raise ValueError(f"Invalid bbox: {bbox}")
                text = cell.get("tokens", None)
                if not (text is None or isinstance(text, list) and all(isinstance(t, str) for t in text)):
                    raise ValueError(f"Invalid text: {text}")

                cell_entry = CellEntry(
                    bbox=bbox,
                    text=text,
                    columns=cell_cols,
                    rows=cell_rows,
                )
                cell_data.append(cell_entry)

                cell_i += 1

                cell_cols = None
                cell_rows = None

            if token == "</tr>":
                rows_i += 1
                cols_i = 0

        return {"image_id": image_id, "filename": filename, "split": split, "cell_data": cell_data}

    def convert_single_line(self, line):
        data = self.single_line_to_data(line)

        filename = data["filename"]
        split = data["split"]
        cell_data = data["cell_data"]

        self.output_dir.joinpath("images", split).mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath("labels", split).mkdir(parents=True, exist_ok=True)

        images_input_path = self.pubtabnet_path.joinpath(split, filename)

        # Get image size
        width, height = imagesize.get(images_input_path)

        # Write to YOLO format
        images_path = self.output_dir.joinpath("images", split, filename)
        json_path = self.output_dir.joinpath("labels", split, images_path.stem + ".json")

        copy_mode(path=images_input_path, destination=images_path, mode="symlink")

        output = self.cell_data_to_bbox_columns_and_rows(cell_data, height, width)

        with json_path.open("w") as f:
            json.dump(output, f)

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
    def _normalize_coords(bbox: list[int | float], size: tuple[int, int]) -> list[float]:
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
        min_x = bbox[0] / width
        min_y = bbox[1] / height
        max_x = bbox[2] / width
        max_y = bbox[3] / height

        return [min_x, min_y, max_x, max_y]

    def convert_jsonl(self):
        with open(self.pubtabnet_jsonl_path, "r") as f:
            lines = f.readlines()
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.convert_single_line, lines),
                    desc="Converting JSONL to JSON",
                    total=len(lines),
                )
            )

    def cell_data_to_bbox_columns_and_rows(self, cell_data, height, width):
        cell_bbox_output = []

        row_coords = defaultdict(lambda: [np.inf, np.inf, -np.inf, -np.inf])
        col_coords = defaultdict(lambda: [np.inf, np.inf, -np.inf, -np.inf])

        for cell in cell_data:
            bbox = cell["bbox"]
            if bbox is None:
                continue

            rows = cell["rows"]
            cols = cell["columns"]

            min_x_cell = bbox[0]
            min_y_cell = bbox[1]
            max_x_cell = bbox[2]
            max_y_cell = bbox[3]

            if min_x_cell < 0:
                min_x_cell = 0
                self.logger.warning(f"min_x_cell < 0: {bbox}")
            if min_y_cell < 0:
                min_y_cell = 0
                self.logger.warning(f"min_y_cell < 0: {bbox}")
            if max_x_cell > width:
                max_x_cell = width
                self.logger.warning(f"max_x_cell > width: {bbox}")
            if max_y_cell > height:
                max_y_cell = height
                self.logger.warning(f"max_y_cell > height: {bbox}")

            if len(rows) == 1:
                row = list(rows)[0]
                row_coords[row] = [
                    min(row_coords[row][0], min_x_cell),
                    min(row_coords[row][1], min_y_cell),
                    max(row_coords[row][2], max_x_cell),
                    max(row_coords[row][3], max_y_cell),
                ]

            if len(cols) == 1:
                col = list(cols)[0]
                col_coords[col] = [
                    min(col_coords[col][0], min_x_cell),
                    min(col_coords[col][1], min_y_cell),
                    max(col_coords[col][2], max_x_cell),
                    max(col_coords[col][3], max_y_cell),
                ]

            cell_bbox = [min_x_cell, min_y_cell, max_x_cell, max_y_cell]

            cell_bbox_output.append(cell_bbox)

        row_bbox_output = list(row_coords.values())

        col_bbox_output = list(col_coords.values())

        return {"cells": cell_bbox_output, "rows": row_bbox_output, "cols": col_bbox_output}

    def convert(self):
        self.convert_jsonl()

        info_yaml = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": ["table_cell", "table_row", "table_column"],
        }

        with open(self.output_dir.joinpath("yolo.yaml"), "w") as f:
            yaml.dump(info_yaml, f)


if __name__ == "__main__":
    pubtabnet_path = Path("/home/stefan/Documents/datasets/pubtabnet")
    output_dir = Path("/tmp/pubtabnet_json")

    converter = PubTabNetToJSON(pubtabnet_path=pubtabnet_path, output_dir=output_dir)
    converter.convert()
