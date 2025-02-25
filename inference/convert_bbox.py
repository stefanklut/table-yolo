import json
import sys
from multiprocessing.pool import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from data.convert_pubtabnet_to_json import PubTabNetToJSON


def convert_bboxes_to_cells(bboxes: dict):
    rows = bboxes["row"]
    cols = bboxes["col"]
    cells = bboxes["cell"]

    sorted_rows = sorted(rows, key=lambda cell: cell[1])
    sorted_cols = sorted(cols, key=lambda cell: cell[0])

    cells = {"cells": [], "rows": {}, "cols": {}}

    for i, row in enumerate(sorted_rows):
        cells["rows"][i] = row
        for j, col in enumerate(sorted_cols):
            if i == 0:
                cells["cols"][j] = col
            cell = {
                "row": i,
                "col": j,
                "bbox": [
                    max(row[0], col[0]),  # x1
                    max(row[1], col[1]),  # y1
                    min(row[2], col[2]),  # x2
                    min(row[3], col[3]),  # y2
                ],
            }
            # Ensure the cell bbox is valid (i.e., x1 < x2 and y1 < y2)
            if cell["bbox"][0] < cell["bbox"][2] and cell["bbox"][1] < cell["bbox"][3]:
                cells["cells"].append(cell)

    return cells


if __name__ == "__main__":

    def load_and_convert(kwargs):
        path = kwargs["path"]
        output_path = kwargs["output_path"]

        with open(path, "r") as f:
            data = json.load(f)

        cells = convert_bboxes_to_cells(data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(cells, f)

        return cells

    pubtabnet_path = Path("/home/stefan/Documents/datasets/pubtabnet")
    conversion_output_dir = Path("/tmp/pubtabnet_json_conversion")

    converter = PubTabNetToJSON(pubtabnet_path=pubtabnet_path, output_dir=conversion_output_dir)
    converter.convert()

    output_train_labels = conversion_output_dir.joinpath("labels", "train")
    output_val_labels = conversion_output_dir.joinpath("labels", "val")

    assert output_train_labels.exists()
    assert output_val_labels.exists()

    prediction_output_dir = Path("/tmp/pubtabnet_json_predictions")

    paths = list(output_val_labels.glob("*.json"))

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    load_and_convert,
                    [{"path": path, "output_path": prediction_output_dir.joinpath("val", path.name)} for path in paths],
                ),
                desc="Converting Bounding Boxes to Cells",
                total=len(paths),
            )
        )
