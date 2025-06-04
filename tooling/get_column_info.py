import json
import sys
from pathlib import Path

import numpy as np
from natsort import natsorted

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.page_xml_editor import PageXMLEditor
from tooling.stretch_column import col_based_on_header, stretch_col
from utils.vector_utils import fraction_line_inside_polygon


def get_column_text_lines(col: np.ndarray, page_xml: PageXMLEditor) -> list[str]:
    """
    Get the values of a column from a pageXML object.

    Args:
        col (np.ndarray): The column coordinates in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        page_xml (PageXMLEditor): The PageXMLEditor object containing the pageXML data.

    Returns:
        list[str]: A list of strings representing the values in the column.
    """
    assert col.shape == (4, 2), f"Column shape is not (4, 2), but {col.shape}"

    text_lines = []

    for node in page_xml.iterfind(".//TextLine"):
        baseline = node.find("Baseline")
        if baseline is None:
            continue  # Skip if no baseline is found

        str_coords = baseline.attrib.get("points")

        # Ignoring empty baselines
        if str_coords is None:
            continue
        split_str_coords = str_coords.split()
        # Ignoring empty baselines
        if len(split_str_coords) == 0:
            continue
        # HACK Doubling single value baselines (otherwise they are not drawn)
        if len(split_str_coords) == 1:
            split_str_coords = split_str_coords * 2  # Double list [value]*2 for cv2.polyline
        coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)

        # Check if the baseline intersects with the column
        if fraction_line_inside_polygon(coords, col) > 0.5:
            text = page_xml.get_text(node)
            if text is None or text.strip() == "":
                continue
            entry = {}
            if text:
                entry["text"] = text
                entry["coords"] = coords.tolist()
                text_lines.append(entry)

    # Sort text lines by the y-coordinate of the first point
    text_lines.sort(key=lambda x: x["coords"][0][1])
    # Extract the text values from the sorted text lines
    text_lines = [line["text"] for line in text_lines]

    return text_lines


def combine_json_and_pagexml(json_path: Path, pagexml_path: Path, output_path: Path) -> None:
    """
    Combine JSON data with PageXML data and save the result.
    Args:
        json_path (Path): Path to the JSON file.
        pagexml_path (Path): Path to the PageXML file.
        output_path (Path): Path to save the combined data.
    """
    json_data = json.loads(json_path.read_text())
    page_xml = PageXMLEditor(pagexml_path)

    polygon_col_ndpohkp = json_data["col-ndpohkp"]
    polygon_header_ndpohkp = json_data["header-ndpohkp"]

    col = col_based_on_header(
        np.array(polygon_col_ndpohkp),
        np.array(polygon_header_ndpohkp),
    )

    if col is None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "col-text-lines": [],
                },
                f,
                ensure_ascii=False,
                indent=4,
            )
        print(f"Empty output saved to {output_path}.")
        return

    image_size = page_xml.get_size()

    if image_size is None:
        raise ValueError("Image size not found in PageXML.")

    col[:, 0] *= image_size[1]  # Scale x-coordinates to image width
    col[:, 1] *= image_size[0]  # Scale y-coordinates to image height

    col_text_lines = get_column_text_lines(col, page_xml)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "col-text-lines": col_text_lines,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    json_dir_path = Path("/home/stefan/Documents/data/kadaster_overijsel_results")
    json_paths = natsorted(json_dir_path.glob("*.json"))

    pagexml_dir_path = Path("/home/stefan/Documents/data/kadaster_overijsel_scans/page")

    output_base_path = Path("/home/stefan/Documents/data/kadaster_overijsel_combined")
    output_base_path.mkdir(parents=True, exist_ok=True)

    for i, json_path in enumerate(json_paths):
        pagexml_path = pagexml_dir_path / f"{json_path.stem}.xml"
        if not pagexml_path.exists():
            print(f"PageXML file not found for {json_path.stem}, skipping.")
            continue

        output_path = output_base_path / f"{json_path.stem}_combined.json"
        combine_json_and_pagexml(json_path, pagexml_path, output_path)
        print(f"Combined data saved to {output_path} {i}/{len(json_paths)}")
