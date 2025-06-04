import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from natsort import natsorted

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.vector_utils import (
    line_line_intersection_vector_point,
    line_segment_intersection_points,
)


def order_bounding_box(points):
    # Sort by y, then x
    sorted_by_y = points[np.argsort(points[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([bottom_left, bottom_right, top_right, top_left])


def is_ordered_bounding_box(points):
    """
    Check if the points are ordered as bottom-left, bottom-right, top-right, top-left.
    """
    if points.shape != (4, 2):
        raise ValueError(f"Points shape is not (4, 2), but {points.shape}")
    ordered_points = order_bounding_box(points)
    return np.allclose(ordered_points, points)


def pick_best_col(cols: np.ndarray) -> np.ndarray:
    """
    Pick the best column from a list of columns.
    """
    if cols.shape[0] == 0:
        return cols

    # TODO: Implement a better picking strategy
    assert cols.shape[1:] == (4, 2), f"Column shape is not (n, 4, 2), but {cols.shape}"
    assert cols.shape[0] >= 1, f"Column missing, got {cols.shape}"

    if cols.shape[0] == 1:
        return cols[0]
    else:
        # For now, just return the first column
        return cols[0]


def pick_best_header(headers: np.ndarray) -> np.ndarray:
    """
    Pick the best header from a list of headers.
    """
    if headers.shape == (0,):
        return headers

    # TODO: Implement a better picking strategy
    assert headers.shape[1:] == (4, 2), f"Header shape is not (n, 4, 2), but {headers.shape}"
    assert headers.shape[0] >= 1, f"Header missing, got {headers.shape}"

    if headers.shape[0] == 1:
        return headers[0]
    else:
        # For now, just return the first header
        return headers[0]


def is_header_above_col(col: np.ndarray, header: np.ndarray) -> bool:
    assert col.shape == (4, 2), f"Column shape is not (4, 2), but {col.shape}"
    assert header.shape == (4, 2), f"Header shape is not (4, 2), but {header.shape}"

    col_top_left = col[3]
    col_bottom_left = col[0]
    col_top_right = col[2]
    col_bottom_right = col[1]
    header_bottom_left = header[0]
    header_bottom_right = header[1]

    center_top_col = (col_top_left + col_top_right) / 2
    center_bottom_col = (col_bottom_left + col_bottom_right) / 2

    col_points = np.array([center_top_col, center_bottom_col])
    header_points = np.array([header_bottom_left, header_bottom_right])

    intersection = line_segment_intersection_points(col_points, header_points)
    if intersection is None:
        return False

    return True


def pick_best_col_based_on_header(
    col: np.ndarray,
    header: np.ndarray,
) -> np.ndarray:
    assert col.shape[1:] == (4, 2), f"Column shape is not (n, 4, 2), but {col.shape}"
    assert col.shape[0] >= 1, f"Column missing, got {col.shape}"
    assert header.shape == (4, 2), f"Header shape is not (4, 2), but {header.shape}"

    header_above = []

    for col_i in col:
        if is_header_above_col(col_i, header):
            header_above.append(col_i)

    if len(header_above) == 1:
        return header_above[0]
    else:
        return pick_best_col(col)


def pick_best_header_based_on_col(
    header: np.ndarray,
    col: np.ndarray,
) -> np.ndarray:
    assert header.shape[1:] == (4, 2), f"Header shape is not (4, 2), but {header.shape}"
    assert header.shape[0] >= 1, f"Header missing, got {header.shape}"
    assert col.shape == (4, 2), f"Column shape is not (4, 2), but {col.shape}"

    header_above = []
    for header_i in header:
        if is_header_above_col(col, header_i):
            header_above.append(header_i)
    if len(header_above) == 1:
        return header_above[0]
    else:
        return pick_best_header(header)


def pick_best_col_header(col: np.ndarray, header: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick the best column and header from a list of columns and headers.
    """
    assert col.shape == (0,) or col.shape[1:] == (4, 2), f"Column shape is not (0,) or (4, 2), but {col.shape}"
    assert header.shape == (0,) or header.shape[1:] == (4, 2), f"Header shape is not (0,) or (4, 2), but {header.shape}"

    if col.shape[0] == 0:
        best_col = col
    elif col.shape[0] == 1:
        best_col = col[0]

    if header.shape[0] == 0:
        best_header = header
    elif header.shape[0] == 1:
        best_header = header[0]

    if col.shape[0] > 1 and header.shape[0] == 0:
        # If there are multiple columns but no header, pick the best column solo
        best_col = pick_best_col(col)

    elif header.shape[0] > 1 and col.shape[0] == 0:
        # If there are multiple headers but no column, pick the best header solo
        best_header = pick_best_header(header)

    elif col.shape[0] > 1 and header.shape[0] == 1:
        best_col = pick_best_col_based_on_header(col, header[0])
    elif header.shape[0] > 1 and col.shape[0] == 1:
        best_header = pick_best_header_based_on_col(header, col[0])
    else:
        # If there are multiple columns and headers, pick the best column and header
        best_col = pick_best_col(col)
        best_header = pick_best_header(header)

    assert best_col.shape == (0,) or best_col.shape == (4, 2), f"Best column shape is not (0,) or (4, 2), but {best_col.shape}"
    assert best_header.shape == (0,) or best_header.shape == (
        4,
        2,
    ), f"Best header shape is not (0,) or (4, 2), but {best_header.shape}"

    return best_col, best_header


def stretch_col(
    col: np.ndarray,
    header: np.ndarray,
) -> Optional[np.ndarray]:
    col, header = pick_best_col_header(col, header)

    stretch_to_header = True

    # Check if the order is bottom-left, bottom-right, top-right, top-left
    if col.shape == (0,) or not is_ordered_bounding_box(col):
        return None
    if header.shape == (0,) or not is_ordered_bounding_box(header):
        stretch_to_header = False

    col_top_left = col[3]
    col_bottom_left = col[0]
    col_top_right = col[2]
    col_bottom_right = col[1]

    col_left_vector = col_bottom_left - col_top_left
    col_right_vector = col_bottom_right - col_top_right

    if stretch_to_header:
        header_bottom_left = header[0]
        header_bottom_right = header[1]

        header_bottom_vector = header_bottom_right - header_bottom_left

        # Calculate the intersection points
        intersection_left_col_bottom_header = line_line_intersection_vector_point(
            col_left_vector,
            col_top_left,
            header_bottom_vector,
            header_bottom_left,
        )
        intersection_right_col_bottom_header = line_line_intersection_vector_point(
            col_right_vector,
            col_top_right,
            header_bottom_vector,
            header_bottom_right,
        )

        if intersection_left_col_bottom_header is None or intersection_right_col_bottom_header is None:
            print(f"Warning: No intersection found between column and header, got \n{col} \n{header}")
            return col

        vector_to_intersection_left = intersection_left_col_bottom_header - col_top_left
        vector_to_intersection_right = intersection_right_col_bottom_header - col_top_right

        distance_left = np.linalg.norm(vector_to_intersection_left)
        distance_right = np.linalg.norm(vector_to_intersection_right)

        if distance_right < distance_left:
            # Move both top points up by the distance to the header
            col_top_left = intersection_left_col_bottom_header
            col_top_right = col_top_right + vector_to_intersection_right
        else:
            # Move both top points up by the distance to the header
            col_top_left = col_top_left + vector_to_intersection_left
            col_top_right = intersection_right_col_bottom_header

    image_bottom_left = np.array([0, 1])
    image_bottom_right = np.array([1, 1])
    image_bottom_vector = image_bottom_right - image_bottom_left

    # Calculate the intersection points
    intersection_left_col_bottom_image = line_line_intersection_vector_point(
        col_left_vector,
        col_bottom_left,
        image_bottom_vector,
        image_bottom_left,
    )
    intersection_right_col_bottom_image = line_line_intersection_vector_point(
        col_right_vector,
        col_bottom_right,
        image_bottom_vector,
        image_bottom_right,
    )

    vector_to_intersection_left = intersection_left_col_bottom_image - col_bottom_left
    vector_to_intersection_right = intersection_right_col_bottom_image - col_bottom_right

    distance_left = np.linalg.norm(vector_to_intersection_left)
    distance_right = np.linalg.norm(vector_to_intersection_right)

    if distance_right < distance_left:
        # Move both bottom points up by the distance to the image
        col_bottom_left = intersection_left_col_bottom_image
        col_bottom_right = col_bottom_right + vector_to_intersection_right
    else:
        # Move both bottom points up by the distance to the image
        col_bottom_left = col_bottom_left + vector_to_intersection_left
        col_bottom_right = intersection_right_col_bottom_image

    # Create the new column
    new_col = np.array([col_bottom_left, col_bottom_right, col_top_right, col_top_left])
    return new_col


def col_based_on_header(
    col: np.ndarray,
    header: np.ndarray,
) -> Optional[np.ndarray]:
    """
    If there is no column, base the column on the header.

    Args:
        col (np.ndarray): The column points.
        header (np.ndarray): The header points.
    Returns:
        np.ndarray: The column points based on the header.
    """

    # TODO pick_best_col_header(col, header)
    assert col.shape == (0,) or col.shape[1:] == (4, 2), f"Column shape is not (0,) or (4, 2), but {col.shape}"
    assert header.shape == (0,) or header.shape[1:] == (4, 2), f"Header shape is not (0,) or (4, 2), but {header.shape}"

    if col.shape[0] == 0:
        col, header = pick_best_col_header(col, header)
        if header.shape[0] == 0:
            return None

        # Create a column based on the header
        col = header.copy()[None]
        header = header[None]
        return stretch_col(col, header)

    return stretch_col(col, header)


if __name__ == "__main__":
    json_dir_path = Path("/home/stefan/Documents/data/kadaster_overijsel_results")
    json_paths = natsorted(json_dir_path.glob("*.json"))

    image_dir_path = Path("/home/stefan/Documents/data/kadaster_overijsel_scans")
    image_paths = natsorted(image_dir_path.glob("*.jpg"))
    assert len(json_paths) == len(
        image_paths
    ), f"Number of json files {len(json_paths)} does not match number of image files {len(image_paths)}"
    mapping = {json_path.stem: image_path for json_path, image_path in zip(json_paths, image_paths)}

    json_paths = json_paths[:10]

    for json_path in json_paths:
        with open(json_path, "r") as f:
            json_data = json.load(f)

        # Get the header
        header = json_data["header-ndpohkp"]
        header = np.array(header)

        # Get the columns
        col = json_data["col-ndpohkp"]
        col = np.array(col)

        old_col = col.copy()
        if old_col.shape[0] > 1:
            # TODO Pick the best column
            old_col = old_col[0]
        if old_col.shape[0] == 1:
            old_col = old_col[0]
        if old_col.shape != (4, 2):
            raise ValueError(f"Column shape is not (4, 2), but {old_col.shape}")

        col = stretch_col(col, header)
        if col is None:
            continue

        # Visualize the column and header
        import cv2

        image_path = mapping[json_path.stem]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        col[:, 0] *= width
        col[:, 1] *= height

        old_col[:, 0] *= width
        old_col[:, 1] *= height

        # Draw the column
        image = cv2.polylines(image, [col.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        # Draw the old column
        image = cv2.polylines(image, [old_col.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

        if header.shape[0] == 0:
            header = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]])
        if header.shape[0] > 1:
            # TODO Pick the best header
            header = header[0]
        if header.shape[0] == 1:
            header = header[0]
        if header.shape != (4, 2):
            raise ValueError(f"Header shape is not (4, 2), but {header.shape}")

        header[:, 0] *= width
        header[:, 1] *= height

        # Draw the header
        # image = cv2.polylines(image, [header.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.show()
