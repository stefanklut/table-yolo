import json
import sys
from pathlib import Path

import numpy as np
from natsort import natsorted

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.vector_utils import line_intersection_vector_point


def stretch_column(
    col: np.ndarray,
    header: np.ndarray,
) -> np.ndarray:
    if col.shape[0] == 0:
        return None
    if header.shape[0] == 0:
        return col[0]
    assert col.shape[1:] == (4, 2), f"Column shape is not (n, 4, 2), but {col.shape}"
    assert header.shape[1:] == (4, 2), f"Header shape is not (n, 4, 2), but {header.shape}"

    if header.shape[0] > 1:
        # Pick the left-most header by the minimum average x-coordinate
        header = header[np.argmin(header.mean(axis=1))]
    if header.shape[0] == 1:
        header = header[0]

    if header.shape != (4, 2):
        raise ValueError(f"Header shape is not (4, 2), but {header.shape}")

    if col.shape[0] > 1:
        # Pick the left-most column by the minimum average x-coordinate
        col = col[np.argmin(col.mean(axis=1))]
    if col.shape[0] == 1:
        col = col[0]

    if col.shape != (4, 2):
        raise ValueError(f"Column shape is not (4, 2), but {col.shape}")

    # Check if the order is bottom-left, bottom-right, top-right, top-left

    def order_bounding_box(points):
        # Sort by y, then x
        sorted_by_y = points[np.argsort(points[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        top_left, top_right = top_two[np.argsort(top_two[:, 0])]
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([bottom_left, bottom_right, top_right, top_left])

    ordered_column = order_bounding_box(col)
    if not np.allclose(ordered_column, col):
        raise ValueError(
            f"Column points are not in bottom-left, bottom-right, top-right, top-left order., got \n{col} \n{ordered_column}"
        )
    ordered_header = order_bounding_box(header)
    if not np.allclose(ordered_header, header):
        raise ValueError(
            f"Header points are not in bottom-left, bottom-right, top-right, top-left order., got \n{header} \n{ordered_header}"
        )

    col_top_left = col[3]
    col_bottom_left = col[0]
    col_top_right = col[2]
    col_bottom_right = col[1]

    col_left_vector = col_bottom_left - col_top_left
    col_right_vector = col_bottom_right - col_top_right

    header_bottom_left = header[0]
    header_bottom_right = header[1]

    header_bottom_vector = header_bottom_right - header_bottom_left

    # Calculate the intersection points
    intersection_left_col_bottom_header = line_intersection_vector_point(
        col_left_vector,
        col_top_left,
        header_bottom_vector,
        header_bottom_left,
    )
    intersection_right_col_bottom_header = line_intersection_vector_point(
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
        # TODO Optimize by just setting it to the intersection point
        col_top_left = col_top_left + vector_to_intersection_right
        col_top_right = col_top_right + vector_to_intersection_right
    else:
        # Move both top points up by the distance to the header
        col_top_left = col_top_left + vector_to_intersection_left
        col_top_right = col_top_right + vector_to_intersection_left

    image_bottom_left = np.array([0, 1])
    image_bottom_right = np.array([1, 1])
    image_bottom_vector = image_bottom_right - image_bottom_left

    # Calculate the intersection points
    intersection_left_col_bottom_image = line_intersection_vector_point(
        col_left_vector,
        col_bottom_left,
        image_bottom_vector,
        image_bottom_left,
    )
    intersection_right_col_bottom_image = line_intersection_vector_point(
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
        col_bottom_left = col_bottom_left + vector_to_intersection_right
        col_bottom_right = col_bottom_right + vector_to_intersection_right
    else:
        # Move both bottom points up by the distance to the image
        col_bottom_left = col_bottom_left + vector_to_intersection_left
        col_bottom_right = col_bottom_right + vector_to_intersection_left

    # Create the new column
    new_col = np.array([col_bottom_left, col_bottom_right, col_top_right, col_top_left])
    return new_col


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

        col = stretch_column(col, header)

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
