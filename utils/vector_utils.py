import numpy as np


def line_intersection_point(line1: np.ndarray, line2: np.ndarray):
    """
    Find the intersection point of two lines defined by two points each.

    Args:
        line1 (np.ndarray): The first line defined by two points.
        line2 (np.ndarray): The second line defined by two points.

    Returns:
        np.ndarray: The intersection point (x, y) if it exists, otherwise None.
    """
    # line1.shape = (2, 2)
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (px, py)


def line_intersection_vector_point(vector1: np.ndarray, point1: np.ndarray, vector2: np.ndarray, point2: np.ndarray):
    """
    Find the intersection point of two lines defined by a vector and a point each.
    Args:
        vector1 (np.ndarray): The first line defined by a vector.
        point1 (np.ndarray): A point on the first line.
        vector2 (np.ndarray): The second line defined by a vector.
        point2 (np.ndarray): A point on the second line.

    Returns:
        np.ndarray: The intersection point (x, y) if it exists, otherwise None.
    """

    # Calculate the intersection point
    denom = (vector1[0] * vector2[1]) - (vector1[1] * vector2[0])
    if denom == 0:
        return None  # Lines are parallel

    t = ((point2[0] - point1[0]) * vector2[1] - (point2[1] - point1[1]) * vector2[0]) / denom

    intersection_point = point1 + t * vector1

    return intersection_point


def point_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using the ray-casting algorithm.

    Args:
        point (np.ndarray): The point to check.
        polygon (np.ndarray): The polygon defined by its vertices.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    assert polygon.shape[1] == 2, "Polygon must be a 2D array of shape (n, 2)"
    assert point.shape == (2,), "Point must be a 1D array of shape (2,)"
    assert polygon.shape[0] >= 3, "Polygon must have at least 3 vertices"
    n = len(polygon)
    inside = False

    x_intercept = point[0]
    y_intercept = point[1]

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y_intercept > min(p1y, p2y):
            if y_intercept <= max(p1y, p2y):
                if x_intercept <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y_intercept - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x_intercept <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


if __name__ == "__main__":
    line1 = np.array([[1, 0], [1, 1]])
    line2 = np.array([[0, 0], [0, 1]])

    line1_point = line1[0]
    line1_vector = line1[0] - line1[1]

    line2_point = line2[0]
    line2_vector = line2[0] - line1[1]

    print(line_intersection_point(line1, line2))
    print(line_intersection_vector_point(line1_vector, line1_point, line2_vector, line2_point))
