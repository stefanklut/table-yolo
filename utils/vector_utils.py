import numpy as np
import shapely


def line_intersection_points(line1: np.ndarray, line2: np.ndarray):
    """
    Find the intersection point of two lines defined by two points each.

    Args:
        line1 (np.ndarray): The first line defined by two points.
        line2 (np.ndarray): The second line defined by two points.

    Returns:
        np.ndarray: The intersection point (x, y) if it exists, otherwise None.
    """
    assert line1.shape == (2, 2), "Line1 must be defined by two points."
    assert line2.shape == (2, 2), "Line2 must be defined by two points."

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


def line_segment_intersection_points(segment1: np.ndarray, segment2: np.ndarray):
    """
    Find the intersection point of two line segments defined by two points each using shapely.

    Args:
        segment1 (np.ndarray): The first line segment defined by two points.
        segment2 (np.ndarray): The second line segment defined by two points.

    Returns:
        np.ndarray: The intersection point (x, y) if it exists, otherwise None.
    """
    assert segment1.shape == (2, 2), "Segment1 must be defined by two points."
    assert segment2.shape == (2, 2), "Segment2 must be defined by two points."

    line1 = shapely.LineString(segment1)
    line2 = shapely.LineString(segment2)
    intersection = line1.intersection(line2)

    if intersection.is_empty:
        return None
    if intersection.geom_type == "Point":
        return np.array([intersection.x, intersection.y])
    return None


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

    assert vector1.shape == (2,), "Vector1 must be a 2D vector."
    assert vector2.shape == (2,), "Vector2 must be a 2D vector."
    assert point1.shape == (2,), "Point1 must be a 2D coordinate."
    assert point2.shape == (2,), "Point2 must be a 2D coordinate."

    # Calculate the intersection point
    denom = (vector1[0] * vector2[1]) - (vector1[1] * vector2[0])
    if denom == 0:
        return None  # Lines are parallel

    t = ((point2[0] - point1[0]) * vector2[1] - (point2[1] - point1[1]) * vector2[0]) / denom

    intersection_point = point1 + t * vector1

    return intersection_point


def point_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using shapely.

    Args:
        point (np.ndarray): The point to check.
        polygon (np.ndarray): The polygon defined by its vertices.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    assert point.shape == (2,), "Point must be a 2D coordinate."
    assert polygon.shape[1] == 2, "Polygon must be defined by its vertices in 2D."

    poly = shapely.Polygon(polygon)
    pt = shapely.Point(point)
    return poly.contains(pt)


def segment_length_inside_polygon(segment: np.ndarray, polygon: np.ndarray) -> float:
    """
    Calculate the length of the portion of a line segment that lies inside a polygon.

    Args:
        segment (np.ndarray): The line segment defined by two points, shape (2, 2).
        polygon (np.ndarray): The polygon defined by its vertices, shape (n, 2).

    Returns:
        float: The length of the segment inside the polygon.
    """

    assert segment.shape == (2, 2), "Segment must be defined by two points."
    assert polygon.shape[1] == 2, "Polygon must be defined by its vertices in 2D."

    line = shapely.LineString(segment)
    poly = shapely.Polygon(polygon)
    intersection = line.intersection(poly)

    if intersection.is_empty:
        return 0.0
    elif intersection.geom_type == "LineString":
        return intersection.length
    elif intersection.geom_type == "MultiLineString":
        return sum(part.length for part in intersection)
    else:
        return 0.0


def line_length_inside_polygon(polyline: np.ndarray, polygon: np.ndarray) -> float:
    """
    Calculate the length of the portion of a polyline that lies inside a polygon.

    Args:
        polyline (np.ndarray): The polyline defined by its points, shape (m, 2).
        polygon (np.ndarray): The polygon defined by its vertices, shape (n, 2).

    Returns:
        float: The total length of the polyline inside the polygon.
    """
    assert polyline.shape[1] == 2, "Polyline must be defined by its vertices in 2D."
    assert polygon.shape[1] == 2, "Polygon must be defined by its vertices in 2D."

    line = shapely.LineString(polyline)
    poly = shapely.Polygon(polygon)
    intersection = line.intersection(poly)
    if intersection.is_empty:
        return 0.0
    elif intersection.geom_type == "LineString":
        return intersection.length
    elif intersection.geom_type == "MultiLineString":
        return sum(part.length for part in intersection.geoms)
    else:
        return 0.0


def fraction_line_inside_polygon(polyline: np.ndarray, polygon: np.ndarray) -> float:
    """
    Calculate the percentage of a polyline that lies inside a polygon.

    Args:
        polyline (np.ndarray): The polyline defined by its points, shape (m, 2).
        polygon (np.ndarray): The polygon defined by its vertices, shape (n, 2).

    Returns:
        float: The percentage of the polyline length that is inside the polygon.
    """
    polyline_length = shapely.LineString(polyline).length

    if polyline_length == 0:
        return 0.0
    inside_length = line_length_inside_polygon(polyline, polygon)
    return inside_length / polyline_length


if __name__ == "__main__":

    def test_line_intersection_points():
        # Intersecting lines
        line1 = np.array([[0, 0], [1, 1]])
        line2 = np.array([[0, 1], [1, 0]])
        intersection = line_intersection_points(line1, line2)
        assert np.allclose(intersection, (0.5, 0.5))

        # Parallel lines
        line3 = np.array([[0, 0], [1, 0]])
        line4 = np.array([[0, 1], [1, 1]])
        assert line_intersection_points(line3, line4) is None

    def test_line_segment_intersection_points():
        # Intersecting segments
        seg1 = np.array([[0, 0], [1, 1]])
        seg2 = np.array([[0, 1], [1, 0]])
        intersection = line_segment_intersection_points(seg1, seg2)
        assert np.allclose(intersection, [0.5, 0.5])

        # Non-intersecting segments
        seg3 = np.array([[0, 0], [1, 0]])
        seg4 = np.array([[0, 1], [1, 1]])
        assert line_segment_intersection_points(seg3, seg4) is None

    def test_line_intersection_vector_point():
        # Intersecting lines
        v1 = np.array([1, 1])
        p1 = np.array([0, 0])
        v2 = np.array([1, -1])
        p2 = np.array([0, 1])
        intersection = line_intersection_vector_point(v1, p1, v2, p2)
        assert np.allclose(intersection, [0.5, 0.5])

        # Parallel lines
        v3 = np.array([1, 0])
        p3 = np.array([0, 0])
        v4 = np.array([1, 0])
        p4 = np.array([0, 1])
        assert line_intersection_vector_point(v3, p3, v4, p4) is None

    def test_point_inside_polygon():
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        point_inside = np.array([1, 1])
        point_outside = np.array([3, 3])
        assert point_inside_polygon(point_inside, polygon) is True
        assert point_inside_polygon(point_outside, polygon) is False

    def test_segment_length_inside_polygon():
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        # Segment fully inside
        seg1 = np.array([[0.5, 0.5], [1.5, 1.5]])
        assert np.isclose(segment_length_inside_polygon(seg1, polygon), np.linalg.norm(seg1[1] - seg1[0]))

        # Segment partially inside
        seg2 = np.array([[-1, 1], [1, 1]])
        assert np.isclose(segment_length_inside_polygon(seg2, polygon), 1.0)

        # Segment outside
        seg3 = np.array([[3, 3], [4, 4]])
        assert segment_length_inside_polygon(seg3, polygon) == 0.0

        # Segment crossing polygon
        seg4 = np.array([[-1, 1], [3, 1]])
        assert np.isclose(segment_length_inside_polygon(seg4, polygon), 2.0)

    def test_line_length_inside_polygon():
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        # Polyline with 3 segments, all inside
        polyline1 = np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 1.5], [1.5, 0.5]])
        expected_length1 = (
            np.linalg.norm(polyline1[1] - polyline1[0])
            + np.linalg.norm(polyline1[2] - polyline1[1])
            + np.linalg.norm(polyline1[3] - polyline1[2])
        )
        assert np.isclose(line_length_inside_polygon(polyline1, polygon), expected_length1)

        # Polyline with 3 segments, partially inside
        polyline2 = np.array([[-1, 1], [1, 1], [2.5, 1], [3, 1]])
        # Only the segment from (0,1) to (2,1) is inside, length 2.0
        assert np.isclose(line_length_inside_polygon(polyline2, polygon), 2.0)

        # Polyline with 3 segments, all outside
        polyline3 = np.array([[3, 3], [4, 4], [5, 5], [6, 6]])
        assert line_length_inside_polygon(polyline3, polygon) == 0.0

    # Run tests
    test_line_intersection_points()
    test_line_segment_intersection_points()
    test_line_intersection_vector_point()
    test_point_inside_polygon()
    test_segment_length_inside_polygon()
    test_line_length_inside_polygon()
