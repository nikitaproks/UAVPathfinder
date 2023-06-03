from typing import Optional
import numpy as np


def get_points_num(number: float, resolution: int) -> int:
    """Calculate the number of points in a circle mesh."""
    points_number: int = int(round(resolution * number))
    if points_number < 3:
        raise ValueError("Resolution too low")
    return points_number


def rotation_matrix(axis, theta: np.ndarray) -> np.ndarray:
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians."""
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c - a * d),
                2 * (b * d + a * c),
            ],
            [
                2 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2 * (c * d - a * b),
            ],
            [
                2 * (b * d - a * c),
                2 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ],
        dtype=np.float64,
    )


def rotate_points(
    points: np.ndarray, axis_vector: np.ndarray
) -> np.ndarray:
    # Rotate points to align with axis_vector
    rot_axis = np.cross([0, 0, 1], axis_vector)
    if np.linalg.norm(rot_axis) > 1e-6:  # if not already aligned...
        rot_angle = np.arccos(np.dot([0, 0, 1], axis_vector))
        rot_matrix = rotation_matrix(rot_axis, rot_angle)
        points = np.dot(points, rot_matrix)
    return points


def project_3d_to_2d(points_3d):
    points_3d = np.array(points_3d)
    points_2d = []
    flat_dim = None
    # Identify the dimension(s) with exactly two distinct values
    for idx in range(points_3d.shape[1]):
        if len(np.unique(points_3d[:, idx])) <= 2:
            flat_dim = idx
            break

    if flat_dim is None:
        raise ValueError("The points do not form a flat 3D polygon")

    # Project the points to 2D
    for point in points_3d:
        points_2d.append(
            [coord for i, coord in enumerate(point) if i != flat_dim]
        )

    return np.array(points_2d), flat_dim


def project_2d_to_3d(
    points_2d, polygon_flat_dim, const_value
) -> np.ndarray:
    points_3d = np.insert(
        points_2d, polygon_flat_dim, const_value, axis=1
    )
    return points_3d
