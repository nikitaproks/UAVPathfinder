from typing import Union
import numpy as np
from shapely.geometry import Polygon, LineString
from UAVpathfinder.visualization import Render3D


class Mesh:
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype=np.float64)
        self.faces = np.empty((0, 3), dtype=np.int32)

    def _get_points_num(self, number: float, resolution: int) -> int:
        """Calculate the number of points in a circle mesh."""
        points_number: int = int(round(resolution * number))
        if points_number < 3:
            raise ValueError("Resolution too low")
        return points_number

    def _rotation_matrix(self, axis, theta: np.ndarray) -> np.ndarray:
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

    def _rotate_points(
        self, points: np.ndarray, axis_vector: np.ndarray
    ) -> np.ndarray:
        # Rotate points to align with axis_vector
        rot_axis = np.cross([0, 0, 1], axis_vector)
        if (
            np.linalg.norm(rot_axis) > 1e-6
        ):  # if not already aligned...
            rot_angle = np.arccos(np.dot([0, 0, 1], axis_vector))
            rot_matrix = self._rotation_matrix(rot_axis, rot_angle)
            points = np.dot(points, rot_matrix)
        return points

    def _create_circle_points(
        self,
        center: Union[list, np.ndarray],
        radius: float,
        resolution: int = 20,
    ) -> np.ndarray:
        """Create a circle mesh with a given radius and resolution."""
        center = np.array(center, dtype=np.float64)
        radius_points = self._get_points_num(radius, resolution)
        x = np.linspace(-radius, radius, radius_points)
        y = np.linspace(-radius, radius, radius_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        mask = X**2 + Y**2 <= radius**2
        Z[mask] = center[2]

        return np.column_stack(
            (
                X[mask],
                Y[mask],
                Z[mask],
            )
        )

    def _create_cylinder_shell_points(
        self,
        height: Union[list, np.ndarray],
        radius: float,
        resolution: int = 20,
    ) -> np.ndarray:
        """Create a cylinder mesh with a given radius and resolution."""
        height_points = self._get_points_num(height, resolution)

        # Create cylinder shell
        theta = np.linspace(
            0, 2 * np.pi, height_points, dtype=np.float64
        )
        z = np.linspace(0, height, height_points, dtype=np.float64)
        Theta, Z = np.meshgrid(theta, z)
        x = radius * np.cos(Theta)
        y = radius * np.sin(Theta)

        return np.column_stack(
            (x.flatten(), y.flatten(), Z.flatten())
        )

    def create_cylinder(
        self,
        start_point: list[int],
        end_point: list[int],
        radius: float,
        resolution: int = 20,
    ):
        """Create a cylinder mesh between two points."""

        # Create empty arrays to store points and faces
        body_points = np.empty((0, 3), dtype=np.float64)
        body_faces = np.empty((0, 3), dtype=np.int32)

        # Convert start and end points to numpy arrays
        start_point = np.array(start_point, dtype=np.float64)
        end_point = np.array(end_point, dtype=np.float64)

        # Calculate the height and axis vector of the cylinder
        height = np.linalg.norm(end_point - start_point)
        axis_vector = (end_point - start_point) / height

        shell_points = self._create_cylinder_shell_points(
            height, radius, resolution
        )

        # Generate a grid of points within the bounding box of the circle
        bottom_points = self._create_circle_points(
            [0, 0, 0], radius, resolution
        )
        top_points = self._create_circle_points(
            [0, 0, height], radius, resolution
        )

        # Add the points to the body points
        body_points = np.vstack(
            (body_points, shell_points, top_points, bottom_points)
        )

        # Rotate the points to align with the axis vector
        body_points = self._rotate_points(body_points, axis_vector)

        points: np.ndarray = body_points
        mesh.add(points)
        return points

    def add(self, vertices: np.ndarray):
        self.vertices = np.vstack((self.vertices, vertices))


render = Render3D()
mesh = Mesh()
points = mesh.create_cylinder(
    [0, 0, 0], [5, 5, 5], 10, resolution=100
)
render.create_point_map(mesh.vertices)
render.visualize()
