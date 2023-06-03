from typing import Union
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


from UAVpathfinder.maps.map import Map
from UAVpathfinder.visualization.render import Render3D
from UAVpathfinder.visualization.dependencies import (
    get_points_num,
    rotate_points,
    project_3d_to_2d,
    project_2d_to_3d,
)


class Mesh:
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype=np.float64)
        self.faces = np.empty((0, 3), dtype=np.int32)

    def _create_circle_points(
        self,
        center: Union[list, np.ndarray],
        radius: float,
        resolution: float = 0.1,
    ) -> np.ndarray:
        """Create a circle mesh with a given radius and resolution."""
        center = np.array(center, dtype=np.float64)
        x = np.arange(-radius, radius, resolution)
        y = np.arange(-radius, radius, resolution)
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
        resolution: float = 0.1,
    ) -> np.ndarray:
        """Create a cylinder mesh with a given radius and resolution."""

        # Create cylinder shell
        theta = np.arange(0, 2 * np.pi, resolution, dtype=np.float64)
        z = np.arange(0, height, resolution, dtype=np.float64)
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
        resolution: float = 0.1,
    ):
        """Create a cylinder mesh between two points."""

        # Create empty arrays to store points and faces
        body_points = np.empty((0, 3), dtype=np.float64)

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
        body_points = rotate_points(body_points, axis_vector)

        points: np.ndarray = body_points
        mesh.add(points)
        return points

    def add_2d_polygon(
        self,
        boundary_points_3d: list[list[float]],
        resolution: float = 0.1,
    ):
        """Add a 2D polygon to the mesh."""

        boundary_points_3d = np.array(boundary_points_3d) / 10
        boundary_points_2d, polygon_flat_dim = project_3d_to_2d(
            boundary_points_3d
        )

        polygon_2d = Polygon(boundary_points_2d)
        flat_dim_value = boundary_points_3d[0, polygon_flat_dim]

        # Bounding box
        min_x, min_y, max_x, max_y = polygon_2d.bounds
        # Generate points
        grid_points_2d = []
        print(boundary_points_3d)
        print(min_x, min_y, max_x, max_y)
        for x in np.arange(min_x, max_x, resolution):
            for y in np.arange(min_y, max_y, resolution):
                point = Point(x, y)
                if polygon_2d.contains(point):
                    grid_points_2d.append((x, y))

        polygon_points_2d = np.empty((0, 2), dtype=np.float64)

        if len(grid_points_2d) != 0:
            polygon_points_2d = np.vstack(
                (polygon_points_2d, grid_points_2d)
            )

        polygon_points_2d = np.vstack(
            (polygon_points_2d, boundary_points_2d)
        )

        polygon_points_3d = project_2d_to_3d(
            polygon_points_2d, polygon_flat_dim, flat_dim_value
        )
        self.add(polygon_points_3d)
        print("Done")
        return polygon_points_3d

    def add(self, vertices: np.ndarray):
        self.vertices = np.vstack((self.vertices, vertices))


render = Render3D()
mesh = Mesh()
new_map = Map(
    start_coord=[48.15454749016991, 11.544871334811818],
    end_coord=[48.15633324537993, 11.545783285821432],
)
# mesh.create_cylinder([0, 0, 0], [0, 0, 10], 1, resolution=0.01)

buildings = new_map.generate_building_faces()
from tqdm import tqdm

for building in buildings:
    for faces in building:
        mesh.add_2d_polygon(faces, resolution=0.0001)

render.create_point_map(mesh.vertices)
render.visualize()
