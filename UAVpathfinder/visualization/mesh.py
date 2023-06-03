from typing import Union
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


from UAVpathfinder.maps.map import Map
from UAVpathfinder.visualization.render import Render3D
from UAVpathfinder.visualization.dependencies import (
    get_points_num,
    rotate_points,
    PlanarPolygon,
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
        mesh.add_vertices(points)
        return points

    def add_2d_polygon(
        self,
        vertices: list[list[float]],
        resolution: float = 0.1,
    ):
        """Add a 2D polygon to the mesh."""

        polygon = PlanarPolygon(vertices)

        # polygon.upsample_mesh(resolution)
        self.add(polygon.vertices_3d, polygon.faces)

    def add_vertices(self, vertices: np.ndarray):
        self.vertices = np.vstack((self.vertices, vertices))

    def add_faces(self, faces: np.ndarray):
        self.faces = np.vstack((self.faces, faces))

    def add(self, vertices: np.ndarray, faces: np.ndarray):
        self.add_vertices(vertices)
        self.add_faces(faces)


render = Render3D()
mesh = Mesh()
new_map = Map(
    start_coord=[48.15454749016991, 11.544871334811818],
    end_coord=[48.15633324537993, 11.545783285821432],
)
# mesh.create_cylinder([0, 0, 0], [0, 0, 10], 1, resolution=0.01)

# buildings = new_map.generate_building_faces()
buildings = [[[[0, 0, 0], [0, 0, 10], [0, 10, 10], [0, 10, 0]]]]
for building in buildings:
    for face in building:
        mesh.add_2d_polygon(face, resolution=1)

print(mesh.vertices)
print(mesh.faces)
render.create_triangular_mesh(mesh.vertices, mesh.faces)
render.visualize()
