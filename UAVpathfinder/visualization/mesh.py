from typing import Union
import numpy as np
from tqdm import tqdm
import trimesh
from shapely.geometry import Polygon
from UAVpathfinder.maps.map import Map, Building
from UAVpathfinder.visualization.render import Render3D
from UAVpathfinder.visualization.dependencies import (
    get_position_rotation_height,
    PlanarPolygon,
)
from UAVpathfinder.maps.free_space_graph import FreeSpaceGraph


class Mesh:
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype=np.float64)
        self.triangles = np.empty((0, 3, 3), dtype=np.float64)

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

    def _create_cylinder(
        self,
        start_point: list[int],
        end_point: list[int],
        radius: float,
    ):
        """Create a cylinder mesh between two points."""

        start_point = np.array(start_point, dtype=np.float64)
        end_point = np.array(end_point, dtype=np.float64)

        position: np.ndarray
        rotation: np.ndarray
        height: float

        position, rotation, height = get_position_rotation_height(
            start_point, end_point
        )

        # Create the transformation matrix
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = rotation[:3, :3]

        cylinder_mesh = trimesh.creation.cylinder(
            radius=radius, height=height, transform=transform
        )
        return cylinder_mesh

    def add_path(self, path: list[list[int]], radius: float):
        """Add a path to the mesh."""
        for i in range(len(path) - 2):
            cylinder_mesh = self._create_cylinder(
                path[i], path[i + 1], radius
            )
            self._add_triangles(cylinder_mesh.triangles)

    def add_box(
        self,
        polygon: Polygon,
        height: float,
    ):
        """Add a box to the mesh."""
        polygon_mesh = trimesh.creation.extrude_polygon(
            polygon, height
        )
        self._add_triangles(polygon_mesh.triangles)

    def _add_vertices(self, vertices: np.ndarray):
        self.vertices = np.vstack((self.vertices, vertices))

    def _add_triangles(self, triangles: np.ndarray):
        self.triangles = np.vstack((self.triangles, triangles))

    def get_vertices_faces(self):
        vertices = np.unique(self.triangles.reshape(-1, 3), axis=0)
        _, faces = np.unique(
            self.triangles.reshape(-1, 3), axis=0, return_inverse=True
        )
        faces = faces.reshape(self.triangles.shape[:-1])
        back_faces = faces[:, ::-1]
        all_faces = np.concatenate((faces, back_faces), axis=0)
        return vertices, all_faces

    def get_bounding_box(self):
        vertices, faces = self.get_vertices_faces()
        min_x = np.min(vertices[:, 0])
        min_y = np.min(vertices[:, 1])
        min_z = np.min(vertices[:, 2])
        max_x = np.max(vertices[:, 0])
        max_y = np.max(vertices[:, 1])
        max_z = np.max(vertices[:, 2])
        return [
            [min_x, min_y, min_z],
            [max_x, max_y, max_z],
        ]


if __name__ == "__main__":
    render = Render3D()
    mesh = Mesh()
    new_map = Map(
        start_coord=[25.273621, 55.316846],
        end_coord=[25.265242, 55.333434],
    )

    # new_graph = FreeSpaceGraph(
    #     x_resolution=30,
    #     y_resolution=30,
    #     z_steps=5,
    #     building_map=new_map,
    #     nogo_zones=[0],  # in meters
    # )
    # path = new_graph.determine_shortest_path()
    # print(path)

    buildings: list[Building] = new_map.generate_building_faces()
    for building in tqdm(buildings):
        mesh.add_box(building.footprint, building.height)

    vertices, faces = mesh.get_vertices_faces()
    bounding_box = mesh.get_bounding_box()

    render.create_triangular_mesh(vertices, faces)
    # render.create_floor_mesh(bounding_box)
    render.visualize()
