import numpy as np
from scipy.spatial import Delaunay
import trimesh


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


class PlanarPolygon:
    def __init__(self, vertices: list[list[float]]):
        vertices = np.array(vertices)
        if not self.is_planar(vertices):
            raise ValueError(
                "The vertices do not form a flat 3D polygon"
            )
        self.vertices_3d = vertices
        self.vertices_2d = self.project_to_2d(vertices)
        self.faces = self.get_triangular_faces(self.vertices_2d)

    def is_planar(self, vertices):
        if len(vertices) < 3:
            return True  # If there are less than 3 vertices, then it's always planar

        # Take three points, form two vectors, compute their cross product
        v1 = np.subtract(vertices[1], vertices[0])
        v2 = np.subtract(vertices[2], vertices[0])
        normal = np.cross(v1, v2)

        for v in vertices[3:]:
            # Compute vector from one of the original points to this one
            v3 = np.subtract(v, vertices[0])
            # Compute dot product of this vector with the normal - should be zero
            # if the point lies in the same plane, within a small tolerance
            if abs(np.dot(normal, v3)) > 1e-6:
                return False

        # If we get through the entire loop without finding a non-coplanar point, it's planar
        return True

    def project_to_2d(self, vertices):
        v1 = np.subtract(vertices[1], vertices[0])
        normal = np.cross(v1, np.subtract(vertices[2], vertices[0]))
        self.v2 = np.cross(normal, v1)

        basis = np.array(
            [
                v1 / np.linalg.norm(v1),
                self.v2 / np.linalg.norm(self.v2),
            ]
        )
        self.basis = basis

        return np.dot(vertices - vertices[0], basis.T)

    def project_to_3d(self, vertices_2d):
        return vertices_2d @ self.basis + self.vertices_3d[0]

    def upsample_mesh(self, resolution: int):
        mesh = trimesh.Trimesh(
            vertices=self.vertices_2d, faces=self.faces
        )

        avg_edge_length = self.get_average_edge_length()
        subdivision_levels = int(avg_edge_length / resolution)

        for _ in range(subdivision_levels):
            mesh = mesh.subdivide()

        self.vertices_2d = mesh.vertices
        self.faces = mesh.faces
        self.vertices_3d = self.project_to_3d(self.vertices_2d)

    def triangulate_2D(self, vertices_2d):
        # Perform Delaunay triangulation on the polygon
        triangulation = Delaunay(vertices_2d)
        # Extract the triangles from the triangulation result
        triangles_indices = triangulation.simplices
        return triangles_indices

    def get_triangular_faces(self, vertices_2d):
        triangles_indices = self.triangulate_2D(vertices_2d)
        return triangles_indices

    def get_average_edge_length(self):
        total_length = 0.0
        num_edges = 0

        for face in self.faces:
            # Get the vertices of the face
            v1, v2, v3 = self.vertices_2d[face]

            # Calculate the lengths of the edges
            edge_lengths = [
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v3 - v2),
                np.linalg.norm(v1 - v3),
            ]

            # Add up the lengths of the edges
            total_length += sum(edge_lengths)
            num_edges += 3

        # Calculate the average edge length
        average_length = total_length / num_edges

        return average_length
