import numpy as np
from scipy.spatial import Delaunay


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
            print("The vertices do not form a flat 3D polygon")
            self.is_valid = False
            return
        if self.is_collinear(vertices):
            # print("The vertices are collinear")
            self.is_valid = False
            return
        self.is_valid = True
        self.vertices_3d = vertices
        self.vertices_2d = self.project_to_2d(vertices)
        self.triangles_2d, self.faces = self.triangulate_2D(
            self.vertices_2d
        )
        self.triangles_3d = self.triangles_to_3d(self.triangles_2d)

    def is_collinear(self, vertices):
        # Ensure points are a numpy array
        vertices = np.asarray(vertices)

        # Iterate over all triples of points
        for i in range(len(vertices) - 2):
            # Get vectors between consecutive points
            v1 = vertices[i + 1] - vertices[i]
            v2 = vertices[i + 2] - vertices[i + 1]

            # If the cross product of the vectors isn't close to zero, the points aren't collinear
            if np.linalg.norm(np.cross(v1, v2)) > 1e-5:
                return False

        return True

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
        v2 = np.subtract(vertices[2], vertices[0])

        normal = np.cross(v1, v2)
        v2 = np.cross(normal, v1)

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        self.basis = np.array([v1, v2])
        self.point_3D = vertices[0]

        return np.dot(vertices - vertices[0], self.basis.T)

    def project_to_3d(self, vertices_2d):
        return np.dot(vertices_2d, self.basis) + self.point_3D

    def triangulate_2D(self, vertices_2d):
        # Perform Delaunay triangulation on the polygon
        triangulation = Delaunay(vertices_2d)
        # Extract the triangles from the triangulation result
        triangles = vertices_2d[triangulation.simplices]
        faces = triangulation.simplices
        return triangles, faces

    def triangles_to_3d(self, triangles):
        return np.array(
            [self.project_to_3d(triangle) for triangle in triangles]
        )
