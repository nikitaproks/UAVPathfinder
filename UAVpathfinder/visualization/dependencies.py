import numpy as np
import trimesh
from scipy.spatial import Delaunay


def get_position_rotation_height(
    start_point: np.ndarray, end_point: np.ndarray
):
    # Calculate the height (distance) between the start and end points
    height = np.linalg.norm(end_point - start_point)

    # Calculate the position as the midpoint between the start and end points
    position = (end_point - start_point) / 2 + start_point

    # Calculate the direction vector from start to end points
    direction = (end_point - start_point) / height

    # Calculate the rotation matrix to align the cylinder with the direction vector
    z_axis = np.array([0, 0, 1])  # Desired Z-axis for the cylinder
    rotation = trimesh.transformations.rotation_matrix(
        trimesh.transformations.angle_between_vectors(
            z_axis, direction
        ),
        np.cross(z_axis, direction),
    )

    return position, rotation, height


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
