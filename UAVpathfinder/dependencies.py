import numpy as np
from shapely.geometry import Polygon, LineString
from UAVpathfinder.visualization import Render3D


class Mesh:
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype=np.float64)
        self.faces = np.empty((0, 3), dtype=np.int32)

    def create_cylinder(
        self, start_point, end_point, radius, resolution=20
    ):
        start_point = np.array(start_point, dtype=np.float64)
        end_point = np.array(end_point, dtype=np.float64)
        body_points = np.empty((0, 3), dtype=np.float64)
        body_faces = np.empty((0, 3), dtype=np.int32)

        height = np.linalg.norm(end_point - start_point)
        axis_vector = (end_point - start_point) / height

        theta = np.linspace(
            0, 2 * np.pi, resolution, dtype=np.float64
        )
        x = np.linspace(0, radius, resolution) * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.linspace(0, height, resolution, dtype=np.float64)

        # Create the cylinder surface
        for z_layer_val in z:
            z_layer = z_layer_val * np.ones(resolution)
            layer = np.vstack((x, y, z_layer)).T
            body_points = np.vstack((body_points, layer))

        # Create the cylinder lids

        body_points = self._rotate_points(body_points, axis_vector)
        # Create the faces
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Split each quad into two triangles
                body_faces.append([i, i + 1, j + resolution])
                body_faces.append(
                    [i + 1, j + resolution, j + resolution + 1]
                )

        # Append points and faces to the vertices and faces in the mesh
        self.vertices = np.vstack((self.vertices, body_points))
        self.faces = np.vstack(
            (self.faces, np.array(body_faces) + len(self.vertices))
        )

    def _rotate_points(self, points, axis_vector):
        # Rotate points to align with axis_vector
        rot_axis = np.cross([0, 0, 1], axis_vector)
        if (
            np.linalg.norm(rot_axis) > 1e-6
        ):  # if not already aligned...
            rot_angle = np.arccos(np.dot([0, 0, 1], axis_vector))
            rot_matrix = self._rotation_matrix(rot_axis, rot_angle)
            points = np.dot(points, rot_matrix)
        return points

    def _rotation_matrix(self, axis, theta):
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


render = Render3D()
mesh = Mesh()
mesh.create_cylinder([0, 0, 0], [3, 3, 3], 1)
render.create_mesh(mesh.vertices, mesh.faces)
render.visualize()
