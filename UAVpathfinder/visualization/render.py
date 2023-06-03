import open3d as o3d
import numpy as np


class Render3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        opt = self.vis.get_render_option()
        opt.point_size = 4

        # get the view control object and change field of view
        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(step=50.0)

    def create_floor_mesh(self, bounding_box):
        # Get the minimum and maximum coordinates in the x and y directions
        x_min, y_min, _ = bounding_box[0]
        x_max, y_max, _ = bounding_box[1]

        x_dist = (x_max - x_min) + 50
        y_dist = (y_max - y_min) + 50
        z_dist = 0.01

        # Create a rectangular mesh representing the floor
        self.floor_mesh = o3d.geometry.TriangleMesh.create_box(
            height=y_dist, width=x_dist, depth=z_dist
        )

        # Set the position of the floor
        self.floor_mesh.translate(
            (x_min - 25, y_min - 25, -z_dist / 2)
        )
        color = np.array([0.3, 0.3, 0.3])  # Grey color
        self.floor_mesh.paint_uniform_color(color)
        self.vis.add_geometry(self.floor_mesh)

    def create_point_cloud(self, vertices):
        self.mesh = o3d.geometry.PointCloud()
        self.mesh.points = o3d.utility.Vector3dVector(vertices)
        self.mesh.paint_uniform_color([0.2, 0.2, 0.5])
        self.vis.add_geometry(self.mesh)

    def create_triangular_mesh(self, vertices, faces):
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh.remove_unreferenced_vertices()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_degenerate_triangles()
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.1, 0.1, 0.7])

        self.vis.add_geometry(self.mesh)

    def visualize(self):
        # run the visualizer
        self.vis.run()
        self.vis.destroy_window()
