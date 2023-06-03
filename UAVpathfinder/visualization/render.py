import open3d as o3d


class Render3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        opt = self.vis.get_render_option()
        opt.point_size = 3

        # get the view control object and change field of view
        ctr = self.vis.get_view_control()
        ctr.change_field_of_view(step=50.0)

    def create_point_cloud(self, vertices):
        self.mesh = o3d.geometry.PointCloud()
        self.mesh.points = o3d.utility.Vector3dVector(vertices)
        self.mesh.paint_uniform_color([0.1, 0.1, 0.7])
        self.vis.add_geometry(self.mesh)

    def create_triangular_mesh(self, vertices, faces):
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.1, 0.1, 0.7])
        self.vis.add_geometry(self.mesh)

    def visualize(self):
        # run the visualizer
        self.vis.run()
        self.vis.destroy_window()
