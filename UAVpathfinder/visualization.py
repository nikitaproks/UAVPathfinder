import open3d as o3d


class Render3D:
    def __init__(self):
        self.mesh = o3d.geometry.PointCloud()

    def create_mesh(self, vertices, faces):
        self.mesh.points = o3d.utility.Vector3dVector(vertices)
        # self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh.paint_uniform_color([0.1, 0.1, 0.7])

    def visualize(self):
        vis = o3d.visualization.Visualizer()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0]
        )
        vis.create_window()
        vis.add_geometry(self.mesh)
        vis.add_geometry(frame)

        # get the view control object and change field of view
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=50.0)

        # run the visualizer
        vis.run()
        vis.destroy_window()
