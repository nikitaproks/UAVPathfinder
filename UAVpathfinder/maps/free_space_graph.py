from typing import List, Dict
import math
from pydantic import BaseModel
from map import Map
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from utils import haversine_distance


class FreeSpaceGraph(BaseModel):
    x_resolution: int  # in meters
    y_resolution: int
    z_steps: int
    building_map: Map
    nogo_zones: list  # placeholders

    def get_distance_steps(self) -> nx.MultiDiGraph:
        bbox = self.building_map.generate_bbox()

        x_distance = self.haversine_distance(bbox[0], [bbox[0][0], bbox[1][1]])
        y_distance = self.haversine_distance(bbox[1], [bbox[0][0], bbox[1][1]])

        dx = x_distance / self.x_resolution
        dy = y_distance / self.y_resolution
        dz = self.building_map.max_known_building_height() / self.z_steps

        return dx, dy, dz

    def generate_equidistant_graph(self):
        G = nx.Graph()
        pos = {}
        for i in range(self.x_resolution):
            for j in range(self.y_resolution):
                for k in range(self.z_steps):
                    x = i * self.get_distance_steps()[0]
                    y = j * self.get_distance_steps()[1]
                    z = k * self.get_distance_steps()[2]
                    node = (i, j, k)
                    pos[node] = np.array([x, y, z])
                    G.add_node(node)

                    # Connect the node to adjacent nodes
                    for dz in [-1, 1]:
                        if k + dz >= 0 and k + dz < self.z_steps:
                            neighbor = (i, j, k + dz)
                            G.add_edge(node, neighbor)

                    for dx in [-1, 1]:
                        if i + dx >= 0 and i + dx < self.x_resolution:
                            neighbor = (i + dx, j, k)
                            G.add_edge(node, neighbor)

                    for dy in [-1, 1]:
                        if j + dy >= 0 and j + dy < self.y_resolution:
                            neighbor = (i, j + dy, k)
                            G.add_edge(node, neighbor)

        return G, pos

    def plot_network_gid(self):
        G, pos = self.generate_equidistant_graph()
        node_xyz = np.array([pos[v] for v in G])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(True)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        _format_axes(ax)
        fig.tight_layout()
        plt.show()
