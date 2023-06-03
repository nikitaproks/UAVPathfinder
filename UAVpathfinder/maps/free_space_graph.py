from typing import List, Dict
import itertools
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

    def get_building_points(self) -> list:
        def _unpack_nested_coordinates(coordinates: list):
            result = []
            for item in coordinates:
                if isinstance(item, list):
                    result.extend(_unpack_nested_coordinates(item))
                else:
                    result.append(item)
            return result

        to_unpack = _unpack_nested_coordinates(
            self.building_map.generate_building_faces()
        )
        return zip(*[to_unpack[i : i + 3] for i in range(0, len(to_unpack), 3)])

    def get_distance_steps(self):
        bbox = self.building_map.generate_bbox()

        x, y, z = self.get_building_points()

        max_y = max(y)
        max_x = max(x)
        min_y = min(y)
        min_x = max(x)

        x_distance = max_x - min_x
        y_distance = max_y - min_y

        dx = x_distance / self.x_resolution
        dy = y_distance / self.y_resolution
        dz = self.building_map.max_known_building_height() / self.z_steps

        return dx, dy, dz

    def generate_equidistant_graph(self):
        G = nx.Graph()
        pos = {}

        x, y, z = self.get_building_points()

        max_y = max(y)
        max_x = max(x)
        min_y = min(y)
        min_x = min(x)

        # node position values
        x_values = np.linspace(min_x, max_x, self.x_resolution)
        y_values = np.linspace(min_y, max_y, self.y_resolution)
        z_values = np.linspace(
            0.0,
            self.building_map.max_known_building_height(),
            self.z_steps,
        )

        # node values
        i_values = np.arange(self.x_resolution).astype(int)
        j_values = np.arange(self.y_resolution).astype(int)
        k_values = np.arange(self.z_steps).astype(int)

        grid_coordinates = np.transpose(
            [
                np.tile(x_values, self.y_resolution * self.z_steps),
                np.tile(np.repeat(y_values, self.x_resolution), self.z_steps),
                np.repeat(z_values, self.x_resolution * self.y_resolution),
            ]
        )

        node_vals = np.transpose(
            [
                np.tile(i_values, self.y_resolution * self.z_steps),
                np.tile(np.repeat(j_values, self.x_resolution), self.z_steps),
                np.repeat(k_values, self.x_resolution * self.y_resolution),
            ]
        )
        # Map the node coordinates to positions
        pos = {tuple(node): coord for node, coord in zip(node_vals, grid_coordinates)}
        for node in pos.keys():
            G.add_node(node)

            i, j, k = node

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

    def plot_network_grid(self):
        G, pos = self.generate_equidistant_graph()
        node_xyz = np.array([pos[v] for v in G])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        print(node_xyz)
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(*node_xyz.T, s=10, ec="w")

        x, y, z = self.get_building_points()
        ax.scatter(x, y, z, s=10, ec="r", c="r")

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
