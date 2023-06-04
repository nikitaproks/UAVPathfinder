from typing import List, Dict, Tuple
from pydantic import BaseModel
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import numpy as np
import osmnx as ox
from UAVpathfinder.maps.map import Map
from UAVpathfinder.maps.map_utils import coord_to_cart


class FreeSpaceGraph(BaseModel):
    # TODO: Make docstring.
    x_resolution: int  # in meters
    y_resolution: int
    z_steps: int
    building_map: Map
    nogo_zones: list  # placeholders
    height_offset = 5.0

    def get_building_points(self) -> List[List[float]]:
        # TODO: this can be done way better, but also means some changes in Map.
        """
        Retrieve the 3D base coordinates of building points from the generated building faces.

        This method retrieves the 3D coordinates of building points from the generated building faces.
        The building faces are obtained from the `generate_building_faces` method of the `building_map` object.

        Returns:
            list: A list of lists representing the 3D coordinates of building points.
                  Each tuple contains the x, y, and z coordinates of a building point.

        Note:
            This method internally uses the `_unpack_nested_coordinates` function to unpack the nested coordinates
            obtained from the `generate_building_faces` method.

        Raises:
            None.
        """

        def _unpack_nested_coordinates(coordinates: list):
            """
            Helper function to unpack nested coordinates.

            Recursively unpacks nested coordinates and returns a flattened list.

            Parameters:
                coordinates (list): A list of nested coordinates.

            Returns:
                list: A flattened list of coordinates.
            """
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
        return zip(
            *[
                to_unpack[i : i + 3]
                for i in range(0, len(to_unpack), 3)
            ]
        )

    def generate_equidistant_graph(
        self,
    ) -> Tuple[
        nx.Graph,
        Dict[Tuple[int, int, int], Tuple[float, float, float]],
    ]:
        """
        Generate an equidistant 3D graph based on building points for an entire region of buildings.

        This method generates an equidistant 3D graph using the building points obtained from the `get_building_points`
        method. The graph represents the connectivity between neighboring nodes in the x, y, and z directions.
        Each node in the graph is a 3D coordinate, and the edges represent the connections between neighboring nodes.

        Returns:
            Tuple[nx.Graph, Dict[Tuple[int, int, int], Tuple[float, float, float]]]: A tuple containing the generated graph
            and the positions of each node in the graph. The graph is represented as an `nx.Graph` object, and the positions
            are stored in a dictionary where the keys are the node coordinates (x, y, z) and the values are the corresponding
            Cartesian coordinates (x, y, z).

        Note:
            The graph generation is based on the specified resolution values (`x_resolution`, `y_resolution`, and `z_steps`)
            and the maximum known building height obtained from the `max_known_building_height` method of the `building_map`.

        Raises:
            None.
        """
        G = nx.Graph()
        pos = {}

        x, y, _ = self.get_building_points()
        print(x, y)
        # node position values
        x_values = np.linspace(min(x), max(x), self.x_resolution)
        y_values = np.linspace(min(y), max(y), self.y_resolution)
        z_values = np.linspace(
            0.0,
            self.building_map.max_known_building_height()
            + self.height_offset,
            self.z_steps,
        )
        all_building_bases = np.array(
            self.building_map.get_all_building_bases(), dtype=object
        )
        building_heights = np.array(
            self.building_map.get_all_building_heights()
        )

        def _is_point_inside_building(coordinates) -> bool:
            """
            Hekper functions that checks if a point (x, y) is inside a closed shape formed by connecting several other
            points.

            Args:
                x (float): x-coordinate of the point.
                y (float): y-coordinate of the point.
                boundary_points (list): List of points forming the boundary of the shape. The points should be ordered
                    in a way that connecting them in sequence would form the closed shape.

            Returns:
                bool: True if the point is inside the shape, False otherwise.
            """
            coordinates = np.array(coordinates)

            for bases, heights in zip(
                all_building_bases, building_heights
            ):
                base_points = np.array(bases)[:, :-1]

                # Create the path object outside the loop
                path = mpl_path.Path(base_points)

                # Perform the containment check and height comparison using numpy broadcasting
                contains_point = path.contains_point(coordinates[:2])
                height_check = coordinates[2] <= heights

                # Combine the conditions using logical AND
                conditions = np.logical_and(
                    contains_point, height_check
                )
                if np.any(conditions):
                    return False
                # Check if any condition is False (i.e., any point fails the conditions)
            return True

        grid_coordinates = np.transpose(
            [
                np.tile(x_values, self.y_resolution * self.z_steps),
                np.tile(
                    np.repeat(y_values, self.x_resolution),
                    self.z_steps,
                ),
                np.repeat(
                    z_values, self.x_resolution * self.y_resolution
                ),
            ]
        )

        # node values
        i_values = np.arange(self.x_resolution).astype(int)
        j_values = np.arange(self.y_resolution).astype(int)
        k_values = np.arange(self.z_steps).astype(int)

        node_vals = np.transpose(
            [
                np.tile(i_values, self.y_resolution * self.z_steps),
                np.tile(
                    np.repeat(j_values, self.x_resolution),
                    self.z_steps,
                ),
                np.repeat(
                    k_values, self.x_resolution * self.y_resolution
                ),
            ]
        )
        # Map the node coordinates to positions
        pos = {
            tuple(node): coord
            for node, coord in zip(node_vals, grid_coordinates)
            if _is_point_inside_building(coord)
        }

        remaining_nodes = pos.keys()
        for node in remaining_nodes:
            G.add_node(node)

            i, j, k = node

            dz_values = np.array([-1, 1])
            dx_values = np.array([-1, 1])
            dy_values = np.array([-1, 1])

            dz_neighbors = (k + dz_values).clip(0, self.z_steps - 1)
            dx_neighbors = (i + dx_values).clip(
                0, self.x_resolution - 1
            )
            dy_neighbors = (j + dy_values).clip(
                0, self.y_resolution - 1
            )

            neighbors = np.transpose(
                [
                    np.repeat(i, len(dz_neighbors)),
                    np.repeat(j, len(dx_neighbors)),
                    dz_neighbors,
                ]
            )
            neighbors = np.concatenate(
                (
                    neighbors,
                    np.transpose(
                        [
                            dx_neighbors,
                            np.repeat(j, len(dx_neighbors)),
                            np.repeat(k, len(dx_neighbors)),
                        ]
                    ),
                )
            )
            neighbors = np.concatenate(
                (
                    neighbors,
                    np.transpose(
                        [
                            np.repeat(i, len(dy_neighbors)),
                            dy_neighbors,
                            np.repeat(k, len(dy_neighbors)),
                        ]
                    ),
                )
            )

            valid_neighbors = set(map(tuple, neighbors)) & set(
                remaining_nodes
            )  # Intersection of neighbors and remaining_set

            for neighbor in valid_neighbors:
                G.add_edge(node, neighbor)
        return G, pos

    def determine_shortest_path(self):
        start_point = [0.0, 0.0, 0.0]  # always the reference point
        end_point = coord_to_cart(
            self.building_map.start_coord,
            [
                self.building_map.end_coord[1],
                self.building_map.end_coord[0],
            ],
        ) + [0.0]
        G, pos = self.generate_equidistant_graph()
        start_distances = {
            node: np.linalg.norm(
                np.array(start_point) - np.array(node)
            )
            for node in G.nodes
        }
        end_distances = {
            node: np.linalg.norm(np.array(end_point) - np.array(node))
            for node in G.nodes
        }

        start_node = min(start_distances, key=start_distances.get)
        end_node = min(end_distances, key=end_distances.get)

        direct_path = nx.shortest_path(
            G, source=start_node, target=end_node
        )
        return np.array([pos[v] for v in direct_path])

    def plot_network_grid(self) -> None:
        """
        Plot the network grid in a 3D visualization.

        This method generates a 3D visualization of the network grid by plotting the equidistant graph
        generated from the `generate_equidistant_graph` method. The building points obtained from the
        `get_building_points` method are also plotted.

        Returns:
            None.

        Note:
            - The equidistant graph is generated using the `generate_equidistant_graph` method.
            - The building points are obtained using the `get_building_points` method.
            - The plot includes scatter points for the nodes of the equidistant graph and the building points,
              as well as lines representing the edges of the equidistant graph.

        Raises:
            None.
        """
        G, pos = self.generate_equidistant_graph()
        node_xyz = np.array([pos[v] for v in G])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        path_graph = self.determine_shortest_path()

        nodes_path = np.array([pos[v] for v in path_graph])
        # edges_path = np.array([(pos[u], pos[v]) for u, v in path_graph.edges()])

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # ax.scatter(*node_xyz.T, s=10, ec="w")
        # Plot the edges
        # for vizedge in edge_xyz:
        # ax.plot(*vizedge.T, color="tab:gray")

        ax.scatter(*nodes_path.T, s=10, ec="g")
        # Plot the edges
        # for vizedge in edges_path:
        # ax.plot(*vizedge.T, color="tab:green")

        # x, y, z = self.get_building_points()
        # ax.scatter(x, y, z, s=10, ec="r", c="r")

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
