# from UAVpathfinder.maps import map
from map import Map
from free_space_graph import FreeSpaceGraph


new_map = Map(
    start_coord=[48.15454749016991, 11.544871334811818],
    end_coord=[48.15633324537993, 11.545783285821432],
)

new_graph = FreeSpaceGraph(
    x_resolution=10,
    y_resolution=10,
    z_steps=3,
    building_map=new_map,
    nogo_zones=[0],  # in meters
)

new_graph.plot_network_gid()


"""

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def generate_equidistant_graph(bbox, resolution, num_layers):
    G = nx.Graph()

    # Generate nodes in the bounding box
    x_min, y_min, x_max, y_max = bbox
    dx, dy = resolution

    for x in range(x_min, x_max, dx):
        for y in range(y_min, y_max, dy):
            for z in range(num_layers):
                node = (x, y, z)
                G.add_node(node)

                # Connect the node to adjacent nodes
                for dz in [-1, 1]:
                    if z + dz >= 0 and z + dz < num_layers:
                        neighbor = (x, y, z + dz)
                        G.add_edge(node, neighbor)

                for dx in [-1, 1]:
                    if x + dx >= x_min and x + dx < x_max:
                        neighbor = (x + dx, y, z)
                        G.add_edge(node, neighbor)

                for dy in [-1, 1]:
                    if y + dy >= y_min and y + dy < y_max:
                        neighbor = (x, y + dy, z)
                        G.add_edge(node, neighbor)

    return G


# Define the parameters
bbox = (0, 0, 10, 10)  # Bounding box coordinates (x_min, y_min, x_max, y_max)
resolution = (1, 1)  # Cartesian distance between nodes (dx, dy)
num_layers = 3  # Number of z-axis layers
layer_distance = 1  # Spacial distance between z-axis layers

# Generate the graph
G = generate_equidistant_graph(bbox, resolution, num_layers, layer_distance)
pos = nx.spring_layout(G, dim=3, seed=779, pos=None)
print(G.nodes())
# Extract node and edge positions from the layout
node_xyz = np.array([np.array(v) for v in G])
edge_xyz = np.array([(np.array(u), np.array(v)) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")


def _format_axes(ax):
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
"""
