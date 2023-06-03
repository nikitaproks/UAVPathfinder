from typing import List
import math
from pydantic import BaseModel
from map import Map
import networkx as nx
import matplotlib.pyplot as plt


class FreeSpaceGraph(BaseModel):
    xy_resolution: float  # in meters
    z_steps: int
    building_map: Map
    nogo_zones: list  # placeholders

    def haversine_distance(self, coord_1: List[float], coord_2: List[float]) -> float:
        """
        Calculate the Haversine distance between two coordinates to help determine the
        initial grid.

        Parameters:
            coord_1 (List[float]): second coordinate in degrees.
            coord_2 (List[float]): first coordinate in degrees.

        Returns:
            float: The distance between the two coordinates in meters.
        """
        lat1_rad = math.radians(coord_1[0])
        lon1_rad = math.radians(coord_1[1])
        lat2_rad = math.radians(coord_2[0])
        lon2_rad = math.radians(coord_2[1])

        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = 6371.1370 * c * 1000  # Earth's radius in kilometers

        return distance

    def initialize_graph(self) -> nx.MultiDiGraph:
        bounding_box = self.building_map.generate_bbox()

        G = nx.grid_graph(dim=(2, 3, 4))
        nx.draw_networkx(G)  # Draw the nodes and edges
        plt.show()
