from typing import List, Union
from pydantic import BaseModel
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd


class Map(BaseModel):
    start_coord: List[float]
    end_coord: List[float]
    level_height = 2.5  # height of a building level
    buffer = 0.001

    def generate_bbox(self) -> List[List[float]]:
        """
        Creates a geographic box around the start and end stations for later graph retrieval.

        The buffer is added to increase the area areound the start and end location and capture relevant
        nodes and edges.
        """
        bbox_vals = [
            [
                min(self.start_coord[0], self.end_coord[0]) - self.buffer,
                min(self.start_coord[1], self.end_coord[1]) - self.buffer,
            ],
            [
                max(self.start_coord[0], self.end_coord[0]) + self.buffer,
                max(self.start_coord[1], self.end_coord[1]) + self.buffer,
            ],
        ]  # [min_lat, min_lon], [max_lat, max_lon]
        return bbox_vals

    def get_2D_building_graph(self) -> nx.MultiDiGraph:
        """
        Return the raw network graph of the buildings given a bounding box from OSMnx,
        """
        box = self.generate_bbox()
        ox.settings.timeout = 100
        ox.settings.use_cache = False
        return ox.graph_from_bbox(
            box[1][0],
            box[0][0],
            box[0][1],
            box[1][1],
            retain_all=False,
            truncate_by_edge=True,
            simplify=False,
            custom_filter='["building"]',
        )
        # ymax, ymin, xmin, xmax

    def get_2d_buildings_data(self) -> gpd.GeoDataFrame:
        box = self.generate_bbox()
        # Get buildings within the bounding box
        buildings = ox.geometries_from_bbox(
            box[1][0], box[0][0], box[0][1], box[1][1], tags={"building": True}
        )
        return buildings

    def plot_building_2D(self) -> None:
        # Plot the graph
        ox.plot_graph(
            self.get_2D_building_graph(),
        )
        plt.show()

        building_series = gpd.GeoSeries(self.get_2d_buildings_data().geometry)
        building_series.plot()
        plt.show()

    def avg_known_building_height(self) -> float:
        """
        Calculate the average height of known buildings based on their heights and level heights.

        Returns:
            float: Average height of known buildings in meters.
        """
        building_levels = self.get_2d_buildings_data()["building:levels"].dropna()
        building_levels = building_levels.astype(float)
        average_height = np.average(building_levels * self.level_height)
        return average_height

    def get_building_height(self, building: gpd.GeoDataFrame) -> float:
        # TODO: what does it mean when a building height is NaN
        if not pd.isna(building["building:levels"]):
            return float(building["building:levels"]) * self.level_height
        return self.avg_known_building_height()

    def generate_building_faces(self) -> List[List[List[float]]]:
        faces = []

        # Iterate over buildings
        for i, building in self.get_2d_buildings_data().iterrows():
            # Extract building height and footprint
            height = self.get_building_height(building)
            footprint = building.geometry

            # Check if footprint is a polygon
            # Points are ignored, only plygons are taken.
            if footprint.geom_type == "Polygon":
                base_coord = footprint.exterior.coords
                # Create vertices for the building's 3D face
                building_faces = [
                    [[coord[0], coord[1], 0] for coord in base_coord],
                    [[coord[0], coord[1], height] for coord in base_coord],
                ]

                for idx, point in enumerate(base_coord):
                    if idx == len(base_coord) - 1:
                        building_faces.append(
                            [
                                [base_coord[0][0], base_coord[1][0], 0],
                                [base_coord[0][0], base_coord[1][0], height],
                                [base_coord[-1][0], base_coord[-1][1], 0],
                                [base_coord[-1][0], base_coord[-1][1], height],
                            ]
                        )
                    else:
                        building_faces.append(
                            [
                                [point[0], point[1], 0],
                                [point[0], point[1], height],
                                [
                                    base_coord[idx + 1][0],
                                    base_coord[idx + 1][1],
                                    0,
                                ],
                                [
                                    base_coord[idx + 1][0],
                                    base_coord[idx + 1][1],
                                    height,
                                ],
                            ]
                        )

                faces.append(building_faces)

        return faces
