from typing import Optional
import logging
from pydantic import BaseModel
from tqdm import tqdm
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from UAVpathfinder.maps.map_utils import coord_to_cart


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Building(BaseModel):
    color: Optional[str]
    height: Optional[float]
    footprint: Polygon

    class Config:
        arbitrary_types_allowed = True


class Map(BaseModel):
    """
    A class representing a map with geographic coordinates and buildings.

    Attributes:
        start_coord (list[float]): The geographic coordinates [latitude, longitude] of the start station.
        end_coord (list[float]): The geographic coordinates [latitude, longitude] of the end station.
        level_height (float): The height of a building level in meters.
        buffer (float): The buffer value used to create a bounding box around the start and end stations.

    Methods:
        generate_bbox() -> list[list[float]]:
            Creates a geographic bounding box around the start and end stations, including a buffer.

        get_2d_building_graph() -> nx.MultiDiGraph:
            Retrieve a 2D network graph of the buildings within a bounding box using OSMnx.

        get_2d_buildings_data() -> gpd.GeoDataFrame:
            Retrieve the 2D building data within a bounding box using OSMnx.

        plot_building_2d() -> None:
            Plot the 2D building graph and buildings in a single figure.

        avg_known_building_height() -> float:
            Calculate the average height of known buildings based on their heights and level heights.

        max_known_building_height() -> float:
            Calculate the maximum height of known buildings.

        get_building_height(building: gpd.GeoDataFrame) -> float:
            Calculate the height of known buildings and places the average height if it does not exist.

        generate_building_faces() -> list[list[list[float]]]:
            Generate 3D faces for buildings based on their 2D footprints.

    Note:
        This class inherits from the `BaseModel` class.

    """

    start_coord: list[float]
    end_coord: list[float]
    level_height = 2.5  # height of a building level
    buffer = 0.0005

    def generate_bbox(self) -> list[list[float]]:
        """
        Creates a geographic bounding box around the start and end stations, including a buffer.

        The bounding box encompasses the minimum and maximum latitude and longitude values
        of the start and end stations, with an additional buffer added to increase the coverage
        area and capture relevant nodes and edges.

        Returns:
            list[list[float]]: A list of lists representing the bounding box coordinates.
            The outer list contains two inner lists:
            - The first inner list represents the minimum latitude and longitude values.
            - The second inner list represents the maximum latitude and longitude values.
        """
        bbox_vals = [
            [
                min(self.start_coord[0], self.end_coord[0])
                - self.buffer,
                min(self.start_coord[1], self.end_coord[1])
                - self.buffer,
            ],
            [
                max(self.start_coord[0], self.end_coord[0])
                + self.buffer,
                max(self.start_coord[1], self.end_coord[1])
                + self.buffer,
            ],
        ]
        return bbox_vals

    def get_2d_building_graph(self) -> nx.MultiDiGraph:
        """
        Retrieve a 2D network graph of the buildings within a bounding box using OSMnx.

        Returns a network graph representation of the buildings within the specified bounding box.
        The bounding box is generated based on the start and end coordinates of the route.

        Returns:
            nx.MultiDiGraph: A multi-directed graph representing the buildings network.

        Note:
            This method uses OSMnx to retrieve the buildings within the bounding box. OSMnx settings
            are temporarily adjusted to control the timeout and cache usage during the retrieval process.
            The buildings are filtered based on the "building" tag.

        Raises:
            Exception: If there is an error retrieving the buildings graph using OSMnx.
        """
        # TODO: Might not even need this
        box = self.generate_bbox()

        ox.settings.timeout = 100
        ox.settings.use_cache = False

        try:
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
        except Exception as e:
            raise Exception(
                "Error retrieving buildings graph using OSMnx."
            ) from e

    def get_2d_buildings_data(self) -> gpd.GeoDataFrame:
        """
        Retrieve the 2D building data within a bounding box using OSMnx.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the building data.

        Note:
            This method uses OSMnx to retrieve the buildings within the bounding box. OSMnx settings
            are temporarily adjusted to control the timeout and cache usage during the retrieval process.
            Only buildings with the "building" tag set to True are included in the result.

        Raises:
            Exception: If there is an error retrieving the building data using OSMnx.
        """
        box = self.generate_bbox()
        try:
            buildings = ox.geometries_from_bbox(
                box[1][0],
                box[0][0],
                box[0][1],
                box[1][1],
                tags={"building": True},
            )
            return buildings
        except Exception as e:
            raise Exception(
                "Error retrieving building data using OSMnx."
            ) from e

    def plot_building_2d(self) -> None:
        """
        Plot the 2D building graph and buildings in a single figure.

        This method retrieves the 2D building graph using `get_2d_building_graph` and plots it using OSMnx's
        `plot_graph` function. It also retrieves the building data using `get_2d_buildings_data` and plots
        the buildings using GeoPandas' `plot` function. The plots are displayed together in a single figure.

        Note:
            The plots are displayed using Matplotlib. Make sure to call this method within a Jupyter Notebook
            or a script where Matplotlib is configured to show the plots inline.

        Raises:
            Exception: If there is an error retrieving the building graph or building data.
        """
        try:
            # Retrieve and plot the 2D building graph
            building_graph = self.get_2d_building_graph()
            ox.plot_graph(building_graph)

            # Retrieve and plot the building data
            building_data = self.get_2d_buildings_data()
            building_series = gpd.GeoSeries(building_data.geometry)
            building_series.plot()

            # Show the plots
            plt.show()

        except Exception as e:
            raise Exception("Error plotting 2D building data.") from e

    def avg_known_building_height(self, building_levels) -> float:
        """
        Calculate the average height of known buildings based on their heights and level heights.

        Returns:
            float: Average height of known buildings in meters.
        """
        building_levels = pd.to_numeric(
            building_levels, errors="coerce"
        )
        building_levels = building_levels.dropna()

        average_height = np.average(
            building_levels.astype(float) * self.level_height
        )
        print(average_height)
        return average_height

    def max_known_building_height(self, building_levels) -> float:
        """
        Calculate the maximum height of known buildings.

        Returns:
            float: Maximum height of known buildings in meters.
        """
        max_height = (
            max(building_levels.astype(float)) * self.level_height
        )
        return max_height

    def get_building_height(
        self, building: gpd.GeoDataFrame, average_height: float
    ) -> float:
        # TODO: what does it mean when a building height is NaN
        """
        Calculate the height of known buildings and places the average height if it does not exist.

        Returns:
            float: Height of building in meters.
        """
        building_levels = building["building:levels"]
        try:
            float(building_levels)
        except ValueError:
            return average_height
        if not pd.isna(building["building:levels"]):
            return (
                float(building["building:levels"]) * self.level_height
            )
        return average_height

    def get_all_building_heights(
        self, building_levels, avg_known_building_height
    ) -> list[float]:
        """
        Retrieve the heights of all buildings.

        Returns:
            list[float]: A list of building heights.
        """
        building_levels = building_levels.astype(float).mul(
            self.level_height,
            fill_value=avg_known_building_height,
        )
        return building_levels.tolist()

    def generate_building_faces(self) -> list[list[list[float]]]:
        """
        Generate 3D faces for buildings based on their 2D footprints.

        This method iterates over the buildings obtained from `get_2d_buildings_data` and generates 3D faces
        for each building based on its 2D footprint. The generated faces include the base of the building,
        the top face, and the side faces. The coordinates of the faces are converted to Cartesian coordinates
        using the reference point generated from `generate_bbox`. The resulting faces are returned as a list
        of nested lists.

        Returns:
            list[list[list[float]]]: A list of nested lists representing the 3D faces of the buildings.

        Note:
            The 2D buildings are obtained from `get_2d_buildings_data` which retrieves the building data.
            The `coord_to_cart` function is used to convert the geographic coordinates to Cartesian coordinates.

        Raises:
            Exception: If there is an error retrieving the 2D building data.
        """
        buildings = []
        min_ref_pt = self.generate_bbox()[0]
        logging.info("Getting buildings data...")
        buildings_data = self.get_2d_buildings_data()
        logging.info("Got buildings data...")
        building_levels = buildings_data["building:levels"]
        avg_known_building_height = self.avg_known_building_height(
            building_levels
        )
        # Iterate over buildings
        with tqdm(total=buildings_data.shape[0], unit="item") as pbar:
            for (
                _,
                building,
            ) in buildings_data.iterrows():
                height = self.get_building_height(
                    building, avg_known_building_height
                )
                if not height:
                    height = avg_known_building_height

                if building.geometry.geom_type == "Polygon":
                    base_coord = building.geometry.exterior.coords
                    base_coord_cart = np.array(
                        [
                            coord_to_cart(min_ref_pt, coord)
                            for coord in base_coord
                        ]
                    )
                    footprint = Polygon(base_coord_cart)
                    buildings.append(
                        Building(footprint=footprint, height=height)
                    )
                pbar.update(1)
        return buildings
