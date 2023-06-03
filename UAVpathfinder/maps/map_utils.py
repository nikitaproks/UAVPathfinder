from typing import List
import numpy as np


def haversine_distance(
    coord_1: List[float], coord_2: List[float]
) -> float:
    """
    Calculate the Haversine distance between two coordinates.

    The Haversine distance is a formula used to determine the great-circle distance between two points
    on the Earth's surface, given their longitudes and latitudes. It provides an approximation of the
    distance along the surface of a sphere.

    Parameters:
        coord_1 (List[float]): The first coordinate in degrees [latitude, longitude].
        coord_2 (List[float]): The second coordinate in degrees [latitude, longitude].

    Returns:
        float: The distance between the two coordinates in meters.

    Note:
        The Haversine formula assumes a spherical Earth with a radius of 6371752.3 meters.
    """
    try:
        lat1_rad = np.radians(coord_1[0])
        lon1_rad = np.radians(coord_1[1])
        lat2_rad = np.radians(coord_2[0])
        lon2_rad = np.radians(coord_2[1])

        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad)
            * np.cos(lat2_rad)
            * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = 6371752.3 * c  # Earth's radius in kilometers

        return abs(distance)
    except Exception as e:
        raise ValueError(
            "Error calculating Haversine distance."
        ) from e


def coord_to_cart(
    relative_coord: List[float], coord: List[float]
) -> List[float]:
    """
    Convert geographic coordinates to Cartesian coordinates.

    This function takes a relative coordinate and a target coordinate, flips their order to match the
    convention of (latitude, longitude), and performs the conversion from geographic coordinates to
    Cartesian coordinates. The conversion involves calculating the x and y values based on the
    difference in longitude and latitude between the target coordinate and the relative coordinate.

    Args:
        relative_coord (List[float]): The relative coordinate representing the reference point.
        coord (List[float]): The target coordinate to be converted.

    Returns:
        List[float]: The converted Cartesian coordinates [x, y].

    Note:
        The conversion formula used assumes a spherical Earth with a radius of 6371752.3 meters.
    """
    try:
        flip_coord = [coord[1], coord[0]]
        x = (
            6371752.3
            * (
                np.radians(flip_coord[1])
                - np.radians(relative_coord[1])
            )
            * np.cos(np.radians(relative_coord[0]))
        )
        y = 6371752.3 * (
            np.radians(flip_coord[0]) - np.radians(relative_coord[0])
        )

        return [x, y]
    except Exception as e:
        raise ValueError(
            "Error converting coordinates to Cartesian."
        ) from e
