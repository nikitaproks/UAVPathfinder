from typing import List
import math


def haversine_distance(
    coord_1: List[float], coord_2: List[float]
) -> float:
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
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371752.3 * c  # Earth's radius in kilometers

    return abs(distance)


def coord_to_cart(
    relative_coord: List[float], coord: List[float]
) -> List[float]:
    # coord and relative coord are flipped!
    flip_coord = [coord[1], coord[0]]
    distance = haversine_distance(flip_coord, relative_coord)
    x = (
        6371752.3
        * (
            math.radians(flip_coord[1])
            - math.radians(relative_coord[1])
        )
        * math.cos(math.radians(relative_coord[0]))
    )
    y = 6371752.3 * (
        math.radians(flip_coord[0]) - math.radians(relative_coord[0])
    )
    print(" ")
    print(distance)
    print([x, y])
    return [x, y]
