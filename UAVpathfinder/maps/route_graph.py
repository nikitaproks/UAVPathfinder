from pydantic import BaseModel
from UAVpathfinder.maps.map import Map
import networkx as nx


class RouteGraph(BaseModel):
    xy_resolution: float  # in meters
    z_resolution: float  # in meters
