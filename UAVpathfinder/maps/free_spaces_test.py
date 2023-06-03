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

new_graph.plot_network_grid()
