# from UAVpathfinder.maps import map
from map import Map

new_map = Map(
    start_coord=[48.14717458684, 11.543753585147323],
    end_coord=[48.14869704375871, 11.542849202512738],
)

new_map.plot_building_2D()

print(new_map.generate_building_faces())
