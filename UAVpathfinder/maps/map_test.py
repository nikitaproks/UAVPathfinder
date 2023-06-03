# from UAVpathfinder.maps import map
import map

new_map = map.Map(
    start_coord=[48.16027671693242, 11.541244989434498],
    end_coord=[48.14533143268084, 11.564247612547494],
    xy_resolution=0.25,  # in meters
    z_resolution=3.0,
)

new_map.plot_building_2D()

print(new_map.generate_building_faces())
