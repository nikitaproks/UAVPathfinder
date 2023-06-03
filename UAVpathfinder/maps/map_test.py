from map import Map
from utils import coord_to_cart
from utils import haversine_distance

print(
    haversine_distance(
        [48.15454749016991, 11.544871334811818], [48.15633324537993, 11.545783285821432]
    )
)

print(
    coord_to_cart(
        [48.15454749016991, 11.544871334811818], [48.15633324537993, 11.545783285821432]
    )
    + [0]
)


new_map = Map(
    start_coord=[48.15454749016991, 11.544871334811818],
    end_coord=[48.15633324537993, 11.545783285821432],
)
faces = new_map.generate_building_faces()
print(len(faces))
print(faces[0])
