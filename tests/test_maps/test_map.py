from UAVpathfinder.maps.map import Map

new_map = Map(
    start_coord=[48.14717458684, 11.543753585147323],
    end_coord=[48.14869704375871, 11.542849202512738],
)


def test_map_generate_bbox() -> None:
    assert new_map.generate_bbox() == [
        [48.14717458684 - new_map.buffer, 11.542849202512738 - new_map.buffer],
        [48.14869704375871 + new_map.buffer, 11.543753585147323 + new_map.buffer],
    ]


def test_map_get_2D_building_graph() -> None:
    assert new_map.get_2D_building_graph() == 0
