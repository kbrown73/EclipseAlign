from eclipse_align.files import numeric_sort_key


def test_numeric_sort_key_orders_camera_sequence_numbers():
    names = ["IMG_100.exr", "IMG_20.exr", "IMG_3.exr"]

    assert sorted(names, key=numeric_sort_key) == ["IMG_3.exr", "IMG_20.exr", "IMG_100.exr"]
