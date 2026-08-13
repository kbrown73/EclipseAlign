from eclipse_align.models import FrameDetection
from eclipse_align.render import (
    add_circle_alpha,
    centered_fixed_square_crop,
    centered_square_crop,
    compute_centered_square_crop,
    compute_safe_crop,
    crop_with_padding,
    parse_manual_crop,
)


def test_compute_safe_crop_intersects_translated_bounds():
    detections = [
        FrameDetection("a.exr", 100, 80, translation_x=10.2, translation_y=-3.4, status="ok", confidence=1),
        FrameDetection("b.exr", 100, 80, translation_x=-5.1, translation_y=7.8, status="ok", confidence=1),
    ]

    crop = compute_safe_crop(detections, mode="safe")

    assert crop is not None
    assert crop.left == 11
    assert crop.top == 8
    assert crop.right == 94
    assert crop.bottom == 76


def test_safe_visible_crop_ignores_clipped_frames():
    detections = [
        FrameDetection(
            "a.exr",
            100,
            80,
            center_x=50,
            center_y=40,
            translation_x=0,
            translation_y=0,
            status="ok",
            confidence=1,
        ),
        FrameDetection(
            "b.exr",
            100,
            80,
            center_x=50,
            center_y=60,
            translation_x=0,
            translation_y=-20,
            status="clipped",
            confidence=1,
        ),
    ]

    crop = compute_safe_crop(detections, mode="safe-visible")

    assert crop is not None
    assert crop.left == 0
    assert crop.top == 0
    assert crop.right == 100
    assert crop.bottom == 80


def test_parse_manual_crop():
    crop = parse_manual_crop("1920x1080+100+50")

    assert crop.left == 100
    assert crop.top == 50
    assert crop.width == 1920
    assert crop.height == 1080


def test_centered_square_crop_uses_smallest_target_margin():
    bounds = parse_manual_crop("2000x1000+100+200")

    crop = centered_square_crop(bounds, target_x=900, target_y=600)

    assert crop is not None
    assert crop.width == 800
    assert crop.height == 800
    assert crop.left == 500
    assert crop.top == 200


def test_centered_square_crop_rejects_too_small_eclipse_margin():
    bounds = parse_manual_crop("2000x1000+100+200")

    crop = centered_square_crop(bounds, target_x=900, target_y=600, min_side=900)

    assert crop is None


def test_centered_fixed_square_crop_centers_on_canvas():
    detections = [FrameDetection("a.exr", 1000, 800)]

    crop = centered_fixed_square_crop(detections, 600)

    assert crop.left == 200
    assert crop.top == 100
    assert crop.width == 600
    assert crop.height == 600


def test_compute_centered_square_crop_adds_margin():
    detections = [
        FrameDetection(
            "a.exr",
            1000,
            1000,
            center_x=500,
            center_y=300,
            radius=100,
            translation_x=0,
            translation_y=200,
            status="ok",
            confidence=1,
        )
    ]

    crop = compute_centered_square_crop(detections, margin=25)

    assert crop is not None
    assert crop.width == 650
    assert crop.height == 650
    assert crop.left == 175
    assert crop.top == 175


def test_crop_with_padding_allows_crop_outside_canvas():
    import numpy as np

    image = np.ones((4, 4, 1), dtype=np.float32)
    crop = parse_manual_crop("6x6+-1+-1")

    output = crop_with_padding(image, crop)

    assert output.shape == (6, 6, 1)
    assert output[0, 0, 0] == 0
    assert output[1:5, 1:5, 0].sum() == 16


def test_add_circle_alpha_appends_filled_disk_after_crop():
    import numpy as np

    image = np.zeros((6, 6, 3), dtype=np.float32)
    crop = parse_manual_crop("6x6+2+2")
    detection = FrameDetection("a.exr", 10, 10, radius=2)

    output = add_circle_alpha(image, detection, crop)

    assert output.shape == (6, 6, 4)
    assert output[3, 3, 3] == 1
    assert output[0, 0, 3] == 0
    assert output[:, :, :3].sum() == 0


def test_add_circle_alpha_replaces_existing_alpha_channel():
    import numpy as np

    image = np.zeros((5, 5, 4), dtype=np.float32)
    image[:, :, 3] = 0.25
    detection = FrameDetection("a.exr", 5, 5, radius=1)

    output = add_circle_alpha(image, detection)

    assert output.shape == (5, 5, 4)
    assert output[2, 2, 3] == 1
    assert output[0, 0, 3] == 0
