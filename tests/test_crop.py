from eclipse_align.models import FrameDetection
from eclipse_align.render import (
    add_circle_alpha,
    centered_fixed_square_crop,
    centered_square_crop,
    compute_centered_square_crop,
    compute_safe_crop,
    correct_dust_pixels,
    crop_with_padding,
    dust_correction_mask_for_frame,
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


def test_add_circle_alpha_uses_accepted_ellipse_metadata():
    import numpy as np

    image = np.zeros((9, 9, 3), dtype=np.float32)
    detection = FrameDetection(
        "a.exr",
        9,
        9,
        radius=4,
        ellipse_major_radius=3,
        ellipse_minor_radius=1,
        ellipse_angle_deg=0,
    )

    output = add_circle_alpha(image, detection)

    assert output[4, 7, 3] == 1
    assert output[6, 5, 3] == 0


def test_correct_dust_pixels_repairs_masked_float_pixels_only():
    import numpy as np

    image = np.ones((9, 9, 3), dtype=np.float32)
    image[4, 4, :] = 0.0
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True

    output = correct_dust_pixels(image, mask, inpaint_radius=2.0, dilation_px=0)

    assert output.shape == image.shape
    assert output[4, 4, 0] > 0.5
    np.testing.assert_array_equal(output[0, 0], image[0, 0])


def test_correct_dust_pixels_resizes_mask_to_source_shape():
    import numpy as np

    image = np.ones((8, 8, 1), dtype=np.float32)
    image[4:6, 4:6, 0] = 0.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 2] = True

    output = correct_dust_pixels(image, mask, inpaint_radius=2.0, dilation_px=0)

    assert output[4, 4, 0] > 0.5
    assert output[5, 5, 0] > 0.5


def test_correct_dust_pixels_clamps_inpaint_overshoot(monkeypatch):
    import numpy as np
    import eclipse_align.render as render_module

    def overshooting_inpaint(channel, mask, radius, method):
        return np.full(channel.shape, 5.0, dtype=np.float32)

    image = np.ones((11, 11, 1), dtype=np.float32) * 0.5
    image[5, 5, 0] = 0.0
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    monkeypatch.setattr(render_module.cv2, "inpaint", overshooting_inpaint)

    output = correct_dust_pixels(image, mask, inpaint_radius=2.0, dilation_px=0)

    assert output[5, 5, 0] == 0.5


def test_dust_correction_mask_for_frame_uses_fitted_solar_disk():
    import numpy as np

    dust_mask = np.ones((9, 9), dtype=bool)
    detection = FrameDetection("a.exr", 9, 9, center_x=4, center_y=4, radius=2)

    mask = dust_correction_mask_for_frame(dust_mask, (9, 9), detection)

    assert mask[4, 4]
    assert mask[4, 6]
    assert not mask[4, 7]
    assert not mask[0, 0]


def test_correct_dust_pixels_clips_dilated_mask_to_limit():
    import numpy as np

    image = np.ones((9, 9, 1), dtype=np.float32)
    image[4, 5, 0] = 0.0
    image[4, 7, 0] = 0.0
    dust_mask = np.zeros((9, 9), dtype=bool)
    dust_mask[4, 5] = True
    limit_mask = np.zeros((9, 9), dtype=bool)
    limit_mask[:, :7] = True

    output = correct_dust_pixels(image, dust_mask, inpaint_radius=2.0, dilation_px=3, limit_mask=limit_mask)

    assert output[4, 5, 0] > 0.5
    assert output[4, 7, 0] == 0.0
