import numpy as np

from eclipse_align.detect import DetectionConfig, adjusted_circle_confidence, detect_image, robust_circle_fit


def test_robust_circle_fit_recovers_circle_with_outliers():
    angles = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    circle = np.column_stack([100.0 + 40.0 * np.cos(angles), 80.0 + 40.0 * np.sin(angles)])
    outliers = np.array([[10.0, 10.0], [200.0, 10.0], [200.0, 180.0], [20.0, 170.0]])
    points = np.vstack([circle, outliers]).astype(np.float32)

    result = robust_circle_fit(points, DetectionConfig(ransac_iterations=300, seed=5))

    assert result is not None
    cx, cy, radius, confidence = result
    assert abs(cx - 100.0) < 1.0
    assert abs(cy - 80.0) < 1.0
    assert abs(radius - 40.0) < 1.0
    assert confidence > 0.9


def test_adjusted_circle_confidence_rejects_high_residual_fit():
    confidence = adjusted_circle_confidence(
        1.0,
        residual_median_px=46.8,
        limb_support_fraction=1.0,
        config=DetectionConfig(),
    )

    assert confidence < 0.35


def test_adjusted_circle_confidence_keeps_clean_supported_fit():
    confidence = adjusted_circle_confidence(
        0.82,
        residual_median_px=1.0,
        limb_support_fraction=0.9,
        config=DetectionConfig(),
    )

    assert confidence == 0.82


def test_detect_image_ignores_frame_border_when_fitting_clipped_disk():
    height = 200
    width = 200
    center_x = 20.0
    center_y = 160.0
    radius = 50.0
    yy, xx = np.mgrid[:height, :width]
    disk = ((xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2).astype(np.float32)
    image = np.dstack([disk, disk, disk])

    detection = detect_image(
        image,
        "clipped.exr",
        DetectionConfig(work_max_dim=200, threshold=0.5, ransac_iterations=300, seed=8),
    )

    assert detection.status == "clipped"
    assert "touches_left_edge" in detection.flags
    assert "touches_bottom_edge" in detection.flags
    assert detection.center_x is not None
    assert detection.center_y is not None
    assert detection.radius is not None
    assert abs(detection.center_x - center_x) < 2.0
    assert abs(detection.center_y - center_y) < 2.0
    assert abs(detection.radius - radius) < 2.0
