import numpy as np

from eclipse_align.detect import DetectionConfig, adjusted_circle_confidence, robust_circle_fit


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
