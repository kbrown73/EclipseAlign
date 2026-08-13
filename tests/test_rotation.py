import numpy as np

from eclipse_align.cli import write_rotation_summary
from eclipse_align.models import FrameDetection
from eclipse_align.render import rotate_image_around_center
from eclipse_align.rotation import (
    BoundaryRotation,
    RotationConfig,
    assign_segments,
    detect_reframe_boundaries,
    estimate_rotations,
    initialize_rotation_metadata,
    registration_confidence,
)


def test_detect_reframe_boundaries_from_center_jumps():
    detections = [
        FrameDetection("a.exr", 100, 100, center_x=10, center_y=10),
        FrameDetection("b.exr", 100, 100, center_x=12, center_y=10),
        FrameDetection("c.exr", 100, 100, center_x=90, center_y=90),
    ]

    assert detect_reframe_boundaries(detections, threshold_px=50) == [2]


def test_detect_reframe_boundaries_merges_tiny_segments():
    detections = [
        FrameDetection("a.exr", 100, 100, center_x=0, center_y=0),
        FrameDetection("b.exr", 100, 100, center_x=100, center_y=0),
        FrameDetection("c.exr", 100, 100, center_x=200, center_y=0),
        FrameDetection("d.exr", 100, 100, center_x=202, center_y=0),
    ]

    assert detect_reframe_boundaries(detections, threshold_px=50, min_segment_length=2) == [2]


def test_assign_segments_increments_at_boundaries():
    detections = [FrameDetection(f"{idx}.exr", 100, 100) for idx in range(5)]

    assign_segments(detections, [2, 4])

    assert [d.segment_id for d in detections] == [0, 0, 1, 1, 2]


def test_rotate_image_around_center_preserves_shape():
    image = np.zeros((20, 20, 1), dtype=np.float32)
    image[5:15, 9:11, 0] = 1.0

    rotated = rotate_image_around_center(image, 10)

    assert rotated.shape == image.shape
    assert rotated.max() > 0


def test_initialize_rotation_metadata_resets_fields():
    detections = [
        FrameDetection(
            "a.exr",
            100,
            100,
            rotation_deg=12,
            rotation_confidence=1,
            rotation_source="registration",
        )
    ]

    initialize_rotation_metadata(detections)

    assert detections[0].rotation_deg == 0
    assert detections[0].rotation_confidence == 0
    assert detections[0].rotation_source == "none"


def test_rotation_config_defaults_are_conservative():
    config = RotationConfig()

    assert config.min_boundary_confidence >= 0.6


def test_registration_confidence_accepts_broad_clear_zero_improvement():
    confidence = registration_confidence(best_score=0.78, next_score=0.779, zero_score=0.685)

    assert confidence >= RotationConfig().min_boundary_confidence


def test_registration_confidence_rejects_tiny_zero_improvement():
    confidence = registration_confidence(best_score=0.92, next_score=0.919, zero_score=0.909)

    assert confidence < RotationConfig().min_boundary_confidence


def synthetic_drift_detections(angles: list[float], frames_per_segment: int = 10) -> list[FrameDetection]:
    detections = []
    for segment_id, angle in enumerate(angles):
        radians = np.radians(angle)
        origin_x = 1000.0 + segment_id * 5000.0
        origin_y = 1000.0 + segment_id * 3000.0
        for offset in range(frames_per_segment):
            detections.append(
                FrameDetection(
                    f"{segment_id}_{offset}.exr",
                    10000,
                    10000,
                    center_x=origin_x + offset * 30.0 * np.cos(radians),
                    center_y=origin_y + offset * 30.0 * np.sin(radians),
                    confidence=1.0,
                    status="ok",
                )
            )
    return detections


def drift_rotation_config() -> RotationConfig:
    return RotationConfig(
        jump_threshold_px=1000.0,
        drift_min_points=6,
        drift_min_distance_px=100.0,
        drift_jump_threshold_degrees=4.0,
        registration_fallback=False,
    )


def test_estimate_rotations_from_persistent_drift_angle_jump():
    detections = synthetic_drift_detections([10.0, 12.0, 24.0, 26.0])

    boundaries = estimate_rotations([], detections, config=drift_rotation_config())

    assert [round(boundary.delta_deg, 1) for boundary in boundaries] == [0.0, -10.0, 0.0]
    assert [round(detections[idx * 10].rotation_deg, 1) for idx in range(4)] == [
        0.0,
        0.0,
        -10.0,
        -10.0,
    ]
    assert detections[20].rotation_source == "drift"


def test_estimate_rotations_from_temporary_drift_angle_jump():
    detections = synthetic_drift_detections([10.0, 12.0, 24.0, 16.0])

    boundaries = estimate_rotations([], detections, config=drift_rotation_config())

    assert [round(boundary.delta_deg, 1) for boundary in boundaries] == [0.0, -10.0, 10.0]
    assert [round(detections[idx * 10].rotation_deg, 1) for idx in range(4)] == [
        0.0,
        0.0,
        -10.0,
        0.0,
    ]


def test_write_rotation_summary_csv(tmp_path):
    detections = synthetic_drift_detections([10.0, 12.0], frames_per_segment=3)
    assign_segments(detections, [3])
    for detection in detections[3:]:
        detection.rotation_deg = 2.5
        detection.rotation_confidence = 0.75
        detection.rotation_source = "registration"
    boundary = BoundaryRotation(
        boundary_index=3,
        from_segment=0,
        to_segment=1,
        delta_deg=2.5,
        confidence=0.75,
        score=0.9,
        source="registration",
        from_drift_angle_deg=10.0,
        to_drift_angle_deg=12.0,
        raw_drift_delta_deg=2.0,
        expected_drift_delta_deg=0.0,
    )

    path = tmp_path / "rotation_summary.csv"
    write_rotation_summary(path, detections, [boundary])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("boundary_index,from_segment,to_segment")
    assert "0_2.exr" in lines[1]
    assert "1_0.exr" in lines[1]
    assert "2.5" in lines[1]
    assert "registration" in lines[1]
